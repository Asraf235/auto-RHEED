"""Single-user background inference jobs for the Flask consumer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid

import numpy as np

from rheed_core import AnalysisStore, StoredInferenceRun
from rheed_core.inference import (
    InferenceCancelled,
    InferenceRunResult,
    ModelSpec,
    cluster_embeddings,
    run_embedding_analysis_with_artifacts,
    run_inference,
)


@dataclass
class AIJob:
    id: str
    dataset_version: int
    model: ModelSpec
    status: str = "queued"
    completed_frames: int = 0
    total_frames: int = 0
    error: str | None = None
    stored_result: StoredInferenceRun | None = None
    result_summary: dict | None = None
    analysis_available: bool = False
    embedding_analysis_summaries: dict[str, dict] = field(default_factory=dict)
    storage_dataset_dir: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def public_dict(self, current_dataset_version: int) -> dict:
        payload = {
            "job_id": self.id,
            "status": self.status,
            "completed_frames": self.completed_frames,
            "total_frames": self.total_frames,
            "progress": (self.completed_frames / self.total_frames
                         if self.total_frames else 0.0),
            "error": self.error,
            "dataset_version": self.dataset_version,
            "stale": self.dataset_version != current_dataset_version,
            "created_utc": self.created_utc,
            "model": self.model.public_dict(),
        }
        if self.result_summary is not None:
            payload["result"] = dict(self.result_summary)
        if self.analysis_available:
            payload["analysis_available"] = True
        if self.embedding_analysis_summaries:
            payload["embedding_analyses"] = dict(self.embedding_analysis_summaries)
        if self.stored_result is not None:
            payload["saved_to"] = str(self.stored_result.path)
        return payload

    def load_result(self) -> InferenceRunResult:
        if self.stored_result is None:
            raise ValueError("Inference result is not available")
        return self.stored_result.load_result()

    def load_analysis(self) -> dict | None:
        if self.stored_result is None:
            return None
        return self.stored_result.load_analysis("pca-kmeans")

    def load_embedding_analysis(self, adapter_name: str) -> dict | None:
        if self.stored_result is None:
            return None
        return self.stored_result.load_analysis(f"embedding-{adapter_name}")

    def load_embedding_analysis_array(
        self,
        adapter_name: str,
        artifact_name: str,
    ) -> np.ndarray | None:
        if self.stored_result is None:
            return None
        return self.stored_result.load_analysis_array(
            f"embedding-{adapter_name}-{artifact_name}"
        )

    def export_analysis(self) -> dict | None:
        """Combine optional analyses without changing the clustering schema."""
        analysis = self.load_analysis()
        payload = dict(analysis) if analysis is not None else {}
        if self.embedding_analysis_summaries:
            payload["embedding_analyses"] = {
                name: self.load_embedding_analysis(name)
                for name in self.embedding_analysis_summaries
            }
        return payload or None


class AIJobManager:
    def __init__(self, analysis_store: AnalysisStore, max_completed_jobs: int = 6):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rheed-ai")
        self._jobs: dict[str, AIJob] = {}
        self._lock = threading.RLock()
        self._max_completed_jobs = max_completed_jobs
        self._analysis_store = analysis_store

    def submit(
        self,
        *,
        frames: np.ndarray,
        timestamps: np.ndarray | None,
        dataset_version: int,
        model: ModelSpec,
        device: str,
        batch_size: int,
        stride: int,
    ) -> AIJob:
        job = AIJob(
            id=uuid.uuid4().hex,
            dataset_version=dataset_version,
            model=model,
            total_frames=len(range(0, len(frames), stride)),
            storage_dataset_dir=self._analysis_store.dataset_dir,
        )
        with self._lock:
            self._trim_completed()
            self._jobs[job.id] = job
        self._executor.submit(
            self._run_job,
            job,
            frames,
            timestamps,
            device,
            batch_size,
            stride,
        )
        return job

    def _run_job(self, job, frames, timestamps, device, batch_size, stride):
        with self._lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                return
            job.status = "running"

        def progress(done: int, total: int):
            with self._lock:
                job.completed_frames = done
                job.total_frames = total

        try:
            result = run_inference(
                frames,
                timestamps,
                job.model,
                device=device,
                batch_size=batch_size,
                stride=stride,
                progress=progress,
                cancelled=job.cancel_event.is_set,
            )
        except InferenceCancelled:
            with self._lock:
                job.status = "cancelled"
            return
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
            return

        try:
            stored = self._analysis_store.save_inference(
                job.id,
                result,
                dataset_dir=job.storage_dataset_dir,
            )
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = f"Inference completed but could not be saved: {exc}"
            return

        with self._lock:
            job.stored_result = stored
            job.result_summary = stored.compact_summary()
            job.completed_frames = job.total_frames
            job.status = "completed"

    def get(self, job_id: str) -> AIJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown AI job: {job_id}")
            return job

    def restore(self, artifact_id: str, *, dataset_version: int) -> AIJob:
        """Re-register a completed file-backed inference run without rerunning it."""
        stored = self._analysis_store.open_inference(artifact_id)
        manifest = stored.manifest()
        job_id = str(manifest.get("job_id") or stored.path.name.split("_", 1)[0])
        model = ModelSpec.from_dict(manifest["model"])
        analysis_names = stored.analysis_names()
        embedding_summaries: dict[str, dict] = {}
        for name in analysis_names:
            if not name.startswith("embedding-"):
                continue
            adapter_name = name[len("embedding-"):]
            analysis = stored.load_analysis(name) or {}
            embedding_summaries[adapter_name] = {
                "task": analysis.get("task"),
                "n_changepoints": len(analysis.get("changepoints", [])),
            }
        job = AIJob(
            id=job_id,
            dataset_version=dataset_version,
            model=model,
            status="completed",
            completed_frames=len(stored.frame_indices),
            total_frames=len(stored.frame_indices),
            stored_result=stored,
            result_summary=stored.compact_summary(),
            analysis_available="pca-kmeans" in analysis_names,
            embedding_analysis_summaries=embedding_summaries,
            storage_dataset_dir=self._analysis_store.dataset_dir,
            created_utc=str(manifest.get("created_utc") or datetime.now(timezone.utc).isoformat()),
        )
        with self._lock:
            self._trim_completed()
            self._jobs[job.id] = job
        return job

    def cancel(self, job_id: str) -> AIJob:
        job = self.get(job_id)
        job.cancel_event.set()
        with self._lock:
            if job.status == "queued":
                job.status = "cancelled"
        return job

    def analyze(self, job_id: str, **parameters) -> dict:
        job = self.get(job_id)
        if job.stored_result is None or job.status != "completed":
            raise ValueError("Inference must complete before PCA/K-means analysis")
        result = job.load_result()
        analysis = cluster_embeddings(result, **parameters).to_dict(
            frame_indices=result.frame_indices,
            timestamps=result.timestamps,
        )
        job.stored_result.save_analysis("pca-kmeans", analysis)
        with self._lock:
            job.analysis_available = True
        return analysis

    def analyze_embeddings(
        self,
        job_id: str,
        adapter_name: str,
        options: dict | None = None,
    ) -> dict:
        job = self.get(job_id)
        if job.stored_result is None or job.status != "completed":
            raise ValueError("Inference must complete before embedding analysis")
        result = job.load_result()
        analysis, array_artifacts = run_embedding_analysis_with_artifacts(
            result, adapter_name, options
        )
        for artifact_name, array in array_artifacts.items():
            job.stored_result.save_analysis_array(
                f"embedding-{adapter_name}-{artifact_name}",
                array,
            )
        job.stored_result.save_analysis(f"embedding-{adapter_name}", analysis)
        with self._lock:
            job.embedding_analysis_summaries[adapter_name] = {
                "task": analysis.get("task"),
                "n_changepoints": len(analysis.get("changepoints", [])),
            }
        return analysis

    def _trim_completed(self):
        terminal = [job for job in self._jobs.values()
                    if job.status in {"completed", "cancelled", "failed"}]
        terminal.sort(key=lambda job: job.created_utc)
        for job in terminal[:-self._max_completed_jobs]:
            self._jobs.pop(job.id, None)
