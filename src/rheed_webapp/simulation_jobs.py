"""Single-user background simulation jobs for the Flask consumer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import BinaryIO, Callable, Mapping
import uuid

from rheed_core.simulation import (
    SimulationCancelled,
    SimulationRunResult,
    SimulationSpec,
    SimulationStore,
    StoredSimulationRun,
    run_simulation,
)


@dataclass
class SimulationJob:
    id: str
    simulation: SimulationSpec
    device: str
    stored_run: StoredSimulationRun
    input_paths: dict[str, Path]
    status: str = "queued"
    completed_steps: int = 0
    total_steps: int = 1
    error: str | None = None
    auto_export_error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def public_dict(self) -> dict:
        return {
            "job_id": self.id,
            "run_id": self.id,
            "status": self.status,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "progress": self.completed_steps / self.total_steps if self.total_steps else 0.0,
            "error": self.error,
            "auto_export_error": self.auto_export_error,
            "created_utc": self.created_utc,
            "simulation": self.simulation.public_dict(),
            "saved_to": str(self.stored_run.path),
        }


class SimulationJobManager:
    def __init__(self, store: SimulationStore, max_completed_jobs: int = 12):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rheed-simulation")
        self._jobs: dict[str, SimulationJob] = {}
        self._lock = threading.RLock()
        self._store = store
        self._max_completed_jobs = max_completed_jobs
        self._completed_callback: (
            Callable[[StoredSimulationRun, SimulationRunResult], None] | None
        ) = None

    def set_completed_callback(
        self,
        callback: Callable[[StoredSimulationRun, SimulationRunResult], None] | None,
    ) -> None:
        """Set optional file-export work performed after a result is safely saved."""

        with self._lock:
            self._completed_callback = callback

    def submit(
        self,
        *,
        simulation: SimulationSpec,
        device: str,
        uploads: Mapping[str, tuple[str, BinaryIO]],
    ) -> SimulationJob:
        job_id = uuid.uuid4().hex
        stored_run, input_paths = self._store.create_run(job_id, simulation, uploads)
        job = SimulationJob(
            id=job_id,
            simulation=simulation,
            device=device,
            stored_run=stored_run,
            input_paths=input_paths,
        )
        with self._lock:
            self._trim_completed()
            self._jobs[job.id] = job
        self._executor.submit(self._run_job, job)
        return job

    def _run_job(self, job: SimulationJob) -> None:
        with self._lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                self._store.update_status(job.id, status="cancelled")
                return
            job.status = "running"
            self._store.update_status(job.id, status="running")

        def progress(completed: int, total: int) -> None:
            with self._lock:
                job.completed_steps = completed
                job.total_steps = total
                self._store.update_status(
                    job.id,
                    completed_steps=completed,
                    total_steps=total,
                )

        try:
            result = run_simulation(
                job.input_paths,
                job.simulation,
                device=job.device,
                progress=progress,
                cancelled=job.cancel_event.is_set,
            )
        except SimulationCancelled:
            with self._lock:
                job.status = "cancelled"
                self._store.update_status(job.id, status="cancelled")
            return
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                self._store.update_status(job.id, status="failed", error=str(exc))
            return

        try:
            stored = self._store.save_result(job.id, result)
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = f"Simulation completed but could not be saved: {exc}"
                self._store.update_status(job.id, status="failed", error=job.error)
            return

        callback = self._completed_callback
        if callback is not None:
            try:
                callback(stored, result)
            except Exception as exc:
                # The scientific result is already complete and file-backed;
                # an optional presentation export must not invalidate it.
                job.auto_export_error = str(exc)
                self._store.update_status(job.id, auto_export_error=job.auto_export_error)

        with self._lock:
            job.stored_run = stored
            job.completed_steps = job.total_steps
            job.status = "completed"

    def get(self, job_id: str) -> SimulationJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown simulation job: {job_id}")
            return job

    def status(self, run_id: str) -> dict:
        try:
            return self.get(run_id).public_dict()
        except KeyError:
            manifest = self._store.open(run_id).manifest()
            completed = int(manifest.get("completed_steps", 0))
            total = int(manifest.get("total_steps", 1))
            return {
                "job_id": run_id,
                "run_id": run_id,
                "status": manifest.get("status"),
                "completed_steps": completed,
                "total_steps": total,
                "progress": completed / total if total else 0.0,
                "error": manifest.get("error"),
                "auto_export_error": manifest.get("auto_export_error"),
                "created_utc": manifest.get("created_utc"),
                "simulation": manifest.get("simulation"),
                "saved_to": str(self._store.open(run_id).path),
            }

    def cancel(self, job_id: str) -> SimulationJob:
        job = self.get(job_id)
        job.cancel_event.set()
        with self._lock:
            if job.status == "queued":
                job.status = "cancelled"
                self._store.update_status(job.id, status="cancelled")
        return job

    def _trim_completed(self) -> None:
        terminal = [
            job for job in self._jobs.values()
            if job.status in {"completed", "cancelled", "failed"}
        ]
        terminal.sort(key=lambda job: job.created_utc)
        for job in terminal[:-self._max_completed_jobs]:
            self._jobs.pop(job.id, None)
