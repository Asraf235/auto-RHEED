"""Persistent, local-first storage for Auto RHEED analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
import csv
import gzip
import json
import os
from pathlib import Path
import re
import threading
import uuid
from typing import Any

import numpy as np

from .inference.types import InferenceRunResult, ModelSpec


_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _safe_part(value: str, fallback: str) -> str:
    safe = _SAFE_PART.sub("-", str(value)).strip(".-")
    return safe[:80] or fallback


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex}.json"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _is_csv_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, np.generic))


def _csv_series(value: Any) -> list[Any] | None:
    """Return a one-dimensional scalar sequence suitable for a CSV column."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return list(value) if all(_is_csv_scalar(item) for item in value) else None


def _csv_matrix(value: Any) -> list[list[Any]] | None:
    """Return a rectangular two-dimensional scalar sequence for CSV columns."""
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or not value.size:
            return None
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    rows = [list(row) for row in value if isinstance(row, (list, tuple))]
    if len(rows) != len(value) or not rows:
        return None
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None
    return rows if all(_is_csv_scalar(cell) for row in rows for cell in row) else None


def _atomic_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex}.csv"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_csv_exports(target_json: Path, result: dict[str, Any]) -> list[Path]:
    """Write human-readable tables found in a classical result.

    Direct one-dimensional arrays become columns in the primary CSV. Lists
    of row dictionaries (for example manual measurements and FWHM records)
    become a separate named CSV. Nested/non-tabular metadata remains in the
    companion JSON file.
    """
    exports: list[Path] = []
    direct: dict[str, list[Any]] = {}
    series_rows = result.get("series")
    if isinstance(series_rows, list):
        for index, row in enumerate(series_rows):
            if not isinstance(row, dict):
                continue
            raw = _csv_series(row.get("intensities"))
            timestamps = _csv_series(row.get("timestamps"))
            if raw is None or timestamps is None:
                continue
            if "time_s" not in direct:
                direct["time_s"] = timestamps
            label = _safe_part(str(row.get("label") or f"series-{index + 1}"), f"series-{index + 1}")
            if f"{label}_raw" in direct:
                label = f"{label}-{index + 1}"
            direct[f"{label}_raw"] = raw
            smoothed = _csv_series(row.get("ema_intensities"))
            if smoothed is not None:
                alpha = row.get("ema_alpha")
                try:
                    suffix = f"_alpha_{float(alpha):.2f}" if alpha is not None else ""
                except (TypeError, ValueError):
                    suffix = ""
                direct[f"{label}_ema{suffix}"] = smoothed
    for key, value in result.items():
        series = _csv_series(value)
        if series is not None:
            direct[str(key)] = series
            continue
        matrix = _csv_matrix(value)
        if matrix is None:
            continue
        for column in range(len(matrix[0])):
            direct[f"{key}_{column + 1}"] = [row[column] for row in matrix]
    if direct:
        row_count = max(len(values) for values in direct.values())
        headers = list(direct)
        rows = [
            [direct[header][index] if index < len(direct[header]) else "" for header in headers]
            for index in range(row_count)
        ]
        csv_path = target_json.with_suffix(".csv")
        _atomic_csv(csv_path, headers, rows)
        exports.append(csv_path)

    for key, value in result.items():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        if not all(isinstance(row, dict) for row in value):
            continue
        headers: list[str] = []
        for row in value:
            for field, cell in row.items():
                if field not in headers and _is_csv_scalar(cell):
                    headers.append(str(field))
        if not headers:
            continue
        rows = [
            [row.get(header, "") if _is_csv_scalar(row.get(header)) else "" for header in headers]
            for row in value
        ]
        csv_path = target_json.with_name(
            f"{target_json.stem}-{_safe_part(str(key), 'table')}.csv"
        )
        _atomic_csv(csv_path, headers, rows)
        exports.append(csv_path)
    return exports


def _remove_stale_csv_exports(target_json: Path, current: list[Path]) -> None:
    """Remove superseded CSV companions after a successful replacement."""
    keep = {path.resolve() for path in current}
    candidates = [target_json.with_suffix(".csv")]
    candidates.extend(target_json.parent.glob(f"{target_json.stem}-*.csv"))
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() not in keep:
            candidate.unlink()


def _write_artifact_index(
    index_path: Path,
    entry: dict[str, Any],
    *,
    replace: bool = False,
) -> None:
    """Append a new artifact or atomically upsert one stable artifact entry."""
    if not replace:
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return

    entries: list[dict[str, Any]] = []
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            entries = [json.loads(line) for line in stream if line.strip()]
    entries = [item for item in entries if item.get("artifact_id") != entry["artifact_id"]]
    entries.append(entry)
    temporary = index_path.parent / f".tmp-{uuid.uuid4().hex}.jsonl"
    with temporary.open("w", encoding="utf-8") as stream:
        for item in entries:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    os.replace(temporary, index_path)


@dataclass(frozen=True)
class StoredInferenceRun:
    """Reference to one file-backed inference result."""

    path: Path

    @cached_property
    def frame_indices(self) -> np.ndarray:
        # These one-dimensional indexes are small and accessed for every frame.
        # Loading them into memory avoids repeatedly opening the files without
        # keeping Windows file handles alive for the lifetime of a web job.
        return np.load(self.path / "frame_indices.npy", allow_pickle=False)

    @cached_property
    def timestamps(self) -> np.ndarray:
        return np.load(self.path / "timestamps_s.npy", allow_pickle=False)

    @property
    def embeddings(self) -> np.ndarray | None:
        path = self.path / "embeddings.npy"
        # Embeddings may be large, so expose a short-lived memory map rather
        # than retaining the full array or an open handle on this object.
        return np.load(path, mmap_mode="r", allow_pickle=False) if path.is_file() else None

    @cached_property
    def frame_offsets(self) -> np.ndarray | None:
        path = self.path / "frame_offsets.npy"
        return np.load(path, allow_pickle=False) if path.is_file() else None

    def manifest(self) -> dict[str, Any]:
        with (self.path / "manifest.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def summary(self) -> dict[str, Any]:
        return dict(self.manifest()["summary"])

    def compact_summary(self) -> dict[str, Any]:
        manifest = self.manifest()
        return dict(manifest.get("compact_summary") or manifest["summary"])

    def load_result(self) -> InferenceRunResult:
        manifest = self.manifest()
        embeddings_path = self.path / "embeddings.npy"
        # Full-sequence analyses need every row. Load a transient plain array
        # so Windows can close the file promptly when the operation finishes.
        embeddings = (
            np.load(embeddings_path, allow_pickle=False)
            if embeddings_path.is_file() else None
        )
        frames = None
        if (self.path / "frame_results.jsonl").is_file():
            frames = [self._frame_at_position(i) for i in range(len(self.frame_indices))]
        return InferenceRunResult(
            model=ModelSpec.from_dict(manifest["model"]),
            task=manifest["task"],
            frame_indices=np.asarray(self.frame_indices),
            timestamps=np.asarray(self.timestamps),
            metadata=dict(manifest.get("metadata") or {}),
            embeddings=np.asarray(embeddings) if embeddings is not None else None,
            frames=frames,
        )

    def frame_result(self, native_frame_index: int) -> dict[str, Any] | None:
        position = int(np.searchsorted(self.frame_indices, native_frame_index))
        if (
            position >= len(self.frame_indices)
            or int(self.frame_indices[position]) != native_frame_index
        ):
            return None
        result: dict[str, Any] = {
            "frame_index": int(self.frame_indices[position]),
            "timestamp_s": float(self.timestamps[position]),
        }
        embeddings = self.embeddings
        if embeddings is not None:
            result["embedding"] = np.asarray(embeddings[position]).tolist()
        if (self.path / "frame_results.jsonl").is_file():
            result.update(self._frame_at_position(position))
        return result

    def _frame_at_position(self, position: int) -> dict[str, Any]:
        if self.frame_offsets is None:
            raise ValueError("This inference run has no per-frame result index")
        start = int(self.frame_offsets[position])
        length = int(self.frame_offsets[position + 1]) - start
        with (self.path / "frame_results.jsonl").open("rb") as stream:
            stream.seek(start)
            return json.loads(stream.read(length).decode("utf-8"))

    def save_analysis(self, name: str, payload: dict[str, Any]) -> Path:
        target = self.path / "analyses" / f"{_safe_part(name, 'analysis')}.json"
        normalized = _json_value(payload)
        _atomic_json(target, normalized)
        csv_paths = _write_csv_exports(target, normalized)
        _remove_stale_csv_exports(target, csv_paths)
        return target

    def load_analysis(self, name: str) -> dict[str, Any] | None:
        stem = _safe_part(name, "analysis")
        target = self.path / "analyses" / f"{stem}.json"
        if target.is_file():
            with target.open(encoding="utf-8") as stream:
                return json.load(stream)
        legacy = self.path / "analyses" / f"{stem}.json.gz"
        if legacy.is_file():
            with gzip.open(legacy, "rt", encoding="utf-8") as stream:
                return json.load(stream)
        return None

    def analysis_names(self) -> list[str]:
        analyses = self.path / "analyses"
        if not analyses.is_dir():
            return []
        names = {path.stem for path in analyses.glob("*.json")}
        names.update(path.name[:-8] for path in analyses.glob("*.json.gz"))
        return sorted(names)

    def save_analysis_array(self, name: str, array: np.ndarray) -> Path:
        """Atomically save one large numeric analysis artifact as NumPy data."""
        target = self.path / "analyses" / f"{_safe_part(name, 'analysis')}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        numeric = np.asarray(array)
        if numeric.dtype.hasobject:
            raise ValueError("Analysis arrays must not use the object dtype")
        temporary = target.parent / f".tmp-{uuid.uuid4().hex}.npy"
        with temporary.open("wb") as stream:
            np.save(stream, numeric, allow_pickle=False)
        os.replace(temporary, target)
        return target

    def load_analysis_array(
        self,
        name: str,
        *,
        mmap_mode: str | None = "r",
    ) -> np.ndarray | None:
        """Open one separately stored numeric analysis artifact."""
        target = self.path / "analyses" / f"{_safe_part(name, 'analysis')}.npy"
        if not target.is_file():
            return None
        return np.load(target, mmap_mode=mmap_mode, allow_pickle=False)


class AnalysisStore:
    """Thread-safe store with one active dataset workspace."""

    def __init__(self, root: str | Path):
        self._root = Path(root).expanduser().resolve()
        self._dataset_dir: Path | None = None
        self._dataset_manifest: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def dataset_dir(self) -> Path | None:
        return self._dataset_dir

    def set_root(self, root: str | Path) -> Path:
        resolved = Path(root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        probe = resolved / f".auto-rheed-write-test-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok", encoding="utf-8")
        finally:
            probe.unlink(missing_ok=True)
        with self._lock:
            self._root = resolved
            self._dataset_dir = None
            self._dataset_manifest = None
        return resolved

    def begin_dataset(
        self,
        label: str,
        summary: dict[str, Any],
        dataset_version: int,
        *,
        source_fingerprint: str | None = None,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{_safe_part(label, 'dataset')}_{uuid.uuid4().hex[:8]}"
        with self._lock:
            dataset_dir = self._root / "runs" / name
            dataset_dir.mkdir(parents=True, exist_ok=False)
            manifest = {
                "schema_version": 1,
                "run_id": name,
                "created_utc": _utc_now(),
                "source_label": label,
                "source_fingerprint": source_fingerprint,
                "dataset_version": int(dataset_version),
                "dataset": _json_value(summary),
            }
            _atomic_json(dataset_dir / "manifest.json", manifest)
            (dataset_dir / "artifacts.jsonl").touch()
            self._dataset_dir = dataset_dir
            self._dataset_manifest = manifest
            return dataset_dir

    def activate_run(
        self,
        run_id: str,
        *,
        runtime_dataset_version: int | None = None,
    ) -> Path:
        """Activate an existing saved run after validating its local path."""
        if (
            not run_id
            or run_id in {".", ".."}
            or "/" in run_id
            or "\\" in run_id
            or Path(run_id).name != run_id
        ):
            raise KeyError(f"Unknown analysis run: {run_id}")
        run_dir = (self._root / "runs" / run_id).resolve()
        runs_root = (self._root / "runs").resolve()
        if run_dir.parent != runs_root:
            raise ValueError("Analysis run path escapes the results folder")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise KeyError(f"Unknown analysis run: {run_id}")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("run_id") != run_id:
            raise ValueError("Analysis run manifest does not match its folder")
        runtime_manifest = dict(manifest)
        if runtime_dataset_version is not None:
            runtime_manifest["stored_dataset_version"] = manifest.get("dataset_version")
            runtime_manifest["dataset_version"] = int(runtime_dataset_version)
        with self._lock:
            self._dataset_dir = run_dir
            self._dataset_manifest = runtime_manifest
        return run_dir

    def begin_or_resume_dataset(
        self,
        label: str,
        summary: dict[str, Any],
        dataset_version: int,
        *,
        source_fingerprint: str | None,
    ) -> tuple[Path, bool]:
        """Resume the newest exact source match, otherwise create a run."""
        if source_fingerprint:
            for run in self.list_runs():
                if run.get("source_fingerprint") == source_fingerprint:
                    path = self.activate_run(
                        str(run["run_id"]),
                        runtime_dataset_version=dataset_version,
                    )
                    return path, True
        return (
            self.begin_dataset(
                label,
                summary,
                dataset_version,
                source_fingerprint=source_fingerprint,
            ),
            False,
        )

    def ensure_dataset(
        self,
        *,
        label: str = "unspecified",
        summary: dict[str, Any] | None = None,
        dataset_version: int = 0,
    ) -> Path:
        with self._lock:
            if self._dataset_dir is not None:
                return self._dataset_dir
        return self.begin_dataset(label, summary or {}, dataset_version)

    def record_json(
        self,
        kind: str,
        result: dict[str, Any],
        *,
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            dataset_dir = self.ensure_dataset()
            safe_kind = _safe_part(kind, "analysis")
            artifact_id = _safe_part(name, "result") if replace and name else (
                f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{uuid.uuid4().hex[:6]}"
            )
            relative = Path("classical") / safe_kind / f"{artifact_id}.json"
            target = dataset_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "artifact_id": f"{safe_kind}/{artifact_id}",
                "kind": safe_kind,
                "name": name,
                "created_utc": _utc_now(),
                "parameters": _json_value(parameters or {}),
                "result": _json_value(result),
            }
            _atomic_json(target, payload)
            csv_paths = _write_csv_exports(target, payload["result"])
            if replace:
                _remove_stale_csv_exports(target, csv_paths)
            index_entry = {
                "artifact_id": payload["artifact_id"],
                "kind": safe_kind,
                "name": name,
                "created_utc": payload["created_utc"],
                "path": relative.as_posix(),
                "exports": [
                    path.relative_to(dataset_dir).as_posix() for path in csv_paths
                ],
            }
            _write_artifact_index(
                dataset_dir / "artifacts.jsonl",
                index_entry,
                replace=replace,
            )
            return index_entry

    def save_inference(
        self,
        job_id: str,
        result: InferenceRunResult,
        *,
        dataset_dir: str | Path | None = None,
    ) -> StoredInferenceRun:
        with self._lock:
            target_dataset_dir = (
                Path(dataset_dir).resolve()
                if dataset_dir is not None else self.ensure_dataset()
            )
            target_dataset_dir.mkdir(parents=True, exist_ok=True)
            run_name = f"{_safe_part(job_id, 'job')}_{_safe_part(result.model.id, 'model')}"
            target = target_dataset_dir / "ai" / run_name
            target.mkdir(parents=True, exist_ok=False)
            np.save(target / "frame_indices.npy", result.frame_indices, allow_pickle=False)
            np.save(target / "timestamps_s.npy", result.timestamps, allow_pickle=False)
            if result.embeddings is not None:
                np.save(
                    target / "embeddings.npy",
                    np.asarray(result.embeddings, dtype=np.float32),
                    allow_pickle=False,
                )
            if result.frames is not None:
                offsets = [0]
                with (target / "frame_results.jsonl").open("wb") as stream:
                    for frame in result.frames:
                        line = (json.dumps(_json_value(frame), separators=(",", ":")) + "\n").encode("utf-8")
                        stream.write(line)
                        offsets.append(offsets[-1] + len(line))
                np.save(
                    target / "frame_offsets.npy",
                    np.asarray(offsets, dtype=np.uint64),
                    allow_pickle=False,
                )
            summary = result.summary()
            compact_summary = {
                "model": result.model.public_dict(),
                "task": result.task,
                "n_results": int(len(result.frame_indices)),
                "metadata": _json_value(result.metadata),
            }
            if result.embeddings is not None:
                compact_summary["embedding_shape"] = list(result.embeddings.shape)
            if result.frames is not None:
                compact_summary["has_frame_summaries"] = True
            manifest = {
                "schema_version": 1,
                "job_id": job_id,
                "created_utc": _utc_now(),
                "task": result.task,
                "model": result.model.public_dict(),
                "metadata": _json_value(result.metadata),
                "summary": summary,
                "compact_summary": compact_summary,
            }
            _atomic_json(target / "manifest.json", manifest)
            index_entry = {
                "artifact_id": f"ai/{run_name}",
                "kind": "ai-inference",
                "name": result.model.id,
                "created_utc": manifest["created_utc"],
                "path": target.relative_to(target_dataset_dir).as_posix(),
            }
            _write_artifact_index(
                target_dataset_dir / "artifacts.jsonl",
                index_entry,
            )
            return StoredInferenceRun(target)

    def open_inference(self, artifact_id: str) -> StoredInferenceRun:
        """Open one AI artifact from the active, validated dataset run."""
        with self._lock:
            if self._dataset_dir is None:
                raise KeyError("No active analysis run")
            entry = next(
                (
                    item for item in reversed(self.list_artifacts())
                    if item["artifact_id"] == artifact_id
                ),
                None,
            )
            if entry is None or entry.get("kind") != "ai-inference":
                raise KeyError(f"Unknown AI artifact: {artifact_id}")
            target = (self._dataset_dir / entry["path"]).resolve()
            if self._dataset_dir not in target.parents or not target.is_dir():
                raise ValueError("Stored AI artifact path is invalid")
            return StoredInferenceRun(target)

    def status(self) -> dict[str, Any]:
        return {
            "root": str(self._root),
            "dataset_dir": str(self._dataset_dir) if self._dataset_dir else None,
            "dataset": dict(self._dataset_manifest) if self._dataset_manifest else None,
        }

    def list_artifacts(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._dataset_dir is None:
                return []
            index_path = self._dataset_dir / "artifacts.jsonl"
            if not index_path.is_file():
                return []
            entries_by_id: dict[str, dict[str, Any]] = {}
            with index_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        entry = json.loads(line)
                        entries_by_id[entry["artifact_id"]] = entry
            return list(entries_by_id.values())

    def load_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Load one classical JSON artifact from the active dataset."""
        with self._lock:
            if self._dataset_dir is None:
                raise KeyError("No active analysis run")
            entry = next(
                (
                    item for item in reversed(self.list_artifacts())
                    if item["artifact_id"] == artifact_id
                ),
                None,
            )
            if entry is None or entry.get("kind") == "ai-inference":
                raise KeyError(f"Unknown classical artifact: {artifact_id}")
            target = (self._dataset_dir / entry["path"]).resolve()
            if self._dataset_dir not in target.parents:
                raise ValueError("Stored artifact path escapes the analysis run")
            if target.suffix == ".gz":
                with gzip.open(target, "rt", encoding="utf-8") as stream:
                    return json.load(stream)
            with target.open(encoding="utf-8") as stream:
                return json.load(stream)

    def list_runs(self) -> list[dict[str, Any]]:
        """List saved dataset runs newest first without activating them."""
        runs_dir = self._root / "runs"
        if not runs_dir.is_dir():
            return []
        runs: list[dict[str, Any]] = []
        for manifest_path in runs_dir.glob("*/manifest.json"):
            try:
                with manifest_path.open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                manifest["path"] = str(manifest_path.parent)
                runs.append(manifest)
            except (OSError, ValueError, TypeError):
                continue
        return sorted(runs, key=lambda item: item.get("created_utc", ""), reverse=True)
