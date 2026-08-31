"""File-backed storage for local simulation inputs and numeric results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, BinaryIO, Mapping
import uuid

import numpy as np

from .types import SimulationRunResult, SimulationSpec


_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_part(value: str | None, fallback: str) -> str:
    return _SAFE_PART.sub("-", str(value or "")).strip(".-") or fallback


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(payload), stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class StoredSimulationRun:
    path: Path

    def manifest(self) -> dict[str, Any]:
        with (self.path / "manifest.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def load_result(self, *, mmap_mode: str | None = None) -> SimulationRunResult:
        manifest = self.manifest()
        if manifest.get("status") != "completed":
            raise ValueError("Simulation result is not complete")
        result_manifest = manifest.get("result") or {}
        coordinates = {
            item["name"]: np.load(
                self.path / item["path"],
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
            for item in result_manifest.get("scan_coordinates", [])
        }

        def optional_array(name: str):
            relative = result_manifest.get(name)
            if not relative:
                return None
            return np.load(
                self.path / relative,
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )

        return SimulationRunResult(
            simulation=SimulationSpec.from_dict(manifest["simulation"]),
            scan_coordinates=coordinates,
            beam_indices=optional_array("beam_indices"),
            intensities=optional_array("intensities"),
            detector_images=optional_array("detector_images"),
            metadata=dict(result_manifest.get("metadata") or {}),
        )

    def detector_frame(self, index: int) -> np.ndarray:
        manifest = self.manifest()
        result_manifest = manifest.get("result") or {}
        relative = result_manifest.get("detector_images")
        if not relative:
            raise ValueError("This simulation has no detector images")
        images = np.load(self.path / relative, allow_pickle=False, mmap_mode="r")
        if not 0 <= int(index) < len(images):
            raise IndexError(f"Detector frame must be between 0 and {len(images) - 1}")
        return np.asarray(images[int(index)], dtype=np.float32)

    def load_measurements(self) -> list[dict[str, Any]]:
        path = self.path / "measurements.json"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        measurements = payload.get("measurements", [])
        if not isinstance(measurements, list):
            raise ValueError("Saved simulation measurements are invalid")
        return measurements


class SimulationStore:
    """Thread-safe root containing independent, recallable simulation runs."""

    def __init__(self, root: str | Path):
        self._root = Path(root).expanduser().resolve()
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def set_root(self, root: str | Path) -> Path:
        resolved = Path(root).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._root = resolved
        return resolved

    def create_run(
        self,
        run_id: str,
        spec: SimulationSpec,
        uploads: Mapping[str, tuple[str, BinaryIO]],
    ) -> tuple[StoredSimulationRun, dict[str, Path]]:
        safe_run_id = self._validate_run_id(run_id)
        with self._lock:
            run_path = self._root / safe_run_id
            run_path.mkdir(parents=True, exist_ok=False)
            input_dir = run_path / "inputs"
            input_dir.mkdir()
            stored_inputs: dict[str, Path] = {}
            input_manifest: dict[str, dict[str, Any]] = {}
            for input_name, (filename, stream) in uploads.items():
                safe_input = _safe_part(input_name, "input")
                safe_filename = _safe_part(Path(filename or "input.dat").name, "input.dat")
                target = input_dir / f"{safe_input}_{safe_filename}"
                size = 0
                with target.open("wb") as output:
                    while chunk := stream.read(4 * 1024 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                stored_inputs[input_name] = target
                input_manifest[input_name] = {
                    "filename": filename,
                    "path": target.relative_to(run_path).as_posix(),
                    "size_bytes": size,
                }
            _atomic_json(run_path / "manifest.json", {
                "schema_version": 1,
                "run_id": safe_run_id,
                "created_utc": _utc_now(),
                "status": "queued",
                "completed_steps": 0,
                "total_steps": 1,
                "simulation": spec.public_dict(),
                "inputs": input_manifest,
            })
            return StoredSimulationRun(run_path), stored_inputs

    def update_status(self, run_id: str, **updates) -> dict[str, Any]:
        with self._lock:
            stored = self.open(run_id)
            manifest = stored.manifest()
            manifest.update(_json_value(updates))
            _atomic_json(stored.path / "manifest.json", manifest)
            return manifest

    def save_result(self, run_id: str, result: SimulationRunResult) -> StoredSimulationRun:
        with self._lock:
            stored = self.open(run_id)
            coordinate_manifest = []
            for index, (name, values) in enumerate(result.scan_coordinates.items()):
                filename = f"scan_coordinate_{index}.npy"
                np.save(stored.path / filename, np.asarray(values), allow_pickle=False)
                coordinate_manifest.append({"name": name, "path": filename})

            result_manifest: dict[str, Any] = {
                "scan_coordinates": coordinate_manifest,
                "metadata": _json_value(result.metadata),
                "n_samples": result.n_samples,
            }
            for name, values in (
                ("beam_indices", result.beam_indices),
                ("intensities", result.intensities),
                ("detector_images", result.detector_images),
            ):
                if values is not None:
                    filename = f"{name}.npy"
                    np.save(stored.path / filename, np.asarray(values), allow_pickle=False)
                    result_manifest[name] = filename
                    result_manifest[f"{name}_shape"] = list(values.shape)

            manifest = stored.manifest()
            manifest.update({
                "status": "completed",
                "completed_steps": 1,
                "total_steps": 1,
                "completed_utc": _utc_now(),
                "result": result_manifest,
            })
            _atomic_json(stored.path / "manifest.json", manifest)
            return stored

    def save_measurements(
        self,
        run_id: str,
        measurements: list[dict[str, Any]],
    ) -> StoredSimulationRun:
        with self._lock:
            stored = self.open(run_id)
            _atomic_json(stored.path / "measurements.json", {
                "schema_version": 1,
                "updated_utc": _utc_now(),
                "measurements": measurements,
            })
            return stored

    def open(self, run_id: str) -> StoredSimulationRun:
        safe_run_id = self._validate_run_id(run_id)
        path = (self._root / safe_run_id).resolve()
        if path.parent != self._root or not (path / "manifest.json").is_file():
            raise KeyError(f"Unknown simulation run: {run_id}")
        return StoredSimulationRun(path)

    def list_runs(self) -> list[dict[str, Any]]:
        if not self._root.is_dir():
            return []
        runs = []
        for manifest_path in self._root.glob("*/manifest.json"):
            try:
                with manifest_path.open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                runs.append({
                    "run_id": manifest["run_id"],
                    "created_utc": manifest.get("created_utc"),
                    "completed_utc": manifest.get("completed_utc"),
                    "status": manifest.get("status"),
                    "error": manifest.get("error"),
                    "simulation": manifest.get("simulation"),
                    "result": manifest.get("result"),
                    "saved_to": str(manifest_path.parent),
                })
            except (OSError, ValueError, KeyError):
                continue
        runs.sort(key=lambda item: str(item.get("created_utc") or ""), reverse=True)
        return runs

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        text = str(run_id or "")
        if not text or _safe_part(text, "") != text or Path(text).name != text:
            raise KeyError(f"Unknown simulation run: {run_id}")
        return text
