"""Data contracts shared by simulation adapters and consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


_SECRET_PARTS = ("token", "secret", "password", "api_key", "apikey")
_SAFE_ARRAY_NAME = re.compile(r"[^A-Za-z0-9_]+")


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


def _redact(mapping: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in mapping.items():
        if any(part in key.lower() for part in _SECRET_PARTS):
            clean[key] = "<redacted>"
        elif isinstance(value, dict):
            clean[key] = _redact(value)
        else:
            clean[key] = _json_value(value)
    return clean


@dataclass(frozen=True)
class SimulationSpec:
    """Manifest selecting a simulation adapter and its scientific options."""

    id: str
    adapter: str
    revision: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationSpec":
        if not isinstance(data, dict):
            raise ValueError("simulation must be a JSON object")
        simulation_id = str(data.get("id") or "").strip()
        adapter = str(data.get("adapter") or "").strip()
        if not simulation_id:
            raise ValueError("simulation.id is required")
        if not adapter:
            raise ValueError("simulation.adapter is required")
        options = data.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("simulation.options must be an object")
        return cls(
            id=simulation_id,
            adapter=adapter,
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            options=dict(options),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SimulationSpec":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "adapter": self.adapter,
            "revision": self.revision,
            "options": _redact(self.options),
        }


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class SimulationContext:
    """Execution state supplied to an adapter for one simulation run."""

    spec: SimulationSpec
    device: str
    progress: ProgressCallback | None = None
    cancelled: CancelCallback | None = None

    def report_progress(self, completed: int, total: int) -> None:
        if self.progress is not None:
            self.progress(int(completed), int(total))

    def is_cancelled(self) -> bool:
        return bool(self.cancelled and self.cancelled())


@dataclass
class SimulationRunResult:
    """Normalized simulator output on one shared scan-position axis."""

    simulation: SimulationSpec
    scan_coordinates: dict[str, np.ndarray]
    metadata: dict[str, Any]
    beam_indices: np.ndarray | None = None
    intensities: np.ndarray | None = None
    detector_images: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.scan_coordinates:
            raise ValueError("simulation results require at least one scan coordinate")
        lengths = {len(np.asarray(values)) for values in self.scan_coordinates.values()}
        if len(lengths) != 1:
            raise ValueError("all simulation scan coordinates must have equal length")
        n_samples = next(iter(lengths))
        if n_samples < 1:
            raise ValueError("simulation results cannot be empty")
        if self.intensities is None and self.detector_images is None:
            raise ValueError("simulation results require intensities or detector images")
        if self.intensities is not None:
            if self.intensities.ndim != 2 or len(self.intensities) != n_samples:
                raise ValueError("intensities must have shape (N, B)")
            if self.beam_indices is None:
                raise ValueError("beam_indices are required with beam intensities")
            if self.beam_indices.shape != (self.intensities.shape[1], 2):
                raise ValueError("beam_indices must have shape (B, 2)")
        elif self.beam_indices is not None:
            raise ValueError("beam_indices require beam intensities")
        if self.detector_images is not None:
            if self.detector_images.ndim != 3 or len(self.detector_images) != n_samples:
                raise ValueError("detector_images must have shape (N, H, W)")

    @property
    def n_samples(self) -> int:
        return len(next(iter(self.scan_coordinates.values())))

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "simulation": self.simulation.public_dict(),
            "n_samples": self.n_samples,
            "scan_coordinates": {
                name: np.asarray(values).tolist()
                for name, values in self.scan_coordinates.items()
            },
            "metadata": _json_value(self.metadata),
        }
        if self.intensities is not None:
            result["intensity_shape"] = list(self.intensities.shape)
            result["beam_indices"] = self.beam_indices.tolist()
        if self.detector_images is not None:
            result["detector_image_shape"] = list(self.detector_images.shape)
        return result

    def to_npz_bytes(self) -> bytes:
        payload: dict[str, Any] = {
            "metadata_json": np.asarray(json.dumps({
                "simulation": self.simulation.public_dict(),
                "metadata": _json_value(self.metadata),
                "scan_coordinate_names": list(self.scan_coordinates),
            })),
        }
        used_names: set[str] = set()
        for position, (name, values) in enumerate(self.scan_coordinates.items()):
            safe_name = _SAFE_ARRAY_NAME.sub("_", name).strip("_") or f"coordinate_{position}"
            array_name = f"scan_{safe_name}"
            if array_name in used_names:
                array_name = f"{array_name}_{position}"
            used_names.add(array_name)
            payload[array_name] = np.asarray(values, dtype=np.float64)
        if self.beam_indices is not None:
            payload["beam_indices"] = self.beam_indices.astype(np.int64, copy=False)
        if self.intensities is not None:
            payload["intensities"] = self.intensities
        if self.detector_images is not None:
            payload["detector_images"] = self.detector_images
        stream = io.BytesIO()
        np.savez_compressed(stream, **payload)
        return stream.getvalue()
