"""Simulation orchestration shared by web, MCP, and script consumers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .base import create_simulation_adapter
from .types import CancelCallback, ProgressCallback, SimulationContext, SimulationRunResult, SimulationSpec


class SimulationCancelled(RuntimeError):
    pass


def _input_provenance(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _numeric_array(value, *, dtype, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise ValueError(f"{label} must not use object dtype")
    return np.ascontiguousarray(array)


def run_simulation(
    inputs: Mapping[str, str | Path],
    spec: SimulationSpec,
    *,
    device: str = "auto",
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> SimulationRunResult:
    """Run one configured local simulation and validate its normalized output."""

    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("simulation inputs must be a non-empty mapping")
    resolved_inputs: dict[str, Path] = {}
    for raw_name, raw_path in inputs.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("simulation input names cannot be empty")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Simulation input {name!r} is not a file: {path}")
        resolved_inputs[name] = path
    input_provenance = {
        name: _input_provenance(path)
        for name, path in resolved_inputs.items()
    }

    adapter = create_simulation_adapter(spec.adapter)
    descriptor = adapter.descriptor
    missing = sorted(set(descriptor.input_names).difference(resolved_inputs))
    if missing:
        raise ValueError(f"Missing simulation inputs: {', '.join(missing)}")
    unexpected = sorted(set(resolved_inputs).difference(descriptor.input_names))
    if unexpected:
        raise ValueError(f"Unexpected simulation inputs: {', '.join(unexpected)}")

    context = SimulationContext(
        spec=spec,
        device=str(device),
        progress=progress,
        cancelled=cancelled,
    )
    runtime_metadata: dict = {}
    started = datetime.now(timezone.utc)
    try:
        if context.is_cancelled():
            raise SimulationCancelled("Simulation cancelled")
        context.report_progress(0, 1)
        runtime_metadata = adapter.load(spec, str(device)) or {}
        if not isinstance(runtime_metadata, dict):
            raise TypeError("SimulationAdapter.load() must return a dictionary")
        if context.is_cancelled():
            raise SimulationCancelled("Simulation cancelled")
        raw_result = adapter.simulate(resolved_inputs, context)
        if not isinstance(raw_result, dict):
            raise TypeError("SimulationAdapter.simulate() must return a dictionary")
        if context.is_cancelled():
            raise SimulationCancelled("Simulation cancelled")

        raw_coordinates = raw_result.get("scan_coordinates")
        if not isinstance(raw_coordinates, dict) or not raw_coordinates:
            raise ValueError("Adapter result requires a scan_coordinates object")
        scan_coordinates: dict[str, np.ndarray] = {}
        for raw_name, values in raw_coordinates.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Scan coordinate names cannot be empty")
            coordinate = _numeric_array(values, dtype=np.float64, label=f"scan coordinate {name}")
            if coordinate.ndim != 1:
                raise ValueError(f"Scan coordinate {name!r} must be one-dimensional")
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(f"Scan coordinate {name!r} must contain finite values")
            scan_coordinates[name] = coordinate

        beam_indices = raw_result.get("beam_indices")
        if beam_indices is not None:
            beam_indices = _numeric_array(beam_indices, dtype=np.int64, label="beam_indices")
        intensities = raw_result.get("intensities")
        if intensities is not None:
            intensities = _numeric_array(intensities, dtype=np.float64, label="intensities")
        detector_images = raw_result.get("detector_images")
        if detector_images is not None:
            detector_images = _numeric_array(
                detector_images,
                dtype=np.float32,
                label="detector_images",
            )
        adapter_metadata = raw_result.get("metadata") or {}
        if not isinstance(adapter_metadata, dict):
            raise TypeError("Simulation adapter result metadata must be a dictionary")

        finished = datetime.now(timezone.utc)
        result = SimulationRunResult(
            simulation=spec,
            scan_coordinates=scan_coordinates,
            beam_indices=beam_indices,
            intensities=intensities,
            detector_images=detector_images,
            metadata={
                "adapter": descriptor.name,
                "adapter_capabilities": list(descriptor.capabilities),
                "runtime": runtime_metadata,
                "inputs": input_provenance,
                "started_utc": started.isoformat(),
                "finished_utc": finished.isoformat(),
                "duration_s": (finished - started).total_seconds(),
                "result": adapter_metadata,
            },
        )
        context.report_progress(1, 1)
        return result
    finally:
        adapter.close()
