"""Adapter for the optional ``torch-rheed`` simulation package."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from .base import SimulationAdapter, SimulationAdapterDescriptor
from .runner import SimulationCancelled
from .types import SimulationContext, SimulationSpec


_TORCH_DEFAULT_DTYPE_LOCK = threading.RLock()


class TorchRheedAdapter(SimulationAdapter):
    descriptor = SimulationAdapterDescriptor(
        name="torch-rheed",
        description="Local rocking-curve and detector simulation using torch-rheed.",
        input_names=("bulk", "surface"),
        capabilities=("rocking-curves", "detector-images", "cpu", "cuda"),
        requirements=("torch_rheed",),
        default_options={
            "solver": "sp6",
            # TiO2_bulk.txt uses DZ=0.05 Å. torch-rheed's implicit SP6 step is
            # 10 * DZ, made explicit here so the web form records 0.5 Å.
            "integration_step_angstrom": 0.5,
            "rhst_threshold": 1000.0,
            "render_detector": True,
            # Screen geometry is not part of the legacy bulk/surface inputs.
            # Spell out torch-rheed's ScreenImageConfig defaults in the form so
            # every value is visible and preserved in the run manifest.
            "screen": {
                "plane_mode": "vertical",
                "screen_distance_mm": 100.0,
                "screen_width_mm": 120.0,
                "screen_height_mm": 90.0,
                "pixels_x": 320,
                "pixels_y": 240,
                "correlation_length_angstrom": 1000.0,
                "rod_profile": "lorentzian",
                "beam_intensity_floor": 0.0,
                "instrument_broadening_fwhm_mm": 0.0,
                "source_glancing_divergence_fwhm_deg": 0.0,
                "source_divergence_samples": 9,
                "reference_angle_deg": None,
                "frame_azimuth_deg": None,
                "display_scale": "sqrt",
                "colormap": "inferno",
            },
        },
        option_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "solver": {"type": "string", "enum": ["sp6", "multislice"]},
                "integration_step_angstrom": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "rhst_threshold": {"type": "number", "exclusiveMinimum": 0},
                "render_detector": {"type": "boolean"},
                "screen": {"type": "object"},
            },
        },
    )

    def __init__(self) -> None:
        self._package = None
        self._torch = None
        self._device = "cpu"

    def load(self, spec: SimulationSpec, device: str) -> dict[str, Any]:
        try:
            torch = import_module("torch")
            # torch-rheed currently selects float64 through PyTorch's process-wide
            # default at import. Restore the caller's default immediately so the
            # optional simulator cannot silently change later AI calculations.
            with _torch_float64_default(torch):
                package = import_module("torch_rheed")
        except ImportError as exc:
            raise RuntimeError(
                "The torch-rheed simulation adapter requires the optional "
                "torch-rheed package. Install it into the Auto RHEED environment."
            ) from exc

        requested = str(device).strip().lower() or "auto"
        if requested == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = str(device)

        try:
            distribution = importlib_metadata.distribution("torch-rheed")
            package_version = distribution.version
            direct_url_text = distribution.read_text("direct_url.json")
            direct_url = json.loads(direct_url_text) if direct_url_text else None
        except (importlib_metadata.PackageNotFoundError, json.JSONDecodeError):
            package_version = getattr(package, "__version__", None)
            direct_url = None

        self._package = package
        self._torch = torch
        self._device = resolved_device
        return {
            "device_requested": device,
            "device": resolved_device,
            "torch_rheed_version": package_version,
            "torch_version": getattr(torch, "__version__", None),
            "package_path": str(Path(package.__file__).resolve()),
            "direct_url": direct_url,
            "requested_revision": spec.revision,
        }

    def simulate(
        self,
        inputs: Mapping[str, Path],
        context: SimulationContext,
    ) -> dict[str, Any]:
        if self._package is None:
            raise RuntimeError("torch-rheed adapter was not loaded")
        if context.is_cancelled():
            raise SimulationCancelled("Simulation cancelled")

        options = dict(self.descriptor.default_options)
        supplied = dict(context.spec.options)
        unknown = sorted(set(supplied).difference(options))
        if unknown:
            raise ValueError(f"Unknown torch-rheed options: {', '.join(unknown)}")
        options.update(supplied)

        solver = str(options["solver"]).strip().lower()
        if solver not in {"sp6", "multislice"}:
            raise ValueError("torch-rheed solver must be 'sp6' or 'multislice'")
        integration_step = options["integration_step_angstrom"]
        if integration_step is not None:
            integration_step = float(integration_step)
            if not np.isfinite(integration_step) or integration_step <= 0:
                raise ValueError("integration_step_angstrom must be positive")
        if solver == "multislice":
            # The integration step belongs only to SP6. The generic web form
            # submits its prefilled value even when the solver is changed.
            integration_step = None
        rhst_threshold = float(options["rhst_threshold"])
        if not np.isfinite(rhst_threshold) or rhst_threshold <= 0:
            raise ValueError("rhst_threshold must be positive")

        render_detector = options["render_detector"]
        if not isinstance(render_detector, (bool, np.bool_)):
            raise ValueError("render_detector must be a boolean")
        screen_options = options["screen"] or {}
        if not isinstance(screen_options, dict):
            raise ValueError("torch-rheed screen options must be an object")
        screen_config = None
        if bool(render_detector):
            screen_config = self._package.ScreenImageConfig(**screen_options)

        with _torch_float64_default(self._torch):
            bulk_input = self._package.load_bulk_input(inputs["bulk"])
            result = self._package.simulate_from_files(
                inputs["bulk"],
                inputs["surface"],
                device=self._device,
                screen_config=screen_config,
                solver=solver,
                integration_step=integration_step,
                rhst_threshold=rhst_threshold,
            )
        if context.is_cancelled():
            raise SimulationCancelled("Simulation cancelled")

        coordinates = _scan_coordinates(bulk_input, result.naz, result.ng)
        detector_images = (
            None if result.screen_images is None else _tensor_to_numpy(result.screen_images)
        )
        resolved_screen = result.screen_config
        if resolved_screen is None:
            screen_metadata = None
        elif is_dataclass(resolved_screen):
            screen_metadata = asdict(resolved_screen)
        else:
            screen_metadata = str(resolved_screen)
        return {
            "scan_coordinates": coordinates,
            "beam_indices": np.asarray(result.beam_indices, dtype=np.int64),
            "intensities": _tensor_to_numpy(result.intensities),
            "detector_images": detector_images,
            "metadata": {
                "solver": solver,
                # Read from the uploaded bulk input; this is the energy used by
                # torch-rheed to construct the relativistic electron wavevector.
                "beam_energy_kev": float(bulk_input.be),
                "integration_step_angstrom": integration_step,
                "rhst_threshold": rhst_threshold,
                "naz": int(result.naz),
                "ng": int(result.ng),
                "screen": screen_metadata,
            },
        }

    def close(self) -> None:
        self._package = None
        self._torch = None

    def recover_result_metadata(self, inputs: Mapping[str, Path]) -> dict[str, Any]:
        """Read BE from an archived legacy bulk input without importing torch-rheed."""

        bulk_path = inputs.get("bulk")
        if bulk_path is None:
            return {}
        with Path(bulk_path).open("r", encoding="utf-8", errors="replace") as stream:
            for _ in range(256):
                line = stream.readline()
                if not line:
                    break
                fields = [field.strip() for field in line.split(",")]
                if any(field.upper() == "BE" for field in fields[1:]):
                    try:
                        energy = float(fields[0])
                    except (TypeError, ValueError):
                        return {}
                    if np.isfinite(energy) and energy > 0:
                        return {"beam_energy_kev": energy}
        return {}


@contextmanager
def _torch_float64_default(torch):
    """Isolate torch-rheed's process-wide default-dtype expectation."""

    with _TORCH_DEFAULT_DTYPE_LOCK:
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            yield
        finally:
            torch.set_default_dtype(previous)


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scan_coordinates(bulk_input, naz: int, ng: int) -> dict[str, np.ndarray]:
    """Reproduce torch-rheed's flattened scan order with explicit coordinates."""

    azimuth = (
        float(bulk_input.azi_deg)
        + float(bulk_input.rdom_deg[0])
        + np.arange(naz, dtype=np.float64) * float(bulk_input.daz_deg)
    )
    glancing = (
        float(bulk_input.gi_deg)
        + np.arange(ng, dtype=np.float64) * float(bulk_input.dg_deg)
    )
    if naz == 1 or ng > naz:
        azimuth_grid = np.broadcast_to(azimuth[:, None], (naz, ng))
        glancing_grid = np.broadcast_to(glancing[None, :], (naz, ng))
    else:
        glancing_grid = np.broadcast_to(glancing[:, None], (ng, naz))
        azimuth_grid = np.broadcast_to(azimuth[None, :], (ng, naz))
    return {
        "azimuth_deg": np.ascontiguousarray(azimuth_grid.reshape(-1)),
        "glancing_angle_deg": np.ascontiguousarray(glancing_grid.reshape(-1)),
    }
