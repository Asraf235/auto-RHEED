"""Simulation-adapter interface and entry-point discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping

from .types import SimulationContext, SimulationSpec


@dataclass(frozen=True)
class SimulationAdapterDescriptor:
    """Small, import-safe description of one simulation implementation."""

    name: str
    description: str
    input_names: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    default_options: dict[str, Any] = field(default_factory=dict)
    option_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["input_names"] = list(self.input_names)
        result["capabilities"] = list(self.capabilities)
        result["requirements"] = list(self.requirements)
        result["available"] = all(find_spec(name) is not None for name in self.requirements)
        return result


class SimulationAdapter(ABC):
    """Stable boundary implemented by simulator-specific integrations."""

    descriptor: SimulationAdapterDescriptor

    def load(self, spec: SimulationSpec, device: str) -> dict[str, Any]:
        """Prepare the optional runtime and return JSON-safe provenance."""

        return {"device": device}

    @abstractmethod
    def simulate(
        self,
        inputs: Mapping[str, Path],
        context: SimulationContext,
    ) -> dict[str, Any]:
        """Return normalized scan coordinates and simulation arrays."""

    def recover_result_metadata(self, inputs: Mapping[str, Path]) -> dict[str, Any]:
        """Recover lightweight provenance omitted by an older saved run.

        This hook must not load the optional simulation runtime. It exists for
        backwards-compatible reads of archived input copies; new runs should
        return the metadata directly from :meth:`simulate`.
        """

        return {}

    def close(self) -> None:
        """Release optional runtime resources after any terminal outcome."""


_BUILTINS = {
    "torch-rheed": "rheed_core.simulation.torch_rheed:TorchRheedAdapter",
}
_ENTRY_POINT_GROUP = "auto_rheed.simulation_adapters"


def _load_object(reference: str):
    module_name, sep, attr_name = reference.partition(":")
    if not sep:
        raise ValueError(f"Invalid simulation adapter reference: {reference!r}")
    return getattr(import_module(module_name), attr_name)


def _entry_points() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    entry_points = importlib_metadata.entry_points()
    selected = (
        entry_points.select(group=_ENTRY_POINT_GROUP)
        if hasattr(entry_points, "select")
        else entry_points.get(_ENTRY_POINT_GROUP, [])
    )
    for entry_point in selected:
        if entry_point.name not in _BUILTINS:
            discovered[entry_point.name] = entry_point
    return discovered


def _adapter_class(name: str):
    if name in _BUILTINS:
        return _load_object(_BUILTINS[name])
    entry_point = _entry_points().get(name)
    if entry_point is None:
        choices = ", ".join(sorted([*_BUILTINS, *_entry_points()]))
        raise ValueError(
            f"Unknown simulation adapter {name!r}; available: {choices or 'none'}"
        )
    return entry_point.load()


def create_simulation_adapter(name: str) -> SimulationAdapter:
    adapter = _adapter_class(name)()
    if not isinstance(adapter, SimulationAdapter):
        raise TypeError(f"Simulation adapter {name!r} does not implement SimulationAdapter")
    return adapter


def list_simulation_adapters() -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for name in sorted([*_BUILTINS, *_entry_points()]):
        try:
            cls = _adapter_class(name)
            descriptor = cls.descriptor.to_dict()
            descriptor["name"] = name
            descriptors.append(descriptor)
        except Exception as exc:
            descriptors.append({
                "name": name,
                "available": False,
                "error": str(exc),
            })
    return descriptors
