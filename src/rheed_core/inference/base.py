"""Adapter interface and discovery through Python entry points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from typing import Any

import numpy as np

from .types import ModelSpec, PreprocessingContext


@dataclass(frozen=True)
class AdapterDescriptor:
    name: str
    task: str
    description: str
    requirements: tuple[str, ...] = ()
    default_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requirements"] = list(self.requirements)
        result["available"] = all(find_spec(name) is not None for name in self.requirements)
        return result


class ModelAdapter(ABC):
    """Stable boundary implemented by framework-specific model packages."""

    descriptor: AdapterDescriptor

    @abstractmethod
    def load(self, spec: ModelSpec, device: str) -> dict[str, Any]:
        """Load model weights and return runtime metadata."""

    @abstractmethod
    def infer_batch(
        self,
        frames: np.ndarray,
        context: PreprocessingContext,
    ) -> dict[str, Any]:
        """Return either {'embeddings': array} or {'frames': list[dict]}."""

    def effective_batch_size(self, requested: int) -> int:
        """Return the runner batch size; sequential adapters may reduce it."""
        return int(requested)

    def close(self) -> None:
        """Release optional framework/GPU resources."""


_BUILTINS = {
    "dinov3": "rheed_core.inference.dinov3:DinoV3Adapter",
    "ultralytics-seg": "rheed_core.inference.ultralytics_seg:UltralyticsSegAdapter",
}
_ENTRY_POINT_GROUP = "auto_rheed.inference_adapters"


def _load_object(reference: str):
    module_name, sep, attr_name = reference.partition(":")
    if not sep:
        raise ValueError(f"Invalid adapter reference: {reference!r}")
    return getattr(import_module(module_name), attr_name)


def _entry_points() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    eps = importlib_metadata.entry_points()
    selected = eps.select(group=_ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(_ENTRY_POINT_GROUP, [])
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
        raise ValueError(f"Unknown inference adapter {name!r}; available: {choices or 'none'}")
    return entry_point.load()


def create_adapter(name: str) -> ModelAdapter:
    adapter = _adapter_class(name)()
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"Adapter {name!r} does not implement ModelAdapter")
    return adapter


def list_adapters() -> list[dict[str, Any]]:
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
