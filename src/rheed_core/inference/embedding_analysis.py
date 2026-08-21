"""Pluggable analyses that consume completed frame-embedding sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from typing import Any

import numpy as np

from .types import InferenceRunResult


@dataclass(frozen=True)
class EmbeddingAnalysisDescriptor:
    """Discovery metadata for a temporal embedding-analysis adapter."""

    name: str
    task: str
    description: str
    requirements: tuple[str, ...] = ()
    default_options: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requirements"] = list(self.requirements)
        result["available"] = all(find_spec(name) is not None for name in self.requirements)
        return result


class EmbeddingAnalysisAdapter(ABC):
    """Stable boundary for analyses that require a complete embedding sequence."""

    descriptor: EmbeddingAnalysisDescriptor

    @abstractmethod
    def analyze(
        self,
        embeddings: np.ndarray,
        frame_indices: np.ndarray,
        timestamps: np.ndarray,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON-safe, frame-aligned analysis result."""

    def array_artifacts(self) -> dict[str, np.ndarray]:
        """Return optional large arrays produced by the most recent analysis.

        Arrays are saved separately from the JSON result so adapters can expose
        large visualizations without forcing them into browser or server memory.
        """
        return {}


_BUILTINS = {
    "rhaapsody-changepoint": (
        "rheed_core.inference.rhaapsody_changepoint:RhaapsodyChangepointAdapter"
    ),
}
_ENTRY_POINT_GROUP = "auto_rheed.embedding_analysis_adapters"


def _load_object(reference: str):
    module_name, sep, attr_name = reference.partition(":")
    if not sep:
        raise ValueError(f"Invalid embedding-analysis adapter reference: {reference!r}")
    return getattr(import_module(module_name), attr_name)


def _entry_points() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    eps = importlib_metadata.entry_points()
    selected = (
        eps.select(group=_ENTRY_POINT_GROUP)
        if hasattr(eps, "select")
        else eps.get(_ENTRY_POINT_GROUP, [])
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
            f"Unknown embedding-analysis adapter {name!r}; available: {choices or 'none'}"
        )
    return entry_point.load()


def create_embedding_analysis_adapter(name: str) -> EmbeddingAnalysisAdapter:
    adapter = _adapter_class(name)()
    if not isinstance(adapter, EmbeddingAnalysisAdapter):
        raise TypeError(
            f"Embedding-analysis adapter {name!r} does not implement "
            "EmbeddingAnalysisAdapter"
        )
    return adapter


def list_embedding_analysis_adapters() -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for name in sorted([*_BUILTINS, *_entry_points()]):
        try:
            cls = _adapter_class(name)
            descriptor = cls.descriptor.to_dict()
            descriptor["name"] = name
            descriptors.append(descriptor)
        except Exception as exc:
            descriptors.append({"name": name, "available": False, "error": str(exc)})
    return descriptors


def _execute_embedding_analysis(
    result: InferenceRunResult,
    adapter_name: str,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if result.embeddings is None or result.task != "embedding":
        raise ValueError("Embedding analysis requires a completed embedding result")
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("Embedding-analysis options must be a JSON object")

    adapter = create_embedding_analysis_adapter(adapter_name)
    analysis = adapter.analyze(
        np.asarray(result.embeddings, dtype=np.float32),
        np.asarray(result.frame_indices, dtype=np.int64),
        np.asarray(result.timestamps, dtype=np.float64),
        dict(options),
    )
    if not isinstance(analysis, dict):
        raise TypeError("Embedding-analysis adapters must return a dictionary")
    analysis.setdefault("adapter", adapter_name)
    analysis.setdefault("task", adapter.descriptor.task)
    analysis.setdefault("provenance", dict(adapter.descriptor.provenance))
    artifacts = {
        str(name): np.asarray(array)
        for name, array in adapter.array_artifacts().items()
    }
    return analysis, artifacts


def run_embedding_analysis(
    result: InferenceRunResult,
    adapter_name: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one adapter and return its JSON-safe result."""
    analysis, _ = _execute_embedding_analysis(result, adapter_name, options)
    return analysis


def run_embedding_analysis_with_artifacts(
    result: InferenceRunResult,
    adapter_name: str,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one adapter and return its JSON plus separately stored arrays."""
    return _execute_embedding_analysis(result, adapter_name, options)
