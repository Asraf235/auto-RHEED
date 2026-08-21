"""Data contracts shared by inference adapters and consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


_SECRET_PARTS = ("token", "secret", "password", "api_key", "apikey")


def _json_value(value: Any) -> Any:
    """Convert NumPy/path values into ordinary JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
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
class ModelSpec:
    """Manifest describing a model without coupling core to its framework."""

    id: str
    adapter: str
    source: str
    task: str | None = None
    revision: str | None = None
    preprocessing: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        if not isinstance(data, dict):
            raise ValueError("model must be a JSON object")
        model_id = str(data.get("id") or "").strip()
        adapter = str(data.get("adapter") or "").strip()
        source = str(data.get("source") or "").strip()
        if not model_id:
            raise ValueError("model.id is required")
        if not adapter:
            raise ValueError("model.adapter is required")
        if not source:
            raise ValueError("model.source is required")
        preprocessing = data.get("preprocessing") or {}
        options = data.get("options") or {}
        if not isinstance(preprocessing, dict) or not isinstance(options, dict):
            raise ValueError("model.preprocessing and model.options must be objects")
        return cls(
            id=model_id,
            adapter=adapter,
            source=source,
            task=str(data["task"]).strip() if data.get("task") else None,
            revision=str(data["revision"]).strip() if data.get("revision") else None,
            preprocessing=dict(preprocessing),
            options=dict(options),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelSpec":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "adapter": self.adapter,
            "source": self.source,
            "task": self.task,
            "revision": self.revision,
            "preprocessing": _redact(self.preprocessing),
            "options": _redact(self.options),
        }

    def local_source_sha256(self) -> str | None:
        """Hash local weights for provenance; remote model IDs return None."""
        path = Path(self.source).expanduser()
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class PreprocessingContext:
    """Dataset-level preprocessing values shared by every inference batch."""

    intensity_low: float
    intensity_high: float
    scaling: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intensity_low": self.intensity_low,
            "intensity_high": self.intensity_high,
            "scaling": self.scaling,
            "config": _redact(self.config),
        }


@dataclass
class InferenceRunResult:
    """One inference result aligned to the selected dataset frames."""

    model: ModelSpec
    task: str
    frame_indices: np.ndarray
    timestamps: np.ndarray
    metadata: dict[str, Any]
    embeddings: np.ndarray | None = None
    frames: list[dict[str, Any]] | None = None

    def __post_init__(self):
        n = len(self.frame_indices)
        if len(self.timestamps) != n:
            raise ValueError("frame_indices and timestamps must have equal length")
        if self.embeddings is not None and len(self.embeddings) != n:
            raise ValueError("embedding rows must align with frame_indices")
        if self.frames is not None and len(self.frames) != n:
            raise ValueError("frame results must align with frame_indices")

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model.public_dict(),
            "task": self.task,
            "n_results": int(len(self.frame_indices)),
            "frame_indices": self.frame_indices.tolist(),
            "timestamps": self.timestamps.tolist(),
            "metadata": _json_value(self.metadata),
        }
        if self.embeddings is not None:
            result["embedding_shape"] = list(self.embeddings.shape)
        if self.frames is not None:
            result["frame_summaries"] = [
                {
                    "frame_index": int(self.frame_indices[i]),
                    "timestamp_s": float(self.timestamps[i]),
                    "instance_count": len(frame.get("instances", [])),
                    "values": _json_value(frame.get("values", {})),
                    "label": frame.get("label"),
                }
                for i, frame in enumerate(self.frames)
            ]
        return result

    def frame_result(self, native_frame_index: int) -> dict[str, Any] | None:
        matches = np.flatnonzero(self.frame_indices == native_frame_index)
        if len(matches) == 0:
            return None
        pos = int(matches[0])
        result = {
            "frame_index": int(self.frame_indices[pos]),
            "timestamp_s": float(self.timestamps[pos]),
        }
        if self.embeddings is not None:
            result["embedding"] = self.embeddings[pos].tolist()
        if self.frames is not None:
            result.update(_json_value(self.frames[pos]))
        return result

    def to_npz_bytes(self, analysis: dict[str, Any] | None = None) -> bytes:
        """Serialize numeric results compactly without requiring pickle."""
        payload: dict[str, Any] = {
            "frame_indices": self.frame_indices.astype(np.int64, copy=False),
            "timestamps_s": self.timestamps.astype(np.float64, copy=False),
            "metadata_json": np.asarray(json.dumps({
                "model": self.model.public_dict(),
                "task": self.task,
                "metadata": _json_value(self.metadata),
            })),
        }
        if self.embeddings is not None:
            payload["embeddings"] = self.embeddings.astype(np.float32, copy=False)
        if self.frames is not None:
            payload["frame_results_json"] = np.asarray(json.dumps(_json_value(self.frames)))
        if analysis is not None:
            payload["analysis_json"] = np.asarray(json.dumps(_json_value(analysis)))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        return buffer.getvalue()
