"""Batch orchestration shared by the web app, MCP server, and scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .base import create_adapter
from .preprocessing import build_preprocessing_context
from .types import InferenceRunResult, ModelSpec


class InferenceCancelled(RuntimeError):
    pass


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


def run_inference(
    frames: np.ndarray,
    timestamps: np.ndarray | None,
    spec: ModelSpec,
    *,
    device: str = "auto",
    batch_size: int = 8,
    stride: int = 1,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> InferenceRunResult:
    """Run a configured local adapter over selected frames in stable order."""
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"frames must have shape (N, H, W), got {frames.shape}")
    if len(frames) == 0:
        raise ValueError("Cannot run inference on an empty dataset")
    batch_size = int(batch_size)
    stride = int(stride)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    frame_indices = np.arange(0, len(frames), stride, dtype=np.int64)
    if timestamps is None:
        selected_timestamps = frame_indices.astype(np.float64)
    else:
        all_timestamps = np.asarray(timestamps, dtype=np.float64)
        if len(all_timestamps) != len(frames):
            raise ValueError("timestamps must contain one value per frame")
        selected_timestamps = all_timestamps[frame_indices]

    context = build_preprocessing_context(frames, frame_indices, spec.preprocessing)
    adapter = create_adapter(spec.adapter)
    descriptor = adapter.descriptor
    if spec.task and spec.task != descriptor.task:
        raise ValueError(
            f"Manifest task {spec.task!r} does not match adapter task {descriptor.task!r}"
        )

    runtime_metadata: dict = {}
    embedding_batches: list[np.ndarray] = []
    frame_results: list[dict] = []
    started = datetime.now(timezone.utc)
    try:
        runtime_metadata = adapter.load(spec, device)
        total = len(frame_indices)
        effective_batch_size = int(adapter.effective_batch_size(batch_size))
        if effective_batch_size < 1:
            raise ValueError("Adapter effective_batch_size must be at least 1")
        if progress:
            progress(0, total)
        for offset in range(0, total, effective_batch_size):
            if cancelled and cancelled():
                raise InferenceCancelled("Inference cancelled")
            batch_indices = frame_indices[offset:offset + effective_batch_size]
            batch_result = adapter.infer_batch(frames[batch_indices], context)
            if descriptor.task == "embedding":
                embeddings = np.asarray(batch_result.get("embeddings"), dtype=np.float32)
                if embeddings.ndim != 2 or len(embeddings) != len(batch_indices):
                    raise ValueError("Embedding adapter must return a (batch, dimensions) array")
                embedding_batches.append(embeddings)
            else:
                batch_frames = batch_result.get("frames")
                if not isinstance(batch_frames, list) or len(batch_frames) != len(batch_indices):
                    raise ValueError("Adapter must return one frame-result object per input frame")
                frame_results.extend(batch_frames)
            if progress:
                progress(min(offset + len(batch_indices), total), total)
    finally:
        adapter.close()

    completed = datetime.now(timezone.utc)
    metadata = {
        "adapter": spec.adapter,
        "device_requested": device,
        "batch_size": effective_batch_size,
        "batch_size_requested": batch_size,
        "stride": stride,
        "native_frame_shape": list(frames.shape[1:]),
        "preprocessing": context.to_dict(),
        "source_sha256": spec.local_source_sha256(),
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "runtime": runtime_metadata,
    }
    return InferenceRunResult(
        model=spec,
        task=descriptor.task,
        frame_indices=frame_indices,
        timestamps=selected_timestamps,
        embeddings=np.concatenate(embedding_batches, axis=0) if embedding_batches else None,
        frames=frame_results if descriptor.task != "embedding" else None,
        metadata=metadata,
    )
