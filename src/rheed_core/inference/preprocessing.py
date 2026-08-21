"""Deterministic preprocessing helpers shared by local model adapters."""

from __future__ import annotations

import numpy as np

from .types import PreprocessingContext


def build_preprocessing_context(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    config: dict | None = None,
) -> PreprocessingContext:
    config = dict(config or {})
    scaling = str(config.get("intensity_scaling", "dataset_percentile"))
    selected = frames[frame_indices]
    if selected.size == 0:
        raise ValueError("No frames selected for preprocessing")

    if scaling == "fixed":
        low = float(config.get("intensity_low", 0.0))
        high = float(config.get("intensity_high", 65535.0))
    elif scaling == "dtype_range":
        if np.issubdtype(selected.dtype, np.integer):
            info = np.iinfo(selected.dtype)
            low, high = float(info.min), float(info.max)
        else:
            low, high = 0.0, 1.0
    elif scaling in {"dataset_percentile", "per_frame_percentile"}:
        low_pct = float(config.get("low_percentile", 0.5))
        high_pct = float(config.get("high_percentile", 99.5))
        if not (0 <= low_pct < high_pct <= 100):
            raise ValueError("Preprocessing percentiles must satisfy 0 <= low < high <= 100")
        # Bound percentile work for very long/high-resolution videos while
        # sampling uniformly across both time and image space.
        max_frames = max(1, int(config.get("percentile_sample_frames", 64)))
        temporal_step = max(1, len(selected) // max_frames)
        sampled = selected[::temporal_step][:max_frames]
        max_pixels = max(10_000, int(config.get("percentile_sample_pixels", 2_000_000)))
        spatial_step = max(1, int(np.sqrt(sampled.size / max_pixels)))
        sample_values = sampled[:, ::spatial_step, ::spatial_step]
        low, high = (float(v) for v in np.percentile(sample_values, [low_pct, high_pct]))
    else:
        raise ValueError(
            "Unknown intensity_scaling; choose dataset_percentile, "
            "per_frame_percentile, dtype_range, or fixed"
        )

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Preprocessing intensity bounds are not finite")
    if high <= low:
        high = low + 1.0
    return PreprocessingContext(low, high, scaling, config)


def frame_to_rgb_uint8(frame: np.ndarray, context: PreprocessingContext) -> np.ndarray:
    """Convert a native grayscale frame to RGB without using display state."""
    if frame.ndim != 2:
        raise ValueError(f"Expected a single (H, W) grayscale frame, got {frame.shape}")
    if context.scaling == "per_frame_percentile":
        low_pct = float(context.config.get("low_percentile", 0.5))
        high_pct = float(context.config.get("high_percentile", 99.5))
        low, high = (float(v) for v in np.percentile(frame, [low_pct, high_pct]))
        if high <= low:
            high = low + 1.0
    else:
        low, high = context.intensity_low, context.intensity_high
    normalized = np.clip((frame.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)
