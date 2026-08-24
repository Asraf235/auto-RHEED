"""Fast, deterministic broad-background estimation for RHEED frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COARSE_PERCENTILE_DEFAULTS: dict[str, Any] = {
    "method": "coarse_percentile",
    "tile_size": 96,
    "percentile": 40.0,
    "smooth_sigma_tiles": 1.25,
    "sample_step": 2,
}


def normalize_background_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a validated, JSON-safe coarse-percentile configuration."""
    normalized = dict(COARSE_PERCENTILE_DEFAULTS)
    normalized.update(dict(config or {}))
    method = str(normalized.get("method", "coarse_percentile"))
    if method != "coarse_percentile":
        raise ValueError("Unknown background subtraction method")
    tile_size = int(normalized.get("tile_size", 96))
    percentile = float(normalized.get("percentile", 40.0))
    smooth_sigma = float(normalized.get("smooth_sigma_tiles", 1.25))
    sample_step = int(normalized.get("sample_step", 2))
    if tile_size < 4:
        raise ValueError("Background tile_size must be at least 4 pixels")
    if tile_size > 4096:
        raise ValueError("Background tile_size must not exceed 4096 pixels")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Background percentile must be between 0 and 100")
    if not np.isfinite(smooth_sigma) or smooth_sigma < 0.0:
        raise ValueError("Background smooth_sigma_tiles must be finite and non-negative")
    if sample_step < 1:
        raise ValueError("Background sample_step must be at least 1")
    if sample_step > 32:
        raise ValueError("Background sample_step must not exceed 32 pixels")
    return {
        "method": method,
        "tile_size": tile_size,
        "percentile": percentile,
        "smooth_sigma_tiles": smooth_sigma,
        "sample_step": sample_step,
    }


def estimate_coarse_percentile_background(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Estimate a smooth 2-D background from one percentile per coarse tile."""
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected a single (H, W) frame, got {frame.shape}")
    if frame.size == 0:
        raise ValueError("Cannot estimate a background for an empty frame")
    settings = normalize_background_config(config)
    tile = settings["tile_size"]
    height, width = frame.shape
    grid_h = (height + tile - 1) // tile
    grid_w = (width + tile - 1) // tile
    pad_h = grid_h * tile - height
    pad_w = grid_w * tile - width
    padded = np.pad(frame, ((0, pad_h), (0, pad_w)), mode="edge")
    tiles = padded.reshape(grid_h, tile, grid_w, tile)
    sampled_tiles = tiles[
        :,
        ::settings["sample_step"],
        :,
        ::settings["sample_step"],
    ]
    coarse = np.percentile(
        sampled_tiles,
        settings["percentile"],
        axis=(1, 3),
    ).astype(np.float32)
    sigma = settings["smooth_sigma_tiles"]
    if sigma > 0 and min(coarse.shape) > 1:
        coarse = cv2.GaussianBlur(
            coarse,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def subtract_coarse_percentile_background(
    frame: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Subtract the broad estimate, clip at zero, and preserve frame dtype."""
    frame = np.asarray(frame)
    background = estimate_coarse_percentile_background(frame, config)
    corrected = np.maximum(frame.astype(np.float32) - background, 0.0)
    if np.issubdtype(frame.dtype, np.integer):
        info = np.iinfo(frame.dtype)
        return np.clip(np.rint(corrected), info.min, info.max).astype(frame.dtype)
    return corrected.astype(frame.dtype, copy=False)


def subtract_coarse_percentile_stack(
    frames: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply coarse-percentile subtraction to every frame in a stack."""
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"Expected frames with shape (N, H, W), got {frames.shape}")
    settings = normalize_background_config(config)
    return np.stack(
        [subtract_coarse_percentile_background(frame, settings) for frame in frames],
        axis=0,
    )
