"""Direct-beam normalization for RHEED intensity series.

The correction is intentionally represented as a per-frame multiplicative
factor.  It can therefore be applied to an analysis trace or to a rendered
frame without mutating the loaded scientific dataset.
"""

from __future__ import annotations

import numpy as np


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    smoothed = np.empty_like(values)
    for index in range(values.size):
        segment = values[max(0, index - radius):min(values.size, index + radius + 1)]
        valid = segment[np.isfinite(segment) & (segment > 0)]
        smoothed[index] = np.median(valid) if valid.size else np.nan
    return smoothed


def build_intensity_correction(
    intensities,
    *,
    smoothing_window: int = 1,
    max_factor: float = 1000.0,
) -> dict:
    """Build a stable direct-beam correction curve.

    Non-finite measurements are linearly interpolated. Non-positive values
    are treated as a fully clipped beam and receive the configured maximum
    correction rather than causing a division by zero.
    """
    raw = np.asarray(intensities, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Direct-beam intensity must be a non-empty 1D series")

    try:
        window = int(smoothing_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("Smoothing window must be a positive odd integer") from exc
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd integer")
    if not np.isfinite(max_factor) or max_factor < 1:
        raise ValueError("Maximum correction factor must be at least 1")

    positive = raw[np.isfinite(raw) & (raw > 0)]
    if not positive.size:
        raise ValueError("Direct-beam ROI has no positive finite intensity values")

    cleaned = raw.copy()
    invalid_finite = ~np.isfinite(cleaned)
    if invalid_finite.any():
        valid_indices = np.flatnonzero(~invalid_finite)
        if not valid_indices.size:
            raise ValueError("Direct-beam ROI has no finite intensity values")
        bad_indices = np.flatnonzero(invalid_finite)
        cleaned[bad_indices] = np.interp(bad_indices, valid_indices, cleaned[valid_indices])

    baseline = _rolling_median(cleaned, window)
    baseline_positive = baseline[np.isfinite(baseline) & (baseline > 0)]
    if not baseline_positive.size:
        raise ValueError("Direct-beam baseline has no positive values")

    # Persist a conventional 0–1 sensitivity curve. Correcting another
    # signal is then simply ``signal / normalized_curve``.
    reference = float(np.max(baseline_positive))

    normalized = np.clip(
        np.where(
            np.isfinite(baseline) & (baseline > 0),
            baseline / reference,
            0.0,
        ),
        0.0,
        1.0,
    )
    # The persisted curve remains an honest 0–1 measurement. Application
    # uses a floored copy so a fully clipped beam cannot divide by zero or
    # amplify a frame without bound.
    application_curve = np.clip(normalized, 1.0 / max_factor, 1.0)
    factors = 1.0 / application_curve
    positive_normalized = normalized[normalized > 0]
    required_max_factor = (
        float(1.0 / np.min(positive_normalized))
        if positive_normalized.size else None
    )
    cap_engaged_frames = int(np.count_nonzero(normalized < 1.0 / max_factor))

    return {
        "measured_intensities": raw.tolist(),
        "baseline_intensities": baseline.tolist(),
        "normalized_curve": normalized.tolist(),
        "application_curve": application_curve.tolist(),
        "correction_factors": factors.tolist(),
        "reference_intensity": reference,
        "normalization": "maximum",
        "smoothing_window": window,
        "max_factor": float(max_factor),
        "required_max_factor": required_max_factor,
        "cap_engaged_frames": cap_engaged_frames,
        "zero_intensity_frames": int(np.count_nonzero(normalized == 0)),
    }
