"""RHAAPSODY changepoint analysis over existing Auto RHEED embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._vendor.rhaapsody_changepoint import ChangepointDetection
from .embedding_analysis import EmbeddingAnalysisAdapter, EmbeddingAnalysisDescriptor


_UPSTREAM_URL = "https://github.com/pnnl/RHAAPSODY"
_UPSTREAM_REVISION = "d1591d16528926be76b106408cbe453842406685"


def _integer_option(options: dict[str, Any], name: str, default: int,
                    minimum: int) -> int:
    value = options.get(name, default)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer of at least {minimum}") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(numeric)


class RhaapsodyChangepointAdapter(EmbeddingAnalysisAdapter):
    descriptor = EmbeddingAnalysisDescriptor(
        name="rhaapsody-changepoint",
        task="changepoint",
        description=(
            "PNNL RHAAPSODY kernel-similarity changepoint detection over existing "
            "frame embeddings"
        ),
        default_options={
            "cost_threshold": 0.06,
            "window_size": 300,
            "min_time_between_changepoints": 10,
            "starting_period": 30,
        },
        provenance={
            "project": "RHAAPSODY",
            "organization": "Pacific Northwest National Laboratory",
            "source_url": _UPSTREAM_URL,
            "source_revision": _UPSTREAM_REVISION,
            "license": "BSD-2-Clause",
            "local_modification": (
                "NumPy detector excerpt only; Auto RHEED supplies embeddings, "
                "frame alignment, visualization, and exports"
            ),
        },
    )

    def __init__(self):
        self._similarity_matrix: np.ndarray | None = None

    def array_artifacts(self) -> dict[str, np.ndarray]:
        if self._similarity_matrix is None:
            return {}
        return {"similarity-matrix": self._similarity_matrix}

    def analyze(
        self,
        embeddings: np.ndarray,
        frame_indices: np.ndarray,
        timestamps: np.ndarray,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"embeddings must have shape (N, D), got {x.shape}")
        n_samples = len(x)
        if len(frame_indices) != n_samples or len(timestamps) != n_samples:
            raise ValueError("Embeddings, frame indices, and timestamps must align")

        starting_period = _integer_option(options, "starting_period", 30, 2)
        min_gap = _integer_option(
            options, "min_time_between_changepoints", 10, 0
        )
        if starting_period >= n_samples:
            raise ValueError(
                f"starting_period must be smaller than the {n_samples} embedding rows"
            )

        threshold = float(options.get("cost_threshold", 0.06))
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("cost_threshold must be a finite, non-negative number")

        raw_window = options.get("window_size", 300)
        if raw_window is None or str(raw_window).strip().lower() in {"full", "inf", "infinity"}:
            window_size = np.inf
            recorded_window = None
        else:
            window_size = _integer_option(options, "window_size", 300, 2)
            recorded_window = int(window_size)

        # RHAAPSODY fixes the center from its initial starting period, then
        # performs changepoint detection on a cosine-similarity kernel.
        centered = x - np.mean(x[:starting_period], axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        zero_rows = np.flatnonzero(norms <= 1e-12)
        if len(zero_rows):
            raise ValueError(
                "Centered embeddings contain zero-length rows, so cosine similarity "
                f"is undefined (positions: {zero_rows[:8].tolist()})"
            )
        normalized = centered / norms[:, np.newaxis]
        similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
        np.fill_diagonal(similarity, 1.0)

        detector = ChangepointDetection(
            cost_threshold=threshold,
            window_size=window_size,
            min_time_between_changepoints=min_gap,
        )
        score_positions: list[int] = []
        score_frame_indices: list[int] = []
        score_timestamps: list[float] = []
        proposed_positions: list[int | None] = []
        proposed_frame_indices: list[int | None] = []
        amplitudes: list[float | None] = []
        detection_flags: list[bool] = []
        changepoints: list[dict[str, Any]] = []

        for seen in range(starting_period + 1, n_samples + 1):
            proposed, amplitude, detected_at, actual = detector.get_changepoint(
                similarity[:seen, :seen], seen
            )
            score_position = seen - 1
            proposed_position = int(proposed) if np.isfinite(proposed) else None
            amplitude_value = float(amplitude) if np.isfinite(amplitude) else None

            score_positions.append(score_position)
            score_frame_indices.append(int(frame_indices[score_position]))
            score_timestamps.append(float(timestamps[score_position]))
            proposed_positions.append(proposed_position)
            proposed_frame_indices.append(
                int(frame_indices[proposed_position])
                if proposed_position is not None and proposed_position < n_samples
                else None
            )
            amplitudes.append(amplitude_value)
            detection_flags.append(bool(actual))

            if actual and proposed_position is not None:
                detected_position = min(int(detected_at) - 1, n_samples - 1)
                changepoints.append({
                    "position": proposed_position,
                    "frame_index": int(frame_indices[proposed_position]),
                    "timestamp_s": float(timestamps[proposed_position]),
                    "detected_position": detected_position,
                    "detected_frame_index": int(frame_indices[detected_position]),
                    "detected_timestamp_s": float(timestamps[detected_position]),
                    "amplitude": amplitude_value,
                })

        # Keep the full matrix as a file artifact. The JSON contains only
        # alignment metadata so clients can request a bounded display view.
        self._similarity_matrix = np.asarray(similarity, dtype=np.float32)
        return {
            "adapter": self.descriptor.name,
            "task": self.descriptor.task,
            "parameters": {
                "cost_threshold": threshold,
                "window_size": recorded_window,
                "min_time_between_changepoints": min_gap,
                "starting_period": starting_period,
                "embedding_rows": n_samples,
                "embedding_dimensions": int(x.shape[1]),
                "similarity": "fixed-initial-mean centered cosine",
            },
            "changepoints": changepoints,
            "score_positions": score_positions,
            "score_frame_indices": score_frame_indices,
            "score_timestamps": score_timestamps,
            "proposed_positions": proposed_positions,
            "proposed_frame_indices": proposed_frame_indices,
            "amplitudes": amplitudes,
            "detection_flags": detection_flags,
            "similarity_matrix": {
                "artifact": "similarity-matrix",
                "shape": [n_samples, n_samples],
                "dtype": "float32",
                "value_range": [-1.0, 1.0],
                "frame_indices": np.asarray(frame_indices, dtype=np.int64).tolist(),
                "timestamps": np.asarray(timestamps, dtype=np.float64).tolist(),
            },
            "provenance": dict(self.descriptor.provenance),
        }
