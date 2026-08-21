"""Lightweight excerpt of PNNL RHAAPSODY's changepoint detector.

Source: https://github.com/pnnl/RHAAPSODY/blob/d1591d16528926be76b106408cbe453842406685/change_detection.py
Copyright 2024 Battelle Memorial Institute
License: BSD-2-Clause; see THIRD_PARTY_LICENSES/RHAAPSODY-BSD-2-Clause.txt.

Only the NumPy-based detection methods are retained. RHAAPSODY's plotting,
HDF5 parsing, pandas aggregation, and ruptures convenience functions are not
included because Auto RHEED supplies its own UI, data loading, and exports.
The detector logic below is otherwise kept equivalent to the cited source.
"""

import numpy as np


class ChangepointDetection:
    """Class for the changepoint detection."""

    def __init__(
        self,
        cost_threshold: float = 0.06,
        window_size=np.inf,
        min_time_between_changepoints: int = 10,
    ):
        """Initialize self."""
        self.cost_threshold = cost_threshold
        self.window_size = window_size
        self.min_time_between_changepoints = min_time_between_changepoints
        self._reset()

    def _reset(self):
        self.changepoints = [0]
        self.detected_times = [0]

    @staticmethod
    def segmented_cost(matrix: np.ndarray, tau: int, start: int = 0, end: int = None):
        """Return the cost of segmenting the square matrix at tau."""
        sub_matrix_1 = matrix[start:tau, start:tau]
        val1 = np.diagonal(sub_matrix_1).sum()
        val1 -= sub_matrix_1.sum() / (tau - start)

        sub_matrix_2 = matrix[tau:end, tau:end]
        val2 = np.diagonal(sub_matrix_2).sum()
        val2 -= sub_matrix_2.sum() / (end - tau)

        sub_matrix_3 = matrix[start:end, start:end]
        val3 = np.diagonal(sub_matrix_3).sum()
        val3 -= sub_matrix_3.sum() / (end - start)

        return (val3 - val1 - val2) / (end - start)

    def maximize_segmented_cost(self, matrix, current_time, window_start):
        """Maximize the segmented cost for the square matrix on an interval."""
        xs = np.arange(window_start + 1, current_time, 1).astype(int)
        vals = [
            self.segmented_cost(matrix, x, start=window_start, end=current_time)
            for x in xs
        ]

        if len(vals) > 0:
            max_idx = np.argmax(vals)
            max_time = xs[max_idx]
            max_val = vals[max_idx]
            return (max_time, max_val)
        return []

    def get_changepoint(self, matrix: np.ndarray, step: int):
        """Get the changepoints on the given interval."""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Cannot do changepoint detection on a rectangular matrix, "
                f"({matrix.shape[0]}, {matrix.shape[1]})."
            )
        current_time = step
        window_start = max([self.changepoints[-1], current_time - self.window_size])
        vals = self.maximize_segmented_cost(
            matrix=matrix, current_time=current_time, window_start=window_start
        )

        actual_changepoint = False
        if len(vals) > 0:
            proposed_changepoint = vals[0]
            changepoint_amplitude = vals[1]

            if current_time - self.changepoints[-1] > self.min_time_between_changepoints:
                if changepoint_amplitude > self.cost_threshold:
                    self.changepoints.append(proposed_changepoint)
                    self.detected_times.append(current_time)
                    actual_changepoint = True
        else:
            proposed_changepoint = np.nan
            changepoint_amplitude = np.nan
            actual_changepoint = False

        return (
            proposed_changepoint,
            changepoint_amplitude,
            current_time,
            actual_changepoint,
        )

    @property
    def current_changepoint(self):
        """Return the most recent changepoint."""
        return self.changepoints[-1]
