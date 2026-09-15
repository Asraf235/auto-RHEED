"""
Strip-profile streak tracking: sum a horizontal strip of rows to get a
1D lateral intensity profile (specular + first-order streaks on each
side), subtract a slowly-varying background, find the three peaks, and
turn the (left, right) distances from the specular peak into an
in-plane lattice parameter — mirrors rheed_rocking_improved_NbSTO.ipynb
Cells 5/6 (strip-sum + per-frame ALS background subtraction), extended
with a few alternative background estimators and frame-to-frame peak
tracking so it can run automatically across an entire growth video.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import grey_opening, minimum_filter1d, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .wavelength import lambda_cm

BG_METHODS = ("als", "rolling_ball", "polynomial", "moving_min", "none")


# ── strip extraction ─────────────────────────────────────────────────────

def strip_sum_profile(frame: np.ndarray, y1: int, y2: int,
                       x1: Optional[int] = None, x2: Optional[int] = None) -> np.ndarray:
    """Sum rows y1:y2 of a single (H, W) frame → 1D profile over columns
    x1:x2 (defaults to the full width)."""
    y1, y2 = sorted((int(y1), int(y2)))
    H, W = frame.shape
    y1, y2 = max(0, y1), min(H, y2)
    x1 = 0 if x1 is None else max(0, int(x1))
    x2 = W if x2 is None else min(W, int(x2))
    if y2 <= y1 or x2 <= x1:
        raise ValueError("Strip ROI too small")
    return frame[y1:y2, x1:x2].astype(np.float64).sum(axis=0)


def strip_sum_profile_vtrack(frame: np.ndarray, y1: int, y2: int,
                             x1: Optional[int] = None, x2: Optional[int] = None,
                             band_px: Optional[int] = None,
                             center_col: Optional[float] = None):
    """
    Like strip_sum_profile, but instead of summing the WHOLE fixed
    vertical band y1:y2 it first locates the specular spot's row inside
    the ROI and sums only a band centered on it. This matters when the
    spots sweep vertically (e.g. a rocking experiment, or any drift in
    incidence angle): a fixed horizontal band loses the first-order spots
    as they move out of it, so the row-sum's lateral peaks jump to
    spurious features and the measured d oscillates even though the true
    in-plane spacing is constant. Following the specular vertically keeps
    all three spots in the summed band, decoupling vertical motion from
    the lateral Δx measurement.

    `center_col` (local strip column, 0 = the ROI's own left edge) seeds
    WHERE to look for the specular vertically; defaults to the ROI centre.
    Returns (profile, band_y_lo, band_y_hi) with the band rows in
    ABSOLUTE frame-pixel coordinates (for drawing the band on the overlay).
    """
    y1, y2 = sorted((int(y1), int(y2)))
    H, W = frame.shape
    y1, y2 = max(0, y1), min(H, y2)
    x1 = 0 if x1 is None else max(0, int(x1))
    x2 = W if x2 is None else min(W, int(x2))
    if y2 <= y1 or x2 <= x1:
        raise ValueError("Strip ROI too small")

    sub = frame[y1:y2, x1:x2].astype(np.float64)
    Hs, Ws = sub.shape

    # Locate the specular row from a central column window (the specular
    # is the bright central spot); a narrow window avoids latching onto a
    # bright first-order spot off to the side.
    cc = Ws // 2 if center_col is None else int(round(center_col))
    cc = max(0, min(Ws - 1, cc))
    col_half = max(3, Ws // 12)
    c_lo, c_hi = max(0, cc - col_half), min(Ws, cc + col_half + 1)
    vprof = sub[:, c_lo:c_hi].sum(axis=1)
    vprof = uniform_filter1d(vprof, size=max(3, Hs // 20), mode="nearest")
    srow = int(np.argmax(vprof))

    # Sum a band centered on that row.
    if band_px is None:
        band_px = max(12, Hs // 3)
    bh = band_px // 2
    r_lo, r_hi = max(0, srow - bh), min(Hs, srow + bh + 1)
    profile = sub[r_lo:r_hi, :].sum(axis=0)
    return profile, y1 + r_lo, y1 + r_hi


# ── background estimators ────────────────────────────────────────────────

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (same as the notebook's Cell 6)."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.T)
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L).tocsc()
    z = y.copy()
    for _ in range(niter):
        W.setdiag(w)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_rolling_ball(y: np.ndarray, radius: Optional[int] = None) -> np.ndarray:
    """Morphological opening (grayscale erosion→dilation) — fast, no
    iteration. `radius` defaults to ~8% of the profile length."""
    if radius is None:
        radius = max(5, len(y) // 12)
    size = 2 * radius + 1
    return grey_opening(y, size=size)


def baseline_polynomial(y: np.ndarray, degree: int = 4, iterations: int = 6) -> np.ndarray:
    """Iteratively-reweighted polynomial fit: refit only to points at or
    below the current baseline, so peaks get progressively excluded."""
    x = np.arange(len(y), dtype=float)
    weights = np.ones(len(y))
    z = y.copy()
    for _ in range(iterations):
        coeffs = np.polyfit(x, y, degree, w=weights)
        z = np.polyval(coeffs, x)
        weights = np.where(y > z, 0.1, 1.0)
    return z


def baseline_moving_min(y: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """Moving-minimum filter, lightly smoothed to avoid stair-steps."""
    if window is None:
        window = max(9, len(y) // 8)
    mn = minimum_filter1d(y, size=window, mode="nearest")
    return uniform_filter1d(mn, size=max(3, window // 3), mode="nearest")


_BASELINE_FUNCS = {
    "als": baseline_als,
    "rolling_ball": baseline_rolling_ball,
    "polynomial": baseline_polynomial,
    "moving_min": baseline_moving_min,
}


def subtract_background(profile: np.ndarray, method: str = "als", params: Optional[dict] = None):
    """Returns (baseline, subtracted). method='none' → baseline is all zeros."""
    params = params or {}
    if method == "none":
        return np.zeros_like(profile), profile.copy()
    fn = _BASELINE_FUNCS.get(method)
    if fn is None:
        raise ValueError(f"Unknown background method: {method!r} (choose from {BG_METHODS})")
    baseline = fn(profile, **params)
    return baseline, profile - baseline


# ── peak finding ──────────────────────────────────────────────────────────

@dataclass
class StreakPeaks:
    specular_x: Optional[float]
    left_x: Optional[float]
    right_x: Optional[float]

    @property
    def left_dx(self):
        if self.specular_x is None or self.left_x is None:
            return None
        return abs(self.specular_x - self.left_x)

    @property
    def right_dx(self):
        if self.specular_x is None or self.right_x is None:
            return None
        return abs(self.right_x - self.specular_x)

    @property
    def avg_dx(self):
        vals = [v for v in (self.left_dx, self.right_dx) if v is not None]
        return sum(vals) / len(vals) if vals else None


def _subpixel_refine(subtracted: np.ndarray, idx: int) -> float:
    """
    Parabolic interpolation around an integer-pixel peak index, using its
    two neighbors. Whole-pixel argmax picks are noticeably coarse —
    sub-pixel refinement is what actually makes the detected position
    track real (sub-pixel) peak drift smoothly instead of jumping in
    discrete 1px steps, which is what shows up as jitter in d(t).
    """
    L = len(subtracted)
    if idx <= 0 or idx >= L - 1:
        return float(idx)
    y0, y1, y2 = subtracted[idx - 1], subtracted[idx], subtracted[idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return float(idx)
    offset = 0.5 * (y0 - y2) / denom
    if abs(offset) > 1:  # degenerate fit — don't trust it
        return float(idx)
    return float(idx) + offset


def find_three_peaks(subtracted: np.ndarray, prev: Optional[StreakPeaks] = None,
                      search_window: int = 20, min_prominence_frac: float = 0.05,
                      min_signal_frac: float = 0.05, min_peak_gap: int = 4) -> StreakPeaks:
    """
    Find the specular + two first-order peaks in a background-subtracted
    profile.

    Cold start (no `prev`): uses scipy.find_peaks (prominence-based) to
    identify the three peaks from scratch — the strongest peak is taken
    as specular, with the nearest peak on each side as the first-order
    streaks.

    Tracking (`prev` given): rather than requiring a *formal* peak near
    each previous position (which is brittle — ordinary frame-to-frame
    noise can suppress a real streak just enough that it doesn't
    register as a local maximum by a global prominence test), each
    expected position is instead refined by a local-window argmax
    directly on the subtracted profile. This follows real intensity
    dips far more robustly; a position only counts as "lost" if its
    window contains no signal above the background at all.

    The specular peak (usually by far the brightest) is located FIRST,
    then the left/right search windows are clipped so they can't cross
    past it — without this, an overlapping search window will simply
    re-find the bright specular peak for "left" and "right" too, which
    silently collapses all three positions onto one point and corrupts
    every subsequent tracked frame (it never recovers on its own).
    """
    L = len(subtracted)
    mx = subtracted.max()

    if prev is not None and prev.specular_x is not None:
        floor = max(mx * min_signal_frac, 1e-9)

        def pick(expected, lo_bound=0, hi_bound=L):
            if expected is None:
                return None
            lo = max(lo_bound, int(round(expected)) - search_window)
            hi = min(hi_bound, int(round(expected)) + search_window + 1)
            if hi <= lo:
                return None
            window = subtracted[lo:hi]
            local_max = window.max()
            if local_max < floor:
                return None  # nothing but background/noise in this window
            peak_idx = lo + int(np.argmax(window))
            return _subpixel_refine(subtracted, peak_idx)

        def reacquire(lo_bound, hi_bound):
            """
            Fallback when a side's narrow tracking window comes up empty
            (it drifted farther than `search_window` since the last
            successful frame). Rather than giving up on that side until
            the WHOLE frame loses tracking, independently re-search the
            full region between the clip bound and the specular peak for
            any real, prominent local feature — this is what lets a
            streak that drifted out of the narrow window re-lock on its
            own, without needing specular to also be lost first.
            """
            lo, hi = max(0, lo_bound), min(L, hi_bound)
            if hi - lo < 3:
                return None
            region = subtracted[lo:hi]
            prominence = max(region.max() * min_prominence_frac, 1e-9)
            cand_idx, _ = find_peaks(region, prominence=prominence)
            if len(cand_idx) == 0:
                return None
            best = lo + cand_idx[np.argmax(region[cand_idx])]
            return _subpixel_refine(subtracted, int(best))

        specular_x = pick(prev.specular_x)
        if specular_x is not None:
            # Clip the side windows so they can't "steal" the specular peak.
            left_hi = int(specular_x) - min_peak_gap
            right_lo = int(specular_x) + min_peak_gap + 1
            left_x = pick(prev.left_x, hi_bound=left_hi) if prev.left_x is not None else None
            if left_x is None and prev.left_x is not None:
                left_x = reacquire(0, left_hi)
            right_x = pick(prev.right_x, lo_bound=right_lo) if prev.right_x is not None else None
            if right_x is None and prev.right_x is not None:
                right_x = reacquire(right_lo, L)
            return StreakPeaks(specular_x, left_x, right_x)
        # fall through to cold-start search if the specular peak was lost

    # Cold start (first frame, or tracking lost): need real peak
    # detection since there's no prior expectation to search around.
    if mx <= 0:
        return StreakPeaks(None, None, None)
    prominence = max(mx * min_prominence_frac, 1e-9)
    idx, _props = find_peaks(subtracted, prominence=prominence)
    if len(idx) == 0:
        return StreakPeaks(None, None, None)

    def nearest(candidates, target):
        if len(candidates) == 0 or target is None:
            return None
        return candidates[np.argmin(np.abs(candidates - target))]

    # Specular = the single strongest peak in the ROI; first-order
    # streaks = the nearest peak on each side of it.
    specular_idx = idx[np.argmax(subtracted[idx])]
    left_candidates = idx[idx < specular_idx]
    right_candidates = idx[idx > specular_idx]
    left_idx = nearest(left_candidates, specular_idx) if len(left_candidates) else None
    right_idx = nearest(right_candidates, specular_idx) if len(right_candidates) else None
    return StreakPeaks(
        specular_x=_subpixel_refine(subtracted, int(specular_idx)),
        left_x=_subpixel_refine(subtracted, int(left_idx)) if left_idx is not None else None,
        right_x=_subpixel_refine(subtracted, int(right_idx)) if right_idx is not None else None,
    )


def find_extra_peaks(subtracted: np.ndarray, specular_x: float,
                      left_x: float | None, right_x: float | None,
                      min_prominence_frac: float = 0.2, min_distance: int = 8,
                      anchor_exclusion: int = 6, max_order: int = 4) -> list[dict]:
    """
    Find genuinely prominent peaks in the strip BEYOND the already-known
    specular/±1 first-order trio (e.g. higher diffraction orders or
    surface-reconstruction streaks that a wide strip ROI also happens to
    capture). These are purely for visualization — the d-spacing
    measurement stays anchored to the ±1 first-order pair only.

    Order numbers are anchored explicitly to the known specular_x /
    left_x / right_x positions (order 0 / -1 / +1) rather than re-derived
    from this function's own (differently-thresholded) peak search —
    using a stricter prominence search to rank order ±1 itself can
    disagree with the windowed-search the 3-peak detector uses (e.g. a
    real first-order peak sitting on the specular's broad shoulder has
    high amplitude but low *prominence*, so a prominence-only search can
    silently skip it and report a *different*, unrelated peak as "-1").
    Anchoring avoids that inconsistency: extra peaks found here always
    start numbering from ±2.
    """
    mx = subtracted.max()
    if mx <= 0:
        return []
    prominence = max(mx * min_prominence_frac, 1e-9)
    idx, _props = find_peaks(subtracted, prominence=prominence, distance=min_distance)
    if len(idx) == 0:
        return []

    known = [v for v in (specular_x, left_x, right_x) if v is not None]
    candidates = []
    for i in idx:
        x = _subpixel_refine(subtracted, int(i))
        if any(abs(x - k) < anchor_exclusion for k in known):
            continue  # this is just one of the 3 already-displayed peaks
        candidates.append(x)

    left_extra = sorted((x for x in candidates if x < specular_x), reverse=True)  # nearest first
    right_extra = sorted(x for x in candidates if x > specular_x)  # nearest first

    peaks = []
    for rank, x in enumerate(left_extra[:max_order - 1]):
        order = -(rank + 2)  # extras start at -2 (since -1 is `left_x`)
        peaks.append({"x": x, "order": order,
                      "intensity": float(subtracted[int(round(x))])})
    for rank, x in enumerate(right_extra[:max_order - 1]):
        order = rank + 2  # extras start at +2 (since +1 is `right_x`)
        peaks.append({"x": x, "order": order,
                      "intensity": float(subtracted[int(round(x))])})
    return peaks


# ── peak width (FWHM) → coherence length ─────────────────────────────────

def fwhm_of_peak(subtracted: np.ndarray, peak_x: Optional[float]) -> Optional[float]:
    """
    Full-Width-at-Half-Maximum (in pixels) of the peak located at
    `peak_x` in a BACKGROUND-SUBTRACTED profile (so the half-max
    reference is the zero baseline). The width is read off at half the
    peak's height, with linear interpolation at each half-max crossing
    for sub-pixel precision — matching the sub-pixel peak positions the
    tracker already produces.

    A diffraction streak's FWHM is an inverse measure of surface order:
    it narrows as the surface gets smoother / more ordered (larger flat
    terraces) and broadens as it roughens. Converted through the same
    pixel→reciprocal-length calibration used for d-spacing, 1/Δk gives an
    in-plane coherence length (see d_from_dx, reused with the FWHM in
    place of the streak separation).

    Returns the width in pixels, or None if there's no usable peak (e.g.
    tracking lost it, or it sits on the profile edge so a half-max
    crossing can't be found on one side).
    """
    if peak_x is None:
        return None
    L = len(subtracted)
    xp = int(round(peak_x))
    if xp <= 0 or xp >= L - 1:
        return None
    peak_val = float(subtracted[xp])
    if peak_val <= 0:
        return None
    half = peak_val / 2.0

    # Walk left until the profile drops to/below half-max; interpolate.
    i = xp
    while i > 0 and subtracted[i] > half:
        i -= 1
    if subtracted[i] > half:           # never crossed before the edge
        return None
    yl0, yl1 = float(subtracted[i]), float(subtracted[i + 1])
    x_left = i + (half - yl0) / (yl1 - yl0) if yl1 != yl0 else float(i)

    # Walk right similarly.
    j = xp
    while j < L - 1 and subtracted[j] > half:
        j += 1
    if subtracted[j] > half:
        return None
    yr0, yr1 = float(subtracted[j]), float(subtracted[j - 1])
    x_right = j - (half - yr0) / (yr1 - yr0) if yr1 != yr0 else float(j)

    width = x_right - x_left
    return width if width > 0 else None


def fwhm_gauss_of_peak(subtracted: np.ndarray, peak_x: Optional[float],
                       window: Optional[int] = None) -> Optional[float]:
    """
    FWHM (px) of the peak at `peak_x` from a least-squares GAUSSIAN FIT
    of a window around it, FWHM = 2·√(2·ln2)·σ ≈ 2.3548·σ. This is the
    kSA-style measurement: fitting a model smooths over single-pixel
    noise at the peak top and gives a more stable width than the raw
    half-max read-off (fwhm_of_peak), at the cost of assuming a roughly
    Gaussian lineshape. The two are offered side by side so the user can
    compare.

    The fit window defaults to ~1.5× the rough half-max width so it spans
    the peak without pulling in neighbouring streaks. Returns None if
    there's no usable peak or the fit fails to converge.
    """
    if peak_x is None:
        return None
    L = len(subtracted)
    xp = int(round(peak_x))
    if xp <= 0 or xp >= L - 1:
        return None
    peak_val = float(subtracted[xp])
    if peak_val <= 0:
        return None

    if window is None:
        rough = fwhm_of_peak(subtracted, peak_x)
        window = int(max(5, round((rough if rough else 6.0) * 1.5)))
    lo, hi = max(0, xp - window), min(L, xp + window + 1)
    xs = np.arange(lo, hi, dtype=float)
    ys = subtracted[lo:hi].astype(float)
    if len(xs) < 5:
        return None

    def _gauss(x, A, mu, sigma, off):
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma)) + off

    p0 = [peak_val, float(peak_x), max(window / 2.0, 1.0), 0.0]
    bounds = ([0.0, lo, 0.5, -abs(peak_val)],
              [peak_val * 3.0, hi, window * 2.0, abs(peak_val)])
    try:
        popt, _ = curve_fit(_gauss, xs, ys, p0=p0, bounds=bounds, maxfev=5000)
    except Exception:
        return None
    sigma = abs(popt[2])
    fw = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    return float(fw) if fw > 0 else None


# ── lattice parameter from a strip profile ───────────────────────────────

def d_from_dx(dx_px: float, scale_cm_px: float, l_cm: float, v_kev: float) -> float:
    """d (Å) = λL / Δx, with Δx given in pixels (converted via scale_cm_px)."""
    dx_cm = dx_px * scale_cm_px
    d_cm = (lambda_cm(v_kev) * l_cm) / dx_cm
    return d_cm * 1e8
