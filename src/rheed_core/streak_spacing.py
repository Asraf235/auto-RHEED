"""
In-plane lattice parameter from RHEED streak spacing: d = λL / Δx.

Mirrors the math in the Streak Spacing tool and the Growth Analysis tab's
substrate-calibration step (rheed_webapp/templates/index.html), so the
same calculation is available to non-browser consumers.
"""
from dataclasses import dataclass

from .wavelength import lambda_cm
from .materials import expected_d, find_material_key


@dataclass
class StreakMeasurement:
    dx_px: float
    scale_cm_px: float
    l_cm: float
    v_kev: float
    d_angstrom: float
    expected_d_angstrom: float | None = None
    deviation_pct: float | None = None


def measure_d(dx_px: float, scale_cm_px: float, l_cm: float, v_kev: float,
               material: str | None = None, zone: str | None = None) -> StreakMeasurement:
    """
    Compute the in-plane d-spacing from a measured streak pixel separation,
    given an already-known pixel scale (e.g. from the 5-point calibration).
    """
    if dx_px <= 0:
        raise ValueError("dx_px must be positive")
    dx_cm = dx_px * scale_cm_px
    lam_cm = lambda_cm(v_kev)
    d_cm = (lam_cm * l_cm) / dx_cm
    d_angstrom = d_cm * 1e8

    exp_d = None
    deviation = None
    if material:
        mat_key = find_material_key(material)
        if mat_key and zone:
            exp_d = expected_d(mat_key, zone)
            if exp_d is not None:
                deviation = (d_angstrom - exp_d) / exp_d * 100.0

    return StreakMeasurement(
        dx_px=dx_px, scale_cm_px=scale_cm_px, l_cm=l_cm, v_kev=v_kev,
        d_angstrom=d_angstrom, expected_d_angstrom=exp_d, deviation_pct=deviation,
    )


def solve_scale_from_standard(dx_px: float, l_cm: float, v_kev: float,
                               material: str, zone: str) -> float:
    """
    The Growth Analysis "known-standard substrate" calibration: given a
    measured streak spacing on a substrate of known material/zone, solve
    for the missing pixel scale (cm/px).
        d_known = λL / (Δx_px · scale_cm_px)
        ⇒ scale_cm_px = λL / (d_known · Δx_px)
    """
    mat_key = find_material_key(material)
    if mat_key is None:
        raise ValueError(f"Unrecognized material: {material!r}")
    d_known = expected_d(mat_key, zone)
    if d_known is None:
        raise ValueError(f"No reference d-spacing for {mat_key} along [{zone}]")
    lam_cm = lambda_cm(v_kev)
    dx_cm = (lam_cm * l_cm) / (d_known * 1e-8)
    return dx_cm / dx_px


def measure_d_by_ratio(dx_px_reference: float, d_reference_angstrom: float,
                        dx_px_unknown: float) -> float:
    """
    Direct-ratio method: when a known-d reference (e.g. substrate) and an
    unknown spacing (e.g. film) are visible in the SAME frame, the camera
    constant cancels entirely:
        d_unknown = d_reference * (Δx_reference / Δx_unknown)
    No wavelength, L, or pixel scale needed.
    """
    if dx_px_unknown <= 0:
        raise ValueError("dx_px_unknown must be positive")
    return d_reference_angstrom * (dx_px_reference / dx_px_unknown)
