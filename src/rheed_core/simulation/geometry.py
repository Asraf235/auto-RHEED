"""Physical measurements on a simulated planar RHEED detector."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from rheed_core.wavelength import lambda_angstrom


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _point(value: Mapping[str, Any], width_px: int, height_px: int) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError("detector points must be objects with x and y")
    x = float(value.get("x"))
    y = float(value.get("y"))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("detector point coordinates must be finite")
    if not 0 <= x <= width_px or not 0 <= y <= height_px:
        raise ValueError("detector points must lie inside the native image")
    return x, y


def measure_detector_distance(
    point1: Mapping[str, Any],
    point2: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    screen: Mapping[str, Any],
    beam_energy_kev: float,
) -> dict[str, Any]:
    """Measure detector-plane and reciprocal-space separation between two pixels.

    Pixel coordinates are native detector-edge coordinates: ``(0, 0)`` is the
    upper-left screen edge and ``(W, H)`` is the lower-right edge. Reciprocal
    separation is the exact difference between the two elastic outgoing
    wavevectors, not a small-angle approximation.
    """

    if len(image_shape) != 2:
        raise ValueError("detector image_shape must contain height and width")
    height_px = int(image_shape[0])
    width_px = int(image_shape[1])
    if height_px < 1 or width_px < 1:
        raise ValueError("detector image dimensions must be positive")
    p1 = _point(point1, width_px, height_px)
    p2 = _point(point2, width_px, height_px)

    distance_mm = _finite_positive(screen.get("screen_distance_mm"), "screen distance")
    width_mm = _finite_positive(screen.get("screen_width_mm"), "screen width")
    height_mm = _finite_positive(screen.get("screen_height_mm"), "screen height")
    energy_kev = _finite_positive(beam_energy_kev, "beam energy")
    plane_mode = str(screen.get("plane_mode") or "vertical")
    if plane_mode not in {"vertical", "specular-normal"}:
        raise ValueError(f"unsupported detector plane mode: {plane_mode}")

    def screen_xy(point: tuple[float, float]) -> tuple[float, float]:
        x_px, y_px = point
        return (
            (x_px / width_px - 0.5) * width_mm,
            (0.5 - y_px / height_px) * height_mm,
        )

    xy1 = screen_xy(p1)
    xy2 = screen_xy(p2)
    dx_mm = xy2[0] - xy1[0]
    dy_mm = xy2[1] - xy1[1]
    detector_distance_mm = math.hypot(dx_mm, dy_mm)

    # Express rays in the detector's orthonormal (normal, right, up) basis.
    # For the specular-normal plane, torch-rheed places its screen origin at
    # the sample-surface ray's intersection with the tilted detector plane.
    reference_angle_deg = float(screen.get("reference_angle_deg") or 0.0)
    origin_up_mm = 0.0
    if plane_mode == "specular-normal":
        origin_up_mm = -distance_mm * math.tan(math.radians(reference_angle_deg))

    def unit_ray(xy: tuple[float, float]) -> np.ndarray:
        ray = np.asarray(
            [distance_mm, xy[0], origin_up_mm + xy[1]],
            dtype=np.float64,
        )
        return ray / np.linalg.norm(ray)

    ray1 = unit_ray(xy1)
    ray2 = unit_ray(xy2)
    wavelength = lambda_angstrom(energy_kev)
    wave_number = 2.0 * math.pi / wavelength
    delta_q = wave_number * (ray2 - ray1)
    reciprocal_distance = float(np.linalg.norm(delta_q))
    dot = float(np.clip(np.dot(ray1, ray2), -1.0, 1.0))
    scattering_angle_deg = math.degrees(math.acos(dot))

    return {
        "point1_px": {"x": p1[0], "y": p1[1]},
        "point2_px": {"x": p2[0], "y": p2[1]},
        "point1_mm": {"x": xy1[0], "y": xy1[1]},
        "point2_mm": {"x": xy2[0], "y": xy2[1]},
        "delta_mm": {"x": dx_mm, "y": dy_mm},
        "distance_mm": detector_distance_mm,
        "delta_q_A_inv": {
            "normal": float(delta_q[0]),
            "horizontal": float(delta_q[1]),
            "vertical": float(delta_q[2]),
        },
        "reciprocal_distance_A_inv": reciprocal_distance,
        "real_space_period_A": (
            None if reciprocal_distance <= 0 else 2.0 * math.pi / reciprocal_distance
        ),
        "scattering_angle_deg": scattering_angle_deg,
        "beam_energy_kev": energy_kev,
        "electron_wavelength_A": wavelength,
        "screen": {
            "plane_mode": plane_mode,
            "screen_distance_mm": distance_mm,
            "screen_width_mm": width_mm,
            "screen_height_mm": height_mm,
            "reference_angle_deg": reference_angle_deg,
        },
    }
