"""
5-point frame calibration: pixel scale (cm/px) and angle of incidence α
from clicked direct-beam / shadow-edge / specular / camera-edge points.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float
    y: float


@dataclass
class CalibrationResult:
    scale_cm_px: float
    d_px: float
    dy_px: float
    dy_cm: float
    alpha_deg: float
    direct_beam_y: float
    shadow_edge_y: float
    specular_y: float


def calibrate(direct_beam: Point, shadow_edge: Point, specular: Point,
              cam_edge_left: Point, cam_edge_right: Point,
              l_cm: float, d_cm: float) -> CalibrationResult:
    """
    Args (all in the SAME native pixel coordinate space — i.e. already
    scaled from display/canvas pixels to the original frame's pixels):
      direct_beam, shadow_edge, specular, cam_edge_left, cam_edge_right
      l_cm — substrate-to-screen distance
      d_cm — physical phosphor screen diameter
    """
    d_px = float(np.hypot(cam_edge_right.x - cam_edge_left.x,
                           cam_edge_right.y - cam_edge_left.y))
    if d_px < 1:
        raise ValueError("Camera edge points too close")
    scale = d_cm / d_px  # cm per pixel

    dy_px = specular.y - shadow_edge.y  # positive = specular below shadow
    dy_cm = dy_px * scale
    alpha = float(np.degrees(np.arctan(dy_cm / l_cm)))

    return CalibrationResult(
        scale_cm_px=round(scale, 6),
        d_px=round(d_px, 1),
        dy_px=round(dy_px, 1),
        dy_cm=round(dy_cm, 4),
        alpha_deg=round(alpha, 4),
        direct_beam_y=round(direct_beam.y, 1),
        shadow_edge_y=round(shadow_edge.y, 1),
        specular_y=round(specular.y, 1),
    )
