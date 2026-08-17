"""
RHEED Studio MCP server.

Exposes task-shaped tools (not a 1:1 mirror of every rheed_core function)
so an agent can drive a RHEED analysis without re-deriving the workflow
order itself. Holds its own RheedSession, independent of any running
rheed_webapp process — see the project's architecture notes.

Run directly for local/stdio use:
    python -m rheed_mcp.server
"""
import base64
from typing import Optional

from mcp.server.fastmcp import FastMCP, Image

from rheed_core import RheedSession
from rheed_core.materials import MATERIALS, expected_d, find_material_key
from rheed_core.streak_spacing import (
    measure_d,
    solve_scale_from_standard,
    measure_d_by_ratio,
)
from rheed_core.calibration import Point, calibrate as _calibrate

mcp = FastMCP("rheed-studio")

# One session for the lifetime of this MCP server process. Independent
# of any rheed_webapp session — the agent and a human looking at the
# browser are not guaranteed to be looking at the same loaded file.
session = RheedSession()


def _b64_to_image(b64_png: str) -> Image:
    return Image(data=base64.b64decode(b64_png), format="png")


# ── file loading / status ────────────────────────────────────────────────

@mcp.tool()
def load_rheed_file(path: str) -> dict:
    """
    Load a RHEED dataset from disk: .h5, .npy, video (.mp4/.avi/.mov/.mkv),
    or a single still image (.png/.jpg/.tif/.bmp/.img — including kSA
    RHEED .img files). Returns n_frames, width, height, duration, fps.
    """
    import os
    suffix = os.path.splitext(path)[1].lower()
    summary = session.load_file(path, suffix)
    return summary


@mcp.tool()
def get_status() -> dict:
    """Report whether a file is loaded and its basic properties."""
    if session.frames is None:
        return {"loaded": False}
    H, W = session.shape
    return {
        "loaded": True,
        "n_frames": session.n_frames,
        "width": W,
        "height": H,
        "clim": list(session.clim),
        "colormap": session.colormap,
    }


@mcp.tool()
def get_frame_image(idx: int = 0) -> Image:
    """Return frame `idx` (0-based) as a PNG image, so a vision-capable
    agent can look directly at the diffraction pattern."""
    if session.frames is None:
        raise ValueError("No file loaded — call load_rheed_file first")
    b64, _t = session.frame_png_b64(idx)
    return _b64_to_image(b64)


@mcp.tool()
def rotate_frame(direction: str = "cw") -> dict:
    """Rotate the loaded dataset 90° ('cw' or 'ccw'). Applies to the
    actual pixel data (all frames), so any pixel coordinates obtained
    before rotating are no longer valid afterward."""
    H, W = session.rotate(direction)
    return {"width": W, "height": H}


# ── calibration ───────────────────────────────────────────────────────────

@mcp.tool()
def calibrate_frame(direct_beam: dict, shadow_edge: dict, specular: dict,
                     cam_edge_left: dict, cam_edge_right: dict,
                     l_cm: float, d_cm: float) -> dict:
    """
    5-point RHEED calibration. Each point is {"x": ..., "y": ...} in the
    NATIVE pixel coordinates of the currently loaded frame (use
    get_frame_image / a spot-detection tool to find these).
      l_cm — substrate-to-screen distance (cm)
      d_cm — physical phosphor screen diameter (cm)
    Returns scale_cm_px (pixel scale) and alpha_deg (angle of incidence).
    """
    result = _calibrate(
        direct_beam=Point(**direct_beam),
        shadow_edge=Point(**shadow_edge),
        specular=Point(**specular),
        cam_edge_left=Point(**cam_edge_left),
        cam_edge_right=Point(**cam_edge_right),
        l_cm=l_cm, d_cm=d_cm,
    )
    return {
        "scale_cm_px": result.scale_cm_px,
        "alpha_deg": result.alpha_deg,
        "d_px": result.d_px,
        "dy_px": result.dy_px,
        "dy_cm": result.dy_cm,
    }


# ── materials reference ──────────────────────────────────────────────────

@mcp.tool()
def list_materials() -> list[str]:
    """List material keys with a known lattice parameter (for streak-
    spacing comparisons), e.g. 'SrTiO3', 'Si', 'MgO'."""
    return sorted(MATERIALS.keys())


@mcp.tool()
def get_expected_d(material: str, zone: str) -> Optional[float]:
    """
    Expected in-plane d-spacing (Å) for a known material + RHEED zone
    axis (zone is e.g. '100', '110', '001' — no brackets).
    Returns null if the material or zone isn't recognized.
    """
    key = find_material_key(material)
    if key is None:
        return None
    return expected_d(key, zone)


# ── streak spacing / lattice parameter ───────────────────────────────────

@mcp.tool()
def measure_streak_spacing(dx_px: float, scale_cm_px: float, l_cm: float,
                            v_kev: float, material: Optional[str] = None,
                            zone: Optional[str] = None) -> dict:
    """
    Compute the in-plane lattice parameter d (Å) from a measured streak
    pixel separation, given an already-known pixel scale (e.g. from
    calibrate_frame). If material+zone are given, also returns the
    expected d and % deviation for comparison.
    """
    result = measure_d(dx_px, scale_cm_px, l_cm, v_kev, material, zone)
    return {
        "d_angstrom": result.d_angstrom,
        "expected_d_angstrom": result.expected_d_angstrom,
        "deviation_pct": result.deviation_pct,
    }


@mcp.tool()
def solve_pixel_scale_from_known_substrate(dx_px: float, l_cm: float, v_kev: float,
                                            material: str, zone: str) -> float:
    """
    Use case: camera/screen edges aren't visible (only a zoomed-in spot
    region), but L (camera length) and beam voltage are known, and the
    frame shows a well-characterized substrate. Solves for the missing
    pixel scale (cm/px) from the substrate's known d-spacing, so later
    frames (e.g. of a growing film) can be measured with
    measure_streak_spacing using the returned scale_cm_px.
    """
    return solve_scale_from_standard(dx_px, l_cm, v_kev, material, zone)


@mcp.tool()
def measure_streak_spacing_by_ratio(dx_px_reference: float, d_reference_angstrom: float,
                                     dx_px_unknown: float) -> float:
    """
    Most robust lattice-parameter method: when a known-d reference (e.g.
    substrate) and an unknown spacing (e.g. growing film) are visible in
    the SAME frame, the camera constant cancels — no wavelength, L, or
    pixel scale needed. Returns d_unknown (Å).
    """
    return measure_d_by_ratio(dx_px_reference, d_reference_angstrom, dx_px_unknown)


# ── ROI intensity ─────────────────────────────────────────────────────────

@mcp.tool()
def compute_roi_intensity(roi_type: str, roi: dict) -> dict:
    """
    Sum pixel intensity inside an ROI for every loaded frame, returning
    a time series. `roi` (in NATIVE pixel coordinates) is:
      circle → {"cx", "cy", "r"}
      rect   → {"x1", "y1", "x2", "y2"}
      line   → {"x1", "y1", "x2", "y2", "width" (optional, default 3)}
    """
    intensities, timestamps = session.compute_intensity(roi_type, roi)
    return {
        "intensities": intensities,
        "timestamps": timestamps,
        "min": min(intensities), "max": max(intensities),
        "mean": sum(intensities) / len(intensities),
    }


if __name__ == "__main__":
    mcp.run()
