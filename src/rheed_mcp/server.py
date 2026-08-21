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
import json
import os
from pathlib import Path
from typing import Optional
import uuid

from mcp.server.fastmcp import FastMCP, Image

from rheed_core import AnalysisStore, RheedSession
from rheed_core.materials import MATERIALS, expected_d, find_material_key
from rheed_core.streak_spacing import (
    measure_d,
    solve_scale_from_standard,
    measure_d_by_ratio,
)
from rheed_core.calibration import Point, calibrate as _calibrate
from rheed_core.inference import (
    ModelSpec,
    cluster_embeddings,
    list_embedding_analysis_adapters as _list_embedding_analysis_adapters,
    list_adapters as _list_ai_adapters,
    run_embedding_analysis_with_artifacts as _run_embedding_analysis_with_artifacts,
    run_inference as _run_inference,
)

mcp = FastMCP("rheed-studio")

# One session for the lifetime of this MCP server process. Independent
# of any rheed_webapp session — the agent and a human looking at the
# browser are not guaranteed to be looking at the same loaded file.
session = RheedSession()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _PROJECT_ROOT / ".auto_rheed_settings.json"


def _analysis_root() -> Path:
    configured = os.environ.get("AUTO_RHEED_DATA_DIR")
    if not configured:
        try:
            with _SETTINGS_PATH.open(encoding="utf-8") as stream:
                configured = json.load(stream).get("analysis_root")
        except (OSError, ValueError, TypeError):
            configured = None
    return Path(configured).expanduser().resolve() if configured else _PROJECT_ROOT / "data"


analysis_store = AnalysisStore(_analysis_root())
_current_dataset_label = "mcp-dataset"
last_ai_result = None
last_ai_analysis = None
last_ai_embedding_analyses = {}


def _begin_analysis_dataset(label: str) -> Path:
    global _current_dataset_label
    _current_dataset_label = label
    return analysis_store.begin_dataset(label, session.summary(), session.dataset_version)


def _record_analysis(kind: str, result: dict, parameters: dict) -> None:
    if analysis_store.dataset_dir is None:
        _begin_analysis_dataset(_current_dataset_label)
    analysis_store.record_json(kind, result, parameters=parameters)


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
    global last_ai_result, last_ai_analysis, last_ai_embedding_analyses
    suffix = os.path.splitext(path)[1].lower()
    summary = session.load_file(path, suffix)
    _begin_analysis_dataset(Path(path).name)
    last_ai_result = None
    last_ai_analysis = None
    last_ai_embedding_analyses = {}
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
        "analysis_storage": analysis_store.status(),
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
    global last_ai_result, last_ai_analysis, last_ai_embedding_analyses
    H, W = session.rotate(direction)
    _begin_analysis_dataset(f"{_current_dataset_label}-rotated")
    last_ai_result = None
    last_ai_analysis = None
    last_ai_embedding_analyses = {}
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
    payload = {
        "scale_cm_px": result.scale_cm_px,
        "alpha_deg": result.alpha_deg,
        "d_px": result.d_px,
        "dy_px": result.dy_px,
        "dy_cm": result.dy_cm,
    }
    _record_analysis("calibration", payload, {
        "direct_beam": direct_beam, "shadow_edge": shadow_edge,
        "specular": specular, "cam_edge_left": cam_edge_left,
        "cam_edge_right": cam_edge_right, "L_cm": l_cm, "D_cm": d_cm,
    })
    return payload


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
    payload = {
        "d_angstrom": result.d_angstrom,
        "expected_d_angstrom": result.expected_d_angstrom,
        "deviation_pct": result.deviation_pct,
    }
    _record_analysis("streak-spacing", payload, {
        "dx_px": dx_px, "scale_cm_px": scale_cm_px, "L_cm": l_cm,
        "V_keV": v_kev, "material": material, "zone": zone,
    })
    return payload


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
    scale = solve_scale_from_standard(dx_px, l_cm, v_kev, material, zone)
    _record_analysis("substrate-scale", {"scale_cm_px": scale}, {
        "dx_px": dx_px, "L_cm": l_cm, "V_keV": v_kev,
        "material": material, "zone": zone,
    })
    return scale


@mcp.tool()
def measure_streak_spacing_by_ratio(dx_px_reference: float, d_reference_angstrom: float,
                                     dx_px_unknown: float) -> float:
    """
    Most robust lattice-parameter method: when a known-d reference (e.g.
    substrate) and an unknown spacing (e.g. growing film) are visible in
    the SAME frame, the camera constant cancels — no wavelength, L, or
    pixel scale needed. Returns d_unknown (Å).
    """
    d_angstrom = measure_d_by_ratio(dx_px_reference, d_reference_angstrom, dx_px_unknown)
    _record_analysis("streak-spacing-ratio", {"d_angstrom": d_angstrom}, {
        "dx_px_reference": dx_px_reference,
        "d_reference_angstrom": d_reference_angstrom,
        "dx_px_unknown": dx_px_unknown,
    })
    return d_angstrom


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
    payload = {
        "intensities": intensities,
        "timestamps": timestamps,
        "min": min(intensities), "max": max(intensities),
        "mean": sum(intensities) / len(intensities),
    }
    _record_analysis("intensity", payload, {"roi_type": roi_type, "roi": roi})
    return payload


# ── optional local AI model adapters ──────────────────────────────────────

@mcp.tool()
def list_ai_adapters() -> list[dict]:
    """List local inference adapters and whether their optional runtimes
    are installed. Model weights are selected separately in a manifest."""
    return _list_ai_adapters()


@mcp.tool()
def list_ai_embedding_analysis_adapters() -> list[dict]:
    """List adapters that analyze a completed frame-embedding sequence."""
    return _list_embedding_analysis_adapters()


@mcp.tool()
def run_ai_inference(model: dict, device: str = "auto",
                     batch_size: int = 8, stride: int = 1) -> dict:
    """Run a local embedding, segmentation, regression, or classification
    adapter over the loaded dataset. `model` is a manifest object with id,
    adapter, source, optional preprocessing, and adapter-specific options.
    Returns aligned frame/timestamp metadata; use save_ai_results for the
    full embeddings or segmentation geometry."""
    global last_ai_result, last_ai_analysis, last_ai_embedding_analyses
    if session.frames is None:
        raise ValueError("No file loaded — call load_rheed_file first")
    spec = ModelSpec.from_dict(model)
    result = _run_inference(
        session.frames,
        session.timestamps,
        spec,
        device=device,
        batch_size=batch_size,
        stride=stride,
    )
    if analysis_store.dataset_dir is None:
        _begin_analysis_dataset(_current_dataset_label)
    last_ai_result = analysis_store.save_inference(uuid.uuid4().hex, result)
    last_ai_analysis = None
    last_ai_embedding_analyses = {}
    return last_ai_result.summary()


@mcp.tool()
def analyze_ai_embeddings(pca_components: float = 10.0, n_clusters: int = 3,
                          auto_clusters: bool = False, max_clusters: int = 10,
                          random_state: int = 0) -> dict:
    """Run PCA and K-means on the most recent embedding inference result.

    `pca_components` may be a whole-number component count or an explained-
    variance target between 0 and 1 (for example, 0.95). Set `auto_clusters`
    to compare a zero-score K=1 baseline with silhouette scores from 2 through
    `max_clusters`. Clustering uses every retained PCA dimension; PC1/PC2 are
    visualization coordinates rather than the only clustering inputs.
    """
    global last_ai_analysis
    if last_ai_result is None:
        raise ValueError("Run an embedding model first")
    result = last_ai_result.load_result()
    clustered = cluster_embeddings(
        result,
        pca_components=pca_components,
        n_clusters="auto" if auto_clusters else n_clusters,
        max_clusters=max_clusters,
        random_state=random_state,
    )
    analysis = clustered.to_dict(
        frame_indices=result.frame_indices,
        timestamps=result.timestamps,
    )
    last_ai_result.save_analysis("pca-kmeans", analysis)
    last_ai_analysis = True
    return analysis


@mcp.tool()
def analyze_ai_embedding_sequence(
    adapter: str = "rhaapsody-changepoint",
    options: Optional[dict] = None,
) -> dict:
    """Run a temporal adapter after an embedding inference has completed.

    The built-in `rhaapsody-changepoint` adapter uses Auto RHEED's existing
    embeddings and PNNL RHAAPSODY's kernel-similarity detector. Its options
    include `cost_threshold`, `window_size`, `starting_period`, and
    `min_time_between_changepoints`.
    """
    global last_ai_embedding_analyses
    if last_ai_result is None:
        raise ValueError("Run an embedding model first")
    result = last_ai_result.load_result()
    if result.embeddings is None:
        raise ValueError("Run an embedding model first")
    analysis, array_artifacts = _run_embedding_analysis_with_artifacts(
        result, adapter, options or {}
    )
    for artifact_name, array in array_artifacts.items():
        last_ai_result.save_analysis_array(
            f"embedding-{adapter}-{artifact_name}",
            array,
        )
    last_ai_result.save_analysis(f"embedding-{adapter}", analysis)
    last_ai_embedding_analyses[adapter] = {
        "task": analysis.get("task"),
        "n_changepoints": len(analysis.get("changepoints", [])),
    }
    return analysis


@mcp.tool()
def get_ai_frame_result(frame_idx: int) -> dict:
    """Get the most recent AI result aligned to one native frame index."""
    if last_ai_result is None:
        raise ValueError("Run AI inference first")
    result = last_ai_result.frame_result(frame_idx)
    if result is None:
        raise ValueError("That frame was not selected by the inference stride")
    analysis = last_ai_result.load_analysis("pca-kmeans") if last_ai_analysis else None
    if analysis is not None:
        try:
            pos = analysis["frame_indices"].index(frame_idx)
        except ValueError:
            pass
        else:
            result["pca_scores"] = analysis["pca_scores"][pos]
            result["cluster"] = analysis["cluster_labels"][pos]
    return result


@mcp.tool()
def save_ai_results(path: str) -> dict:
    """Save the latest full local AI result as a compressed NPZ file.
    Embedding matrices and segmentation geometry are included along with
    model/preprocessing provenance and optional PCA/K-means analysis."""
    if last_ai_result is None:
        raise ValueError("Run AI inference first")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    result = last_ai_result.load_result()
    stored_analysis = last_ai_result.load_analysis("pca-kmeans")
    analysis = dict(stored_analysis) if stored_analysis is not None else {}
    if last_ai_embedding_analyses:
        analysis["embedding_analyses"] = {
            name: last_ai_result.load_analysis(f"embedding-{name}")
            for name in last_ai_embedding_analyses
        }
    target.write_bytes(result.to_npz_bytes(analysis or None))
    return {"path": str(target), "bytes": target.stat().st_size}


if __name__ == "__main__":
    mcp.run()
