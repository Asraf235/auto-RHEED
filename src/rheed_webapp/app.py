"""
RHEED Studio — Flask web app.

This module is intentionally thin: every route parses the HTTP
request, calls into a `rheed_core.RheedSession`, and serializes the
result. All RHEED domain logic (file formats, calibration math,
ROI intensity, peak tracking) lives in rheed_core and has no
knowledge of Flask — see src/rheed_core/.
"""
import io
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from flask import Flask, request, jsonify, render_template, send_file
import numpy as np

from rheed_core import AnalysisStore, RheedSession
from rheed_core.constants import CV2_CMAPS, IMAGE_EXTS
from rheed_core.inference import (
    ModelManifestRegistry,
    ModelSpec,
    list_adapters,
    list_embedding_analysis_adapters,
)
from rheed_core.library import RheedLibrary
from rheed_webapp.ai_jobs import AIJobManager

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024  # 16 GB
# Werkzeug's default multipart limit is 1000 parts — far too low for a
# growth video reassembled from thousands of individual frame images
# (/upload_image_stack sends one multipart "part" per image). Without
# raising this, a large image-stack upload is rejected by Werkzeug
# itself (a 413, before our route code even runs) with an HTML error
# page instead of JSON, which the frontend can't parse.
app.request_class.max_form_parts = 50_000


@app.after_request
def _prevent_stale_dynamic_api_reads(response):
    """Polling and recall indexes must never be satisfied from browser cache."""
    path = request.path
    is_job_status = bool(re.fullmatch(r"/ai/jobs/[^/]+", path))
    is_analysis_listing = path in {
        "/analysis/storage", "/analysis/runs", "/analysis/artifacts",
    }
    if is_job_status or is_analysis_listing:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# One session per running web app process — matches the original
# single-user, single-dataset-at-a-time behavior of this app.
session = RheedSession()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _PROJECT_ROOT / ".auto_rheed_settings.json"
_DEFAULT_ANALYSIS_DIR = _PROJECT_ROOT / "data"


def _load_analysis_root() -> Path:
    try:
        with _SETTINGS_PATH.open(encoding="utf-8") as stream:
            configured = json.load(stream).get("analysis_root")
        return Path(configured).expanduser().resolve() if configured else _DEFAULT_ANALYSIS_DIR
    except (OSError, ValueError, TypeError):
        return _DEFAULT_ANALYSIS_DIR


def _save_analysis_root(root: Path) -> None:
    temporary = _SETTINGS_PATH.with_name(f".{_SETTINGS_PATH.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"analysis_root": str(root)}, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, _SETTINGS_PATH)


analysis_store = AnalysisStore(_load_analysis_root())
_current_dataset_label = "loaded-dataset"
_current_source_fingerprint: str | None = None
_current_run_resumed = False


def _begin_analysis_dataset(label: str) -> Path:
    global _current_dataset_label, _current_run_resumed
    _current_dataset_label = label or "loaded-dataset"
    _current_run_resumed = False
    return analysis_store.begin_dataset(
        _current_dataset_label,
        session.summary(),
        session.dataset_version,
        source_fingerprint=_current_source_fingerprint,
    )


def _resume_or_begin_analysis_dataset(
    label: str,
    source_fingerprint: str,
) -> tuple[Path, bool]:
    global _current_dataset_label, _current_source_fingerprint, _current_run_resumed
    _current_dataset_label = label or "loaded-dataset"
    _current_source_fingerprint = source_fingerprint
    dataset_dir, resumed = analysis_store.begin_or_resume_dataset(
        _current_dataset_label,
        session.summary(),
        session.dataset_version,
        source_fingerprint=source_fingerprint,
    )
    _current_run_resumed = resumed
    return dataset_dir, resumed


def _ensure_analysis_dataset() -> Path:
    status = analysis_store.status()
    manifest = status.get("dataset") or {}
    if (
        status.get("dataset_dir") is None
        or int(manifest.get("dataset_version", -1)) != session.dataset_version
    ):
        return _begin_analysis_dataset(_current_dataset_label)
    return Path(status["dataset_dir"])


def _record_analysis(
    kind: str,
    result: dict,
    *,
    parameters: dict | None = None,
    name: str | None = None,
    replace: bool = False,
) -> dict:
    _ensure_analysis_dataset()
    return analysis_store.record_json(
        kind,
        result,
        parameters=parameters,
        name=name,
        replace=replace,
    )

# Persistent reference gallery of RHEED patterns (RHEED Library tab),
# stored in a folder at the project root.
_LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "rheed_library")
library = RheedLibrary(os.path.abspath(_LIBRARY_DIR))

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "rheed_models")
model_registry = ModelManifestRegistry(os.path.abspath(_MODEL_DIR))
ai_jobs = AIJobManager(analysis_store)


def _save_upload_to_tmp(f):
    suffix = os.path.splitext(f.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := f.stream.read(4 * 1024 * 1024):
            tmp.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    finally:
        tmp.close()
    return tmp.name, suffix, (size, digest.digest())


def _source_fingerprint(parts: list[tuple[int, bytes]]) -> str:
    """Order-sensitive content identity assembled during upload streaming."""
    digest = hashlib.sha256()
    digest.update(b"auto-rheed-source-v1\0")
    for size, content_digest in parts:
        digest.update(size.to_bytes(8, "big", signed=False))
        digest.update(content_digest)
        digest.update(b"\0next-file\0")
    return f"sha256:{digest.hexdigest()}"


# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis/storage", methods=["GET", "POST"])
def analysis_storage():
    """Inspect or change the local folder used for automatic result storage."""
    if request.method == "POST":
        data = request.get_json() or {}
        requested_root = str(data.get("root") or "").strip()
        if not requested_root:
            return jsonify(error="root is required"), 400
        try:
            root = analysis_store.set_root(requested_root)
            _save_analysis_root(root)
            if session.frames is not None:
                if _current_source_fingerprint:
                    _resume_or_begin_analysis_dataset(
                        _current_dataset_label,
                        _current_source_fingerprint,
                    )
                else:
                    _begin_analysis_dataset(_current_dataset_label)
        except Exception as exc:
            return jsonify(error=f"Could not use analysis folder: {exc}"), 400
    payload = analysis_store.status()
    payload["default_root"] = str(_DEFAULT_ANALYSIS_DIR)
    payload["artifact_count"] = len(analysis_store.list_artifacts())
    payload["run_count"] = len(analysis_store.list_runs())
    return jsonify(payload)


@app.route("/analysis/runs")
def analysis_runs():
    active_dir = analysis_store.dataset_dir
    runs = []
    for run in analysis_store.list_runs():
        item = dict(run)
        item["compatible"] = _analysis_run_compatible(item)
        item["active"] = bool(
            active_dir and Path(item.get("path", "")).resolve() == active_dir.resolve()
        )
        runs.append(item)
    return jsonify(runs=runs)


def _analysis_run_compatible(run: dict) -> bool:
    if session.frames is None:
        return False
    fingerprint = run.get("source_fingerprint")
    if fingerprint and _current_source_fingerprint:
        return fingerprint == _current_source_fingerprint
    if str(run.get("source_label") or "") != _current_dataset_label:
        return False
    saved = run.get("dataset") or {}
    current = session.summary()
    return all(
        int(saved.get(key, -1)) == int(current.get(key, -2))
        for key in ("n_frames", "height", "width")
    )


@app.route("/analysis/runs/<run_id>/activate", methods=["POST"])
def analysis_activate_run(run_id):
    """Activate a compatible saved run for the currently loaded pixels."""
    run = next(
        (item for item in analysis_store.list_runs() if item.get("run_id") == run_id),
        None,
    )
    if run is None:
        return jsonify(error="Unknown analysis run"), 404
    if not _analysis_run_compatible(run):
        return jsonify(error="That saved run does not match the loaded dataset"), 409
    try:
        path = analysis_store.activate_run(
            run_id,
            runtime_dataset_version=session.dataset_version,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        ok=True,
        analysis_dir=str(path),
        artifacts=analysis_store.list_artifacts(),
    )


@app.route("/analysis/artifacts")
def analysis_artifacts():
    return jsonify(
        storage=analysis_store.status(),
        artifacts=analysis_store.list_artifacts(),
    )


@app.route("/analysis/artifacts/<path:artifact_id>")
def analysis_artifact(artifact_id):
    try:
        return jsonify(analysis_store.load_artifact(artifact_id))
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route("/analysis/ai/recall", methods=["POST"])
def analysis_recall_ai():
    """Re-register a saved AI run for the currently loaded dataset."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json() or {}
    artifact_id = str(data.get("artifact_id") or "").strip()
    if not artifact_id:
        return jsonify(error="artifact_id is required"), 400
    try:
        job = ai_jobs.restore(
            artifact_id,
            dataset_version=session.dataset_version,
        )
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(job.public_dict(session.dataset_version))


@app.route("/analysis/record", methods=["POST"])
def analysis_record():
    """Persist browser-created analysis such as manual growth measurements."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json() or {}
    kind = str(data.get("kind") or "").strip()
    result = data.get("result")
    if not kind or not isinstance(result, dict):
        return jsonify(error="kind and object-valued result are required"), 400
    try:
        artifact = _record_analysis(
            kind,
            result,
            parameters=data.get("parameters") or {},
            name=str(data.get("name") or "").strip() or None,
            replace=bool(data.get("replace", False)),
        )
    except Exception as exc:
        return jsonify(error=f"Could not save analysis: {exc}"), 500
    return jsonify(ok=True, artifact=artifact)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file"), 400

    path, suffix, fingerprint_part = _save_upload_to_tmp(f)
    try:
        source_fingerprint = _source_fingerprint([fingerprint_part])
        summary = session.load_file(path, suffix)
    except Exception as e:
        return jsonify(error=f"Failed to load {suffix} file: {e}"), 400
    finally:
        os.unlink(path)

    try:
        dataset_dir, resumed = _resume_or_begin_analysis_dataset(
            f.filename or "loaded-dataset",
            source_fingerprint,
        )
    except Exception as exc:
        return jsonify(error=f"Dataset loaded but its analysis folder could not be created: {exc}"), 500
    return jsonify(
        **summary,
        dataset_name=f.filename or "loaded-dataset",
        analysis_dir=str(dataset_dir),
        analysis_resumed=resumed,
        saved_artifact_count=len(analysis_store.list_artifacts()),
    )


@app.route("/frame/<int:idx>")
def get_frame(idx):
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    auto = request.args.get("auto", "0") == "1"
    b64, t = session.frame_png_b64(idx, auto_contrast=auto)
    resp = {"image": b64, "timestamp": t}
    if auto:
        resp["clim"] = list(session.clim)
    return jsonify(**resp)


@app.route("/frame_image/<int:idx>")
def get_frame_image(idx):
    """Return a binary display frame without Base64/JSON playback overhead."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    auto = request.args.get("auto", "0") == "1"
    encoding = request.args.get("format", "jpeg").lower()
    if encoding not in {"jpeg", "png"}:
        return jsonify(error="format must be jpeg or png"), 400
    try:
        quality = int(request.args.get("quality", 90))
        payload, timestamp, clim = session.frame_image_bytes(
            idx,
            auto_contrast=auto,
            encoding=encoding,
            jpeg_quality=quality,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    response = app.response_class(
        payload,
        mimetype="image/jpeg" if encoding == "jpeg" else "image/png",
    )
    response.headers["X-Rheed-Timestamp"] = str(timestamp)
    if auto:
        response.headers["X-Rheed-Clim-Lo"] = str(clim[0])
        response.headers["X-Rheed-Clim-Hi"] = str(clim[1])
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/intensity", methods=["POST"])
def intensity():
    data = request.get_json()
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    try:
        intensities, timestamps = session.compute_intensity(
            roi_type=data.get("type", "circle"),
            roi=data,
            display_w=data.get("display_w"),
            display_h=data.get("display_h"),
        )
    except ValueError as e:
        return jsonify(error=str(e)), 400
    result = {"intensities": intensities, "timestamps": timestamps}
    try:
        artifact = _record_analysis(
            "intensity",
            result,
            parameters=data,
            name="current",
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"Intensity computed but could not be saved: {exc}"), 500
    return jsonify(**result, _saved_artifact=artifact)


@app.route("/clim", methods=["POST"])
def set_clim():
    data = request.get_json()
    session.set_clim(data["lo"], data["hi"])
    return jsonify(ok=True)


@app.route("/rotate", methods=["POST"])
def rotate():
    global _current_source_fingerprint
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json() or {}
    H, W = session.rotate(data.get("direction", "cw"))
    if _current_source_fingerprint:
        _current_source_fingerprint = (
            f"{_current_source_fingerprint}:rotated:{session.dataset_version}"
        )
    try:
        dataset_dir = _begin_analysis_dataset(f"{_current_dataset_label}-rotated")
    except Exception as exc:
        return jsonify(error=f"Frames rotated but the new analysis run could not be created: {exc}"), 500
    return jsonify(ok=True, width=W, height=H, analysis_dir=str(dataset_dir))


@app.route("/mean_frame")
def mean_frame():
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    b64, w, h = session.mean_frame_png_b64()
    return jsonify(image=b64, width=w, height=h)


@app.route("/export_frame/<int:idx>")
def export_frame(idx):
    """Download a single frame as a PNG, rendered with the current contrast/colormap."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    png = session.export_frame_png(idx)
    return send_file(io.BytesIO(png), mimetype="image/png",
                      as_attachment=True, download_name=f"frame_{idx:04d}.png")


@app.route("/export_frames_zip")
def export_frames_zip():
    """Download every loaded frame as a ZIP of numbered PNGs."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    zip_bytes = session.export_all_frames_zip()
    return send_file(io.BytesIO(zip_bytes), mimetype="application/zip",
                      as_attachment=True, download_name="rheed_frames.zip")


@app.route("/upload_image_stack", methods=["POST"])
def upload_image_stack():
    """
    Build a 'video' from a batch of individual still images (e.g. a
    folder of per-frame exports) instead of a single video/h5 file.
    Files are ordered by a natural sort of their filenames (so
    'frame2' sorts before 'frame10') before being stacked.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files"), 400

    def natural_key(f):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", f.filename)]

    files_sorted = sorted(files, key=natural_key)

    tmp_paths = []
    source_fingerprint = None
    try:
        specs = []
        fingerprint_parts = []
        for f in files_sorted:
            suffix = os.path.splitext(f.filename)[1].lower()
            if suffix not in IMAGE_EXTS:
                return jsonify(error=f"Not a supported image type: {f.filename}"), 400
            path, suffix, fingerprint_part = _save_upload_to_tmp(f)
            tmp_paths.append(path)
            specs.append((path, suffix))
            fingerprint_parts.append(fingerprint_part)
        source_fingerprint = _source_fingerprint(fingerprint_parts)
        summary = session.load_file_stack(specs)
    except Exception as e:
        return jsonify(error=f"Failed to build video from images: {e}"), 400
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    label = files_sorted[0].filename if len(files_sorted) == 1 else f"image-stack-{len(files_sorted)}-frames"
    try:
        dataset_dir, resumed = _resume_or_begin_analysis_dataset(
            label,
            source_fingerprint,
        )
    except Exception as exc:
        return jsonify(error=f"Image stack loaded but its analysis folder could not be created: {exc}"), 500
    return jsonify(
        **summary,
        dataset_name=label,
        analysis_dir=str(dataset_dir),
        analysis_resumed=resumed,
        saved_artifact_count=len(analysis_store.list_artifacts()),
    )


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """
    Body JSON:
      points: { direct_beam, shadow_edge, specular, cam_edge_left, cam_edge_right } each {x,y}
      params: { L_cm, D_cm, display_w, display_h }
    """
    data = request.get_json()
    params = data["params"]
    try:
        result = session.calibrate_points(
            points=data["points"],
            l_cm=float(params["L_cm"]),
            d_cm=float(params["D_cm"]),
            display_w=params.get("display_w"),
            display_h=params.get("display_h"),
        )
    except ValueError as e:
        return jsonify(error=str(e)), 400

    payload = {
        "scale_cm_px": result.scale_cm_px,
        "D_px": result.d_px,
        "dy_px": result.dy_px,
        "dy_cm": result.dy_cm,
        "alpha_deg": result.alpha_deg,
        "rows": {
            "direct_beam": result.direct_beam_y,
            "shadow_edge": result.shadow_edge_y,
            "specular": result.specular_y,
        },
    }
    try:
        artifact = _record_analysis(
            "calibration",
            payload,
            parameters=data,
            name="current",
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"Calibration completed but could not be saved: {exc}"), 500
    return jsonify(**payload, _saved_artifact=artifact)


@app.route("/load_image", methods=["POST"])
def load_image():
    """Preview a single still image (used by the Calibration tab's
    standalone 'Open File' button) without committing it as the loaded
    dataset."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file"), 400
    path, suffix, _ = _save_upload_to_tmp(f)
    try:
        b64, w, h = session.preview_image(path, suffix)
        return jsonify(image=b64, width=w, height=h)
    except Exception as e:
        return jsonify(error=f"Failed to load {suffix} file: {e}"), 400
    finally:
        os.unlink(path)


@app.route("/track_peak", methods=["POST"])
def track_peak():
    """
    Body JSON:
      specular    : {x1,y1,x2,y2} in display pixels  (required)
      direct_beam : {x1,y1,x2,y2} in display pixels  (optional)
      display_w, display_h : canvas dimensions
      calibration : {shadow_edge_y, direct_beam_y, scale_cm_px, L_cm}
    """
    data = request.get_json()
    if session.frames is None:
        return jsonify(error="No file loaded"), 400

    calib = data.get("calibration", {})
    try:
        result = session.track_peak(
            specular_roi=data.get("specular"),
            direct_beam_roi=data.get("direct_beam"),
            shadow_edge_y=calib.get("shadow_edge_y"),
            direct_beam_y=calib.get("direct_beam_y"),
            scale_cm_px=calib.get("scale_cm_px", 1.0),
            l_cm=calib.get("L_cm", 10.5),
            display_w=data.get("display_w"),
            display_h=data.get("display_h"),
        )
    except ValueError as e:
        return jsonify(error=str(e)), 400
    try:
        artifact = _record_analysis(
            "peak-tracking",
            result,
            parameters=data,
            name="current",
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"Peak tracking completed but could not be saved: {exc}"), 500
    return jsonify(**result, _saved_artifact=artifact)


@app.route("/clim_auto")
def clim_auto():
    return jsonify(lo=session.clim_auto[0], hi=session.clim_auto[1])


@app.route("/clim_auto/<int:idx>")
def clim_auto_frame(idx):
    """Auto-contrast range for a SPECIFIC frame (not just frame 0) — lets
    the 'Auto' button rescale to whatever frame is currently on screen."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    lo, hi = session.clim_auto_for_frame(idx)
    return jsonify(lo=lo, hi=hi)


@app.route("/colormap", methods=["POST"])
def set_colormap():
    data = request.get_json()
    cmap = data.get("colormap", "gray")
    try:
        session.set_colormap(cmap)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


# ── local AI inference ─────────────────────────────────────────────────────

@app.route("/ai/adapters")
def ai_adapters():
    """List built-in and third-party model adapters without loading models."""
    return jsonify(adapters=list_adapters())


@app.route("/ai/embedding-analysis-adapters")
def ai_embedding_analysis_adapters():
    """List analyses that consume a completed embedding sequence."""
    return jsonify(adapters=list_embedding_analysis_adapters())


@app.route("/ai/models", methods=["GET", "POST"])
def ai_models():
    """List or save small JSON manifests; model weights remain external."""
    if request.method == "GET":
        return jsonify(models=model_registry.list())
    try:
        data = request.get_json() or {}
        spec = ModelSpec.from_dict(data.get("model") or data)
        filename = model_registry.save(spec, data.get("filename"))
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True, filename=filename, model=spec.public_dict())


def _ai_model_from_request(data) -> ModelSpec:
    manifest = data.get("manifest")
    if manifest:
        return model_registry.get(str(manifest))
    return ModelSpec.from_dict(data.get("model") or {})


@app.route("/ai/run", methods=["POST"])
def ai_run():
    """Start local inference against the currently loaded frame stack."""
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json() or {}
    try:
        _ensure_analysis_dataset()
        spec = _ai_model_from_request(data)
        batch_size = int(data.get("batch_size", 8))
        stride = int(data.get("stride", 1))
        if not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if not 1 <= stride <= session.n_frames:
            raise ValueError(f"stride must be between 1 and {session.n_frames}")
        job = ai_jobs.submit(
            frames=session.frames,
            timestamps=session.timestamps,
            dataset_version=session.dataset_version,
            model=spec,
            device=str(data.get("device", "auto")),
            batch_size=batch_size,
            stride=stride,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(job.public_dict(session.dataset_version)), 202


@app.route("/ai/jobs/<job_id>")
def ai_job_status(job_id):
    try:
        job = ai_jobs.get(job_id)
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    return jsonify(job.public_dict(session.dataset_version))


@app.route("/ai/jobs/<job_id>/series")
def ai_job_series(job_id):
    """Load time-series summaries on demand instead of keeping them in job memory."""
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        if job.stored_result is None:
            raise ValueError("Inference result is not available")
        summary = job.stored_result.summary()
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        frame_indices=summary.get("frame_indices", []),
        timestamps=summary.get("timestamps", []),
        frame_summaries=summary.get("frame_summaries", []),
    )


@app.route("/ai/jobs/<job_id>/cancel", methods=["POST"])
def ai_job_cancel(job_id):
    try:
        job = ai_jobs.cancel(job_id)
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    return jsonify(job.public_dict(session.dataset_version))


@app.route("/ai/jobs/<job_id>/analyze", methods=["POST"])
def ai_job_analyze(job_id):
    data = request.get_json() or {}
    try:
        analysis = ai_jobs.analyze(
            job_id,
            pca_components=data.get("pca_components", 10),
            n_clusters=data.get("n_clusters", 3),
            max_clusters=data.get("max_clusters", 10),
            random_state=int(data.get("random_state", 0)),
            normalize=bool(data.get("normalize", True)),
            whiten=bool(data.get("whiten", False)),
        )
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(analysis=analysis)


@app.route("/ai/jobs/<job_id>/analysis")
def ai_job_saved_analysis(job_id):
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        analysis = job.load_analysis()
        if analysis is None:
            return jsonify(error="This embedding run has no saved PCA/K-means analysis"), 404
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(analysis=analysis)


@app.route("/ai/jobs/<job_id>/embedding-analysis", methods=["POST"])
def ai_job_embedding_analysis(job_id):
    data = request.get_json() or {}
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        adapter_name = str(data.get("adapter") or "").strip()
        if not adapter_name:
            raise ValueError("adapter is required")
        analysis = ai_jobs.analyze_embeddings(
            job_id,
            adapter_name,
            data.get("options") or {},
        )
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(analysis=analysis)


@app.route("/ai/jobs/<job_id>/embedding-analysis/<adapter_name>")
def ai_job_saved_embedding_analysis(job_id, adapter_name):
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        analysis = job.load_embedding_analysis(adapter_name)
        if analysis is None:
            return jsonify(error=f"No saved embedding analysis for {adapter_name!r}"), 404
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(analysis=analysis)


@app.route("/ai/jobs/<job_id>/embedding-analysis/<adapter_name>/csv")
def ai_job_embedding_analysis_csv(job_id, adapter_name):
    try:
        job = ai_jobs.get(job_id)
        analysis = job.load_embedding_analysis(adapter_name)
        if analysis is None:
            raise ValueError(f"Run embedding-analysis adapter {adapter_name!r} first")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow([
            "changepoint_frame", "changepoint_t_s", "detected_frame",
            "detected_t_s", "amplitude",
        ])
        for event in analysis.get("changepoints", []):
            writer.writerow([
                int(event["frame_index"]) + 1,
                event["timestamp_s"],
                int(event["detected_frame_index"]) + 1,
                event["detected_timestamp_s"],
                event["amplitude"],
            ])
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    payload = io.BytesIO(stream.getvalue().encode("utf-8"))
    safe_adapter = re.sub(r"[^A-Za-z0-9._-]+", "-", adapter_name).strip(".-")
    return send_file(
        payload,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"rheed_ai_{safe_adapter or 'embedding-analysis'}.csv",
    )


@app.route(
    "/ai/jobs/<job_id>/embedding-analysis/<adapter_name>/similarity-matrix"
)
def ai_job_embedding_similarity_matrix(job_id, adapter_name):
    """Serve a bounded binary view of a file-backed similarity matrix."""
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        analysis = job.load_embedding_analysis(adapter_name)
        matrix_info = (analysis or {}).get("similarity_matrix")
        if not matrix_info:
            raise ValueError(
                f"Embedding analysis {adapter_name!r} has no similarity matrix"
            )
        artifact_name = str(matrix_info.get("artifact") or "similarity-matrix")
        matrix = job.load_embedding_analysis_array(adapter_name, artifact_name)
        if matrix is None:
            raise ValueError("The saved similarity-matrix artifact is missing")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Similarity matrix must be square, got {matrix.shape}")
        requested_size = int(request.args.get("max_size", 600))
        max_size = max(32, min(requested_size, 800))
        source_rows = int(matrix.shape[0])
        if source_rows <= max_size:
            positions = np.arange(source_rows, dtype=np.int64)
        else:
            positions = np.linspace(0, source_rows - 1, max_size).astype(np.int64)
        sampled = np.ascontiguousarray(
            matrix[np.ix_(positions, positions)],
            dtype="<f4",
        )
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    response = app.response_class(sampled.tobytes(), mimetype="application/octet-stream")
    response.headers["X-Rheed-Matrix-Size"] = str(len(positions))
    response.headers["X-Rheed-Matrix-Source-Rows"] = str(source_rows)
    response.headers["X-Rheed-Matrix-Positions"] = ",".join(map(str, positions))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/ai/jobs/<job_id>/frame/<int:frame_idx>")
def ai_job_frame(job_id, frame_idx):
    try:
        job = ai_jobs.get(job_id)
        if job.dataset_version != session.dataset_version:
            return jsonify(error="AI result is stale because the loaded pixels changed"), 409
        if job.stored_result is None:
            raise ValueError("Inference result is not available")
        frame_result = job.stored_result.frame_result(frame_idx)
        if frame_result is None:
            return jsonify(error="Frame was not selected by this inference run"), 404
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(frame_result)


@app.route("/ai/jobs/<job_id>/download")
def ai_job_download(job_id):
    try:
        job = ai_jobs.get(job_id)
        result = job.load_result()
        payload = result.to_npz_bytes(job.export_analysis())
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", job.model.id).strip(".-") or "model"
    return send_file(
        io.BytesIO(payload),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"rheed_ai_{safe_id}.npz",
    )


@app.route("/ai/jobs/<job_id>/csv")
def ai_job_csv(job_id):
    try:
        job = ai_jobs.get(job_id)
        result = job.load_result()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        if result.task == "embedding":
            analysis = job.load_analysis()
            if analysis is None:
                raise ValueError("Run PCA/K-means before exporting embedding CSV")
            scores = analysis["pca_scores"]
            labels = analysis["cluster_labels"]
            n_pc = len(scores[0]) if scores else 0
            writer.writerow(["frame", "t_s", *[f"pc{i + 1}" for i in range(n_pc)], "cluster"])
            for i, frame_index in enumerate(result.frame_indices):
                writer.writerow([
                    int(frame_index) + 1,
                    float(result.timestamps[i]),
                    *scores[i],
                    labels[i],
                ])
        elif result.task in {"segmentation", "detection"}:
            writer.writerow([
                "frame", "t_s", "track_id", "class_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2", "mask_area_px", "polygon_xy_json",
            ])
            for i, frame in enumerate(result.frames or []):
                for instance in frame.get("instances", []):
                    box = instance.get("box_xyxy") or [None] * 4
                    writer.writerow([
                        int(result.frame_indices[i]) + 1,
                        float(result.timestamps[i]),
                        instance.get("track_id"),
                        instance.get("class_id"),
                        instance.get("class_name"),
                        instance.get("confidence"),
                        *box,
                        instance.get("mask_area_px"),
                        json.dumps(instance.get("polygon_xy") or [], separators=(",", ":")),
                    ])
        else:
            value_names = sorted({
                str(name)
                for frame in (result.frames or [])
                for name in (frame.get("values") or {})
            })
            writer.writerow(["frame", "t_s", "label", "confidence", *value_names])
            for i, frame in enumerate(result.frames or []):
                values = frame.get("values") or {}
                writer.writerow([
                    int(result.frame_indices[i]) + 1,
                    float(result.timestamps[i]),
                    frame.get("label"),
                    frame.get("confidence"),
                    *[values.get(name) for name in value_names],
                ])
        payload = stream.getvalue().encode("utf-8-sig")
    except KeyError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return send_file(
        io.BytesIO(payload),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"rheed_ai_{result.task}.csv",
    )


@app.route("/strip_profile", methods=["POST"])
def strip_profile():
    """
    Live single-frame strip-profile preview (raw / baseline / subtracted
    + the 3 detected peaks), used by the Growth Analysis tab's "Streak
    Profile" panel.
    Body JSON:
      frame_idx, y1, y2, x1, x2 (display px), bg_method, bg_params,
      prev_peaks (optional, for drift-tracked search), display_w, display_h
    """
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json()
    try:
        result = session.strip_profile(
            idx=data.get("frame_idx", 0),
            y1=data["y1"], y2=data["y2"],
            x1=data.get("x1"), x2=data.get("x2"),
            bg_method=data.get("bg_method", "als"),
            bg_params=data.get("bg_params"),
            prev_peaks=data.get("prev_peaks"),
            vertical_track=data.get("vertical_track", True),
            display_w=data.get("display_w"), display_h=data.get("display_h"),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400
    # This endpoint is a transient live preview and is called repeatedly
    # while scrubbing/playing. Persisting every preview produced hundreds of
    # compressed files that were not independent analyses. The durable
    # whole-video result is saved by /strip_track; the current profile also
    # remains explicitly exportable from its chart window.
    return jsonify(**result)


@app.route("/strip_track", methods=["POST"])
def strip_track():
    """
    Automatic strip-profile peak tracking across EVERY loaded frame —
    returns d(t), specular/first-order FWHM, and background-subtracted
    tracked-peak intensities for the whole video in one call.
    Body JSON:
      y1, y2, x1, x2 (display px), bg_method, bg_params,
      scale_cm_px, L_cm, V_keV, display_w, display_h
    """
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json()
    try:
        result = session.strip_track(
            y1=data["y1"], y2=data["y2"],
            x1=data.get("x1"), x2=data.get("x2"),
            bg_method=data.get("bg_method", "als"),
            bg_params=data.get("bg_params"),
            scale_cm_px=float(data["scale_cm_px"]),
            l_cm=float(data["L_cm"]),
            v_kev=float(data["V_keV"]),
            vertical_track=data.get("vertical_track", True),
            display_w=data.get("display_w"), display_h=data.get("display_h"),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400
    try:
        artifact = _record_analysis(
            "strip-tracking",
            result,
            parameters=data,
            name="current",
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"Strip tracking completed but could not be saved: {exc}"), 500
    return jsonify(**result, _saved_artifact=artifact)


@app.route("/material_lookup", methods=["POST"])
def material_lookup():
    """
    Look up a material's lattice constant from the Materials Project REST
    API (https://api.materialsproject.org). Needs the caller's own MP API
    key (passed per-request; never stored). Returns the CONVENTIONAL cubic
    `a` where it can be inferred from the primitive cell, plus the raw
    lattice for the user to sanity-check. This is best-effort: MP's stored
    `structure` is often the primitive cell, so non-cubic results in
    particular should be verified against the conventional setting.
    """
    import math

    data = request.get_json() or {}
    formula = (data.get("formula") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    if not formula:
        return jsonify(error="No formula given."), 400
    if not api_key:
        return jsonify(error="A Materials Project API key is required (get one free at materialsproject.org)."), 400

    try:
        import requests as _rq
    except ImportError:
        return jsonify(error="The 'requests' package is needed for Materials Project lookup."), 400

    try:
        resp = _rq.get(
            "https://api.materialsproject.org/materials/summary/",
            headers={"X-API-KEY": api_key},
            params={"formula": formula,
                    "_fields": "material_id,formula_pretty,symmetry,structure,energy_above_hull",
                    "_limit": 8},
            timeout=25,
        )
    except Exception as e:
        return jsonify(error=f"Could not reach Materials Project: {e}"), 400

    if resp.status_code in (401, 403):
        return jsonify(error="Materials Project rejected the API key (check it at materialsproject.org/api)."), 400
    if not resp.ok:
        return jsonify(error=f"Materials Project returned HTTP {resp.status_code}."), 400

    docs = (resp.json() or {}).get("data", [])
    if not docs:
        return jsonify(error=f"No Materials Project entry found for '{formula}'."), 404

    # Most stable polymorph first (lowest energy above hull).
    docs.sort(key=lambda d: d.get("energy_above_hull") if d.get("energy_above_hull") is not None else 1e9)
    doc = docs[0]
    latt = (doc.get("structure") or {}).get("lattice") or {}
    a = latt.get("a"); b = latt.get("b"); c = latt.get("c")
    gamma = latt.get("gamma")
    sym = doc.get("symmetry") or {}
    system = (sym.get("crystal_system") or "").lower()

    # MP often stores the PRIMITIVE cell. For cubic, recover the
    # conventional edge from the primitive cell angle: FCC primitive has
    # 60° angles, BCC primitive ~109.47°; simple-cubic is already 90°.
    a_conv = a
    if system == "cubic" and a and gamma is not None:
        if abs(gamma - 60.0) < 5:
            a_conv = a * math.sqrt(2.0)        # FCC primitive → conventional
        elif abs(gamma - 109.4712) < 5:
            a_conv = a * 2.0 / math.sqrt(3.0)  # BCC primitive → conventional

    result = {
        "material_id": doc.get("material_id"),
        "formula": doc.get("formula_pretty"),
        "a": a,
        "b": b,
        "c": c,
        "gamma": gamma,
        "a_conventional": a_conv,
        "system": system,
        "spacegroup": sym.get("symbol"),
        "note": "Materials Project value (DFT-relaxed, typically ~1% large). Verify the conventional cell before citing.",
    }
    try:
        artifact = _record_analysis(
            "material-lookup",
            result,
            parameters={"formula": formula},
            name="current",
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"Material found but the lookup could not be saved: {exc}"), 500
    return jsonify(**result, _saved_artifact=artifact)


@app.route("/library")
def library_list():
    return jsonify(entries=library.list())


@app.route("/library/add", methods=["POST"])
def library_add():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No image file"), 400
    path, suffix, _ = _save_upload_to_tmp(f)
    try:
        entry = library.add(
            path, suffix,
            material=request.form.get("material", ""),
            zone=request.form.get("zone", ""),
            surface=request.form.get("surface", ""),
            energy=request.form.get("energy", ""),
            notes=request.form.get("notes", ""),
        )
    except Exception as e:
        return jsonify(error=f"Failed to add image: {e}"), 400
    finally:
        os.unlink(path)
    return jsonify(entry=entry)


@app.route("/library/image/<entry_id>")
def library_image(entry_id):
    path = library.image_path(entry_id)
    if path is None:
        return jsonify(error="Not found"), 404
    return send_file(path, mimetype="image/png")


@app.route("/library/delete", methods=["POST"])
def library_delete():
    data = request.get_json() or {}
    ok = library.delete(data.get("id", ""))
    return jsonify(ok=ok)


@app.route("/fft", methods=["POST"])
def fft_route():
    """
    FFT of an intensity-vs-time series → dominant frequency / growth rate.
    Body JSON: intensities[], timestamps[], detrend ('ema'|'linear'|'mean'),
    alpha (EMA factor, only used for detrend='ema').
    """
    from rheed_core import spectra as _spec
    data = request.get_json()
    try:
        result = _spec.intensity_fft(
            data["intensities"], data["timestamps"],
            detrend=data.get("detrend", "ema"),
            alpha=float(data.get("alpha", 0.15)),
        )
    except Exception as e:
        return jsonify(error=str(e)), 400
    try:
        layer_spacing = float(data.get("layer_spacing_A") or 0)
        if result.get("f0") and layer_spacing > 0:
            result["layer_spacing_A"] = layer_spacing
            result["deposition_rate_A_s"] = result["f0"] * layer_spacing
        analysis_name = str(data.get("analysis_name") or "current")
        artifact = _record_analysis(
            "fft",
            result,
            parameters={
                "detrend": data.get("detrend", "ema"),
                "alpha": data.get("alpha", 0.15),
                "series_label": data.get("series_label"),
                "series_source": data.get("series_source"),
                "growth_direction": data.get("growth_direction"),
                "layer_spacing_A": layer_spacing,
                "input_intensities": data.get("intensities", []),
                "input_timestamps": data.get("timestamps", []),
            },
            name=analysis_name,
            replace=True,
        )
    except Exception as exc:
        return jsonify(error=f"FFT completed but could not be saved: {exc}"), 500
    return jsonify(**result, _saved_artifact=artifact)


@app.route("/create_shortcut", methods=["POST"])
def create_shortcut():
    """
    Create a Desktop shortcut to the RHEED Studio launcher on the local
    machine. The server runs on the user's own computer, so it can place a
    .lnk on their Desktop pointing at 'Launch RHEED Studio.bat'. Windows
    only; uses PowerShell's WScript.Shell (no extra Python dependency).
    """
    import platform
    import subprocess

    if platform.system() != "Windows":
        return jsonify(error="Desktop shortcuts are only supported on Windows."), 400

    # project root = two levels up from this file (src/rheed_webapp/app.py)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bat = os.path.join(root, "Launch Auto RHEED.bat")
    ico = os.path.join(root, "auto_rheed.ico")
    if not os.path.exists(bat):
        return jsonify(error=f"Launcher not found at {bat}"), 400

    def ps_quote(p):
        return "'" + p.replace("'", "''") + "'"

    ps = (
        '$d = [Environment]::GetFolderPath("Desktop"); '
        '$lnk = Join-Path $d "Auto RHEED.lnk"; '
        '$w = New-Object -ComObject WScript.Shell; '
        '$s = $w.CreateShortcut($lnk); '
        f'$s.TargetPath = {ps_quote(bat)}; '
        f'$s.WorkingDirectory = {ps_quote(root)}; '
        '$s.Description = "Launch Auto RHEED"; '
        f'if (Test-Path {ps_quote(ico)}) {{ $s.IconLocation = {ps_quote(ico + ",0")} }}; '
        '$s.Save(); Write-Output $lnk'
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return jsonify(error=(r.stderr.strip() or "PowerShell failed")), 500
        return jsonify(path=r.stdout.strip())
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
