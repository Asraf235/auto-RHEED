"""
RHEED Studio — Flask web app.

This module is intentionally thin: every route parses the HTTP
request, calls into a `rheed_core.RheedSession`, and serializes the
result. All RHEED domain logic (file formats, calibration math,
ROI intensity, peak tracking) lives in rheed_core and has no
knowledge of Flask — see src/rheed_core/.
"""
import io
import os
import re
import tempfile

from flask import Flask, request, jsonify, render_template, send_file

from rheed_core import RheedSession
from rheed_core.constants import CV2_CMAPS, IMAGE_EXTS
from rheed_core.library import RheedLibrary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024  # 16 GB
# Werkzeug's default multipart limit is 1000 parts — far too low for a
# growth video reassembled from thousands of individual frame images
# (/upload_image_stack sends one multipart "part" per image). Without
# raising this, a large image-stack upload is rejected by Werkzeug
# itself (a 413, before our route code even runs) with an HTML error
# page instead of JSON, which the frontend can't parse.
app.request_class.max_form_parts = 50_000

# One session per running web app process — matches the original
# single-user, single-dataset-at-a-time behavior of this app.
session = RheedSession()

# Persistent reference gallery of RHEED patterns (RHEED Library tab),
# stored in a folder at the project root.
_LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "rheed_library")
library = RheedLibrary(os.path.abspath(_LIBRARY_DIR))


def _save_upload_to_tmp(f):
    suffix = os.path.splitext(f.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()
    return tmp.name, suffix


# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file"), 400

    path, suffix = _save_upload_to_tmp(f)
    try:
        summary = session.load_file(path, suffix)
    except Exception as e:
        return jsonify(error=f"Failed to load {suffix} file: {e}"), 400
    finally:
        os.unlink(path)

    return jsonify(**summary)


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
    return jsonify(intensities=intensities, timestamps=timestamps)


@app.route("/clim", methods=["POST"])
def set_clim():
    data = request.get_json()
    session.set_clim(data["lo"], data["hi"])
    return jsonify(ok=True)


@app.route("/rotate", methods=["POST"])
def rotate():
    if session.frames is None:
        return jsonify(error="No file loaded"), 400
    data = request.get_json() or {}
    H, W = session.rotate(data.get("direction", "cw"))
    return jsonify(ok=True, width=W, height=H)


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
    try:
        specs = []
        for f in files_sorted:
            suffix = os.path.splitext(f.filename)[1].lower()
            if suffix not in IMAGE_EXTS:
                return jsonify(error=f"Not a supported image type: {f.filename}"), 400
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.save(tmp.name)
            tmp.close()
            tmp_paths.append(tmp.name)
            specs.append((tmp.name, suffix))
        summary = session.load_file_stack(specs)
    except Exception as e:
        return jsonify(error=f"Failed to build video from images: {e}"), 400
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return jsonify(**summary)


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

    return jsonify(
        scale_cm_px=result.scale_cm_px,
        D_px=result.d_px,
        dy_px=result.dy_px,
        dy_cm=result.dy_cm,
        alpha_deg=result.alpha_deg,
        rows={
            "direct_beam": result.direct_beam_y,
            "shadow_edge": result.shadow_edge_y,
            "specular": result.specular_y,
        },
    )


@app.route("/load_image", methods=["POST"])
def load_image():
    """Preview a single still image (used by the Calibration tab's
    standalone 'Open File' button) without committing it as the loaded
    dataset."""
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No file"), 400
    path, suffix = _save_upload_to_tmp(f)
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
    return jsonify(result)


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
    return jsonify(result)


@app.route("/strip_track", methods=["POST"])
def strip_track():
    """
    Automatic strip-profile peak tracking across EVERY loaded frame —
    returns d(t) for the whole video in one call.
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
    return jsonify(result)


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

    return jsonify(
        material_id=doc.get("material_id"),
        formula=doc.get("formula_pretty"),
        a=a, b=b, c=c, gamma=gamma,
        a_conventional=a_conv,
        system=system,
        spacegroup=sym.get("symbol"),
        note="Materials Project value (DFT-relaxed, typically ~1% large). Verify the conventional cell before citing.",
    )


@app.route("/library")
def library_list():
    return jsonify(entries=library.list())


@app.route("/library/add", methods=["POST"])
def library_add():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="No image file"), 400
    path, suffix = _save_upload_to_tmp(f)
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
    return jsonify(result)


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
