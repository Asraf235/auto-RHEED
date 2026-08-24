"""
RheedSession — the in-memory representation of "currently loaded RHEED
dataset", plus the analysis operations that act on it (frame rendering,
ROI intensity, 5-point calibration, peak tracking, rotation).

This replaces the module-level `store` dict the original Flask app used
directly. Each consumer (the web app, an MCP server) owns its own
RheedSession instance — there is no hidden global state in this module.
"""
import base64
import io
import zipfile

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _find_peaks
from numpy.fft import rfft as _rfft, rfftfreq as _rfftfreq

from . import io as rheed_io
from .constants import CV2_CMAPS, IMAGE_EXTS, VIDEO_EXTS
from .calibration import Point, calibrate as _calibrate
from .background import (
    COARSE_PERCENTILE_DEFAULTS,
    normalize_background_config,
    subtract_coarse_percentile_background,
)
from . import streak_profile as _sp


def frame_to_display_uint8(frame: np.ndarray, clim, colormap: str = "gray") -> np.ndarray:
    """Apply display contrast/colormap and return an encodable 8-bit image."""
    lo, hi = clim
    arr = np.clip(frame.astype(np.float32), lo, hi)
    arr = ((arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
    cmap_id = CV2_CMAPS.get(colormap)
    if cmap_id is not None:
        arr = cv2.applyColorMap(arr, cmap_id)  # → BGR uint8
    return arr


def frame_to_image_bytes(
    frame: np.ndarray,
    clim,
    colormap: str = "gray",
    *,
    encoding: str = "png",
    jpeg_quality: int = 90,
) -> bytes:
    """Encode a display frame as lossless PNG or fast playback JPEG."""
    arr = frame_to_display_uint8(frame, clim, colormap)
    if encoding == "jpeg":
        ok, buf = cv2.imencode(
            ".jpg",
            arr,
            [cv2.IMWRITE_JPEG_QUALITY, max(60, min(int(jpeg_quality), 100))],
        )
    elif encoding == "png":
        ok, buf = cv2.imencode(".png", arr)
    else:
        raise ValueError("encoding must be 'png' or 'jpeg'")
    if not ok:
        raise RuntimeError(f"Could not encode frame as {encoding}")
    return buf.tobytes()


def frame_to_png_bytes(frame: np.ndarray, clim, colormap: str = "gray") -> bytes:
    return frame_to_image_bytes(frame, clim, colormap, encoding="png")


def frame_to_png_b64(frame: np.ndarray, clim, colormap: str = "gray") -> str:
    return base64.b64encode(frame_to_png_bytes(frame, clim, colormap)).decode()


class RheedSession:
    """Holds one loaded RHEED dataset (a stack of frames) and operates on it."""

    def __init__(self):
        self.frames = None        # np.ndarray (N, H, W)
        self.timestamps = None    # np.ndarray (N,)
        self.n_frames = 0
        self.shape = (0, 0)       # (H, W)
        self.clim = (0, 65535)
        self.clim_auto = (0, 65535)
        self.colormap = "gray"
        self.background_subtraction = {
            "enabled": False,
            **COARSE_PERCENTILE_DEFAULTS,
        }
        # Monotonic identity for results tied to native pixel data. Loading
        # or rotating replaces the coordinate system and invalidates cached
        # calibration, ROI, streak, and AI inference results.
        self.dataset_version = 0

    # ── loading ──────────────────────────────────────────────────────
    def load_file(self, path: str, suffix: str) -> dict:
        suffix = suffix.lower()
        if suffix == ".h5":
            frames, ts = rheed_io.load_h5(path)
        elif suffix in VIDEO_EXTS:
            frames, ts = rheed_io.load_video(path)
        elif suffix == ".npy":
            frames, ts = rheed_io.load_npy(path)
        elif suffix in IMAGE_EXTS:
            frames, ts = rheed_io.load_single_frame_image(path, suffix)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        self._set_frames(frames, ts)
        return self.summary()

    def load_file_stack(self, file_specs) -> dict:
        """Build a 'video' from a batch of individual still images —
        file_specs is a list of (path, suffix) tuples, in temporal order."""
        frames, ts = rheed_io.load_image_stack(file_specs)
        self._set_frames(frames, ts)
        return self.summary()

    def _set_frames(self, frames: np.ndarray, ts: np.ndarray):
        self.frames = frames
        self.timestamps = ts
        self.n_frames = len(frames)
        self.shape = frames.shape[1], frames.shape[2]
        f0 = frames[0].astype(np.float32)
        clim = (int(f0.min()), int(f0.max()))
        self.clim = clim
        self.clim_auto = clim
        # A fresh file load should mean fresh display defaults — otherwise
        # the frontend resets its colormap dropdown to "Gray" on load but
        # the backend keeps rendering with whatever colormap was set for
        # the PREVIOUS file, and the two visibly disagree.
        self.colormap = "gray"
        self.background_subtraction = {
            "enabled": False,
            **COARSE_PERCENTILE_DEFAULTS,
        }
        self.dataset_version += 1

    def summary(self) -> dict:
        H, W = self.shape
        ts, n = self.timestamps, self.n_frames
        duration = float(ts[-1] - ts[0]) if ts is not None and n > 1 else n
        fps = n / duration if duration > 0 else 1
        return {
            "n_frames": n, "height": H, "width": W,
            "duration": round(duration, 3), "fps": round(fps, 2),
            "clim": list(self.clim), "clim_auto": list(self.clim_auto),
            "background_subtraction": dict(self.background_subtraction),
        }

    def preview_image(self, path: str, suffix: str):
        """
        Decode a single still image to a base64 PNG WITHOUT committing it
        as the loaded dataset (n_frames/timestamps untouched) — used by
        the Calibration tab's standalone "Open File" preview. Still
        updates `shape` so coordinate scaling in calibrate() is correct.
        Returns (b64_png, width, height).
        """
        if suffix == ".img":
            img = rheed_io.decode_ksa_img(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image")
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            img = ((img.astype(np.float32) - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode()
        h, w = img.shape[:2]
        self.shape = (h, w)
        return b64, w, h

    # ── frame access ─────────────────────────────────────────────────
    def require_frames(self):
        if self.frames is None:
            raise RuntimeError("No file loaded")
        return self.frames

    def display_frame(self, idx: int) -> np.ndarray:
        """Return one frame with the active non-destructive display transform."""
        frames = self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        frame = frames[idx]
        if self.background_subtraction["enabled"]:
            return subtract_coarse_percentile_background(
                frame,
                self.background_subtraction,
            )
        return frame

    def set_background_subtraction(
        self,
        enabled: bool,
        config: dict | None = None,
    ) -> dict:
        settings = normalize_background_config(config)
        self.background_subtraction = {"enabled": bool(enabled), **settings}
        return dict(self.background_subtraction)

    def inference_background_config(self) -> dict:
        """Return the active transform without its display-only enabled flag."""
        return normalize_background_config(self.background_subtraction)

    def analysis_frame(self, idx: int, use_background_subtraction: bool = False) -> np.ndarray:
        """Return raw or corrected pixels for an explicitly configured analysis."""
        frames = self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        if not use_background_subtraction:
            return frames[idx]
        if not self.background_subtraction["enabled"]:
            raise ValueError(
                "Background subtraction must be enabled before ROI analyses can use it"
            )
        return self.display_frame(idx)

    def frame_png_b64(self, idx: int, auto_contrast: bool = False):
        """auto_contrast=True recomputes clim from THIS frame (and stores
        it as the current clim, so the UI sliders stay in sync) instead
        of using the existing self.clim — lets a "per-frame auto" mode
        rescale contrast on every frame of a video without the user
        re-clicking Auto each time."""
        frames = self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        clim = self.clim
        if auto_contrast:
            clim = self.clim_auto_for_frame(idx)
            self.clim = clim
        b64 = frame_to_png_b64(self.display_frame(idx), clim, self.colormap)
        t = float(self.timestamps[idx]) if self.timestamps is not None else idx
        return b64, round(t, 4)

    def frame_image_bytes(
        self,
        idx: int,
        *,
        auto_contrast: bool = False,
        encoding: str = "jpeg",
        jpeg_quality: int = 90,
    ):
        """Binary frame response used by the latency-sensitive browser player."""
        self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        clim = self.clim_auto_for_frame(idx) if auto_contrast else self.clim
        if auto_contrast:
            self.clim = clim
        payload = frame_to_image_bytes(
            self.display_frame(idx),
            clim,
            self.colormap,
            encoding=encoding,
            jpeg_quality=jpeg_quality,
        )
        timestamp = float(self.timestamps[idx]) if self.timestamps is not None else idx
        return payload, round(timestamp, 4), clim

    def mean_frame_png_b64(self):
        frames = self.require_frames()
        n_use = min(len(frames), 100)
        if self.background_subtraction["enabled"]:
            mf = np.zeros(self.shape, dtype=np.float32)
            for i in range(n_use):
                mf += self.display_frame(i).astype(np.float32)
            mf /= max(n_use, 1)
        else:
            mf = frames[:n_use].astype(np.float32).mean(axis=0)
        mf_norm = mf / max(mf.max(), 1)
        arr = (mf_norm ** 0.4 * 255).astype(np.uint8)  # gamma 0.4 enhancement
        cmap_id = CV2_CMAPS.get(self.colormap)
        if cmap_id is not None:
            arr = cv2.applyColorMap(arr, cmap_id)
        _, buf = cv2.imencode(".png", arr)
        b64 = base64.b64encode(buf).decode()
        return b64, int(arr.shape[1]), int(arr.shape[0])

    # ── export ───────────────────────────────────────────────────────
    def export_frame_png(self, idx: int) -> bytes:
        """Raw PNG bytes for a single frame, rendered with the current
        contrast/colormap (i.e. exactly what's currently displayed)."""
        self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        return frame_to_png_bytes(self.display_frame(idx), self.clim, self.colormap)

    def export_all_frames_zip(self) -> bytes:
        """ZIP archive containing every loaded frame as a numbered PNG,
        rendered with the current contrast/colormap."""
        self.require_frames()
        n_digits = max(4, len(str(self.n_frames - 1)))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(self.n_frames):
                name = f"frame_{i:0{n_digits}d}.png"
                zf.writestr(name, self.export_frame_png(i))
        return buf.getvalue()

    # ── transforms ───────────────────────────────────────────────────
    def rotate(self, direction: str = "cw"):
        frames = self.require_frames()
        k = -1 if direction == "cw" else 1  # np.rot90 k=1 is counter-clockwise
        self.frames = np.rot90(frames, k=k, axes=(1, 2)).copy()
        self.shape = self.frames.shape[1], self.frames.shape[2]
        self.dataset_version += 1
        return self.shape

    def set_clim(self, lo: int, hi: int):
        self.clim = (int(lo), int(hi))

    def clim_auto_for_frame(self, idx: int):
        """Auto-contrast range computed from a SPECIFIC frame, not just
        frame 0. RHEED spot intensity commonly decays/shifts a lot over a
        growth run, so the frame-0 auto range can leave later frames
        clipped outside [lo, hi] entirely — no slider tweak can recover
        signal that's been clipped to flat black/white. Letting the user
        re-auto on whichever frame they're currently viewing fixes that."""
        self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        f = self.display_frame(idx)
        return int(f.min()), int(f.max())

    def set_colormap(self, cmap: str):
        if cmap not in CV2_CMAPS:
            raise ValueError(f"Unknown colormap: {cmap}")
        self.colormap = cmap

    # ── ROI intensity ────────────────────────────────────────────────
    def compute_intensity(self, roi_type: str, roi: dict,
                           display_w: float | None = None, display_h: float | None = None,
                           use_background_subtraction: bool = False):
        """
        roi_type: 'circle' | 'rect' | 'line'
        roi: circle→{cx,cy,r}  rect→{x1,y1,x2,y2}  line→{x1,y1,x2,y2,width}
        display_w/h: size of the canvas the click coords came from; pass
        None (or the native W/H) if `roi` is already in native pixel space.
        """
        frames = self.require_frames()
        H, W = self.shape
        dw = display_w or W
        dh = display_h or H
        sx, sy = W / dw, H / dh

        if roi_type == "circle":
            cx, cy = roi["cx"] * sx, roi["cy"] * sy
            r = roi["r"] * min(sx, sy)
            ys, xs = np.ogrid[:H, :W]
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
            if not mask.any():
                raise ValueError("ROI empty")
            if use_background_subtraction:
                intensities = [
                    float(self.analysis_frame(i, True)[mask].astype(np.float64).sum())
                    for i in range(self.n_frames)
                ]
            else:
                intensities = frames[:, mask].astype(np.float64).sum(axis=1).tolist()

        elif roi_type == "rect":
            x1 = int(round(min(roi["x1"], roi["x2"]) * sx))
            x2 = int(round(max(roi["x1"], roi["x2"]) * sx))
            y1 = int(round(min(roi["y1"], roi["y2"]) * sy))
            y2 = int(round(max(roi["y1"], roi["y2"]) * sy))
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("ROI too small")
            if use_background_subtraction:
                intensities = [
                    float(self.analysis_frame(i, True)[y1:y2, x1:x2].astype(np.float64).sum())
                    for i in range(self.n_frames)
                ]
            else:
                intensities = frames[:, y1:y2, x1:x2].astype(np.float64).sum(axis=(1, 2)).tolist()

        elif roi_type == "line":
            x1, y1 = roi["x1"] * sx, roi["y1"] * sy
            x2, y2 = roi["x2"] * sx, roi["y2"] * sy
            half_w = max(1, int(round(roi.get("width", 3) * min(sx, sy))))
            length = int(np.hypot(x2 - x1, y2 - y1))
            if length < 2:
                raise ValueError("Line too short")
            t_vals = np.linspace(0, 1, length)
            xs_line = x1 + t_vals * (x2 - x1)
            ys_line = y1 + t_vals * (y2 - y1)
            dx = -(y2 - y1) / length
            dy = (x2 - x1) / length
            intensities = []
            for frame_index, raw_frame in enumerate(frames):
                frame = self.analysis_frame(
                    frame_index,
                    use_background_subtraction,
                ) if use_background_subtraction else raw_frame
                total, count = 0.0, 0
                for offset in range(-half_w, half_w + 1):
                    ys_off = ys_line + offset * dy
                    xs_off = xs_line + offset * dx
                    vals = map_coordinates(frame.astype(np.float64),
                                            [ys_off, xs_off], order=1, mode="nearest")
                    total += vals.sum()
                    count += len(vals)
                intensities.append(total / count if count else 0.0)
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")

        return intensities, self.timestamps.tolist()

    # ── 5-point calibration ──────────────────────────────────────────
    def calibrate_points(self, points: dict, l_cm: float, d_cm: float,
                          display_w: float | None = None, display_h: float | None = None):
        H, W = self.shape
        dw = display_w or W
        dh = display_h or H
        sx, sy = W / dw, H / dh

        def s(pt):
            return Point(pt["x"] * sx, pt["y"] * sy)

        result = _calibrate(
            direct_beam=s(points["direct_beam"]),
            shadow_edge=s(points["shadow_edge"]),
            specular=s(points["specular"]),
            cam_edge_left=s(points["cam_edge_left"]),
            cam_edge_right=s(points["cam_edge_right"]),
            l_cm=l_cm, d_cm=d_cm,
        )
        return result

    # ── peak tracking / α(t) ─────────────────────────────────────────
    def track_peak(self, specular_roi=None, direct_beam_roi=None,
                    shadow_edge_y=None, direct_beam_y=None,
                    scale_cm_px=1.0, l_cm=10.5,
                    display_w=None, display_h=None):
        frames = self.require_frames()
        H, W = self.shape
        dw = display_w or W
        dh = display_h or H
        sx, sy = W / dw, H / dh

        shadow_y = float(shadow_edge_y) if shadow_edge_y is not None else H / 2
        scale = float(scale_cm_px)
        l_cm = float(l_cm)
        t_arr = self.timestamps

        def crop(roi):
            x1 = int(round(min(roi["x1"], roi["x2"]) * sx))
            x2 = int(round(max(roi["x1"], roi["x2"]) * sx))
            y1 = int(round(min(roi["y1"], roi["y2"]) * sy))
            y2 = int(round(max(roi["y1"], roi["y2"]) * sy))
            return max(0, x1), min(W, x2), max(0, y1), min(H, y2)

        def find_peak(frame, x1, x2, y1, y2):
            sub = frame[y1:y2, x1:x2].astype(np.float64)
            if sub.size == 0:
                return (y1 + y2) / 2, (x1 + x2) / 2
            flat = np.argmax(sub)
            pr0, pc0 = np.unravel_index(flat, sub.shape)
            r0, c0 = int(pr0), int(pc0)
            win = 5
            rs = slice(max(0, r0 - win), min(sub.shape[0], r0 + win + 1))
            cs = slice(max(0, c0 - win), min(sub.shape[1], c0 + win + 1))
            patch = sub[rs, cs]
            patch = np.maximum(patch - patch.min(), 0)
            total = patch.sum()
            if total == 0:
                return float(y1 + pr0), float(x1 + pc0)
            ys_idx, xs_idx = np.mgrid[rs, cs]
            cy = float((patch * ys_idx).sum() / total) + y1
            cx = float((patch * xs_idx).sum() / total) + x1
            return cy, cx

        def roi_intensity(frame, x1, x2, y1, y2):
            return float(frame[y1:y2, x1:x2].astype(np.float64).sum())

        def sine_fit_alpha(t_arr, alpha_arr):
            t = np.array(t_arr, dtype=float)
            a = np.array(alpha_arr, dtype=float)
            if len(a) < 10:
                return a.tolist(), None
            A0 = max((a.max() - a.min()) / 2.0, 0.01)
            off0 = (a.max() + a.min()) / 2.0
            dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
            freqs = _rfftfreq(len(a), d=dt)
            power = np.abs(_rfft(a - a.mean()))
            power[0] = 0
            f0 = float(freqs[np.argmax(power)]) if power.max() > 0 else 0.1
            if f0 <= 0:
                f0 = 0.1
            peaks, _ = _find_peaks(a, prominence=0.3 * A0)
            phi0 = float(-2 * np.pi * f0 * t[peaks[0]] + np.pi / 2) if len(peaks) > 0 else 0.0
            phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi

            def model(tt, A, f, phi, off):
                return A * np.sin(2 * np.pi * f * tt + phi) + off

            p0 = [A0, f0, phi0, off0]
            bounds = ([A0 * 0.2, f0 * 0.4, -np.pi, off0 - A0 * 2],
                      [A0 * 3.0, f0 * 2.5, np.pi, off0 + A0 * 2])
            try:
                popt, _ = curve_fit(model, t, a, p0=p0, bounds=bounds, maxfev=8000)
                fitted = model(t, *popt).tolist()
                params = {"A": round(popt[0], 4), "f_Hz": round(popt[1], 5),
                          "phi": round(popt[2], 4), "offset": round(popt[3], 4)}
                return fitted, params
            except Exception:
                return a.tolist(), None

        result = {"timestamps": t_arr.tolist()}

        if specular_roi:
            x1, x2, y1, y2 = crop(specular_roi)
            peak_y, peak_x, alphas, intens = [], [], [], []
            for frame in frames:
                cy, cx = find_peak(frame, x1, x2, y1, y2)
                dy = cy - shadow_y
                alpha = float(np.degrees(np.arctan2(dy * scale, l_cm)))
                peak_y.append(round(cy, 3)); peak_x.append(round(cx, 3))
                alphas.append(round(alpha, 4))
                intens.append(round(roi_intensity(frame, x1, x2, y1, y2), 1))
            intens_arr = np.array(intens, dtype=float)
            mx = intens_arr.max()
            norm_intens = (intens_arr / mx).tolist() if mx > 0 else intens
            result["specular"] = {
                "peak_row": peak_y, "peak_col": peak_x, "alpha_deg": alphas,
                "intensity": intens, "intensity_norm": [round(v, 6) for v in norm_intens],
            }

        if direct_beam_roi:
            x1, x2, y1, y2 = crop(direct_beam_roi)
            peak_y, peak_x, alphas, intens = [], [], [], []
            for frame in frames:
                cy, cx = find_peak(frame, x1, x2, y1, y2)
                dy = shadow_y - cy
                alpha = float(np.degrees(np.arctan2(dy * scale, l_cm)))
                peak_y.append(round(cy, 3)); peak_x.append(round(cx, 3))
                alphas.append(round(alpha, 4))
                intens.append(round(roi_intensity(frame, x1, x2, y1, y2), 1))
            alpha_sine, sine_params = sine_fit_alpha(t_arr, alphas)
            intens_arr = np.array(intens, dtype=float)
            mx = intens_arr.max()
            norm_intens = (intens_arr / mx).tolist() if mx > 0 else intens
            result["direct_beam"] = {
                "peak_row": peak_y, "peak_col": peak_x, "alpha_deg": alphas,
                "alpha_sine": [round(v, 4) for v in alpha_sine], "sine_params": sine_params,
                "intensity": intens, "intensity_norm": [round(v, 6) for v in norm_intens],
            }

        return result

    # ── strip-profile streak tracking ────────────────────────────────
    def strip_profile(self, idx: int, y1: float, y2: float,
                       x1: float | None = None, x2: float | None = None,
                       bg_method: str = "als", bg_params: dict | None = None,
                       prev_peaks: dict | None = None, vertical_track: bool = True,
                       display_w: float | None = None, display_h: float | None = None,
                       use_background_subtraction: bool = False):
        """
        Live single-frame preview: strip-sum a horizontal ROI on frame
        `idx`, subtract a background, and find the specular + two
        first-order peaks. Used to show the raw/baseline/subtracted
        profile panel and to preview a background-subtraction method
        before committing to a full-video track.
        """
        self.require_frames()
        idx = max(0, min(idx, self.n_frames - 1))
        frame = self.analysis_frame(idx, use_background_subtraction)
        H, W = self.shape
        dw = display_w or W
        dh = display_h or H
        sx, sy = W / dw, H / dh

        ny1, ny2 = sorted((y1 * sy, y2 * sy))
        nx1 = None if x1 is None else x1 * sx
        nx2 = None if x2 is None else x2 * sx

        x0 = int(nx1) if nx1 is not None else 0
        band_lo = band_hi = None
        if vertical_track:
            # Seed the vertical search at the previous specular column if
            # we have one (shifted to LOCAL strip coords), else ROI centre.
            center_col = None
            if prev_peaks and prev_peaks.get("specular_x") is not None:
                center_col = prev_peaks["specular_x"] - x0
            profile, band_lo_n, band_hi_n = _sp.strip_sum_profile_vtrack(
                frame, ny1, ny2, nx1, nx2, center_col=center_col)
            band_lo, band_hi = band_lo_n / sy, band_hi_n / sy  # → display px
        else:
            profile = _sp.strip_sum_profile(frame, ny1, ny2, nx1, nx2)
        baseline, subtracted = _sp.subtract_background(profile, bg_method, bg_params)

        # find_three_peaks works in LOCAL indices (0 = the strip's own
        # left edge), but prev_peaks comes from the frontend in ABSOLUTE
        # native-frame pixel coordinates (what it draws/displays) — shift
        # back to local before searching.
        prev = None
        if prev_peaks:
            def _to_local(v):
                return None if v is None else v - x0
            prev = _sp.StreakPeaks(
                specular_x=_to_local(prev_peaks.get("specular_x")),
                left_x=_to_local(prev_peaks.get("left_x")),
                right_x=_to_local(prev_peaks.get("right_x")),
            )
        peaks = _sp.find_three_peaks(subtracted, prev=prev)

        # Shift back to ABSOLUTE native-frame pixel coordinates before
        # returning — the chart/canvas overlay both work in that space.
        def _to_abs(v):
            return None if v is None else v + x0
        specular_abs = _to_abs(peaks.specular_x)
        left_abs = _to_abs(peaks.left_x)
        right_abs = _to_abs(peaks.right_x)

        # Any EXTRA prominent peaks beyond the known ±1 first-order pair
        # — e.g. higher diffraction orders or surface-reconstruction
        # streaks that a wide ROI also captures. Purely for display; the
        # d-spacing measurement above is unaffected and still anchored
        # to the ±1 pair only.
        all_peaks = []
        if peaks.specular_x is not None:
            for p in _sp.find_extra_peaks(subtracted, peaks.specular_x, peaks.left_x, peaks.right_x):
                all_peaks.append({"x": p["x"] + x0, "order": p["order"], "intensity": p["intensity"]})

        return {
            "x0": x0,
            "raw": profile.tolist(),
            "baseline": baseline.tolist(),
            "subtracted": subtracted.tolist(),
            "peaks": {
                "specular_x": specular_abs, "left_x": left_abs, "right_x": right_abs,
                "left_dx": peaks.left_dx, "right_dx": peaks.right_dx, "avg_dx": peaks.avg_dx,
            },
            "all_peaks": all_peaks,
            "band_y": None if band_lo is None else [band_lo, band_hi],
            "input_background_subtracted": bool(use_background_subtraction),
        }

    def strip_track(self, y1: float, y2: float, x1: float | None, x2: float | None,
                     bg_method: str, bg_params: dict | None,
                     scale_cm_px: float, l_cm: float, v_kev: float,
                     vertical_track: bool = True,
                     display_w: float | None = None, display_h: float | None = None,
                     use_background_subtraction: bool = False):
        """
        Run the strip-profile measurement across EVERY loaded frame,
        tracking the three peaks frame-to-frame (each frame's search is
        seeded from the previous frame's found positions), and convert
        the average specular↔first-order distance to d (Å) per frame.
        This is the automatic equivalent of clicking two streaks on
        every frame by hand.
        """
        frames = self.require_frames()
        H, W = self.shape
        dw = display_w or W
        dh = display_h or H
        sx, sy = W / dw, H / dh

        ny1, ny2 = sorted((y1 * sy, y2 * sy))
        nx1 = None if x1 is None else x1 * sx
        nx2 = None if x2 is None else x2 * sx
        x0 = int(nx1) if nx1 is not None else 0

        # Last known-good expected positions, tracked independently per
        # peak so a momentary loss on ONE side (e.g. a noisy frame)
        # doesn't permanently null out that side for every later frame —
        # the search keeps retrying around its last good position. These
        # stay in LOCAL strip-relative indices (matching what
        # find_three_peaks works in); only the output lists below get
        # shifted to absolute native-frame pixel coordinates.
        last_specular = last_left = last_right = None
        specular_x, left_x, right_x = [], [], []
        avg_dx_list, d_list = [], []
        fwhm_half_list, fwhm_gauss_list = [], []
        left_fwhm_half_list, right_fwhm_half_list = [], []
        first_order_fwhm_half_list = []
        left_fwhm_gauss_list, right_fwhm_gauss_list = [], []
        first_order_fwhm_gauss_list = []
        specular_intensity_list = []
        left_intensity_list, right_intensity_list = [], []
        first_order_intensity_list = []
        lost_frames = 0

        for frame_index, raw_frame in enumerate(frames):
            frame = self.analysis_frame(
                frame_index,
                use_background_subtraction,
            ) if use_background_subtraction else raw_frame
            if vertical_track:
                # Center the band on the last-known specular column so the
                # band follows the spots vertically across the sweep.
                profile, _bl, _bh = _sp.strip_sum_profile_vtrack(
                    frame, ny1, ny2, nx1, nx2, center_col=last_specular)
            else:
                profile = _sp.strip_sum_profile(frame, ny1, ny2, nx1, nx2)
            _baseline, subtracted = _sp.subtract_background(profile, bg_method, bg_params)
            prev = (_sp.StreakPeaks(last_specular, last_left, last_right)
                    if last_specular is not None else None)
            peaks = _sp.find_three_peaks(subtracted, prev=prev)

            specular_x.append(peaks.specular_x + x0 if peaks.specular_x is not None else None)
            left_x.append(peaks.left_x + x0 if peaks.left_x is not None else None)
            right_x.append(peaks.right_x + x0 if peaks.right_x is not None else None)

            if peaks.specular_x is not None:
                last_specular = peaks.specular_x
            if peaks.left_x is not None:
                last_left = peaks.left_x
            if peaks.right_x is not None:
                last_right = peaks.right_x

            if peaks.avg_dx is not None and peaks.avg_dx > 0:
                d = _sp.d_from_dx(peaks.avg_dx, scale_cm_px, l_cm, v_kev)
                avg_dx_list.append(peaks.avg_dx)
                d_list.append(d)
            else:
                avg_dx_list.append(None)
                d_list.append(None)
                lost_frames += 1

            # Measure the specular and both nearest first-order streaks by
            # both supported methods. Keep the two first-order sides in the
            # result for provenance, and provide their available-side mean
            # for plotting. Coherence length remains derived from the
            # specular width in the frontend.
            specular_fwhm_half = _sp.fwhm_of_peak(subtracted, peaks.specular_x)
            left_fwhm_half = _sp.fwhm_of_peak(subtracted, peaks.left_x)
            right_fwhm_half = _sp.fwhm_of_peak(subtracted, peaks.right_x)
            specular_fwhm_gauss = _sp.fwhm_gauss_of_peak(subtracted, peaks.specular_x)
            left_fwhm_gauss = _sp.fwhm_gauss_of_peak(subtracted, peaks.left_x)
            right_fwhm_gauss = _sp.fwhm_gauss_of_peak(subtracted, peaks.right_x)

            half_values = [
                value for value in (left_fwhm_half, right_fwhm_half)
                if value is not None
            ]
            gauss_values = [
                value for value in (left_fwhm_gauss, right_fwhm_gauss)
                if value is not None
            ]
            fwhm_half_list.append(specular_fwhm_half)
            fwhm_gauss_list.append(specular_fwhm_gauss)
            left_fwhm_half_list.append(left_fwhm_half)
            right_fwhm_half_list.append(right_fwhm_half)
            first_order_fwhm_half_list.append(
                float(np.mean(half_values)) if half_values else None
            )
            left_fwhm_gauss_list.append(left_fwhm_gauss)
            right_fwhm_gauss_list.append(right_fwhm_gauss)
            first_order_fwhm_gauss_list.append(
                float(np.mean(gauss_values)) if gauss_values else None
            )

            # Peak heights from the same background-subtracted profile and
            # tracked positions used for spacing/FWHM. Report both ±1 sides
            # for provenance and their available-side mean for the UI series.
            specular_intensity = _sp.peak_intensity(subtracted, peaks.specular_x)
            left_intensity = _sp.peak_intensity(subtracted, peaks.left_x)
            right_intensity = _sp.peak_intensity(subtracted, peaks.right_x)
            first_order_values = [
                value for value in (left_intensity, right_intensity)
                if value is not None
            ]
            specular_intensity_list.append(specular_intensity)
            left_intensity_list.append(left_intensity)
            right_intensity_list.append(right_intensity)
            first_order_intensity_list.append(
                float(np.mean(first_order_values)) if first_order_values else None
            )

        return {
            "input_background_subtracted": bool(use_background_subtraction),
            "timestamps": self.timestamps.tolist(),
            "specular_x": specular_x, "left_x": left_x, "right_x": right_x,
            "avg_dx_px": avg_dx_list, "d_angstrom": d_list,
            "fwhm_halfmax_px": fwhm_half_list, "fwhm_gauss_px": fwhm_gauss_list,
            "left_first_order_fwhm_halfmax_px": left_fwhm_half_list,
            "right_first_order_fwhm_halfmax_px": right_fwhm_half_list,
            "first_order_fwhm_halfmax_px": first_order_fwhm_half_list,
            "left_first_order_fwhm_gauss_px": left_fwhm_gauss_list,
            "right_first_order_fwhm_gauss_px": right_fwhm_gauss_list,
            "first_order_fwhm_gauss_px": first_order_fwhm_gauss_list,
            "specular_intensity_bgsub": specular_intensity_list,
            "left_first_order_intensity_bgsub": left_intensity_list,
            "right_first_order_intensity_bgsub": right_intensity_list,
            "first_order_intensity_bgsub": first_order_intensity_list,
            "lost_frames": lost_frames,
        }
