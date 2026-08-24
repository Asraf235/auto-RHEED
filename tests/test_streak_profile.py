from __future__ import annotations

import numpy as np
import pytest

from rheed_core.session import RheedSession
from rheed_core.streak_profile import peak_intensity


def _three_streak_frames() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(128, dtype=float)
    frames = []
    for frame_index, scale in enumerate((0.8, 1.0, 1.2, 1.4)):
        profile = (
            scale * 700 * np.exp(-0.5 * ((x - 32) / 2.2) ** 2)
            + scale * 1500 * np.exp(-0.5 * ((x - 64) / 2.8) ** 2)
        )
        # Deliberately lose the right first-order peak on the final frame.
        if frame_index < 3:
            profile += scale * 550 * np.exp(-0.5 * ((x - 96) / 3.4) ** 2)
        row = np.rint(profile).astype(np.uint16)
        frames.append(np.repeat(row[np.newaxis, :], 8, axis=0))
    return np.stack(frames), np.arange(4, dtype=float) * 0.25


def _track(session: RheedSession, use_background_subtraction: bool = False) -> dict:
    return session.strip_track(
        y1=0,
        y2=8,
        x1=0,
        x2=128,
        bg_method="none",
        bg_params=None,
        scale_cm_px=0.01,
        l_cm=10.0,
        v_kev=15.0,
        vertical_track=False,
        display_w=128,
        display_h=8,
        use_background_subtraction=use_background_subtraction,
    )


def test_peak_intensity_handles_subpixels_and_missing_positions():
    profile = np.asarray([0.0, 4.0, 8.0, 0.0])
    assert peak_intensity(profile, 1.5) == pytest.approx(6.0)
    assert peak_intensity(profile, None) is None
    assert peak_intensity(profile, -0.1) is None


def test_strip_track_returns_peak_intensities_and_first_order_fwhm():
    frames, timestamps = _three_streak_frames()
    session = RheedSession()
    session._set_frames(frames, timestamps)

    result = _track(session)

    assert len(result["specular_intensity_bgsub"]) == len(frames)
    assert result["specular_intensity_bgsub"][:3] == sorted(
        result["specular_intensity_bgsub"][:3]
    )
    for i in range(3):
        assert result["first_order_intensity_bgsub"][i] == pytest.approx(
            np.mean([
                result["left_first_order_intensity_bgsub"][i],
                result["right_first_order_intensity_bgsub"][i],
            ])
        )
        assert result["first_order_fwhm_halfmax_px"][i] == pytest.approx(
            np.mean([
                result["left_first_order_fwhm_halfmax_px"][i],
                result["right_first_order_fwhm_halfmax_px"][i],
            ])
        )
        assert result["first_order_fwhm_gauss_px"][i] == pytest.approx(
            np.mean([
                result["left_first_order_fwhm_gauss_px"][i],
                result["right_first_order_fwhm_gauss_px"][i],
            ])
        )

    assert result["right_first_order_intensity_bgsub"][-1] is None
    assert result["first_order_intensity_bgsub"][-1] == pytest.approx(
        result["left_first_order_intensity_bgsub"][-1]
    )
    assert result["right_first_order_fwhm_halfmax_px"][-1] is None
    assert result["first_order_fwhm_halfmax_px"][-1] == pytest.approx(
        result["left_first_order_fwhm_halfmax_px"][-1]
    )

    # The intentionally broader right streak should remain broader than the
    # left streak for both measurement methods.
    assert result["right_first_order_fwhm_halfmax_px"][0] > (
        result["left_first_order_fwhm_halfmax_px"][0]
    )
    assert result["right_first_order_fwhm_gauss_px"][0] > (
        result["left_first_order_fwhm_gauss_px"][0]
    )


def test_strip_preview_and_tracking_can_use_corrected_frames():
    frames, timestamps = _three_streak_frames()
    frames = frames + np.uint16(500)
    session = RheedSession()
    session._set_frames(frames, timestamps)
    raw = session.strip_profile(
        0, 0, 8, 0, 128,
        bg_method="none",
        vertical_track=False,
        display_w=128,
        display_h=8,
    )
    session.set_background_subtraction(True)

    corrected = session.strip_profile(
        0, 0, 8, 0, 128,
        bg_method="none",
        vertical_track=False,
        display_w=128,
        display_h=8,
        use_background_subtraction=True,
    )
    tracked = _track(session, use_background_subtraction=True)

    assert corrected["input_background_subtracted"] is True
    assert tracked["input_background_subtracted"] is True
    assert np.median(corrected["raw"]) < np.median(raw["raw"])


def test_strip_track_route_persists_new_series(analysis_test_dir):
    import rheed_webapp.app as webapp

    frames, timestamps = _three_streak_frames()
    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    webapp.session._set_frames(frames, timestamps)

    client = webapp.app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    page_text = page.get_data(as_text=True)
    assert "Profile specular" in page_text
    assert "Profile 1st order (avg)" in page_text
    assert page_text.count('id="gr-fwhm-method"') == 1
    assert page_text.count('id="gr-roi-use-background"') == 1
    assert page_text.index('id="gr-fwhm-method"') < page_text.index('id="gr-roi-btn"')

    response = client.post("/strip_track", json={
        "y1": 0,
        "y2": 8,
        "x1": 0,
        "x2": 128,
        "bg_method": "none",
        "scale_cm_px": 0.01,
        "L_cm": 10.0,
        "V_keV": 15.0,
        "vertical_track": False,
        "display_w": 128,
        "display_h": 8,
    })

    assert response.status_code == 200, response.get_json()
    payload = response.get_json()
    artifact_id = payload["_saved_artifact"]["artifact_id"]
    recalled = client.get(f"/analysis/artifacts/{artifact_id}")
    assert recalled.status_code == 200
    saved_result = recalled.get_json()["result"]
    assert saved_result["specular_intensity_bgsub"] == payload["specular_intensity_bgsub"]
    assert saved_result["first_order_fwhm_gauss_px"] == payload["first_order_fwhm_gauss_px"]
    assert "left_first_order_fwhm_gauss_px" in saved_result
    assert "right_first_order_fwhm_gauss_px" in saved_result


def test_live_strip_preview_does_not_create_saved_artifacts(analysis_test_dir):
    import rheed_webapp.app as webapp

    frames, timestamps = _three_streak_frames()
    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    webapp.session._set_frames(frames, timestamps)
    response = webapp.app.test_client().post("/strip_profile", json={
        "frame_idx": 0,
        "y1": 0,
        "y2": 8,
        "x1": 0,
        "x2": 128,
        "bg_method": "none",
        "vertical_track": False,
        "display_w": 128,
        "display_h": 8,
    })
    assert response.status_code == 200, response.get_json()
    assert "_saved_artifact" not in response.get_json()
    assert webapp.analysis_store.list_artifacts() == []
