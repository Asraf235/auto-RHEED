import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from rheed_core.intensity_correction import build_intensity_correction
from rheed_core.session import RheedSession


def test_direct_beam_correction_removes_linear_current_drift():
    beam = np.linspace(100.0, 50.0, 7)
    result = build_intensity_correction(
        beam, smoothing_window=1, max_factor=10
    )

    corrected = beam * np.asarray(result["correction_factors"])
    assert corrected == pytest.approx(np.full(7, 100.0))
    assert result["reference_intensity"] == pytest.approx(100.0)
    assert result["normalized_curve"] == pytest.approx(beam / 100.0)


def test_default_correction_makes_same_roi_constant_through_deep_clipping():
    beam = np.asarray([
        320e6, 285e6, 230e6, 18e6, 7e6, 22e6,
        240e6, 310e6, 275e6, 16e6, 5e6, 24e6,
    ])
    result = build_intensity_correction(beam)

    corrected = beam * np.asarray(result["correction_factors"])
    assert corrected == pytest.approx(np.full(beam.size, beam.max()))
    assert result["smoothing_window"] == 1
    assert result["cap_engaged_frames"] == 0
    assert result["required_max_factor"] == pytest.approx(64.0)


def test_direct_beam_correction_caps_fully_clipped_measurements():
    result = build_intensity_correction(
        [100.0, 0.0, 50.0], smoothing_window=1, max_factor=4
    )

    assert result["normalized_curve"][1] == 0.0
    assert result["application_curve"][1] == pytest.approx(0.25)
    assert result["correction_factors"][1] == pytest.approx(4.0)
    assert result["cap_engaged_frames"] == 1
    assert all(0.25 <= value <= 4.0 for value in result["correction_factors"])


@pytest.mark.parametrize("window", [0, 2, 4])
def test_direct_beam_correction_rejects_invalid_smoothing_windows(window):
    with pytest.raises(ValueError, match="positive odd"):
        build_intensity_correction([1, 2, 3], smoothing_window=window)


def test_render_correction_is_display_only_and_supported_by_zip_export():
    session = RheedSession()
    session.frames = np.array([[[0, 10], [20, 30]], [[0, 5], [10, 15]]], dtype=np.uint16)
    session.timestamps = np.array([0.0, 1.0])
    session.n_frames = 2
    session.shape = (2, 2)
    session.clim = (0, 30)
    original = session.frames.copy()

    corrected = session.corrected_display_frame(1, 2.0)
    assert corrected.tolist() == [[0.0, 10.0], [20.0, 30.0]]
    assert np.array_equal(session.frames, original)

    archive = session.export_all_frames_zip([1.0, 2.0])
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert zf.namelist() == ["frame_0000.png", "frame_0001.png"]


def test_analysis_store_saves_reusable_correction_under_root(analysis_test_dir):
    from rheed_core import AnalysisStore

    store = AnalysisStore(analysis_test_dir / "data")
    paths = store.save_correction("sample-direct-beam", {
        "timestamps": [0.0, 1.0],
        "normalized_curve": [1.0, 0.8],
    })

    assert Path(paths["json"]).parent == analysis_test_dir / "data" / "corrections"
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    assert "normalized_curve" in Path(paths["csv"]).read_text(encoding="utf-8-sig")

    run = store.begin_dataset("sample.npy", {"n_frames": 2}, dataset_version=1)
    artifact = store.record_json("intensity", {"series": [{
        "label": "specular",
        "timestamps": [0.0, 1.0],
        "intensities": [10.0, 8.0],
        "current_corrected_intensities": [10.0, 10.0],
    }]})
    exported = (run / artifact["exports"][0]).read_text(encoding="utf-8-sig")
    assert "specular_raw,specular_current_corrected" in exported


def test_intensity_correction_route_saves_normalized_curve(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "data")
    frames = np.stack([
        np.full((8, 8), value, dtype=np.uint16)
        for value in (100, 80, 50)
    ])
    webapp.session._set_frames(frames, np.asarray([0.0, 1.0, 2.0]))
    response = webapp.app.test_client().post("/intensity_correction", json={
        "source": "active",
        "roi": {"x1": 0, "y1": 0, "x2": 4, "y2": 4},
        "display_w": 8,
        "display_h": 8,
        "smoothing_window": 1,
        "max_factor": 10,
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["normalized_curve"] == pytest.approx([1.0, 0.8, 0.5])
    assert data["correction_factors"] == pytest.approx([1.0, 1.25, 2.0])
    same_roi = webapp.app.test_client().post("/intensity", json={
        "type": "rect",
        "x1": 0,
        "y1": 0,
        "x2": 4,
        "y2": 4,
        "display_w": 8,
        "display_h": 8,
    }).get_json()["intensities"]
    assert np.asarray(same_roi) * np.asarray(data["correction_factors"]) == pytest.approx(
        np.full(3, max(same_roi))
    )
    assert Path(data["correction_files"]["csv"]).parent.name == "corrections"
    assert Path(data["correction_files"]["csv"]).is_file()


def test_separate_reference_video_is_time_aligned_to_active_dataset(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "data")
    webapp.session._set_frames(
        np.ones((3, 4, 4), dtype=np.uint16),
        np.asarray([0.0, 0.5, 1.0]),
    )
    reference = io.BytesIO()
    np.save(reference, np.stack([
        np.full((4, 4), 100, dtype=np.uint16),
        np.full((4, 4), 50, dtype=np.uint16),
    ]))
    reference.seek(0)
    client = webapp.app.test_client()
    loaded = client.post(
        "/intensity_correction/reference",
        data={"file": (reference, "beam-reference.npy")},
        content_type="multipart/form-data",
    )
    assert loaded.status_code == 200
    reference_frame = client.get("/intensity_correction/reference/frame/1")
    assert reference_frame.status_code == 200
    assert reference_frame.get_json()["image"]

    response = client.post("/intensity_correction", json={
        "source": "reference",
        "roi": {"x1": 0, "y1": 0, "x2": 4, "y2": 4},
        "display_w": 4,
        "display_h": 4,
        "smoothing_window": 1,
        "max_factor": 10,
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["source"] == "reference"
    assert len(data["application_curve"]) == 3
    assert len(data["correction_factors"]) == 3


def test_page_exposes_intensity_correction_workflow():
    import rheed_webapp.app as webapp

    page = webapp.app.test_client().get("/").get_data(as_text=True)
    assert 'data-tab="calib">Intensity Correction' in page
    assert 'id="corr-measure"' in page
    assert 'name="corr-source" value="reference"' in page
    assert 'id="corr-apply-curves"' in page
    assert 'id="corr-apply-video"' in page
    assert 'id="corr-frame-slider"' in page
    assert 'id="corr-play"' in page
    assert 'id="corr-zoom-in"' in page
    assert 'id="corr-zoom-out"' in page
    assert 'id="corr-zoom-reset"' in page
    assert 'id="corr-fps-slider"' in page
    assert "drawResizeHandles(octx, shape)" in page
    assert "roiMoveState" in page
    assert "roiResizeState" in page
    assert "measured intensity ÷ sensitivity" in page
    assert "fetch('/intensity_correction/reference'" in page
    assert "fetch('/intensity/batch'" in page
    assert "displayedIntensityValues(s)" in page
    assert "showCorrectionFrame(next)" in page
