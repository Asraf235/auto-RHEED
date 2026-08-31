from __future__ import annotations

import numpy as np
import pytest

from rheed_core.background import (
    COARSE_PERCENTILE_DEFAULTS,
    normalize_background_config,
    subtract_coarse_percentile_background,
    subtract_coarse_percentile_stack,
)
from rheed_core.session import RheedSession


def test_coarse_percentile_subtraction_preserves_spot_and_dtype():
    y, x = np.indices((192, 256))
    frame = (900 + 2 * x + y).astype(np.uint16)
    frame[80:84, 120:124] += 5000

    corrected = subtract_coarse_percentile_background(frame)

    assert corrected.shape == frame.shape
    assert corrected.dtype == frame.dtype
    assert corrected[81, 121] > 4000
    assert np.median(corrected) < 150


def test_coarse_percentile_stack_processes_every_frame_without_mutating_source():
    frames = np.stack([
        np.full((64, 80), 500, dtype=np.uint16),
        np.full((64, 80), 900, dtype=np.uint16),
    ])
    original = frames.copy()

    corrected = subtract_coarse_percentile_stack(frames)

    assert np.array_equal(frames, original)
    assert corrected.shape == frames.shape
    assert np.count_nonzero(corrected) == 0


def test_background_config_defaults_and_validation():
    assert normalize_background_config() == COARSE_PERCENTILE_DEFAULTS
    assert normalize_background_config({
        "tile_size": 128,
        "percentile": 35,
        "smooth_sigma_tiles": 1.5,
        "sample_step": 3,
    }) == {
        "method": "coarse_percentile",
        "tile_size": 128,
        "percentile": 35.0,
        "smooth_sigma_tiles": 1.5,
        "sample_step": 3,
    }
    with pytest.raises(ValueError, match="tile_size"):
        normalize_background_config({"tile_size": 2})
    with pytest.raises(ValueError, match="tile_size"):
        normalize_background_config({"tile_size": 5000})
    with pytest.raises(ValueError, match="percentile"):
        normalize_background_config({"percentile": 101})
    with pytest.raises(ValueError, match="sample_step"):
        normalize_background_config({"sample_step": 33})


def test_session_background_is_display_only_and_resets_on_new_dataset():
    frames = np.stack([
        np.full((64, 80), 500, dtype=np.uint16),
        np.full((64, 80), 900, dtype=np.uint16),
    ])
    session = RheedSession()
    session._set_frames(frames.copy(), np.asarray([0.0, 1.0]))
    dataset_version = session.dataset_version

    settings = session.set_background_subtraction(True)

    assert settings["tile_size"] == 96
    assert session.dataset_version == dataset_version
    assert np.count_nonzero(session.display_frame(0)) == 0
    assert np.array_equal(session.frames, frames)
    assert session.summary()["background_subtraction"]["enabled"] is True

    session._set_frames(frames.copy(), np.asarray([0.0, 1.0]))
    assert session.background_subtraction["enabled"] is False


@pytest.mark.parametrize(
    ("roi_type", "roi"),
    [
        ("circle", {"cx": 20, "cy": 20, "r": 8}),
        ("rect", {"x1": 8, "y1": 8, "x2": 30, "y2": 30}),
        ("line", {"x1": 8, "y1": 8, "x2": 30, "y2": 30, "width": 3}),
    ],
)
def test_normal_roi_intensity_can_use_corrected_pixels(roi_type, roi):
    frames = np.stack([
        np.full((48, 64), 500, dtype=np.uint16),
        np.full((48, 64), 900, dtype=np.uint16),
    ])
    session = RheedSession()
    session._set_frames(frames, np.asarray([0.0, 1.0]))

    raw, _ = session.compute_intensity(roi_type, roi)
    session.set_background_subtraction(True)
    corrected, _ = session.compute_intensity(
        roi_type,
        roi,
        use_background_subtraction=True,
    )

    assert all(value > 0 for value in raw)
    assert corrected == [0.0, 0.0]


def test_corrected_roi_batch_estimates_background_once_per_frame(monkeypatch):
    import rheed_core.session as session_module

    frames = np.stack([
        np.full((48, 64), value, dtype=np.uint16)
        for value in (500, 700, 900)
    ])
    session = RheedSession()
    session._set_frames(frames, np.asarray([0.0, 1.0, 2.0]))
    session.set_background_subtraction(True)
    original = session_module.subtract_coarse_percentile_background
    calls = 0

    def counted(frame, config):
        nonlocal calls
        calls += 1
        return original(frame, config)

    monkeypatch.setattr(
        session_module,
        "subtract_coarse_percentile_background",
        counted,
    )
    series, timestamps = session.compute_intensities([
        {"type": "circle", "cx": 20, "cy": 20, "r": 8},
        {"type": "rect", "x1": 8, "y1": 8, "x2": 30, "y2": 30},
        {"type": "line", "x1": 8, "y1": 8, "x2": 30, "y2": 30, "width": 3},
    ], use_background_subtraction=True)

    assert calls == len(frames)
    assert timestamps == [0.0, 1.0, 2.0]
    assert series == [[0.0, 0.0, 0.0]] * 3


def test_intensity_route_records_corrected_roi_input(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    frames = np.stack([
        np.full((48, 64), 500, dtype=np.uint16),
        np.full((48, 64), 900, dtype=np.uint16),
    ])
    webapp.session._set_frames(frames, np.asarray([0.0, 1.0]))
    client = webapp.app.test_client()
    background_response = client.post("/background_subtraction", json={
        "enabled": True,
        "config": {
            "tile_size": 32,
            "percentile": 35,
            "smooth_sigma_tiles": 1.5,
            "sample_step": 2,
        },
    })
    assert background_response.status_code == 200
    assert background_response.get_json()["background_subtraction"]["tile_size"] == 32
    assert background_response.get_json()["background_subtraction"]["percentile"] == 35.0

    response = client.post("/intensity", json={
        "type": "rect",
        "x1": 8,
        "y1": 8,
        "x2": 30,
        "y2": 30,
        "use_background_subtraction": True,
    })

    assert response.status_code == 200
    assert response.get_json()["input_background_subtracted"] is True
    assert response.get_json()["intensities"] == [0.0, 0.0]


def test_intensity_batch_route_returns_aligned_series(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    frames = np.stack([
        np.full((48, 64), value, dtype=np.uint16)
        for value in (500, 900)
    ])
    webapp.session._set_frames(frames, np.asarray([0.0, 1.0]))
    webapp.session.set_background_subtraction(True)

    response = webapp.app.test_client().post("/intensity/batch", json={
        "rois": [
            {"type": "circle", "cx": 20, "cy": 20, "r": 8},
            {"type": "rect", "x1": 8, "y1": 8, "x2": 30, "y2": 30},
        ],
        "use_background_subtraction": True,
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["timestamps"] == [0.0, 1.0]
    assert data["input_background_subtracted"] is True
    assert [row["intensities"] for row in data["series"]] == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_background_setting_change_can_preserve_contrast(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    frames = np.stack([
        np.arange(48 * 64, dtype=np.uint16).reshape(48, 64),
        np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 100,
    ])
    webapp.session._set_frames(frames, np.asarray([0.0, 1.0]))
    webapp.session.set_clim(123, 2345)

    response = webapp.app.test_client().post("/background_subtraction", json={
        "enabled": True,
        "frame_index": 0,
        "rescale_contrast": False,
        "config": {
            "tile_size": 24,
            "percentile": 30,
            "smooth_sigma_tiles": 1.0,
            "sample_step": 2,
        },
    })

    assert response.status_code == 200
    assert response.get_json()["clim"] == [123, 2345]
    assert webapp.session.clim == (123, 2345)
