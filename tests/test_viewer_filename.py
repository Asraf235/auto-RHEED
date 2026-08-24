from __future__ import annotations

import io

import numpy as np


def test_upload_returns_filename_and_viewer_header_is_wired(analysis_test_dir):
    import rheed_webapp.app as webapp

    webapp.analysis_store.set_root(analysis_test_dir / "analysis")
    payload = io.BytesIO()
    np.save(payload, np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8))
    payload.seek(0)

    client = webapp.app.test_client()
    uploaded = client.post(
        "/upload",
        data={"file": (payload, "growth_run_042.npy")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200, uploaded.get_json()
    assert uploaded.get_json()["dataset_name"] == "growth_run_042.npy"
    assert uploaded.get_json()["analysis_resumed"] is False
    analysis_dir = webapp.analysis_store.dataset_dir

    figure = client.post(
        "/analysis/figures",
        data={
            "filename": "cluster-map.png",
            "file": (io.BytesIO(b"\x89PNG\r\n\x1a\nplot-bytes"), "cluster-map.png"),
        },
        content_type="multipart/form-data",
    )
    assert figure.status_code == 200, figure.get_json()
    assert figure.get_json()["relative_path"] == "figures/cluster-map.png"
    assert (analysis_dir / "figures" / "cluster-map.png").is_file()

    frame = client.post("/analysis/figures/frame/0", json={})
    assert frame.status_code == 200, frame.get_json()
    assert (analysis_dir / frame.get_json()["relative_path"]).is_file()

    all_frames = client.post("/analysis/figures/all-frames", json={})
    assert all_frames.status_code == 200, all_frames.get_json()
    assert (analysis_dir / all_frames.get_json()["relative_path"]).is_file()

    saved = client.post("/intensity", json={
        "type": "rect", "x1": 0, "y1": 0, "x2": 4, "y2": 4,
        "display_w": 8, "display_h": 6,
    })
    assert saved.status_code == 200

    reopened_payload = io.BytesIO()
    np.save(
        reopened_payload,
        np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8),
    )
    reopened_payload.seek(0)
    reopened = client.post(
        "/upload",
        data={"file": (reopened_payload, "growth_run_042.npy")},
        content_type="multipart/form-data",
    )
    assert reopened.status_code == 200, reopened.get_json()
    assert reopened.get_json()["analysis_resumed"] is True
    assert reopened.get_json()["saved_artifact_count"] == 1

    runs = client.get("/analysis/runs").get_json()["runs"]
    assert sum(run["active"] for run in runs) == 1
    assert any(run["compatible"] for run in runs)

    page_text = client.get("/").get_data(as_text=True)
    assert 'id="gr-viewer-title"' in page_text
    assert "function setLoadedDatasetName(name)" in page_text
    assert "Viewer — ${datasetName}" in page_text
    assert 'id="gr-recall-artifact-btn"' in page_text
    assert 'id="chart-export-width"' in page_text
    assert 'id="chart-export-height"' in page_text
    assert 'id="chart-export-width" type="number" value="3.5"' in page_text
    assert 'id="chart-export-height" type="number" value="3.5"' in page_text
    assert 'id="chart-export-font" type="number" value="10"' in page_text
    assert 'id="chart-export-transparent" type="checkbox" checked' in page_text
    assert "fetch('/analysis/figures', { method: 'POST', body: form })" in page_text
    assert "figures/all-frames" in page_text
    assert "const AUTO_PLOT_EXPORT = Object.freeze" in page_text
    assert "function schedulePlotAutosave(div, filename" in page_text
    assert "schedulePlotAutosave('intensity-chart', 'intensity_vs_time.png')" in page_text
    assert "schedulePlotAutosave('alpha-plotly', 'alpha_vs_time.png')" in page_text
    assert "schedulePlotAutosave('gr-profile-plotly', 'streak_profile_current.png')" in page_text
