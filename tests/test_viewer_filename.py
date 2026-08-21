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
