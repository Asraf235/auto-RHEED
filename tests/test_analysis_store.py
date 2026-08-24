from __future__ import annotations

import numpy as np
import pytest

from rheed_core import AnalysisStore
from rheed_core.inference import InferenceRunResult, ModelSpec


def _result(task="segmentation"):
    spec = ModelSpec.from_dict({
        "id": "test/model",
        "adapter": "dummy",
        "source": "local.pt",
        "task": task,
    })
    indices = np.asarray([0, 2, 4], dtype=np.int64)
    timestamps = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)
    if task == "embedding":
        return InferenceRunResult(
            model=spec,
            task=task,
            frame_indices=indices,
            timestamps=timestamps,
            metadata={"device": "cpu"},
            embeddings=np.arange(12, dtype=np.float32).reshape(3, 4),
        )
    return InferenceRunResult(
        model=spec,
        task=task,
        frame_indices=indices,
        timestamps=timestamps,
        metadata={"device": "cpu"},
        frames=[
            {"instances": [{"class_id": i, "confidence": 0.9}]}
            for i in range(3)
        ],
    )


def test_analysis_store_round_trips_classical_and_inference(analysis_test_dir):
    store = AnalysisStore(analysis_test_dir / "results")
    run_dir = store.begin_dataset(
        "sample.mp4",
        {"n_frames": 5, "height": 12, "width": 16},
        dataset_version=1,
    )

    artifact = store.record_json(
        "fft",
        {"frequency_hz": [0.0, 1.0], "amplitude": np.asarray([2.0, 3.0])},
        parameters={"detrend": "mean"},
    )
    loaded_artifact = store.load_artifact(artifact["artifact_id"])
    assert loaded_artifact["result"]["amplitude"] == [2.0, 3.0]
    assert artifact["path"].endswith(".json")
    assert not artifact["path"].endswith(".json.gz")
    assert len(artifact["exports"]) == 1
    csv_path = run_dir / artifact["exports"][0]
    assert csv_path.suffix == ".csv"
    assert "frequency_hz,amplitude" in csv_path.read_text(encoding="utf-8-sig")

    stored = store.save_inference("job-1", _result(), dataset_dir=run_dir)
    assert stored.compact_summary()["n_results"] == 3
    assert "frame_summaries" not in stored.compact_summary()
    assert stored.frame_result(2)["instances"][0]["class_id"] == 1
    assert stored.frame_result(3) is None
    restored = stored.load_result()
    assert restored.frame_indices.tolist() == [0, 2, 4]
    assert len(restored.frames) == 3

    stored.save_analysis("pca-kmeans", {
        "frame_indices": [0, 2, 4],
        "cluster_labels": [0, 1, 1],
        "pca_scores": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    })
    assert stored.load_analysis("pca-kmeans")["cluster_labels"] == [0, 1, 1]
    analysis_csv = stored.path / "analyses" / "pca-kmeans.csv"
    assert analysis_csv.is_file()
    assert "pca_scores_1,pca_scores_2" in analysis_csv.read_text(encoding="utf-8-sig")
    matrix = np.eye(3, dtype=np.float32)
    stored.save_analysis_array("similarity-matrix", matrix)
    restored_matrix = stored.load_analysis_array("similarity-matrix", mmap_mode=None)
    assert np.array_equal(restored_matrix, matrix)
    assert len(store.list_artifacts()) == 2
    assert store.list_runs()[0]["source_label"] == "sample.mp4"


def test_analysis_store_saves_named_exports_in_active_run(analysis_test_dir):
    store = AnalysisStore(analysis_test_dir / "results")
    run_dir = store.begin_dataset("sample.mp4", {"n_frames": 1}, dataset_version=1)

    first = store.save_export("../../publication-figure.png", b"first-png")
    assert first == run_dir / "figures" / "publication-figure.png"
    assert first.read_bytes() == b"first-png"

    repeated = store.save_export("publication-figure.png", b"latest-png")
    assert repeated == first
    assert repeated.read_bytes() == b"latest-png"

    with pytest.raises(ValueError, match="PNG images and ZIP"):
        store.save_export("figure.svg", b"svg")


def test_analysis_store_resumes_exact_source_fingerprint(analysis_test_dir):
    store = AnalysisStore(analysis_test_dir / "results")
    first, resumed = store.begin_or_resume_dataset(
        "sample.mp4",
        {"n_frames": 5, "height": 12, "width": 16},
        dataset_version=1,
        source_fingerprint="sha256:abc123",
    )
    assert resumed is False
    store.record_json("intensity", {"timestamps": [0.0], "intensities": [4.0]})

    store.set_root(analysis_test_dir / "results")
    second, resumed = store.begin_or_resume_dataset(
        "renamed-sample.mp4",
        {"n_frames": 5, "height": 12, "width": 16},
        dataset_version=9,
        source_fingerprint="sha256:abc123",
    )
    assert resumed is True
    assert second == first
    assert len(store.list_artifacts()) == 1
    assert store.status()["dataset"]["dataset_version"] == 9


def test_replace_mode_keeps_one_latest_artifact_and_cleans_stale_exports(
    analysis_test_dir,
):
    store = AnalysisStore(analysis_test_dir / "results")
    run_dir = store.begin_dataset("sample.mp4", {"n_frames": 2}, dataset_version=1)
    first = store.record_json(
        "intensity",
        {
            "timestamps": [0.0, 1.0],
            "intensities": [2.0, 4.0],
            "rows": [{"frame": 1, "value": 2.0}],
        },
        name="current",
        replace=True,
    )
    stale_table = run_dir / "classical" / "intensity" / "current-rows.csv"
    assert stale_table.is_file()

    second = store.record_json(
        "intensity",
        {
            "timestamps": [0.0, 1.0],
            "intensities": [3.0, 9.0],
        },
        parameters={"ema_alpha": 0.3},
        name="current",
        replace=True,
    )
    assert second["artifact_id"] == first["artifact_id"] == "intensity/current"
    assert len(store.list_artifacts()) == 1
    assert len((run_dir / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert not stale_table.exists()
    recalled = store.load_artifact("intensity/current")
    assert recalled["parameters"]["ema_alpha"] == 0.3
    assert recalled["result"]["intensities"] == [3.0, 9.0]


def test_analysis_store_embedding_arrays_are_file_backed(analysis_test_dir):
    store = AnalysisStore(analysis_test_dir)
    store.begin_dataset("stack", {"n_frames": 5}, dataset_version=2)
    stored = store.save_inference("job-embed", _result("embedding"))

    assert (stored.path / "embeddings.npy").is_file()
    assert stored.frame_result(4)["embedding"] == [8.0, 9.0, 10.0, 11.0]
    restored = stored.load_result()
    assert restored.embeddings.shape == (3, 4)
    assert restored.frames is None


def test_web_storage_controls_and_classical_auto_save(analysis_test_dir, monkeypatch):
    import rheed_webapp.app as webapp

    monkeypatch.setattr(webapp, "_SETTINGS_PATH", analysis_test_dir / "settings.json")
    webapp.session._set_frames(
        np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8),
        np.asarray([0.0, 0.5]),
    )
    client = webapp.app.test_client()
    target = analysis_test_dir / "chosen-results"
    configured = client.post("/analysis/storage", json={"root": str(target)})
    assert configured.status_code == 200
    assert configured.get_json()["root"] == str(target.resolve())

    response = client.post("/intensity", json={
        "type": "rect",
        "x1": 0,
        "y1": 0,
        "x2": 4,
        "y2": 4,
        "display_w": 8,
        "display_h": 6,
    })
    assert response.status_code == 200, response.get_json()
    artifact_id = response.get_json()["_saved_artifact"]["artifact_id"]
    listed = client.get("/analysis/artifacts").get_json()["artifacts"]
    assert artifact_id in {item["artifact_id"] for item in listed}
    recalled = client.get(f"/analysis/artifacts/{artifact_id}")
    assert recalled.status_code == 200
    assert len(recalled.get_json()["result"]["intensities"]) == 2

    repeated = client.post("/intensity", json={
        "type": "rect",
        "x1": 1,
        "y1": 1,
        "x2": 5,
        "y2": 5,
        "display_w": 8,
        "display_h": 6,
    })
    assert repeated.status_code == 200
    assert repeated.get_json()["_saved_artifact"]["artifact_id"] == "intensity/current"
    assert len(client.get("/analysis/artifacts").get_json()["artifacts"]) == 1

    interactive = client.post("/analysis/record", json={
        "kind": "intensity",
        "name": "current",
        "replace": True,
        "parameters": {"plot_options": {"log_y": False}},
        "result": {
            "plot_options": {"log_y": False},
            "series": [{
                "source": "roi",
                "label": "ROI 1",
                "color": "#38bdf8",
                "timestamps": [0.0, 0.5],
                "intensities": [4.0, 8.0],
                "show": True,
                "ema_enabled": True,
                "ema_alpha": 0.4,
                "ema_intensities": [4.0, 5.6],
            }],
        },
    })
    assert interactive.status_code == 200, interactive.get_json()
    latest = client.get("/analysis/artifacts/intensity/current").get_json()
    assert latest["result"]["series"][0]["ema_alpha"] == 0.4
    artifacts = client.get("/analysis/artifacts").get_json()["artifacts"]
    assert len(artifacts) == 1
    export_path = webapp.analysis_store.dataset_dir / artifacts[0]["exports"][0]
    export_text = export_path.read_text(encoding="utf-8-sig")
    assert "ROI-1_raw" in export_text
    assert "ROI-1_ema_alpha_0.40" in export_text
