from __future__ import annotations

from io import BytesIO
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rheed_core.inference import (
    InferenceCancelled,
    InferenceRunResult,
    ModelAdapter,
    ModelSpec,
    cluster_embeddings,
    list_embedding_analysis_adapters,
    run_embedding_analysis_with_artifacts,
    run_inference,
)
from rheed_core.inference._vendor.rhaapsody_changepoint import ChangepointDetection
from rheed_core.inference.base import AdapterDescriptor
from rheed_core.inference.preprocessing import build_preprocessing_context, frame_to_rgb_uint8
from rheed_core.inference import base as adapter_base
from rheed_core.inference.ultralytics_seg import UltralyticsSegAdapter
from rheed_core.session import RheedSession


class DummyEmbeddingAdapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="dummy-embedding",
        task="embedding",
        description="Dependency-free test embedding adapter",
    )

    def load(self, spec, device):
        return {"device": device, "source": spec.source}

    def infer_batch(self, frames, context):
        rows = []
        for frame in frames:
            image = frame_to_rgb_uint8(frame, context)[:, :, 0].astype(float)
            rows.append([image.mean(), image.std(), image[0, 0], image[-1, -1]])
        return {"embeddings": np.asarray(rows, dtype=np.float32)}


class DummySegmentationAdapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="dummy-segmentation",
        task="segmentation",
        description="Dependency-free test segmentation adapter",
    )

    def load(self, spec, device):
        return {"device": device}

    def infer_batch(self, frames, context):
        return {"frames": [{"instances": [{
            "track_id": 17,
            "class_id": 0,
            "class_name": "spot",
            "confidence": 0.9,
            "box_xyxy": [1.0, 1.0, 3.0, 3.0],
            "polygon_xy": [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0]],
            "mask_area_px": 3.0,
        }]} for _ in frames]}


class DummyRegressionAdapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="dummy-regression",
        task="regression",
        description="Dependency-free test regression adapter",
    )

    def load(self, spec, device):
        return {"device": device}

    def infer_batch(self, frames, context):
        return {"frames": [
            {"values": {"roughness_nm": float(frame.mean())}}
            for frame in frames
        ]}


@pytest.fixture(autouse=True)
def dummy_adapters(monkeypatch):
    monkeypatch.setitem(
        adapter_base._BUILTINS,
        "dummy-embedding",
        "test_inference:DummyEmbeddingAdapter",
    )
    monkeypatch.setitem(
        adapter_base._BUILTINS,
        "dummy-segmentation",
        "test_inference:DummySegmentationAdapter",
    )
    monkeypatch.setitem(
        adapter_base._BUILTINS,
        "dummy-regression",
        "test_inference:DummyRegressionAdapter",
    )


def _frames(n=8):
    y, x = np.indices((12, 16))
    return np.stack([(x * (i + 1) + y * (i + 2) + i * 50).astype(np.uint16)
                     for i in range(n)])


def _spec(adapter="dummy-embedding", task="embedding"):
    return ModelSpec.from_dict({
        "id": "test-model",
        "adapter": adapter,
        "source": "memory",
        "task": task,
        "preprocessing": {
            "intensity_scaling": "fixed",
            "intensity_low": 0,
            "intensity_high": 1000,
        },
    })


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def _fake_ultralytics_result(track_id=None):
    boxes = SimpleNamespace(
        xyxy=_FakeTensor([[1.0, 2.0, 5.0, 7.0]]),
        conf=_FakeTensor([0.875]),
        cls=_FakeTensor([2]),
        id=_FakeTensor([track_id]) if track_id is not None else None,
    )
    masks = SimpleNamespace(
        xy=[np.asarray([[1.0, 2.0], [5.0, 2.0], [5.0, 7.0]])],
        data=_FakeTensor(np.ones((1, 3, 4), dtype=np.float32)),
    )
    return SimpleNamespace(boxes=boxes, masks=masks, names={2: "streak"})


class _FakeUltralyticsModel:
    def __init__(self):
        self.predict_calls = []
        self.track_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [_fake_ultralytics_result() for _ in kwargs["source"]]

    def track(self, **kwargs):
        self.track_calls.append(kwargs)
        return [_fake_ultralytics_result(len(self.track_calls))]


def test_embedding_alignment_clustering_and_npz_roundtrip():
    frames = _frames()
    timestamps = np.arange(len(frames), dtype=float) * 0.25
    progress = []
    result = run_inference(
        frames,
        timestamps,
        _spec(),
        batch_size=2,
        stride=2,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert result.embeddings.shape == (4, 4)
    assert result.frame_indices.tolist() == [0, 2, 4, 6]
    assert result.timestamps.tolist() == [0.0, 0.5, 1.0, 1.5]
    assert progress == [(0, 4), (2, 4), (4, 4)]

    analysis = cluster_embeddings(result, pca_components=3, n_clusters=2, random_state=7)
    assert analysis.pca_scores.shape == (4, 3)
    assert sorted(set(analysis.cluster_labels.tolist())) == [0, 1]
    assert len(analysis.representative_positions) == 2

    payload = result.to_npz_bytes(analysis.to_dict())
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        assert archive["embeddings"].shape == (4, 4)
        assert archive["frame_indices"].tolist() == [0, 2, 4, 6]
        assert "analysis_json" in archive


def test_variance_target_pca_and_silhouette_cluster_selection():
    rng = np.random.default_rng(12)
    centers = np.asarray([
        [-5.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
    ])
    embeddings = np.vstack([
        center + rng.normal(scale=0.08, size=(12, 4))
        for center in centers
    ]).astype(np.float32)
    result = InferenceRunResult(
        model=_spec(),
        task="embedding",
        frame_indices=np.arange(len(embeddings), dtype=np.int64),
        timestamps=np.arange(len(embeddings), dtype=float),
        metadata={},
        embeddings=embeddings,
    )

    analysis = cluster_embeddings(
        result,
        pca_components=0.95,
        n_clusters="auto",
        max_clusters=5,
        random_state=4,
        normalize=False,
    )

    assert analysis.parameters["pca_selection"] == "explained_variance"
    assert analysis.parameters["pca_components_requested"] == pytest.approx(0.95)
    assert analysis.parameters["explained_variance_total"] >= 0.95
    assert analysis.parameters["cluster_selection"] == "silhouette"
    assert analysis.parameters["n_clusters"] == 3
    assert set(analysis.parameters["silhouette_scores"]) == {"2", "3", "4", "5"}
    assert analysis.parameters["silhouette_score"] == pytest.approx(
        analysis.parameters["silhouette_scores"]["3"]
    )


def test_automatic_cluster_selection_uses_one_cluster_for_two_frames():
    result = InferenceRunResult(
        model=_spec(),
        task="embedding",
        frame_indices=np.asarray([0, 1]),
        timestamps=np.asarray([0.0, 1.0]),
        metadata={},
        embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )

    analysis = cluster_embeddings(result, pca_components=0.95, n_clusters="auto")

    assert analysis.parameters["n_clusters"] == 1
    assert analysis.parameters["cluster_selection"] == "silhouette"
    assert analysis.parameters["single_cluster_baseline"] == 0.0
    assert analysis.parameters["silhouette_score"] is None
    assert analysis.parameters["silhouette_scores"] == {}
    assert analysis.parameters["max_clusters_evaluated"] == 1
    assert analysis.cluster_labels.tolist() == [0, 0]


def test_automatic_cluster_selection_accepts_maximum_k_of_one():
    result = InferenceRunResult(
        model=_spec(),
        task="embedding",
        frame_indices=np.arange(4, dtype=np.int64),
        timestamps=np.arange(4, dtype=float),
        metadata={},
        embeddings=np.eye(4, dtype=np.float32),
    )

    analysis = cluster_embeddings(
        result,
        pca_components=0.95,
        n_clusters="auto",
        max_clusters=1,
    )

    assert analysis.parameters["n_clusters"] == 1
    assert analysis.parameters["max_clusters_requested"] == 1
    assert analysis.parameters["max_clusters_evaluated"] == 1
    assert analysis.representative_positions.shape == (1,)


def test_rhaapsody_detector_finds_block_similarity_change():
    matrix = np.block([
        [np.ones((8, 8)), np.zeros((8, 8))],
        [np.zeros((8, 8)), np.ones((8, 8))],
    ])
    detector = ChangepointDetection(
        cost_threshold=0.01,
        window_size=np.inf,
        min_time_between_changepoints=2,
    )
    detections = []
    for seen in range(4, len(matrix) + 1):
        proposed, amplitude, _, actual = detector.get_changepoint(
            matrix[:seen, :seen], seen
        )
        if actual:
            detections.append((int(proposed), float(amplitude)))

    assert detections
    assert any(position == 8 for position, _ in detections)


def test_rhaapsody_embedding_analysis_uses_aligned_existing_embeddings():
    embeddings = np.asarray([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        [5.0, 5.0], [5.2, 5.0], [5.0, 5.2], [5.1, 5.1],
    ], dtype=np.float32)
    result = InferenceRunResult(
        model=_spec(),
        task="embedding",
        frame_indices=np.arange(0, 16, 2, dtype=np.int64),
        timestamps=np.arange(8, dtype=float) * 0.5,
        metadata={},
        embeddings=embeddings,
    )

    analysis, artifacts = run_embedding_analysis_with_artifacts(
        result,
        "rhaapsody-changepoint",
        {
            "starting_period": 4,
            "cost_threshold": 10.0,
            "window_size": "full",
            "min_time_between_changepoints": 2,
        },
    )

    assert analysis["task"] == "changepoint"
    assert analysis["score_positions"] == [4, 5, 6, 7]
    assert analysis["score_frame_indices"] == [8, 10, 12, 14]
    assert analysis["score_timestamps"] == [2.0, 2.5, 3.0, 3.5]
    assert analysis["parameters"]["window_size"] is None
    assert analysis["provenance"]["project"] == "RHAAPSODY"
    assert analysis["similarity_matrix"]["shape"] == [8, 8]
    similarity = artifacts["similarity-matrix"]
    assert similarity.dtype == np.float32
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    descriptors = list_embedding_analysis_adapters()
    assert descriptors[0]["name"] == "rhaapsody-changepoint"
    assert descriptors[0]["available"] is True


def test_segmentation_result_uses_native_frame_indices():
    result = run_inference(
        _frames(3),
        np.asarray([0.0, 0.1, 0.2]),
        _spec("dummy-segmentation", "segmentation"),
        stride=2,
    )
    assert result.frame_indices.tolist() == [0, 2]
    assert result.frame_result(1) is None
    frame = result.frame_result(2)
    assert frame["instances"][0]["track_id"] == 17
    assert frame["instances"][0]["class_name"] == "spot"
    assert frame["instances"][0]["box_xyxy"] == [1.0, 1.0, 3.0, 3.0]


def test_ultralytics_tracking_persists_across_runner_batches():
    frames = _frames(3)
    context = build_preprocessing_context(
        frames,
        np.arange(len(frames)),
        {"intensity_scaling": "fixed", "intensity_low": 0, "intensity_high": 1000},
    )
    model = _FakeUltralyticsModel()
    adapter = UltralyticsSegAdapter()
    adapter.model = model
    adapter.options = {
        **adapter.descriptor.default_options,
        "tracking": True,
        "tracker": "bytetrack.yaml",
    }
    assert adapter.effective_batch_size(32) == 1

    first = adapter.infer_batch(frames[:2], context)
    second = adapter.infer_batch(frames[2:], context)

    assert model.predict_calls == []
    assert len(model.track_calls) == 3
    assert all(call["persist"] is True for call in model.track_calls)
    assert all(call["tracker"] == "bytetrack.yaml" for call in model.track_calls)
    assert all(isinstance(call["source"], np.ndarray) for call in model.track_calls)
    instances = [
        *[frame["instances"][0] for frame in first["frames"]],
        *[frame["instances"][0] for frame in second["frames"]],
    ]
    assert [instance["track_id"] for instance in instances] == [1, 2, 3]
    assert instances[0]["class_name"] == "streak"
    assert instances[0]["mask_area_px"] == 12.0


def test_ultralytics_prediction_keeps_track_id_schema_without_tracking():
    frames = _frames(2)
    context = build_preprocessing_context(
        frames,
        np.arange(len(frames)),
        {"intensity_scaling": "fixed", "intensity_low": 0, "intensity_high": 1000},
    )
    model = _FakeUltralyticsModel()
    adapter = UltralyticsSegAdapter()
    adapter.model = model
    adapter.options = dict(adapter.descriptor.default_options)
    assert adapter.effective_batch_size(32) == 32

    result = adapter.infer_batch(frames, context)

    assert len(model.predict_calls) == 1
    assert len(model.predict_calls[0]["source"]) == 2
    assert model.track_calls == []
    assert [frame["instances"][0]["track_id"] for frame in result["frames"]] == [None, None]


def test_new_regression_adapter_uses_generic_frame_result_contract():
    result = run_inference(
        _frames(4),
        np.arange(4, dtype=float),
        _spec("dummy-regression", "regression"),
        batch_size=3,
    )
    assert result.task == "regression"
    assert len(result.frames) == 4
    assert result.frame_result(2)["values"]["roughness_nm"] > 0
    assert "values" in result.summary()["frame_summaries"][0]


def test_inference_applies_recorded_background_subtraction_to_every_batch():
    frames = np.stack([
        np.full((32, 40), 500, dtype=np.uint16),
        np.full((32, 40), 900, dtype=np.uint16),
        np.full((32, 40), 1300, dtype=np.uint16),
    ])
    source = frames.copy()
    model = _spec("dummy-regression", "regression")
    model = ModelSpec.from_dict({
        **model.public_dict(),
        "preprocessing": {
            **model.preprocessing,
            "background_subtraction": {
                "method": "coarse_percentile",
                "tile_size": 16,
                "percentile": 40.0,
                "smooth_sigma_tiles": 1.25,
                "sample_step": 2,
            },
        },
    })

    result = run_inference(
        frames,
        np.arange(3, dtype=float),
        model,
        batch_size=2,
    )

    assert np.array_equal(frames, source)
    assert [frame["values"]["roughness_nm"] for frame in result.frames] == [0.0, 0.0, 0.0]
    assert result.metadata["preprocessing"]["config"]["background_subtraction"]["tile_size"] == 16


def test_cancellation_stops_before_first_batch():
    with pytest.raises(InferenceCancelled):
        run_inference(
            _frames(3),
            None,
            _spec(),
            cancelled=lambda: True,
        )


def test_session_dataset_version_changes_on_load_and_rotation():
    session = RheedSession()
    session._set_frames(_frames(2), np.asarray([0.0, 1.0]))
    loaded_version = session.dataset_version
    session.rotate("cw")
    assert session.dataset_version == loaded_version + 1
    assert session.frames.shape == (2, 16, 12)


def test_flask_background_embedding_routes(analysis_test_dir):
    from rheed_webapp.app import ai_jobs, analysis_store, app, session

    analysis_store.set_root(analysis_test_dir / "analysis")
    session._set_frames(_frames(6), np.arange(6, dtype=float))
    client = app.test_client()
    background = client.post("/background_subtraction", json={
        "enabled": True,
        "frame_index": 0,
    })
    assert background.status_code == 200
    assert background.get_json()["background_subtraction"] == {
        "enabled": True,
        "method": "coarse_percentile",
        "tile_size": 96,
        "percentile": 40.0,
        "smooth_sigma_tiles": 1.25,
        "sample_step": 2,
    }
    background_model = _spec().public_dict()
    background_model["id"] = "background-test-model"
    background_response = client.post("/ai/run", json={
        "model": background_model,
        "batch_size": 2,
        "stride": 1,
        "use_background_subtraction": True,
    })
    assert background_response.status_code == 202
    assert background_response.get_json()["model"]["preprocessing"]["background_subtraction"]["tile_size"] == 96

    response = client.post("/ai/run", json={
        "model": _spec().public_dict(),
        "batch_size": 2,
        "stride": 1,
        "use_background_subtraction": False,
    })
    assert response.status_code == 202
    assert "background_subtraction" not in response.get_json()["model"]["preprocessing"]
    job_id = response.get_json()["job_id"]

    status = None
    for _ in range(100):
        status = client.get(f"/ai/jobs/{job_id}").get_json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert status["status"] == "completed", status
    assert status["result"]["embedding_shape"] == [6, 4]
    assert "frame_indices" not in status["result"]
    assert status["saved_to"]
    status_response = client.get(f"/ai/jobs/{job_id}")
    assert status_response.headers["Cache-Control"] == "no-store, max-age=0"

    analyzed = client.post(f"/ai/jobs/{job_id}/analyze", json={
        "pca_components": 3,
        "n_clusters": 2,
        "random_state": 0,
    })
    assert analyzed.status_code == 200
    assert len(analyzed.get_json()["analysis"]["cluster_labels"]) == 6

    auto_analyzed = client.post(f"/ai/jobs/{job_id}/analyze", json={
        "pca_components": 0.95,
        "n_clusters": "auto",
        "max_clusters": 4,
        "random_state": 0,
    })
    assert auto_analyzed.status_code == 200
    auto_parameters = auto_analyzed.get_json()["analysis"]["parameters"]
    assert auto_parameters["pca_selection"] == "explained_variance"
    assert auto_parameters["cluster_selection"] == "silhouette"
    assert 1 <= auto_parameters["n_clusters"] <= 4

    temporal = client.post(f"/ai/jobs/{job_id}/embedding-analysis", json={
        "adapter": "rhaapsody-changepoint",
        "options": {
            "starting_period": 2,
            "cost_threshold": 10.0,
            "window_size": "full",
            "min_time_between_changepoints": 1,
        },
    })
    assert temporal.status_code == 200, temporal.get_json()
    temporal_analysis = temporal.get_json()["analysis"]
    assert temporal_analysis["task"] == "changepoint"
    assert temporal_analysis["similarity_matrix"]["shape"] == [6, 6]
    matrix_response = client.get(
        f"/ai/jobs/{job_id}/embedding-analysis/rhaapsody-changepoint/"
        "similarity-matrix?max_size=32"
    )
    assert matrix_response.status_code == 200
    assert matrix_response.mimetype == "application/octet-stream"
    assert matrix_response.headers["X-Rheed-Matrix-Size"] == "6"
    similarity = np.frombuffer(matrix_response.data, dtype="<f4").reshape(6, 6)
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert client.get(
        f"/ai/jobs/{job_id}/embedding-analysis/rhaapsody-changepoint/csv"
    ).status_code == 200
    assert client.get(f"/ai/jobs/{job_id}/csv").status_code == 200
    download = client.get(f"/ai/jobs/{job_id}/download")
    assert download.status_code == 200
    with np.load(BytesIO(download.data), allow_pickle=False) as archive:
        exported_analysis = json.loads(archive["analysis_json"].item())
    assert "embedding_analyses" in exported_analysis
    assert "rhaapsody-changepoint" in exported_analysis["embedding_analyses"]

    artifacts = client.get("/analysis/artifacts").get_json()["artifacts"]
    assert client.get("/analysis/artifacts").headers["Cache-Control"] == "no-store, max-age=0"
    inference_artifact = next(
        item for item in artifacts
        if item["kind"] == "ai-inference" and item["name"] == _spec().id
    )
    saved_path = analysis_store.dataset_dir / inference_artifact["path"]
    assert (saved_path / "analyses" / "pca-kmeans.json").is_file()
    assert (saved_path / "analyses" / "pca-kmeans.csv").is_file()
    assert (saved_path / "analyses" / "embedding-rhaapsody-changepoint.json").is_file()
    assert (saved_path / "analyses" / "embedding-rhaapsody-changepoint.csv").is_file()

    # Simulate restarting the web process: recall re-registers the disk-backed
    # inference job and exposes its saved downstream analyses without rerunning.
    with ai_jobs._lock:
        ai_jobs._jobs.pop(job_id)
    recalled = client.post(
        "/analysis/ai/recall",
        json={"artifact_id": inference_artifact["artifact_id"]},
    )
    assert recalled.status_code == 200, recalled.get_json()
    recall_data = recalled.get_json()
    assert recall_data["job_id"] == job_id
    assert recall_data["analysis_available"] is True
    assert "rhaapsody-changepoint" in recall_data["embedding_analyses"]
    assert client.get(f"/ai/jobs/{job_id}/analysis").status_code == 200
    assert client.get(
        f"/ai/jobs/{job_id}/embedding-analysis/rhaapsody-changepoint"
    ).status_code == 200


def test_flask_generic_regression_frame_and_csv_routes(analysis_test_dir):
    from rheed_webapp.app import analysis_store, app, session

    analysis_store.set_root(analysis_test_dir / "analysis")
    session._set_frames(_frames(3), np.asarray([0.0, 0.5, 1.0]))
    client = app.test_client()
    response = client.post("/ai/run", json={
        "model": _spec("dummy-regression", "regression").public_dict(),
        "batch_size": 2,
        "stride": 1,
    })
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    status = None
    for _ in range(100):
        status = client.get(f"/ai/jobs/{job_id}").get_json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert status["status"] == "completed", status
    frame = client.get(f"/ai/jobs/{job_id}/frame/1").get_json()
    assert "roughness_nm" in frame["values"]
    series = client.get(f"/ai/jobs/{job_id}/series")
    assert series.status_code == 200
    assert len(series.get_json()["frame_summaries"]) == 3
    csv_response = client.get(f"/ai/jobs/{job_id}/csv")
    assert csv_response.status_code == 200
    assert b"roughness_nm" in csv_response.data
    temporal = client.post(f"/ai/jobs/{job_id}/embedding-analysis", json={
        "adapter": "rhaapsody-changepoint",
        "options": {"starting_period": 2},
    })
    assert temporal.status_code == 400
    assert b"completed embedding result" in temporal.data
    session.rotate("cw")
    assert client.get(f"/ai/jobs/{job_id}/frame/1").status_code == 409


def test_flask_segmentation_track_id_frame_and_csv_routes(analysis_test_dir):
    from rheed_webapp.app import analysis_store, app, session

    analysis_store.set_root(analysis_test_dir / "analysis")
    session._set_frames(_frames(2), np.asarray([0.0, 0.5]))
    client = app.test_client()
    response = client.post("/ai/run", json={
        "model": _spec("dummy-segmentation", "segmentation").public_dict(),
        "batch_size": 2,
        "stride": 1,
    })
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    status = None
    for _ in range(100):
        status = client.get(f"/ai/jobs/{job_id}").get_json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert status["status"] == "completed", status

    frame = client.get(f"/ai/jobs/{job_id}/frame/0").get_json()
    assert frame["instances"][0]["track_id"] == 17
    csv_response = client.get(f"/ai/jobs/{job_id}/csv")
    assert csv_response.status_code == 200
    assert b"frame,t_s,track_id,class_id" in csv_response.data
    assert b"1,0.0,17,0,spot" in csv_response.data


def test_growth_page_contains_ai_controls():
    from rheed_webapp.app import app

    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"gr-ai-adapter" in response.data
    assert b"gr-ai-plotly" in response.data
    assert b"gr-ai-pca-mode" in response.data
    assert b"gr-ai-k-mode" in response.data
    assert b"Auto (silhouette)" in response.data
    assert b"gr-fps-slider" in response.data
    assert b"gr-ai-temporal-adapter" in response.data
    assert b"gr-ai-temporal-run" in response.data
    assert b"gr-ai-temporal-matrix" in response.data
    assert b"gr-ai-sim-plotly" in response.data
    assert b"gr-background-subtract" in response.data
    assert b"gr-ai-use-background" in response.data
    assert b"gr-roi-use-background" in response.data
    assert b"gr-background-settings" in response.data
    assert b"gr-background-tile" in response.data
    assert b"gr-background-percentile" in response.data
    assert b"gr-background-smooth" in response.data
    assert b"gr-background-sample" in response.data
    assert b"gr-analysis-root" in response.data
    assert b"Automatic local saving" in response.data

    page = response.get_data(as_text=True)
    assert '<div class="tab active" data-tab="growth">Analysis' in page
    assert 'id="gr-viewer-title"' in page
    assert '>Viewer</span>' in page
    assert ".sidebar {\n    width: 520px;" in page
    assert "#gr-viewer-center {\n    flex: 0 1 50vw;" in page
    assert "height: 50vh;" in page
    assert "window._focusRegister(element.id)" in page
    assert "let grAiOverlayJobId = null;" in page
    assert "fetch(`/ai/jobs/${overlayJobId}/frame/${frameIndex}`)" in page
    assert "grAiOverlayJobId = jobId;" in page
    assert "grAiRunPending = false;" in page
    assert "fetch(`/ai/jobs/${jobId}`, { cache: 'no-store' })" in page
    assert "frames (${progressPct}%)" in page
    assert "await grAiPoll(jobId);" in page
    assert "await grAnalysisRefreshStorage();" in page
    assert "fetch('/analysis/artifacts', { cache: 'no-store' })" in page
    assert "fetch('/analysis/storage'" in page
    assert "fetch('/analysis/record'" in page
    assert "function persistIntensityAnalysis()" in page
    assert "ema_intensities: emaSeries" in page
    assert "function capturePlotAxisRanges(plot)" in page
    assert "function restorePlotAxisRanges(layout, ranges)" in page
    assert "drawIntensityChart(state.currentFrame, true);" in page
    assert "grRenderChart(true);" in page
    assert "grRenderFwhm(true);" in page
    intensity_save = page[
        page.index("async function flushIntensityAnalysisSave()"):
        page.index("function persistIntensityAnalysis()")
    ]
    assert "window._grAnalysisRefreshStorage();" in intensity_save
    assert "\n    grAnalysisRefreshStorage();" not in intensity_save
    assert "window._grAnalysisRefreshStorage = grAnalysisRefreshStorage;" in page
    assert "function grMeasurementPersistenceSnapshot()" in page
    assert "function grRestoreStripRoi(parameters)" in page
    assert "const restoredRoi = grRestoreStripRoi(payload.parameters);" in page
    assert "Recalled the saved strip ROI and tracking settings" in page
    assert "selected_specular_fwhm_px_ema" in page
    assert "analysis_name: `${s.source || 'series'}-${s.label || i + 1}`" in page
    assert "fetch(`/frame_image/${idx}?${query}`)" in page
    assert "requestAnimationFrame(grPlaybackLoop)" in page
    assert "grPrefetchAhead();" in page
    assert "FRAME_CACHE_MAX_BYTES" in page
    assert "setInterval(grPlaybackTick" not in page
    assert "function suppressPlaybackShortcut(target)" in page
    assert "input, textarea, select, button, a, summary" in page
    assert "if (suppressPlaybackShortcut(e.target)) return;" in page
    assert "name: 'current frame'" in page
    assert "grAiSchedulePcaPlaybackMarker(idx)" in page
    assert "Plotly.restyle(plot" in page
    assert "grAiRenderSimilarityMatrix" in page
    assert "similarity-matrix?max_size=600" in page
    assert "colorscale: 'Viridis'" in page
    run_handler = page[page.index("grAiRun.addEventListener('click'"):
                       page.index("grAiCancel.addEventListener('click'")]
    assert "grAiPrepareRun();" in run_handler
    assert "grAiReset(" not in run_handler
    assert "color: labels.map(label => clusterColors.get(label))" in page
    assert "grAiK.min = silhouetteMode ? '1' : '2';" in page
    assert "zero-score K = 1 baseline" in page
    assert "instance.track_id == null" in page
    assert "` #${instance.track_id}`" in page
    viewer_start = page.index('id="tab-viewer"')
    growth_start = page.index('id="tab-growth"')
    ai_card = page.index('id="gr-ai-card"')
    post_embedding_heading = page.index('id="gr-ai-post-embedding-heading"')
    pca_controls = page.index('id="gr-ai-pca-mode"')
    temporal_controls = page.index('id="gr-ai-temporal-controls"')
    growth_canvas = page.index('id="gr-viewer-center"')
    growth_contrast = page.index('id="gr-viewer-contrast"')
    growth_colormap = page.index('id="gr-viewer-colormap"')
    growth_sidebar = page.index('id="gr-sidebar"')
    assert viewer_start < growth_start < ai_card < growth_canvas
    assert growth_sidebar < ai_card < growth_canvas < growth_colormap < growth_contrast
    assert 'id="gr-reset-clim"' not in page[growth_sidebar:growth_canvas]
    assert 'id="gr-cmap-btn"' not in page[growth_sidebar:growth_canvas]
    assert page.count('id="gr-viewer-display-controls"') == 1
    assert page.count('id="gr-viewer-colormap"') == 1
    assert page.count('id="gr-cmap-btn"') == 1
    assert page.count('id="gr-cmap-select"') == 1
    assert page.count('id="gr-viewer-contrast"') == 1
    assert page.count('id="gr-clim-lo"') == 1
    assert page.count('id="gr-clim-hi"') == 1
    assert page.count('id="gr-clim-auto-every"') == 1
    assert page.count('id="gr-reset-clim"') == 1
    assert 'class="growth-playback-layout"' in page
    assert 'class="growth-playback-options"' in page
    assert page.index('class="growth-playback-options"') < growth_colormap
    assert ".viewer-display-controls {\n    flex: 0 1 780px;" in page
    assert "bottom: calc(100% + 4px);" in page
    assert post_embedding_heading < pca_controls < temporal_controls
    assert page.count("Post-embedding analysis") == 1
    assert page.count('id="gr-ai-card"') == 1


def test_growth_page_uses_one_reference_configuration_for_streak_calibration():
    from rheed_webapp.app import app

    response = app.test_client().get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)

    substrate_card = page.index('id="gr-substrate-calibration-card"')
    streak_measurement = page.index('id="gr-streak-measurement"')
    frame_calibration = page.index('<h3>Frame Calibration</h3>')
    assert substrate_card < streak_measurement < frame_calibration
    assert page.count('id="gr-energy"') == 1
    assert page.count('id="gr-material"') == 1
    assert page.count('id="gr-zone"') == 1
    assert 'id="gr-ss-energy"' not in page
    assert 'id="gr-ss-material"' not in page
    assert 'id="gr-ss-zone"' not in page
    assert "const calib = activeSpacingCalibration();" in page
    assert "V_keV: grCalib.V_keV" in page
    assert "grInvalidateSubstrateCalibration();" in page


def test_binary_playback_frame_route_supports_fast_jpeg_and_lossless_png():
    from rheed_webapp.app import app, session

    session._set_frames(_frames(2), np.asarray([0.0, 0.25]))
    client = app.test_client()

    jpeg = client.get("/frame_image/1?format=jpeg&quality=85")
    assert jpeg.status_code == 200
    assert jpeg.mimetype == "image/jpeg"
    assert jpeg.data.startswith(b"\xff\xd8")
    assert float(jpeg.headers["X-Rheed-Timestamp"]) == 0.25
    assert jpeg.headers["Cache-Control"] == "no-store"

    png = client.get("/frame_image/0?format=png&auto=1")
    assert png.status_code == 200
    assert png.mimetype == "image/png"
    assert png.data.startswith(b"\x89PNG\r\n\x1a\n")
    assert "X-Rheed-Clim-Lo" in png.headers
    assert "X-Rheed-Clim-Hi" in png.headers

    invalid = client.get("/frame_image/0?format=webp")
    assert invalid.status_code == 400
