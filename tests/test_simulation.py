from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rheed_core.simulation import (
    SimulationAdapter,
    SimulationAdapterDescriptor,
    SimulationCancelled,
    SimulationContext,
    SimulationSpec,
    SimulationStore,
    list_simulation_adapters,
    measure_detector_distance,
    run_simulation,
)
from rheed_core.simulation import base as adapter_base
from rheed_core.simulation import torch_rheed as torch_rheed_adapter


class DummySimulationAdapter(SimulationAdapter):
    descriptor = SimulationAdapterDescriptor(
        name="dummy-simulation",
        description="Dependency-free simulation adapter",
        input_names=("structure",),
        capabilities=("rocking-curves", "detector-images"),
    )
    loaded = 0
    closed = 0

    def load(self, spec, device):
        type(self).loaded += 1
        return {"device": device, "revision": spec.revision}

    def simulate(self, inputs, context):
        assert inputs["structure"].is_file()
        return {
            "scan_coordinates": {
                "glancing_angle_deg": np.asarray([1.0, 1.5, 2.0]),
            },
            "beam_indices": [[0, 0], [1, 0]],
            "intensities": [[10.0, 2.0], [12.0, 3.0], [8.0, 4.0]],
            "detector_images": np.ones((3, 4, 5), dtype=np.float64),
            "metadata": {
                "method": "dummy",
                "beam_energy_kev": 15.8,
                "screen": {
                    "plane_mode": "vertical",
                    "screen_distance_mm": 100.0,
                    "screen_width_mm": 50.0,
                    "screen_height_mm": 40.0,
                    "pixels_x": 5,
                    "pixels_y": 4,
                    "reference_angle_deg": 3.5,
                },
            },
        }

    def close(self):
        type(self).closed += 1


class InvalidSimulationAdapter(DummySimulationAdapter):
    descriptor = SimulationAdapterDescriptor(
        name="invalid-simulation",
        description="Invalid test output",
        input_names=("structure",),
    )

    def simulate(self, inputs, context):
        return {
            "scan_coordinates": {"angle_deg": [1.0, 2.0]},
            "beam_indices": [[0, 0]],
            "intensities": [[1.0], [2.0], [3.0]],
        }


@pytest.fixture(autouse=True)
def dummy_adapters(monkeypatch):
    DummySimulationAdapter.loaded = 0
    DummySimulationAdapter.closed = 0
    InvalidSimulationAdapter.loaded = 0
    InvalidSimulationAdapter.closed = 0
    monkeypatch.setitem(
        adapter_base._BUILTINS,
        "dummy-simulation",
        "test_simulation:DummySimulationAdapter",
    )
    monkeypatch.setitem(
        adapter_base._BUILTINS,
        "invalid-simulation",
        "test_simulation:InvalidSimulationAdapter",
    )


def _spec(adapter="dummy-simulation"):
    return SimulationSpec.from_dict({
        "id": "test-simulation",
        "adapter": adapter,
        "revision": "abc123",
        "options": {"access_token": "private", "value": 2},
    })


def test_run_simulation_validates_arrays_hashes_inputs_and_exports_npz(analysis_test_dir):
    structure = analysis_test_dir / "structure.txt"
    structure.write_text("test structure\n", encoding="utf-8")
    progress = []

    result = run_simulation(
        {"structure": structure},
        _spec(),
        device="cpu",
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert result.n_samples == 3
    assert result.intensities.shape == (3, 2)
    assert result.intensities.dtype == np.float64
    assert result.beam_indices.tolist() == [[0, 0], [1, 0]]
    assert result.detector_images.shape == (3, 4, 5)
    assert result.detector_images.dtype == np.float32
    assert result.metadata["inputs"]["structure"]["sha256"] == hashlib.sha256(
        structure.read_bytes()
    ).hexdigest()
    assert result.simulation.public_dict()["options"]["access_token"] == "<redacted>"
    assert progress == [(0, 1), (1, 1)]
    assert DummySimulationAdapter.loaded == 1
    assert DummySimulationAdapter.closed == 1

    with np.load(BytesIO(result.to_npz_bytes()), allow_pickle=False) as archive:
        assert archive["scan_glancing_angle_deg"].tolist() == [1.0, 1.5, 2.0]
        assert archive["intensities"].shape == (3, 2)
        assert archive["detector_images"].shape == (3, 4, 5)


def test_run_simulation_rejects_input_and_output_misalignment_and_closes(analysis_test_dir):
    structure = analysis_test_dir / "structure.txt"
    structure.write_text("input", encoding="utf-8")

    with pytest.raises(ValueError, match="intensities must have shape"):
        run_simulation({"structure": structure}, _spec("invalid-simulation"))

    assert InvalidSimulationAdapter.loaded == 1
    assert InvalidSimulationAdapter.closed == 1


def test_run_simulation_cancellation_closes_without_loading(analysis_test_dir):
    structure = analysis_test_dir / "structure.txt"
    structure.write_text("input", encoding="utf-8")

    with pytest.raises(SimulationCancelled):
        run_simulation(
            {"structure": structure},
            _spec(),
            cancelled=lambda: True,
        )

    assert DummySimulationAdapter.loaded == 0
    assert DummySimulationAdapter.closed == 1


def test_adapter_discovery_exposes_inputs_capabilities_and_schema():
    descriptors = {
        descriptor["name"]: descriptor
        for descriptor in list_simulation_adapters()
    }

    assert descriptors["dummy-simulation"]["available"] is True
    assert descriptors["dummy-simulation"]["input_names"] == ["structure"]
    assert descriptors["dummy-simulation"]["capabilities"] == [
        "rocking-curves",
        "detector-images",
    ]
    assert descriptors["torch-rheed"]["option_schema"]["properties"]["solver"]["enum"] == [
        "sp6",
        "multislice",
    ]
    defaults = descriptors["torch-rheed"]["default_options"]
    assert defaults["integration_step_angstrom"] == 0.5
    assert defaults["rhst_threshold"] == 1000.0
    assert defaults["screen"] == {
        "plane_mode": "vertical",
        "screen_distance_mm": 100.0,
        "screen_width_mm": 120.0,
        "screen_height_mm": 90.0,
        "pixels_x": 320,
        "pixels_y": 240,
        "correlation_length_angstrom": 1000.0,
        "rod_profile": "lorentzian",
        "beam_intensity_floor": 0.0,
        "instrument_broadening_fwhm_mm": 0.0,
        "source_glancing_divergence_fwhm_deg": 0.0,
        "source_divergence_samples": 9,
        "reference_angle_deg": None,
        "frame_azimuth_deg": None,
        "display_scale": "sqrt",
        "colormap": "inferno",
    }


def test_detector_measurement_reports_mm_and_exact_reciprocal_distance():
    measured = measure_detector_distance(
        {"x": 50.0, "y": 50.0},
        {"x": 60.0, "y": 50.0},
        image_shape=(100, 100),
        screen={
            "plane_mode": "vertical",
            "screen_distance_mm": 100.0,
            "screen_width_mm": 100.0,
            "screen_height_mm": 100.0,
        },
        beam_energy_kev=15.8,
    )

    assert measured["distance_mm"] == pytest.approx(10.0)
    assert measured["delta_mm"]["x"] == pytest.approx(10.0)
    assert measured["delta_mm"]["y"] == pytest.approx(0.0)
    assert measured["reciprocal_distance_A_inv"] > 0
    assert measured["real_space_period_A"] == pytest.approx(
        2 * np.pi / measured["reciprocal_distance_A_inv"]
    )
    assert measured["beam_energy_kev"] == 15.8


@dataclass(frozen=True)
class _FakeScreenConfig:
    pixels_x: int = 320
    pixels_y: int = 240


def test_torch_rheed_adapter_translates_public_result_without_real_dependency(monkeypatch):
    calls = []
    bulk_input = SimpleNamespace(
        be=15.8,
        azi_deg=15.0,
        daz_deg=1.0,
        gi_deg=0.5,
        dg_deg=0.25,
        rdom_deg=[2.0],
    )
    fake_package = SimpleNamespace(
        __file__=__file__,
        ScreenImageConfig=_FakeScreenConfig,
        load_bulk_input=lambda path: bulk_input,
    )

    def simulate_from_files(bulk, surface, **kwargs):
        calls.append((bulk, surface, kwargs))
        return SimpleNamespace(
            beam_indices=[(0, 0), (1, 0)],
            intensities=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            screen_images=np.ones((3, 2, 4)),
            screen_config=kwargs["screen_config"],
            naz=1,
            ng=3,
        )

    fake_package.simulate_from_files = simulate_from_files
    class FakeTorch:
        __version__ = "test"
        float32 = "float32"
        float64 = "float64"
        cuda = SimpleNamespace(is_available=lambda: False)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

        def __init__(self):
            self.default_dtype = self.float32

        def get_default_dtype(self):
            return self.default_dtype

        def set_default_dtype(self, value):
            self.default_dtype = value

    fake_torch = FakeTorch()
    monkeypatch.setattr(
        torch_rheed_adapter,
        "import_module",
        lambda name: {"torch_rheed": fake_package, "torch": fake_torch}[name],
    )
    monkeypatch.setattr(
        torch_rheed_adapter.importlib_metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(torch_rheed_adapter.importlib_metadata.PackageNotFoundError),
    )

    adapter = torch_rheed_adapter.TorchRheedAdapter()
    spec = SimulationSpec(
        id="torch-test",
        adapter="torch-rheed",
        options={
            "solver": "multislice",
            "render_detector": True,
            "screen": {"pixels_x": 64, "pixels_y": 48},
        },
    )
    runtime = adapter.load(spec, "auto")
    result = adapter.simulate(
        {"bulk": Path("bulk.txt"), "surface": Path("surf.txt")},
        SimulationContext(spec=spec, device="auto"),
    )

    assert runtime["device"] == "cpu"
    assert result["scan_coordinates"]["azimuth_deg"].tolist() == [17.0, 17.0, 17.0]
    assert result["scan_coordinates"]["glancing_angle_deg"].tolist() == [0.5, 0.75, 1.0]
    assert result["intensities"].shape == (3, 2)
    assert result["detector_images"].shape == (3, 2, 4)
    assert calls[0][2]["device"] == "cpu"
    assert calls[0][2]["solver"] == "multislice"
    assert calls[0][2]["integration_step"] is None
    assert calls[0][2]["screen_config"] == _FakeScreenConfig(64, 48)
    assert fake_torch.get_default_dtype() == fake_torch.float32


def test_torch_rheed_adapter_recovers_energy_from_archived_bulk_input(analysis_test_dir):
    bulk = analysis_test_dir / "bulk.txt"
    bulk.write_text(
        "19 ,NB\n15.8,45,45,0,0.1,7,0.01 ,BE,AZI,AZF,DAZ,GI,GF,DG\n",
        encoding="utf-8",
    )

    assert torch_rheed_adapter.TorchRheedAdapter().recover_result_metadata(
        {"bulk": bulk}
    ) == {"beam_energy_kev": 15.8}


def test_torch_rheed_scan_coordinates_preserve_two_axis_flattening_order():
    bulk_input = SimpleNamespace(
        azi_deg=10.0,
        daz_deg=5.0,
        gi_deg=1.0,
        dg_deg=0.5,
        rdom_deg=[2.0],
    )

    coordinates = torch_rheed_adapter._scan_coordinates(bulk_input, naz=3, ng=2)

    assert coordinates["azimuth_deg"].tolist() == [12.0, 17.0, 22.0, 12.0, 17.0, 22.0]
    assert coordinates["glancing_angle_deg"].tolist() == [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]


def test_simulation_store_persists_inputs_arrays_and_random_access_frames(analysis_test_dir):
    store = SimulationStore(analysis_test_dir / "simulations")
    stored, inputs = store.create_run(
        "stored-test",
        _spec(),
        {"structure": ("structure.txt", BytesIO(b"stored input\n"))},
    )
    result = run_simulation(inputs, _spec())
    store.save_result("stored-test", result)

    loaded = stored.load_result(mmap_mode="r")
    assert loaded.intensities.shape == (3, 2)
    assert loaded.detector_images.shape == (3, 4, 5)
    assert stored.detector_frame(2).shape == (4, 5)
    assert store.list_runs()[0]["status"] == "completed"
    assert store.list_runs()[0]["result"]["detector_images_shape"] == [3, 4, 5]


def test_flask_simulation_routes_and_tab(analysis_test_dir):
    import time

    from rheed_webapp import app as webapp

    webapp.simulation_store.set_root(analysis_test_dir / "web-simulations")
    client = webapp.app.test_client()
    adapters = client.get("/simulation/adapters")
    assert adapters.status_code == 200
    assert any(item["name"] == "dummy-simulation" for item in adapters.get_json()["adapters"])

    response = client.post(
        "/simulation/run",
        data={
            "simulation": json.dumps(_spec().public_dict()),
            "device": "cpu",
            "structure": (BytesIO(b"web input\n"), "structure.txt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202, response.get_data(as_text=True)
    run_id = response.get_json()["run_id"]
    status = None
    # Background scientific work in another process can briefly contend for CPU
    # on developer workstations; allow the dependency-free job a few seconds.
    for _ in range(500):
        status = client.get(f"/simulation/jobs/{run_id}").get_json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert status["status"] == "completed", status
    auto_gif = analysis_test_dir / "web-simulations" / run_id / "exports" / "screen_animation.gif"
    assert auto_gif.read_bytes().startswith((b"GIF87a", b"GIF89a"))

    result = client.get(f"/simulation/jobs/{run_id}/result")
    assert result.status_code == 200
    assert result.get_json()["detector_image_shape"] == [3, 4, 5]
    frame = client.get(f"/simulation/jobs/{run_id}/frame/1?scale=log&cmap=viridis")
    assert frame.status_code == 200
    assert frame.mimetype == "image/png"
    assert frame.data.startswith(b"\x89PNG")
    preview = client.get(
        f"/simulation/jobs/{run_id}/frame/1?scale=log&cmap=viridis&format=jpeg&quality=82"
    )
    assert preview.status_code == 200
    assert preview.mimetype == "image/jpeg"
    assert preview.data.startswith(b"\xff\xd8")
    measurement = client.put(
        f"/simulation/jobs/{run_id}/measurements",
        json={"measurements": [{
            "label": "M1",
            "frame_index": 1,
            "point1": {"x": 1.0, "y": 2.0},
            "point2": {"x": 4.0, "y": 2.0},
        }]},
    )
    assert measurement.status_code == 200, measurement.get_data(as_text=True)
    measured = measurement.get_json()["measurements"][0]["physical"]
    assert measured["distance_mm"] == pytest.approx(30.0)
    assert measured["reciprocal_distance_A_inv"] > 0
    recalled = client.get(f"/simulation/jobs/{run_id}/measurements")
    assert recalled.get_json()["measurements"][0]["label"] == "M1"
    image = client.post(
        f"/simulation/jobs/{run_id}/image/1",
        data={"image": (BytesIO(frame.data), "screen.png")},
        content_type="multipart/form-data",
    )
    assert image.status_code == 200, image.get_data(as_text=True)
    assert image.data.startswith(b"\x89PNG")
    gif = client.get(f"/simulation/jobs/{run_id}/gif?fps=5&scale=sqrt&cmap=inferno")
    assert gif.status_code == 200, gif.get_data(as_text=True)
    assert gif.data.startswith((b"GIF87a", b"GIF89a"))
    assert client.get(f"/simulation/jobs/{run_id}/csv").status_code == 200
    assert client.get(f"/simulation/jobs/{run_id}/download").status_code == 200
    assert client.get("/simulation/runs").get_json()["runs"][0]["run_id"] == run_id

    page = client.get("/").get_data(as_text=True)
    assert 'data-tab="simulation"' in page
    assert 'id="tab-simulation"' in page
    assert 'id="sim-adapter"' in page
    assert 'id="sim-btn-play"' in page
    assert 'id="sim-fps-slider"' in page
    assert 'id="sim-detector-canvas"' in page
    assert 'id="sim-save-gif"' in page
    assert 'id="sim-mode-measure"' in page
    assert 'id="sim-plot"' in page
    assert "detectorPrefetchActive < 3" in page
    assert "displayGeneration !== detectorDisplayGeneration" in page
    assert "const format = 'png';" in page
