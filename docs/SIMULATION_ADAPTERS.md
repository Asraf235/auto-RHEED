# Local simulation adapters

Auto RHEED exposes simulation packages through a framework-neutral adapter
contract. Simulators remain optional and local: `rheed_core` does not import a
simulation runtime until a selected adapter is run.

Simulation runs are independent of the experimental frame stack. The web
application's Simulation tab uses the same process, but running or recalling a
simulation does not load, rotate, or clear the current `RheedSession`.

## Normalized result contract

An adapter returns a dictionary containing:

- `scan_coordinates`: one or more named one-dimensional arrays with equal
  length, such as `azimuth_deg` and `glancing_angle_deg`.
- `beam_indices`: optional integer array shaped `(B, 2)`.
- `intensities`: optional numeric array shaped `(N, B)`.
- `detector_images`: optional numeric array shaped `(N, H, W)`.
- `metadata`: optional JSON-safe scientific metadata.

At least one of `intensities` or `detector_images` is required. The core runner
validates alignment, records input paths, sizes, and SHA-256 hashes, captures
runtime provenance, and closes the adapter after success, failure, or
cancellation.

## torch-rheed adapter

`torch-rheed` is Auto RHEED's recommended dynamical RHEED simulation backend.
The built-in bridge uses the public Python API from
[sumner-harris/torch-rheed](https://github.com/sumner-harris/torch-rheed). It
accepts two input files named `bulk` and `surface`, then normalizes the returned
rocking curve and optional detector stack. The package performs dynamical bulk
and surface diffraction calculations locally in PyTorch and currently offers
SP6 and multislice surface solvers for its supported electron/RHEED,
single-domain, `p1` subset.

The package is not part of Auto RHEED's default installation. Install the
pinned GitHub revision with:

```powershell
uv sync --extra torch-rheed
```

## Using the Simulation tab

After installing the optional dependency, restart Auto RHEED and open the
**Simulation** tab. Select an available adapter, provide the input files it
requests, adjust its exposed options, and start the run. Controls are generated
from the adapter descriptor instead of being hard-coded for `torch-rheed`, so
other simulation packages can expose different inputs and options through the
same page.

Completed runs are file-backed below `data/simulations/` by default and can be
recalled without rerunning the solver. Rocking curves are plotted in the tab;
detector frames, when available, are loaded individually on demand. The detector
viewer supports playback, FPS control, fit-relative zoom, resizing, and screen
measurements in millimetres and reciprocal ångströms using the run's recorded
screen geometry and beam energy. The annotated current frame can be saved as a
PNG. A detector-stack GIF is generated automatically when the simulation
completes; the GIF button regenerates and downloads it with the viewer's current
FPS and display settings. Normalized numeric results remain available as CSV
curves or an NPZ bundle.

For local development of both repositories, install a checkout into Auto
RHEED's existing environment instead:

```powershell
uv pip install --python .venv\Scripts\python.exe -e C:\path\to\torch-rheed
```

The current package and optional extra require Python 3.13. Updating the pinned
package revision does not require changing the adapter contract unless its
public result API changes.

Example request:

```python
from rheed_core.simulation import SimulationSpec, run_simulation

result = run_simulation(
    {"bulk": "bulk.txt", "surface": "surf.txt"},
    SimulationSpec.from_dict({
        "id": "sto-rocking-curve",
        "adapter": "torch-rheed",
        "options": {
            "solver": "sp6",
            "integration_step_angstrom": 0.2,
            "rhst_threshold": 1000.0,
            "render_detector": True,
            "screen": {
                "pixels_x": 512,
                "pixels_y": 384
            }
        }
    }),
    device="cpu",
)
```

The adapter records the installed `torch-rheed` and PyTorch versions, resolved
device, package path, installation source metadata, requested revision, solver
settings, detector configuration, and scan dimensions.

## External adapters

An external distribution subclasses `SimulationAdapter` and registers its class
through the `auto_rheed.simulation_adapters` entry-point group:

```toml
[project.entry-points."auto_rheed.simulation_adapters"]
my-simulator = "my_simulator.auto_rheed:MySimulationAdapter"
```

The descriptor supplies required input names, capabilities, default options,
and an optional JSON schema that consumers can use to construct controls. The
adapter owns all simulator-specific imports and converts its result at the
boundary; Flask, MCP, and `RheedSession` should not contain simulator-specific
conditionals.
