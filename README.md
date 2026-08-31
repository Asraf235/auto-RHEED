# Auto RHEED

A browser-based analysis suite for **Reflection High-Energy Electron Diffraction (RHEED)** data, built for thin-film growth researchers (MBE / PLD). Auto RHEED turns raw growth videos into quantitative, time-resolved physics — pixel calibration, angle of incidence α, in-plane lattice parameter *d*(t), streak coherence length, and diffracted-intensity growth rates — through an interactive interface and a matching programmatic API.

Developed at Oak Ridge National Laboratory (ORNL).

---

## Features

- **Multi-format loading** — HDF5 (`.h5`), video (`.mp4/.avi/.mov/.mkv`), NumPy (`.npy`), single images, or a folder of per-frame images stacked into one video; raw `.img` / TIFF detector frames supported.
- **Calibration** — 5-point screen calibration (pixel scale + α) and a unified substrate/streak reference via `d = λL/Δx` (relativistic electron wavelength), with reference *d* from a built-in table, the Materials Project (your own free API key), or a custom value. Manual and automatic strip measurements use the same energy, material, and zone-axis settings.
- **Peak tracking & α(t)** — track specular / direct-beam positions frame-by-frame to follow angle-of-incidence evolution, with an optional intensity-vs-α view.
- **Lattice parameter d(t)** — automatic strip-profile tracking (ALS / rolling-ball / polynomial / moving-min background subtraction) with sub-pixel peak detection.
- **Streak FWHM & coherence length** — specular and nearest ±1 first-order streak widths (Gaussian fit or half-max), with specular width converted to in-plane coherence length, evolving per frame.
- **Intensity vs time & growth rate** — per-ROI intensity oscillations plus background-subtracted specular/first-order strip-fit peak intensities (stacked, EMA-smoothed), and per-series FFT when the trace is complete.
- **Local AI inference** — replaceable model adapters for frame embeddings, PCA/K-means clustering, RHAAPSODY changepoint detection with an interactive similarity-matrix plot, segmentation/tracking overlays, and future regression/classification models; frames and results remain on the local machine.
- **Dynamical RHEED simulation** — the recommended `torch-rheed` adapter runs local PyTorch dynamical diffraction calculations from conventional `bulk.txt` and `surf.txt` inputs, with SP6 or multislice surface solvers and CPU/CUDA support. The dedicated Simulation tab plays, zooms, resizes, and measures synthetic detector stacks in millimetres and reciprocal-space units, recalls file-backed runs, automatically saves a GIF, and exports annotated PNG or normalized CSV/NPZ results. Other simulators can be installed through the same adapter contract.
- **Automatic result storage** — successful classical and AI analyses are written to a configurable local run folder instead of being retained only in browser/server memory.
- **Library** — a persistent reference gallery of RHEED patterns by substrate.
- **Publication-ready export** — every chart exports to CSV and to a transparent PNG with explicit physical figure size, font size, and embedded DPI; figures and high-resolution annotated Viewer images are saved beside the active dataset analysis.
- **Colormaps** — perceptually-uniform Viridis/Plasma/Inferno/Magma and colorblind-safe Cividis, among others.

---

## How it works

### Structural monitoring — lattice parameter, streak width & coherence length

![Auto RHEED analysis pipeline: frame viewer, streak profile, d(t), and FWHM/coherence panels](docs/analysis-pipeline.png)

The analysis proceeds in four stages: **(a)** raw-data ingestion, **(b)** a synchronized
frame viewer and playback timeline, **(c–d)** a single geometric calibration, and per-frame
structural measurement. On the first bare-substrate frame the user marks a known substrate
reflection; the measured streak separation Δx fixes the pixel-to-reciprocal-space scale
through the small-angle RHEED relation `d = λL/Δx` (here SrTiO₃ along [110]). A strip ROI
spanning the specular spot and the two first-order streaks is summed column-wise into a 1-D
profile **(d)**; a slowly varying background is removed (asymmetric least squares shown), and
the specular and ±1 peaks are detected and refined to sub-pixel precision. Averaging the two
specular-to-first-order distances gives Δx per frame, converted to the in-plane lattice
parameter **d(t)** **(e)** — resolving monolayer-periodic oscillations on a slow relaxation
toward the substrate value. From the same profiles, the specular and nearest ±1 first-order
streak **FWHM** values are tracked every frame **(f)**; the plot uses the available-side mean
for the first-order trace while saved data retain each side. The specular width also yields
the derived in-plane **coherence length L**, reporting the periodic sharpening and broadening
of the diffraction features as each layer nucleates and coalesces.

### Kinetic monitoring — intensity oscillations & growth rate

![Auto RHEED growth-rate analysis: two ROIs, intensity-vs-time transients, and FFT spectrum](docs/growth-rate-fft.png)

The diffracted intensity itself carries the most direct signature of layer-by-layer growth.
**(a)** Rectangular ROIs are placed on distinct diffraction features (specular spot and an
adjacent streak) and integrated frame-by-frame. **(b)** The resulting intensity transients are
plotted as time-synchronized panels — both showing pronounced periodic oscillations on a slow
decay — since one full oscillation corresponds to the completion of a single atomic layer.
**(c)** A discrete Fourier transform of each transient (after an EMA detrend) yields a sharp
fundamental at `f₀` that is directly the growth rate (ML/s); combined with the out-of-plane
interlayer spacing from the substrate calibration it gives an absolute deposition rate (Å/s).
A separate FFT window can be opened per ROI to compare rates from different features.

*(These correspond to Figures 3 and 4 of the associated manuscript.)*

### Dynamical simulation with torch-rheed

[`torch-rheed`](https://github.com/sumner-harris/torch-rheed) is the recommended
dynamical simulation backend for Auto RHEED. It is a Python-native PyTorch
implementation of the RHEED forward calculation described by
sim-trhepd-rheed. Auto RHEED supplies the adapter and Simulation-tab workflow;
the optional package supplies the dynamical bulk and surface diffraction
calculation. It currently supports the electron/RHEED, single-domain, `p1`
subset with SP6 and multislice surface solvers.

Each run remains local and file-backed. The tab plots rocking curves and shows
the optional synthetic detector stack with low-latency lossless playback, FPS
control, zooming, resizing, and persistent line measurements in millimetres,
Å⁻¹, and the corresponding `2π/|Δq|` real-space period. It automatically saves
the detector stack as a GIF; the current annotated detector frame can also be
saved as PNG. The simulation inputs, settings, package/runtime provenance,
normalized arrays, measurements, and exports remain together under the run's
`data/simulations/` folder.

A ready-to-run SrTiO₃ [100] input pair is included under
[`examples/simulations/srtio3-100`](examples/simulations/srtio3-100).
Additional example systems can be added beside it without placing generated
simulation results under version control.

---

## Installation

Requires **Python ≥ 3.10** and [uv](https://docs.astral.sh/uv/). From the project root:

```bash
uv sync --frozen
```

This creates the local `.venv` and installs the exact dependency versions recorded in
`uv.lock`. After intentionally changing dependencies in `pyproject.toml`, run `uv lock`
followed by `uv sync` and commit the updated lockfile.

AI runtimes are optional so the classical installation stays lightweight:

```bash
uv sync --extra dinov3  # local DINOv3 embeddings + PCA/K-means
uv sync --extra yolo    # local YOLO-compatible instance segmentation
uv sync --extra ai-all  # both adapters
```

For dynamical RHEED simulation, the recommended installation is the optional
`torch-rheed` extra. It requires Python 3.13 or newer and installs the pinned,
tested GitHub revision without adding PyTorch to analysis-only installations:

```bash
uv sync --extra torch-rheed
```

In the app, open **Analysis**, load a dataset, and use the **Local AI
Inference** card directly below **Growth Video** in the left sidebar.

See [Local AI model inference](docs/AI_MODELS.md) for model manifests, offline
operation, output formats, and writing adapters for new model families.

Local scientific simulators use a separate, framework-neutral adapter contract.
See [Local simulation adapters](docs/SIMULATION_ADAPTERS.md) for the normalized
result schema, the Simulation tab, the recommended `torch-rheed` dynamical
backend, local editable installation, and instructions for additional adapters.

---

## Running the app

```bash
uv run --frozen --no-sync python -m rheed_webapp.app
```

`--no-sync` preserves whichever optional model adapters you selected during
installation. Then open **http://localhost:5000** (set `PORT` to use a different
port). The frontend is a single self-contained page; just refresh the browser to
pick up template changes.

### Desktop launchers

All launchers run the same frozen UV environment without synchronizing away
optional AI dependencies. They open the browser automatically and keep a terminal
window visible; closing that terminal stops the local server. Desktop shortcuts
store the repository's current absolute path, so rerun the shortcut creator after
moving the repository.

**Windows:** Double-click `Launch Auto RHEED.bat`. Run `Create Desktop
Shortcut.bat` once to create an **Auto RHEED** Desktop shortcut.

**macOS:** On first use, make the scripts executable and create the Desktop app:

```bash
chmod +x "Launch Auto RHEED.sh" "Launch Auto RHEED.command" \
  "Create macOS Desktop Shortcut.command"
./"Create macOS Desktop Shortcut.command"
```

You can then open **Auto RHEED.app** from the Desktop, or double-click `Launch
Auto RHEED.command` directly. If macOS blocks the first launch, Control-click the
app and choose **Open**.

**Ubuntu:** On first use, make the scripts executable and create the Desktop
launcher:

```bash
chmod +x "Launch Auto RHEED.sh" "Create Ubuntu Desktop Shortcut.sh"
./"Create Ubuntu Desktop Shortcut.sh"
```

Double-click **Auto RHEED** on the Desktop. Some GNOME versions require one
right-click → **Allow Launching** before the first use. The shared `Launch Auto
RHEED.sh` script also works directly on other Linux desktops that provide
`xdg-open` or `gio`.

### Automatic analysis storage

Every successful analysis is saved automatically. By default Auto RHEED creates
timestamped dataset runs under `data/runs/` in the repository. In **Analysis →
Analysis Storage**, enter another absolute local folder and click
**Use This Folder** to change the destination. The selection is remembered in the
local, untracked `.auto_rheed_settings.json` file. The MCP server uses the same
setting; `AUTO_RHEED_DATA_DIR` can override it for an MCP process.

Saved plots and images use that same active run. Every analysis plot is
automatically rendered to a stable, replaceable PNG after it is created or its
interactive settings change. Automatic figures are **3.5 × 3.5 inches**, use
**10-point fonts**, are rendered at **600 DPI**, and have transparent backgrounds.
The manual PNG dialog can override those settings and optionally download a copy.
PNGs are written to the run's `figures/` folder. Viewer and calibration images,
individual rendered frames, and the all-frame ZIP also default to `figures/`.

Reopening the exact same source file reconnects to its newest matching run by a
content fingerprint. Use **Saved Results for this dataset** to activate an older
matching run and recall a classical result or AI inference run. AI recall also
restores any saved PCA/K-means and RHAAPSODY changepoint/similarity analysis
attached to that inference run; it does not rerun the model. A rotation starts a
new run because it changes native pixel coordinates.

Each run directory contains:

```text
manifest.json                 dataset identity, shape, timing, and version
artifacts.jsonl               lightweight index of saved results
classical/<analysis>/*.json    complete result and metadata
classical/<analysis>/*.csv     readable tabular columns when available
ai/<job>/{manifest.json,*.npy,frame_results.jsonl}
ai/<job>/analyses/*.json       PCA/K-means and temporal metadata/results
ai/<job>/analyses/*.csv        readable tabular post-analysis columns
ai/<job>/analyses/*.npy        large matrices such as similarity data
figures/*.png                   saved plots, Viewer frames, calibration images
figures/rheed_frames.zip        optional rendered all-frame archive
```

Classical files include intensity traces, calibration, peak and strip tracking,
FFT, material lookup, and manual growth-spacing/FWHM measurements. AI embeddings
are stored as numeric arrays; segmentation/detection records are stored as an
indexed JSONL stream. The viewer reads individual AI frames from disk on demand,
and loads a complete time series only when its plot is opened. The original video
is not duplicated into the results folder. Live strip-profile previews are not
saved; running strip tracking creates the durable result. Older `.json.gz`
artifacts remain readable but are not rewritten or deleted. `data/` and the local
settings file are ignored by Git.

Interactive classical plots use stable artifact names rather than timestamped
revisions. For example, `classical/intensity/current.json` and its CSV companion
contain the latest raw series, EMA series and alpha, visibility choices, and axis
mode. Moving an EMA slider or changing a plot option replaces that artifact
atomically. Lattice-spacing/FWHM, calibration, peak tracking, strip tracking, and
material lookup follow the same latest-version rule; each FFT source series has
one stable artifact that is replaced when its detrend or growth settings change.
Existing historical timestamped artifacts are preserved.

### Desktop shortcut (Windows)

For a one-click launch, either:

- Double-click **`Launch Auto RHEED.bat`** (starts the server and opens your browser), or
- Run **`Create Desktop Shortcut.bat`** once to add an *Auto RHEED* icon to your Desktop, or
- Open the app and use **About → Create Desktop Shortcut**.

---

## Project structure

```
src/
  rheed_core/      # All RHEED physics; no Flask. Owns the dataset + math.
    analysis_store.py   # file-backed analysis runs and random-access AI results
    inference/          # stable local-model contracts, adapters, PCA/K-means
    simulation/         # stable simulator contracts, discovery, and adapters
    session.py         # frame render, ROI intensity, calibration, tracking, strip_track
    streak_profile.py  # strip-sum profiles, background subtraction, peaks, FWHM
    spectra.py         # intensity FFT -> growth rate
    library.py         # RHEED Library storage
    materials.py       # substrate lattice constants + zone-axis geometry
    calibration.py, wavelength.py, constants.py, io/
  rheed_webapp/    # Thin Flask routes + the single-page frontend (templates/index.html)
  rheed_mcp/       # MCP server exposing the core to AI agents
```

`rheed_core` never imports Flask — the web app and the MCP server are independent
consumers of the same analysis core, which makes every measurement scriptable and
usable inside automated / agentic growth workflows.

---

## MCP server

An [MCP](https://modelcontextprotocol.io) server (`src/rheed_mcp/server.py`) exposes
the analysis core to AI agents so the same routines available in the GUI can be
driven programmatically.

---

## Data availability

Example RHEED datasets are large and are **not** included in this repository; they
are archived separately (see the dataset DOI in the associated publication).

## Citation

If you use Auto RHEED in your research, please cite the associated paper
*(citation / DOI to be added on publication)*.

## License

*(To be added — pending ORNL/DOE open-source approval.)*

Bundled third-party components retain their own licenses and attribution; see
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/).

## Acknowledgments

This work was supported by the Center for Nanophase Materials Sciences (CNMS), a US
Department of Energy Office of Science User Facility at Oak Ridge National Laboratory.
