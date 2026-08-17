# Auto RHEED

A browser-based analysis suite for **Reflection High-Energy Electron Diffraction (RHEED)** data, built for thin-film growth researchers (MBE / PLD). Auto RHEED turns raw growth videos into quantitative, time-resolved physics — pixel calibration, angle of incidence α, in-plane lattice parameter *d*(t), streak coherence length, and diffracted-intensity growth rates — through an interactive interface and a matching programmatic API.

Developed at Oak Ridge National Laboratory (ORNL).

---

## Features

- **Multi-format loading** — HDF5 (`.h5`), video (`.mp4/.avi/.mov/.mkv`), NumPy (`.npy`), single images, or a folder of per-frame images stacked into one video; raw `.img` / TIFF detector frames supported.
- **Calibration** — 5-point screen calibration (pixel scale + α) and substrate calibration via `d = λL/Δx` (relativistic electron wavelength), with reference *d* from a built-in table, the Materials Project (your own free API key), or a custom value.
- **Peak tracking & α(t)** — track specular / direct-beam positions frame-by-frame to follow angle-of-incidence evolution, with an optional intensity-vs-α view.
- **Lattice parameter d(t)** — automatic strip-profile tracking (ALS / rolling-ball / polynomial / moving-min background subtraction) with sub-pixel peak detection.
- **Streak FWHM & coherence length** — specular-streak width (Gaussian fit or half-max) converted to in-plane coherence length, evolving per frame.
- **Intensity vs time & growth rate** — per-ROI intensity oscillations (stacked, EMA-smoothed) and per-ROI FFT giving growth rate (ML/s), period, and deposition rate (Å/s).
- **Library** — a persistent reference gallery of RHEED patterns by substrate.
- **Publication-ready export** — every chart exports to CSV and to PNG with selectable font size and **600 DPI** (embedded); the frame viewer exports high-resolution annotated images.
- **Colormaps** — perceptually-uniform Viridis/Plasma/Inferno/Magma and colorblind-safe Cividis, among others.

---

## Installation

Requires **Python ≥ 3.10**. From the project root:

```bash
pip install -e .
```

This installs the runtime dependencies (NumPy, SciPy, OpenCV, h5py, Flask, mcp).

---

## Running the app

```bash
python -m rheed_webapp.app
```

Then open **http://localhost:5000** (set `PORT` to use a different port). The frontend is a single self-contained page; just refresh the browser to pick up template changes.

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

## Acknowledgments

This work was supported by the Center for Nanophase Materials Sciences (CNMS), a US
Department of Energy Office of Science User Facility at Oak Ridge National Laboratory.
