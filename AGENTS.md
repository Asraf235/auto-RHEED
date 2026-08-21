# Auto RHEED contributor guidance

## Preserve the package's shape

- Keep `rheed_core` independent of Flask, MCP, and browser concerns. Scientific calculations, loaders, and dataset operations belong there.
- Keep `rheed_webapp` and `rheed_mcp` as separate consumers. Each process owns its own `RheedSession`; do not imply that the GUI and MCP server share a loaded dataset.
- Preserve the local-first, single-user workflow unless a requested feature explicitly changes it. The web app intentionally uses one in-memory dataset per process and a persistent local image library.
- Successful analysis outputs are file-backed through `rheed_core.analysis_store.AnalysisStore`. Keep the default at the repository's ignored `data/` folder, preserve configurable roots, and do not regress completed AI jobs to holding full embeddings or per-frame results in the job manager.
- Prefer focused additions over introducing a frontend framework or build pipeline. The current UI is a self-contained Flask template using vanilla JavaScript, Canvas, and Plotly.

## AI model adapters

- Before adding or changing an adapter, read `docs/AI_MODELS.md` and the contracts in `src/rheed_core/inference/`. The repository uses the spelling "adapter" in code and configuration.
- Treat model families as optional adapters behind the stable core contract. Do not hard-code checkpoint IDs, weight paths, framework objects, or model-specific conditionals into `RheedSession`, Flask routes, MCP tools, or the browser UI.
- Keep heavyweight frameworks optional and lazily imported inside an adapter's `load()` method. Importing `rheed_core` must not require PyTorch, Transformers, Ultralytics, or another model runtime.
- Select concrete models with a `ModelSpec` manifest. Preserve the manifest's model source, revision, task, preprocessing, and options as run provenance; never commit weights, access tokens, or credentials.
- Built-in adapters live in `src/rheed_core/inference/`. External adapters should register through the `auto_rheed.inference_adapters` Python entry-point group rather than modifying core discovery code.
- Analyses that require a completed embedding sequence are not `ModelAdapter`s. Implement `EmbeddingAnalysisAdapter` and register external analyzers through `auto_rheed.embedding_analysis_adapters`. Keep their results aligned to the embedding run's native frame indices and timestamps.

A per-frame model adapter must subclass `ModelAdapter` and provide:

- `descriptor`: an `AdapterDescriptor` with a stable name, task, description, optional requirements, and default options.
- `load(spec, device)`: initialize the runtime and return JSON-safe model/provenance metadata. Make downloads explicit through the model source; inference remains local after artifacts are available.
- `infer_batch(frames, context)`: accept native grayscale frames shaped `(B, H, W)` and return the task contract below.
- `effective_batch_size(requested)`: optional; override only when the adapter must reduce runner batches for sequential state, per-frame progress, or responsive cancellation. The default preserves the requested size.
- `close()`: release resources when needed; it must be safe to call during cleanup after a failed or cancelled run.

A post-embedding analyzer must subclass `EmbeddingAnalysisAdapter`, provide an
`EmbeddingAnalysisDescriptor`, and implement `analyze(embeddings, frame_indices,
timestamps, options)`. It must return a JSON-safe dictionary and must not load,
replace, or silently recompute the source embeddings.

Use these result contracts unless a deliberate core schema extension is required:

- Embedding: `{"embeddings": array}` with shape `(B, D)`.
- Segmentation, detection, regression, classification, or another per-frame task: `{"frames": [result, ...]}` with exactly one JSON-safe result dictionary per input frame.
- Regression values belong under `{"values": {name: number}}`; classification results should include a label and confidence or an explicit score mapping.
- Spatial instances use native-frame coordinates and the established fields `class_id`, `class_name`, `confidence`, `box_xyxy`, `polygon_xy`, and `mask_area_px` where applicable.

Preserve these inference invariants:

- Every result must remain aligned with its source `frame_indices` and `timestamps_s`, including strided and batched runs.
- Preprocessing must be deterministic and recorded. Use `PreprocessingContext`; do not derive model inputs from browser display contrast or colormaps. Dataset-level fixed scaling is the default because per-frame normalization can erase real intensity evolution.
- Convert model-space boxes, polygons, and masks back to native frame pixels exactly once.
- Loading or rotating a dataset changes `dataset_version`. Results from an older version are stale and must not be displayed over the current data.
- The built-in path is local inference. Do not transmit frames to a remote service unless a future feature explicitly requests, labels, and obtains authorization for remote processing.
- Standard task types should work through the generic runner and serialization path. Change the web app or MCP layer only when a genuinely new output schema or user interaction requires it.
- New classical analyses must record their parameters and JSON-safe results in the active analysis run. New AI or embedding-analysis output must be stored beside its source inference run and loaded from there for visualization/export.
- Put new runtime dependencies in an appropriate optional extra in `pyproject.toml`, update `uv.lock`, and document the matching `uv sync --extra ...` command. Do not add a large AI stack to the default installation.
- Preserve third-party scientific code provenance, pinned source revision, license text, and local-modification notes. Keep bundled third-party license files in the source tree and installed distribution.

## Scientific and coordinate invariants

- Frame arrays use `(N, H, W)`; individual frames use `(H, W)`. Loaded analysis data is represented as `uint16` where practical, and timestamps are seconds zeroed to the first frame.
- Browser-drawn points and ROIs may be in display coordinates. Convert them exactly once to native frame pixels using the matching canvas `display_w` and `display_h`.
- Rotation mutates all loaded frames. Any calibration, ROI, streak marker, or cached render tied to old pixel coordinates must be invalidated afterward.
- The core RHEED relation is `d = λL/Δx`, using the relativistic electron wavelength. Keep units explicit: `L` and `Δx` in cm produce `d` in cm, then convert to Å.
- Treat material lattice constants as convenient reference defaults, not authoritative publication citations. Preserve the workbook's uncertainty and pseudo-cubic caveats.

## Synchronized data and behavior

- The 53-entry materials table currently exists in `src/rheed_core/materials.py`, twice in `src/rheed_webapp/templates/index.html`, and in `materials_database.xlsx`. If one copy changes, update and compare all copies.
- The hidden Frame Viewer DOM and JavaScript are still load-bearing infrastructure for shared state, frame caching, ROI rendering, intensity charts, and exports. Do not delete it merely because Growth Analysis is the visible default workspace.
- The visible Analysis tab (`tab-growth`) and the hidden Frame Viewer share frontend `state`, backend contrast/colormap settings, frame cache, and ROI data. Keep both views synchronized when loading, rotating, or clearing state.
- Auto strip tracking uses the specular peak plus the nearest first-order peak on each side. Extra peaks are display-only and must not silently change the measured spacing.
- The SSIM azimuth-alignment notebook is research-stage functionality, not yet part of the core or UI. Preserve its beam masking, tilt correction, DoG filtering, standardized mirror-half SSIM, and peak-refinement intent if promoting it into the package.

## Verification

- An automated test suite exists under `tests/`. Add focused unit tests for new scientific logic and regression tests for bugs.
- At minimum, run `python -m compileall -q src` after Python edits.
- For adapter changes, add dependency-free tests using a small dummy adapter. Cover output validation, frame/timestamp alignment, stride and batching, cancellation or failure cleanup where relevant, and empty model results. Do not make standard tests download model weights.
- When changing optional dependencies, run `uv lock --check`; before handoff, run `uv run --frozen --no-sync python -m pytest -q` and `uv run --frozen --no-sync python -m compileall -q src`.
- For core changes, exercise the affected calculation with a small synthetic dataset and check edge cases such as empty ROIs, constant frames, lost peaks, invalid spacing, and non-monotonic timestamps.
- For web changes, verify the Flask route and the corresponding browser interaction. Pay particular attention to load/reset/rotate behavior and native/display coordinate scaling.
- Keep generated datasets, videos, exports, downloaded model weights, caches, and local `rheed_library/` contents out of version control.
