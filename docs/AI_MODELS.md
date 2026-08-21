# Local AI model inference

Auto RHEED treats model inference as another frame-aligned analysis. Every
result is keyed by native frame index and timestamp, so it stays synchronized
with intensity, streak spacing, FWHM, coherence length, and FFT results.

Inference runs locally. Auto RHEED does not upload frames, embeddings, masks,
or model outputs. A remote repository identifier may download and cache model
weights on first use; a local path works offline.

## Install an optional runtime

The classical analysis installation remains lightweight. Install only the AI
runtime needed on a machine:

```powershell
# PCA and K-means only (also used by the test suite)
uv sync --extra ai

# DINOv3 embeddings
uv sync --extra dinov3

# Ultralytics instance segmentation
uv sync --extra yolo

# Both model adapters
uv sync --extra ai-all
```

PyTorch chooses CPU or an available accelerator at runtime. Hardware-specific
PyTorch/CUDA installations can still be managed with UV before syncing the
remaining extras.

## Run inference in the app

Open **Analysis**, load a video, image stack, or single image, then use
the **Local AI Inference** card directly below **Growth Video** in the left
sidebar. Choose a saved manifest or configure an adapter and model source,
then select **Run Inference**. Embedding runs expose PCA and K-means controls
when inference finishes; spatial results are drawn over their matching frames.

## Model manifests

Small JSON manifests live in `rheed_models/`; model weight extensions are
ignored by Git. A manifest records the adapter, model source, deterministic
preprocessing, and adapter options:

```json
{
  "id": "my-dinov3-run",
  "adapter": "dinov3",
  "source": "facebook/dinov3-vits16-pretrain-lvd1689m",
  "task": "embedding",
  "preprocessing": {
    "intensity_scaling": "dataset_percentile",
    "low_percentile": 0.5,
    "high_percentile": 99.5
  },
  "options": {
    "embedding_source": "cls",
    "l2_normalize": true,
    "local_files_only": false
  }
}
```

`source` may instead be a local model directory or weight file. Set
`local_files_only` for DINOv3 to prevent network access. Pin `revision` when a
repository revision must be exactly reproducible.

The fixed dataset-level percentile transform is intentionally independent of
the viewer's contrast and color map. `per_frame_percentile`, `dtype_range`, and
explicit `fixed` bounds are also supported, but per-frame scaling can remove
real intensity evolution and should be selected deliberately.

## DINOv3 results

The DINOv3 adapter supports these embedding policies:

- `cls` — the whole-image CLS token (default).
- `mean_patch` — the mean of spatial patch tokens, excluding register tokens.
- `cls_mean_patch` — concatenated CLS and mean-patch embeddings.

Grayscale RHEED frames are mapped to three identical channels before the
checkpoint's own image processor is applied. The output is an `(N, D)`
`float32` matrix. PCA and K-means run after all selected embeddings are ready.
PCA can retain either a fixed number of dimensions or the fewest dimensions
needed to reach an explained-variance target such as `0.95`. K-means uses every
retained PCA dimension; PC1/PC2 are only the default visualization axes.

The cluster count can be fixed or selected automatically by silhouette score.
In automatic mode, Auto RHEED starts with `K = 1` as a zero-score
single-cluster baseline, then fits candidate values from `K = 2` through the
configured maximum (capped at one less than the number of analyzed frames).
Because silhouette is undefined for one cluster, `K = 1` has no fabricated
silhouette value; it is retained unless a valid multi-cluster candidate has a
positive silhouette score. The selected K, winning score, score for every valid
multi-cluster candidate, and the baseline are saved in the analysis metadata.
Silhouette selection is a useful data-driven heuristic, but the resulting
clusters should still be checked against the RHEED patterns and growth history.

The Analysis plot shows embedding space and cluster assignment versus
time. Clicking a plotted point jumps the frame viewer to that native frame.

## Post-embedding changepoint analysis

Temporal analysis adapters are separate from frame model adapters. They become
available only after an embedding inference completes and consume that result's
`(N, D)` embedding matrix, native frame indices, and timestamps. PCA/K-means and
temporal analysis are independent, so either can run first and both remain
aligned to the same selected frames and inference stride.

The built-in `rhaapsody-changepoint` adapter uses the NumPy detector from
[PNNL RHAAPSODY](https://github.com/pnnl/RHAAPSODY) at pinned revision
`d1591d16528926be76b106408cbe453842406685`. Auto RHEED supplies its existing
embeddings rather than RHAAPSODY's TensorFlow embedding model. Following the
upstream pipeline, the adapter fixes a center from the initial starting period,
builds a cosine-similarity matrix, and passes the growing kernel matrix to
RHAAPSODY's segmented-cost detector.

The configurable options are:

- `cost_threshold` — minimum segmented cost that triggers a detection (`0.06`).
- `window_size` — maximum detection window in embedding rows (`300`); use
  `null` or `"full"` for the complete history.
- `min_time_between_changepoints` — minimum row separation enforced by the
  upstream detector (`10`).
- `starting_period` — initial embedding rows used to fix the similarity center
  before detection begins (`30`).

Detected changepoints are shown on the embedding/time plot and matching frames,
and can be exported as CSV or with the full NPZ result. The **Similarity Matrix**
button opens the same centered cosine-similarity kernel used by the detector as a
`[-1, 1]` Viridis heatmap with RHEED frame axes and red changepoint lines, following
RHAAPSODY's upstream presentation. Clicking the matrix jumps to that frame, and a
white crosshair follows video playback.

The complete float32 matrix is saved as a separate `.npy` artifact beside the
compressed changepoint JSON. It is not kept in the job manager or loaded with the
normal analysis response. Opening the plot requests at most 600 evenly sampled
rows and columns from the saved matrix; closing it releases the browser plot data.
This keeps long videos responsive while retaining the full-resolution matrix on
disk.

Each result records the upstream project, pinned revision, BSD-2-Clause license,
parameters, proposed positions, detection delay, amplitudes, native frames, and
timestamps. Only the detector excerpt is bundled; RHAAPSODY's embeddings,
messaging, Matplotlib plotting implementation, graph clustering, and filesystem
pipeline are not copied. Auto RHEED supplies its own interactive Plotly renderer
for the saved RHAAPSODY matrix.

External temporal analyzers can register an
`auto_rheed.embedding_analysis_adapters` entry point returning a subclass of
`EmbeddingAnalysisAdapter`. This contract receives a complete aligned embedding
sequence and returns a JSON-safe temporal result; it should not be used for
per-frame model inference.

## Segmentation results

The `ultralytics-seg` adapter accepts any segmentation model source supported
by the installed Ultralytics release. It translates framework results into a
stable per-instance schema containing class, confidence, native-pixel box,
native-pixel polygon, mask area, and an optional `track_id`. Masks are overlaid
on the shared Analysis frame and included in NPZ/CSV exports.

Set `options.tracking` to `true` to run the model's Ultralytics video tracker
sequentially with persistent state. `options.tracker` selects the tracker
configuration and defaults to Ultralytics' bundled `bytetrack.yaml`; a custom
local YAML path is also accepted. Tracked overlays are colored by ID and label
instances as `class #track_id`. Tracking IDs are scoped to one inference run.
See `rheed_models/README.md` for the complete configuration and stride guidance.

## Outputs

- Every completed inference is automatically written to the active analysis run
  under `data/runs/.../ai/` by default. Change the root in the Analysis
  **Analysis Storage** card. Model arrays are served from these files after the
  background job completes rather than being retained in the job manager.
- Embeddings and full RHAAPSODY similarity matrices use `.npy`; per-frame spatial/scalar records use indexed JSONL so a
  requested overlay frame can be read without loading the entire result. PCA,
  K-means, and post-embedding analyses use readable JSON plus automatically
  generated CSV tables beside their source inference run. Older `.json.gz`
  post-analysis files remain readable.
- Reopening the exact same input reconnects to its saved dataset run. In
  **Analysis Storage → Saved results for this dataset**, choose an AI inference
  artifact and click **Recall Selected Analysis**. The inference output and any
  saved PCA/K-means and RHAAPSODY changepoint/similarity results are restored from
  disk without loading the model or running inference again.
- NPZ contains the complete embedding matrix or segmentation geometry plus
  frame indices, timestamps, model provenance, preprocessing, and optional
  clustering and temporal-analysis results. This is an explicit portable export
  assembled from the automatically saved run files.
- Embedding CSV contains PCA coordinates and cluster labels.
- Changepoint CSV contains estimated and detection frames/timestamps plus the
  segmented-cost amplitude.
- Segmentation CSV contains one row per instance, including `track_id` when
  tracking was enabled.

Loading a different dataset or rotating the loaded pixels invalidates AI
results for the current viewer, but does not delete their saved run. Changing
display contrast or color map does not invalidate results.

## Writing another adapter

An adapter distribution implements `rheed_core.inference.ModelAdapter` and
registers its class through a Python entry point:

```toml
[project.entry-points."auto_rheed.inference_adapters"]
cnn-regression = "my_rheed_models:CnnRegressionAdapter"
```

```python
from rheed_core.inference import AdapterDescriptor, ModelAdapter

class CnnRegressionAdapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="cnn-regression",
        task="regression",
        description="Local surface-property regression",
        requirements=("torch",),
        default_options={"output_name": "roughness_nm"},
    )

    def load(self, spec, device):
        # Load spec.source and return reproducibility metadata.
        ...

    def infer_batch(self, frames, context):
        # Return one stable frame-result object per input frame.
        return {"frames": [
            {"values": {"roughness_nm": float(value)}}
            for value in self.model(...)
        ]}
```

Classification, regression, embedding, detection, and segmentation adapters
should translate framework objects into plain NumPy arrays and JSON-safe frame
results at this boundary. A genuinely new output type may need a new UI
renderer, but does not require changing frame loading or job orchestration.
