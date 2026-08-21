# Local AI model manifests

Completed model outputs are not stored in this manifest folder. Auto RHEED saves
them automatically in the active analysis run (`data/runs/.../ai/` by default),
or under the folder selected in the **Analysis** tab's **Analysis Storage** card.
Embeddings are `.npy` arrays, per-frame segmentation/scalar results are indexed
JSONL, and PCA/K-means or changepoint outputs use readable JSON and CSV beside
the inference run. Reopen the same input and use **Analysis Storage → Saved
results for this dataset** to recall an inference artifact and its saved
PCA/K-means or changepoint/similarity plots without rerunning the model. A
RHAAPSODY run also saves its complete float32 similarity matrix
as a separate `.npy` file. The **Analysis** tab's **Similarity Matrix** button reads
a bounded plot view from that file only when opened; closing the matrix window
releases its browser data. Model weights remain external and are never copied
into result runs.

Auto RHEED discovers JSON model manifests in this directory. Model weights are
not committed: place them anywhere on the local machine and set `source` to the
path, or use a model repository identifier supported by the selected adapter.

The package currently includes these optional adapters:

- `dinov3` — whole-frame embeddings using Hugging Face Transformers.
- `ultralytics-seg` — instance segmentation using an Ultralytics-compatible
  model file.

Install only the runtime needed on this machine:

```powershell
uv sync --extra dinov3
uv sync --extra yolo
# or both
uv sync --extra ai-all
```

Inference is local. A repository identifier may download and cache weights the
first time it is used; setting `options.local_files_only` to `true` makes a
DINOv3 manifest strictly offline.

## DINOv3: Hugging Face access and offline use

The included `dinov3-example.json` manifest uses
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
a gated Hugging Face repository. Before its first use, sign in on that page and
request/accept access, create a Hugging Face token with read access, and authenticate
the environment:

```powershell
uv run --frozen --no-sync hf auth login
uv run --frozen --no-sync hf auth whoami
```

Do not add the token to a manifest or commit it to the repository. Authentication is
stored in the user's Hugging Face configuration rather than in Auto RHEED.

With `local_files_only: false` (the example default), the first inference run may
download the processor configuration and model weights into the normal Hugging Face
cache. Later runs reuse those cached files. Once the complete model is cached, set
`local_files_only: true` to prohibit network access; loading will then fail instead of
downloading if any required file is missing. The manifest's `source` can also be set
to a local model directory for explicitly managed offline weights.

Inference runs locally in both modes. Auto RHEED does not upload video frames or
images to Hugging Face; Hugging Face is contacted only to authenticate and obtain
model files when a remote model source is used and network access is allowed.

## YOLO segmentation tracking

The `ultralytics-seg` adapter normally runs independent prediction on each
frame. To keep instance identities across a video, enable tracking in the
manifest options:

```json
"options": {
  "confidence": 0.25,
  "iou": 0.7,
  "image_size": 640,
  "retina_masks": true,
  "tracking": true,
  "tracker": "bytetrack.yaml"
}
```

`bytetrack.yaml` is a tracker configuration bundled with Ultralytics, so it
does not need to exist beside the manifest. Auto RHEED passes the name to the
installed Ultralytics release. Set `tracker` to a local YAML path only when
using a customized tracker configuration; `botsort.yaml`, also bundled by
Ultralytics, is another supported choice.

When tracking is enabled, Auto RHEED processes selected video frames in order
and calls the tracker with persistent state. The saved instance records,
Analysis overlay labels, CSV, and NPZ export include `track_id`. Track
IDs restart with every inference run and are meaningful only within that run.
Use inference stride `1` when temporal identity matters; a larger stride makes
the tracker treat the sampled frames as consecutive and can make identities
less stable. On a single image an assigned track ID has no temporal meaning.

Tracking mode intentionally runs frames one at a time because tracker state is
sequential. Auto RHEED therefore reports progress and checks cancellation after
every tracked frame; the inference batch-size control applies only to ordinary
batched prediction when `tracking` is disabled.

Third-party adapter distributions can register an entry point in the
`auto_rheed.inference_adapters` group. The entry point must return a subclass of
`rheed_core.inference.ModelAdapter` with a stable `AdapterDescriptor`.
