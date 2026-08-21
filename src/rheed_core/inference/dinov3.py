"""Optional Hugging Face DINOv3 embedding adapter.

No model identifier is hard-coded: the manifest's ``source`` selects any
compatible local directory or Hugging Face repository.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import AdapterDescriptor, ModelAdapter
from .preprocessing import frame_to_rgb_uint8
from .types import ModelSpec, PreprocessingContext


class DinoV3Adapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="dinov3",
        task="embedding",
        description="DINOv3 whole-frame embeddings through Hugging Face Transformers",
        requirements=("torch", "transformers"),
        default_options={
            "embedding_source": "cls",
            "l2_normalize": True,
            "local_files_only": False,
        },
    )

    def __init__(self):
        self.model = None
        self.processor = None
        self.torch = None
        self.device = "cpu"
        self.options: dict[str, Any] = {}
        self._runtime_metadata: dict[str, Any] = {}

    @staticmethod
    def _resolve_device(torch, requested: str) -> str:
        requested = (requested or "auto").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
        return requested

    def load(self, spec: ModelSpec, device: str) -> dict[str, Any]:
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "The DINOv3 adapter requires optional dependencies; run "
                "`uv sync --extra dinov3`"
            ) from exc

        self.torch = torch
        self.device = self._resolve_device(torch, device)
        self.options = {**self.descriptor.default_options, **spec.options}
        common: dict[str, Any] = {
            "local_files_only": bool(self.options.get("local_files_only", False)),
            "trust_remote_code": bool(self.options.get("trust_remote_code", False)),
        }
        if spec.revision:
            common["revision"] = spec.revision

        self.processor = AutoImageProcessor.from_pretrained(spec.source, **common)
        self.model = AutoModel.from_pretrained(spec.source, **common)
        self.model.eval()
        self.model.to(self.device)

        dtype_name = str(self.options.get("dtype", "float32")).lower()
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError("DINOv3 dtype must be float32, float16, or bfloat16")
        if self.device == "cpu" and dtype_name == "float16":
            raise ValueError("float16 DINOv3 inference is not supported on CPU; use float32")
        self.model.to(dtype=dtype)

        self._runtime_metadata = {
            "framework": "transformers",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": self.device,
            "dtype": dtype_name,
            "embedding_source": self.options.get("embedding_source", "cls"),
            "l2_normalize": bool(self.options.get("l2_normalize", True)),
            "model_type": getattr(self.model.config, "model_type", None),
        }
        return dict(self._runtime_metadata)

    def _select_embedding(self, outputs):
        torch = self.torch
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            feature_maps = getattr(outputs, "feature_maps", None)
            if feature_maps:
                hidden = feature_maps[-1]
        if hidden is None:
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                return pooler
            raise RuntimeError("DINOv3 model did not return a recognized feature tensor")

        source = str(self.options.get("embedding_source", "cls")).lower()
        if hidden.ndim == 4:
            # ConvNeXt-like feature map: (B, C, H, W).
            pooled = hidden.mean(dim=(-2, -1))
            if source not in {"mean_patch", "mean", "cls"}:
                raise ValueError("Spatial DINOv3 backbones support mean_patch pooling")
            return pooled
        if hidden.ndim != 3:
            raise RuntimeError(f"Unexpected DINOv3 hidden-state shape: {tuple(hidden.shape)}")

        cls_token = hidden[:, 0, :]
        n_registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        patch_tokens = hidden[:, 1 + n_registers:, :]
        if patch_tokens.shape[1] == 0:
            raise RuntimeError("DINOv3 output did not contain patch tokens")
        mean_patch = patch_tokens.mean(dim=1)
        if source == "cls":
            return cls_token
        if source in {"mean_patch", "mean"}:
            return mean_patch
        if source in {"cls_mean_patch", "concat"}:
            return torch.cat([cls_token, mean_patch], dim=1)
        raise ValueError("embedding_source must be cls, mean_patch, or cls_mean_patch")

    def infer_batch(
        self,
        frames: np.ndarray,
        context: PreprocessingContext,
    ) -> dict[str, Any]:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("DINOv3 adapter is not loaded")
        images = [frame_to_rgb_uint8(frame, context) for frame in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        model_dtype = next(self.model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            embedding = self._select_embedding(outputs)
            if bool(self.options.get("l2_normalize", True)):
                embedding = self.torch.nn.functional.normalize(embedding, p=2, dim=1)
        return {"embeddings": embedding.detach().float().cpu().numpy()}

    def close(self) -> None:
        self.model = None
        self.processor = None
        if self.torch is not None and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
