"""Optional Ultralytics instance-segmentation adapter."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .base import AdapterDescriptor, ModelAdapter
from .preprocessing import frame_to_rgb_uint8
from .types import ModelSpec, PreprocessingContext


class UltralyticsSegAdapter(ModelAdapter):
    descriptor = AdapterDescriptor(
        name="ultralytics-seg",
        task="segmentation",
        description="Ultralytics-compatible local instance-segmentation models",
        requirements=("ultralytics",),
        default_options={
            "confidence": 0.25,
            "iou": 0.7,
            "image_size": 640,
            "retina_masks": True,
            "tracking": False,
            "tracker": "bytetrack.yaml",
        },
    )

    def __init__(self):
        self.model = None
        self.options: dict[str, Any] = {}
        self.device: str | None = None
        self.version: str | None = None

    def load(self, spec: ModelSpec, device: str) -> dict[str, Any]:
        # Ultralytics creates a settings file at import time. Keep that
        # framework cache in a writable local temp location unless the user
        # already selected a YOLO_CONFIG_DIR explicitly.
        config_dir = Path(spec.options.get(
            "config_dir", Path(tempfile.gettempdir()) / "auto-rheed-ultralytics"
        )).expanduser()
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        matplotlib_dir = config_dir / "matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The segmentation adapter requires optional dependencies; run "
                "`uv sync --extra yolo`"
            ) from exc

        self.options = {**self.descriptor.default_options, **spec.options}
        tracking = bool(self.options.get("tracking", False))
        tracker = str(self.options.get("tracker", "bytetrack.yaml")).strip()
        if tracking and not tracker:
            raise ValueError("options.tracker must name a tracker when tracking is enabled")
        self.options["tracker"] = tracker
        self.device = None if (device or "auto").lower() == "auto" else device
        self.version = ultralytics.__version__
        self.model = YOLO(spec.source, task="segment")
        names = getattr(self.model, "names", {})
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}
        return {
            "framework": "ultralytics",
            "ultralytics_version": self.version,
            "device": self.device or "auto",
            "classes": {str(k): v for k, v in dict(names).items()},
            "tracking": tracking,
            "tracker": tracker if tracking else None,
        }

    def effective_batch_size(self, requested: int) -> int:
        # Tracking calls the model once per ordered frame. Let the shared
        # runner use that same granularity for progress and cancellation.
        return 1 if bool(self.options.get("tracking", False)) else int(requested)

    @staticmethod
    def _polygon_list(mask_polygon) -> list[list[float]]:
        polygon = np.asarray(mask_polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            return []
        return [[round(float(x), 3), round(float(y), 3)] for x, y in polygon]

    def infer_batch(
        self,
        frames: np.ndarray,
        context: PreprocessingContext,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Ultralytics segmentation adapter is not loaded")
        # Ultralytics treats NumPy inputs as OpenCV images (BGR).
        images = [np.ascontiguousarray(frame_to_rgb_uint8(frame, context)[:, :, ::-1])
                  for frame in frames]
        kwargs: dict[str, Any] = {
            "stream": False,
            "verbose": False,
            "conf": float(self.options.get("confidence", 0.25)),
            "iou": float(self.options.get("iou", 0.7)),
            "imgsz": int(self.options.get("image_size", 640)),
            "retina_masks": bool(self.options.get("retina_masks", True)),
        }
        if self.device is not None:
            kwargs["device"] = self.device
        tracking = bool(self.options.get("tracking", False))
        if tracking:
            # Ultralytics trackers carry state between calls only when persist=True.
            # Process frames individually and in order so identities also persist
            # across the batches chosen by the shared inference runner.
            results = []
            tracker = str(self.options.get("tracker", "bytetrack.yaml")).strip()
            if not tracker:
                raise ValueError("options.tracker must name a tracker when tracking is enabled")
            for image in images:
                tracked = self.model.track(
                    source=image,
                    persist=True,
                    tracker=tracker,
                    **kwargs,
                )
                if len(tracked) != 1:
                    raise RuntimeError(
                        "Ultralytics tracking returned an unexpected number of frame results"
                    )
                results.extend(tracked)
        else:
            results = self.model.predict(source=images, **kwargs)
        frame_results: list[dict[str, Any]] = []
        for result in results:
            instances: list[dict[str, Any]] = []
            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None else np.empty((0, 4))
            scores = boxes.conf.detach().cpu().numpy() if boxes is not None else np.empty(0)
            classes = boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None else np.empty(0, dtype=int)
            track_ids = (
                boxes.id.detach().cpu().numpy().astype(int)
                if boxes is not None and getattr(boxes, "id", None) is not None
                else np.full(len(xyxy), -1, dtype=int)
            )
            polygons = result.masks.xy if result.masks is not None else []
            mask_data = (result.masks.data.detach().cpu().numpy()
                         if result.masks is not None else np.empty((0,)))
            names = result.names or {}
            for i in range(len(xyxy)):
                class_id = int(classes[i])
                polygon = self._polygon_list(polygons[i]) if i < len(polygons) else []
                mask_area = float(mask_data[i].sum()) if i < len(mask_data) else None
                track_id = int(track_ids[i]) if i < len(track_ids) and track_ids[i] >= 0 else None
                instances.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id),
                    "confidence": round(float(scores[i]), 6),
                    "box_xyxy": [round(float(v), 3) for v in xyxy[i]],
                    "polygon_xy": polygon,
                    "mask_area_px": mask_area,
                })
            frame_results.append({"instances": instances})
        if len(frame_results) != len(frames):
            raise RuntimeError("Ultralytics returned an unexpected number of frame results")
        return {"frames": frame_results}

    def close(self) -> None:
        self.model = None
