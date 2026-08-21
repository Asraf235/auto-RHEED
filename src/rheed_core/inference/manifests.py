"""Local JSON model-manifest discovery."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .types import ModelSpec


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ModelManifestRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        manifests = []
        for path in sorted(self.root.glob("*.json")):
            try:
                spec = ModelSpec.from_json_file(path)
                manifests.append({"filename": path.name, "model": spec.public_dict()})
            except Exception as exc:
                manifests.append({"filename": path.name, "error": str(exc)})
        return manifests

    def get(self, filename: str) -> ModelSpec:
        path = self._path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Model manifest not found: {filename}")
        return ModelSpec.from_json_file(path)

    def save(self, spec: ModelSpec, filename: str | None = None) -> str:
        safe = _SAFE_NAME.sub("-", filename or spec.id).strip(".-") or "model"
        if not safe.lower().endswith(".json"):
            safe += ".json"
        path = self._path(safe)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(spec.public_dict(), stream, indent=2)
            stream.write("\n")
        return path.name

    def _path(self, filename: str) -> Path:
        safe = Path(filename).name
        if safe != filename or not safe.lower().endswith(".json"):
            raise ValueError("Manifest filename must be a plain .json filename")
        path = (self.root / safe).resolve()
        if path.parent != self.root.resolve():
            raise ValueError("Manifest path escapes the model registry")
        return path
