"""
RHEED image library — a persistent, on-disk reference gallery of RHEED
patterns for different substrates (similar to kSA's library).

Images are decoded to PNG on add (so .img / 16-bit .tif display in a
browser), stored under <root>/images/, with metadata in <root>/library.json.
No Flask here — the web layer handles HTTP; this is pure storage logic.
"""
import datetime
import json
import os
import uuid

import cv2
import numpy as np

from . import io as rheed_io


class RheedLibrary:
    def __init__(self, root_dir: str):
        self.root = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.meta_path = os.path.join(root_dir, "library.json")

    # ── storage helpers ──────────────────────────────────────────────
    def _ensure(self):
        os.makedirs(self.images_dir, exist_ok=True)
        if not os.path.exists(self.meta_path):
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load(self) -> list:
        self._ensure()
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, entries: list):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    # ── decode any supported image → display PNG bytes ───────────────
    def _decode_to_png(self, path: str, suffix: str) -> bytes:
        suffix = suffix.lower()
        if suffix == ".img":
            img = rheed_io.decode_ksa_img(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not read image")
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            img = ((img.astype(np.float32) - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("PNG encode failed")
        return buf.tobytes()

    # ── public API ───────────────────────────────────────────────────
    def list(self) -> list:
        # Newest first.
        return list(reversed(self._load()))

    def add(self, src_path: str, suffix: str, *, material: str = "", zone: str = "",
            surface: str = "", energy: str = "", notes: str = "") -> dict:
        self._ensure()
        png = self._decode_to_png(src_path, suffix)
        entry_id = uuid.uuid4().hex[:12]
        fname = entry_id + ".png"
        with open(os.path.join(self.images_dir, fname), "wb") as f:
            f.write(png)
        entry = {
            "id": entry_id, "file": fname,
            "material": material.strip(), "zone": zone.strip(),
            "surface": surface.strip(), "energy": str(energy).strip(),
            "notes": notes.strip(),
            "added": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        entries = self._load()
        entries.append(entry)
        self._save(entries)
        return entry

    def delete(self, entry_id: str) -> bool:
        entries = self._load()
        removed = next((e for e in entries if e["id"] == entry_id), None)
        if removed is None:
            return False
        try:
            os.unlink(os.path.join(self.images_dir, removed["file"]))
        except OSError:
            pass
        self._save([e for e in entries if e["id"] != entry_id])
        return True

    def image_path(self, entry_id: str):
        for e in self._load():
            if e["id"] == entry_id:
                p = os.path.join(self.images_dir, e["file"])
                return p if os.path.exists(p) else None
        return None
