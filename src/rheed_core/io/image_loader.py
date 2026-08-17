import cv2
import numpy as np

from .ksa_img import decode_ksa_img


def decode_single_image(path: str, suffix: str) -> np.ndarray:
    """Decode any supported still image to a single (H, W) uint16 array."""
    if suffix == ".img":
        arr = decode_ksa_img(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("Could not read image")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint16:
            a = arr.astype(np.float32)
            lo, hi = a.min(), a.max()
            arr = ((a - lo) / max(hi - lo, 1) * 65535).astype(np.uint16)
    return arr.astype(np.uint16, copy=False)


def load_single_frame_image(path: str, suffix: str):
    """Load any supported still image as a 1-frame (1, H, W) stack."""
    arr = decode_single_image(path, suffix)
    frames = arr[np.newaxis]  # (1, H, W)
    ts = np.array([0.0])
    return frames, ts


def load_image_stack(file_specs):
    """
    Build a (N, H, W) frame stack from a list of (path, suffix) tuples,
    in the given order — e.g. a folder of per-frame exports being
    reassembled into a 'video' for further analysis. All images must be
    the same size. Timestamps are synthetic (1 second per frame), since
    individual still images carry no real acquisition timing.
    """
    if not file_specs:
        raise ValueError("No images given")
    frames_list = []
    shape = None
    for path, suffix in file_specs:
        arr = decode_single_image(path, suffix)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"Image size mismatch: expected {shape}, got {arr.shape} ({path})"
            )
        frames_list.append(arr)
    frames = np.stack(frames_list)
    ts = np.arange(len(frames), dtype=float)
    return frames, ts
