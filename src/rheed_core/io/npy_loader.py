import numpy as np


def load_npy(path: str):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]  # single frame → (1, H, W)
    if arr.dtype != np.uint16:
        a = arr.astype(np.float32)
        lo, hi = a.min(), a.max()
        arr = ((a - lo) / max(hi - lo, 1) * 65535).astype(np.uint16)
    ts = np.arange(len(arr), dtype=float)
    return arr, ts
