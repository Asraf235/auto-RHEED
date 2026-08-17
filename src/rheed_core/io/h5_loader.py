import h5py
import numpy as np


def load_h5(path: str):
    """
    Supports two layouts:
      - 'frames' [+ optional 'timestamps'] datasets at the root
      - 'RHEED/data' (H, W, N) with optional 'RHEED/timestamp data'
        (seen from some kSA-style instruments — frame axis is LAST)
    """
    with h5py.File(path, "r") as h:
        if "frames" in h:
            frames = h["frames"][:]
            ts = h["timestamps"][:] if "timestamps" in h else np.arange(len(frames), dtype=float)
        elif "RHEED" in h and "data" in h["RHEED"]:
            raw = h["RHEED"]["data"][:]
            frames = np.transpose(raw, (2, 0, 1))  # → (N, H, W), no copy
            if "timestamp data" in h["RHEED"]:
                ts = np.asarray(h["RHEED"]["timestamp data"]).reshape(-1).astype(float)
            else:
                ts = np.arange(frames.shape[0], dtype=float)
        else:
            found = []

            def _list_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    found.append(f"{name} {obj.shape}")

            h.visititems(_list_datasets)
            raise KeyError(
                "No recognized frame dataset found (expected 'frames' or 'RHEED/data'). "
                f"Datasets in this file: {found}"
            )

    ts = ts - ts[0]
    return frames, ts
