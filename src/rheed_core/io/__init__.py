"""File loaders. Each returns a (frames, timestamps) tuple:
    frames     — np.ndarray (N, H, W), uint16
    timestamps — np.ndarray (N,), float seconds, zeroed to start at 0
None of these touch any session/global state — callers decide what to
do with the result (RheedSession stores it; an MCP tool might not).
"""
from .h5_loader import load_h5
from .npy_loader import load_npy
from .video_loader import load_video
from .image_loader import load_single_frame_image, load_image_stack, decode_single_image
from .ksa_img import decode_ksa_img

__all__ = [
    "load_h5", "load_npy", "load_video",
    "load_single_frame_image", "load_image_stack", "decode_single_image",
    "decode_ksa_img",
]
