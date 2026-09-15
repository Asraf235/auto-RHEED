"""Shared constants: colormaps and recognized image-file extensions."""
import cv2

CV2_CMAPS = {
    "gray":    None,                    # no colormap — plain grayscale
    "hot":     cv2.COLORMAP_HOT,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma":  cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma":   cv2.COLORMAP_MAGMA,
    "cividis": cv2.COLORMAP_CIVIDIS,    # perceptually uniform + colorblind-safe
    "jet":     cv2.COLORMAP_JET,
    "ocean":   cv2.COLORMAP_OCEAN,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "cool":    cv2.COLORMAP_COOL,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".img"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
