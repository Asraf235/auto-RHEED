import cv2
import numpy as np


def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    if not frames:
        raise ValueError("No frames could be read from this video")

    arr = np.stack(frames).astype(np.uint16) * 257  # scale 0-255 → 0-65535
    ts = np.arange(len(arr)) / fps
    return arr, ts
