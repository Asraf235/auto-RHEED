"""
Decoder for K-Space Associates (kSA RHEED) .img files.

Format (reverse-engineered against a known image, cross-checked against
its .bmp export — see project history for the byte-offset derivation):
  - 640-byte header, magic 'KSA00F' at offset 39
  - width  stored as big-endian uint16 at offset 49
  - height stored as big-endian uint16 at offset 321
  - followed by raw little-endian uint16 pixel data, row-major (H, W)
"""
import numpy as np

HEADER_LEN = 640
WIDTH_OFFSET = 49
HEIGHT_OFFSET = 321


def decode_ksa_img(path: str) -> np.ndarray:
    """Returns a single (H, W) uint16 array."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) <= HEADER_LEN:
        raise ValueError("File too small to be a valid kSA .img")
    header = raw[:HEADER_LEN]
    width = int.from_bytes(header[WIDTH_OFFSET:WIDTH_OFFSET + 2], "big")
    height = int.from_bytes(header[HEIGHT_OFFSET:HEIGHT_OFFSET + 2], "big")
    expected = width * height * 2
    data = raw[HEADER_LEN:]
    if width <= 0 or height <= 0 or len(data) < expected:
        raise ValueError(
            f"Unrecognized .img layout (parsed width={width}, height={height}, "
            f"but only {len(data)} bytes of pixel data available)"
        )
    arr = np.frombuffer(data[:expected], dtype="<u2").reshape(height, width)
    return arr.copy()  # own the memory — frombuffer is read-only
