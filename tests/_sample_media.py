"""Tiny in-memory samples for image / audio / video tests.

These are *byte streams*, not real perceptual content — they're enough to
exercise the agent code paths without dragging real media files into the
repo.
"""

from __future__ import annotations

# Minimal 1x1 transparent PNG.
TINY_PNG: bytes = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x5a\xf4\x8a\xf3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal WAV header + 100 silent samples.
def _silent_wav() -> bytes:
    import struct
    sample_count = 100
    byte_rate = 16000 * 2
    data = b"\x00\x00" * sample_count
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, byte_rate, 2, 16)
    chunk = b"data" + struct.pack("<I", len(data))
    return header + fmt + chunk + data


TINY_WAV: bytes = _silent_wav()

# Placeholder MP4-ish bytes (ftyp box). Not playable but enough for tests
# that only check our code paths.
TINY_MP4: bytes = (
    b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    b"\x00\x00\x00\x08free"
)
