from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd

from audio.preprocess import DEFAULT_SR, TARGET_DURATION_SEC


class RingBuffer:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_samples)

    def extend(self, data: np.ndarray) -> None:
        self.buffer.extend(data.tolist())

    def get(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(self.max_samples, dtype=np.float32)
        buf = np.array(self.buffer, dtype=np.float32)
        if buf.shape[0] < self.max_samples:
            pad = np.zeros(self.max_samples - buf.shape[0], dtype=buf.dtype)
            buf = np.concatenate([pad, buf])
        else:
            buf = buf[-self.max_samples :]
        return buf


def stream_audio(
    callback: Callable[[np.ndarray, int], None],
    sr: int = DEFAULT_SR,
    window_sec: float = TARGET_DURATION_SEC,
    hop_sec: float = 0.5,
) -> None:
    """Continuously stream audio and invoke callback with each window."""
    max_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    ring = RingBuffer(max_samples=max_samples)

    def sd_callback(indata, frames, time, status):  # type: ignore[no-untyped-def]
        if status:
            print(status)
        mono = indata[:, 0].copy()
        ring.extend(mono)
        # Only trigger when we have at least hop_samples new data
        if len(ring.buffer) >= hop_samples:
            window = ring.get()
            callback(window, sr)

    with sd.InputStream(
        channels=1,
        samplerate=sr,
        dtype="float32",
        callback=sd_callback,
    ):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopping stream.")

