import numpy as np
import librosa


DEFAULT_SR = 16000
TARGET_DURATION_SEC = 1.0


def load_and_resample(path: str, target_sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono and resample to target_sr."""
    signal, sr = librosa.load(path, sr=target_sr, mono=True)
    return signal, target_sr


def normalize_volume(signal: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    """Normalize signal to a target RMS level to reduce loudness variation."""
    rms = np.sqrt(np.mean(signal**2) + 1e-9)
    if rms < 1e-6:
        return signal
    gain = target_rms / rms
    normalized = signal * gain
    # Clip to valid range
    return np.clip(normalized, -1.0, 1.0)


def pad_or_crop(signal: np.ndarray, sr: int = DEFAULT_SR, target_duration: float = TARGET_DURATION_SEC) -> np.ndarray:
    """Pad with zeros or crop center to obtain a fixed-length signal."""
    target_len = int(target_duration * sr)
    current_len = signal.shape[0]

    if current_len == target_len:
        return signal
    if current_len < target_len:
        pad_width = target_len - current_len
        return np.pad(signal, (0, pad_width), mode="constant")

    # Crop from center if longer
    start = (current_len - target_len) // 2
    end = start + target_len
    return signal[start:end]


def preprocess_file(path: str, target_sr: int = DEFAULT_SR, target_duration: float = TARGET_DURATION_SEC) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline for a file: load, normalize, pad/crop."""
    signal, sr = load_and_resample(path, target_sr=target_sr)
    signal = normalize_volume(signal)
    signal = pad_or_crop(signal, sr=sr, target_duration=target_duration)
    return signal, sr

