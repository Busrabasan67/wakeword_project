from __future__ import annotations

import numpy as np
import librosa


def compute_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
) -> np.ndarray:
    """Compute MFCC matrix (n_mfcc x T)."""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return mfcc


def mfcc_to_stats_vector(
    mfcc: np.ndarray,
    use_minmax: bool = False,
    use_delta: bool = True,
) -> np.ndarray:
    """Convert MFCC matrix to a fixed-length feature vector via statistics over time.

    With use_delta=True (default):
        26 base + 26 delta + 26 delta-delta = 78 features
    Without delta:
        26 features (mean + std only)

    Delta features capture temporal dynamics, which strongly distinguishes
    a short "Hey" from the full "Hey Pakize" phrase.
    """
    features: list[np.ndarray] = [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if use_minmax:
        features.extend([mfcc.min(axis=1), mfcc.max(axis=1)])

    if use_delta:
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend([
            delta1.mean(axis=1), delta1.std(axis=1),
            delta2.mean(axis=1), delta2.std(axis=1),
        ])

    return np.concatenate(features, axis=0)


def extract_mfcc_features(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    use_minmax: bool = False,
    use_delta: bool = True,
) -> np.ndarray:
    """Convenience wrapper to go from signal to feature vector.

    Default n_mfcc=20 gives 120 features (20×6 with delta+delta-delta),
    providing enough spectral resolution to distinguish 'pakize' from
    phonetically similar words like 'paris' or 'dünya'.
    """
    mfcc = compute_mfcc(signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc_to_stats_vector(mfcc, use_minmax=use_minmax, use_delta=use_delta)

