import numpy as np
import librosa


def add_white_noise(signal: np.ndarray, noise_level: float = 0.03) -> np.ndarray:
    """Add low-level white noise to the signal."""
    noise = np.random.randn(len(signal)).astype(signal.dtype)
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    return np.clip(signal + noise_level * noise, -1.0, 1.0)


def pitch_shift(signal: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Apply small pitch shift in semitones."""
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


def time_stretch(signal: np.ndarray, rate: float) -> np.ndarray:
    """Change speed/tempo slightly."""
    # librosa.time_stretch may change length; caller should pad/crop after this
    stretched = librosa.effects.time_stretch(signal, rate=rate)
    return stretched


def generate_augmentations(
    signal: np.ndarray,
    sr: int,
    num_variants: int = 5,
    noise_range: tuple[float, float] = (0.02, 0.05),
    pitch_steps: tuple[float, float] = (-2.0, 2.0),
    stretch_range: tuple[float, float] = (0.9, 1.1),
) -> list[np.ndarray]:
    """Generate multiple augmented versions of a signal."""
    variants: list[np.ndarray] = []
    rng = np.random.default_rng()
    for _ in range(num_variants):
        augmented = signal.copy()

        # Randomly choose which augmentations to apply
        if rng.random() < 0.8:
            level = rng.uniform(*noise_range)
            augmented = add_white_noise(augmented, noise_level=level)

        if rng.random() < 0.5:
            steps = rng.uniform(*pitch_steps)
            augmented = pitch_shift(augmented, sr=sr, n_steps=steps)

        if rng.random() < 0.5:
            rate = rng.uniform(*stretch_range)
            augmented = time_stretch(augmented, rate=rate)

        variants.append(augmented)

    return variants

