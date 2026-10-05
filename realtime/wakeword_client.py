from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from audio.preprocess import normalize_volume, pad_or_crop, DEFAULT_SR, TARGET_DURATION_SEC
from audio.feature_extraction import extract_mfcc_features
from .audio_stream import stream_audio


class WakeWordDetector:
    def __init__(self, model_path: Path, cooldown_sec: float = 1.5, prob_threshold: float = 0.5):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols = payload.get("feature_cols")
        self.cooldown_sec = cooldown_sec
        self.prob_threshold = prob_threshold
        self._last_trigger_time = 0.0

    def should_trigger(self, signal: np.ndarray, sr: int) -> bool:
        signal = normalize_volume(signal)
        signal = pad_or_crop(signal, sr=sr, target_duration=TARGET_DURATION_SEC)
        feats = extract_mfcc_features(signal, sr=sr, n_mfcc=13)
        X = feats.reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        prob_positive = float(probs[1])
        now = time.time()
        if prob_positive >= self.prob_threshold and (now - self._last_trigger_time) >= self.cooldown_sec:
            self._last_trigger_time = now
            print(f"Prob(positive)={prob_positive:.3f} -> TRIGGER")
            return True
        else:
            print(f"Prob(positive)={prob_positive:.3f}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time wake-word detector client.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/wakeword_model.joblib",
        help="Trained model path.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help="Sample rate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive detection.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="Cooldown time in seconds after a trigger.",
    )
    args = parser.parse_args()

    detector = WakeWordDetector(
        model_path=Path(args.model),
        cooldown_sec=args.cooldown,
        prob_threshold=args.threshold,
    )

    def on_window(window: np.ndarray, sr: int) -> None:
        triggered = detector.should_trigger(window, sr)
        if triggered:
            print("Wake word detected!")

    stream_audio(on_window, sr=args.sr)


if __name__ == "__main__":
    main()

