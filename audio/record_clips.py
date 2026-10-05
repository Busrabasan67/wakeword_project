import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf  # type: ignore


DEFAULT_SR = 16000
DEFAULT_DURATION_SEC = 1.2


def record_clip(duration: float, sr: int) -> np.ndarray:
    """Record a single mono clip from the default microphone."""
    print(f"Recording {duration:.2f} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    return audio.flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled wake-word clips.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/raw",
        help="Base directory for saving recordings.",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["positive", "negative"],
        required=True,
        help="Label for this recording session.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of clips to record in this run.",
    )
    parser.add_argument(
        "--student-id",
        type=str,
        default="student",
        help="Identifier to embed in filenames.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help="Clip duration in seconds.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help="Sample rate.",
    )

    args = parser.parse_args()

    target_dir = Path(args.base_dir) / args.label
    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Recording {args.count} {args.label} clips.\n"
        f"Positive = phrase 'Hey Pakize'; Negative = anything else / noise.\n"
    )

    for idx in range(args.count):
        input(f"Press ENTER to record clip {idx + 1}/{args.count}...")
        audio = record_clip(duration=args.duration, sr=args.sr)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.student_id}_{args.label}_{ts}_{idx}.wav"
        path = target_dir / filename
        sf.write(path, audio, args.sr)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

