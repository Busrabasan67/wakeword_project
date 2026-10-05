from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .preprocess import preprocess_file, DEFAULT_SR, TARGET_DURATION_SEC
from .feature_extraction import extract_mfcc_features


Label = Literal["positive", "negative"]


def iter_audio_files(base_dir: Path, label: Label):
    label_dir = base_dir / label
    if not label_dir.exists():
        return
    for path in label_dir.rglob("*.wav"):
        yield path


def build_dataset(
    base_dir: Path,
    output_csv: Path,
    n_mfcc: int = 20,
    use_minmax: bool = False,
) -> None:
    rows: list[dict] = []

    for label in ("positive", "negative"):
        for path in iter_audio_files(base_dir, label):  # type: ignore[arg-type]
            signal, sr = preprocess_file(
                str(path),
                target_sr=DEFAULT_SR,
                target_duration=TARGET_DURATION_SEC,
            )
            features = extract_mfcc_features(signal, sr=sr, n_mfcc=n_mfcc, use_minmax=use_minmax)

            row: dict[str, object] = {"label": label, "filepath": str(path)}
            for i, value in enumerate(features):
                row[f"f_{i}"] = float(value)
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No audio files found under {base_dir}.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved dataset with {len(df)} samples to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MFCC feature CSV from raw audio.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/raw",
        help="Base directory containing 'positive' and 'negative' folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/csv/mfcc_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=20,
        help="Number of MFCC coefficients.",
    )
    parser.add_argument(
        "--use-minmax",
        action="store_true",
        help="Include min/max stats in addition to mean/std.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_csv = Path(args.output)

    build_dataset(base_dir, output_csv, n_mfcc=args.n_mfcc, use_minmax=bool(args.use_minmax))


if __name__ == "__main__":
    main()

