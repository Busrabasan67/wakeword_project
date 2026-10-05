"""Run audio augmentation: raw -> processed (original + augmented variants)."""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from .augment import generate_augmentations
from .preprocess import preprocess_file, pad_or_crop, DEFAULT_SR, TARGET_DURATION_SEC


def run_augmentation(
    input_dir: Path,
    output_dir: Path,
    num_variants: int = 5,
) -> tuple[int, int]:
    """Process raw audio: preprocess, augment, save to processed.
    Returns (count_positive, count_negative) files processed."""
    counts = {"positive": 0, "negative": 0}
    for label in ("positive", "negative"):
        label_in = input_dir / label
        label_out = output_dir / label
        if not label_in.exists():
            continue
        label_out.mkdir(parents=True, exist_ok=True)

        for path in label_in.rglob("*.wav"):
            signal, sr = preprocess_file(str(path), target_sr=DEFAULT_SR, target_duration=TARGET_DURATION_SEC)

            # Save preprocessed original
            orig_path = label_out / f"{path.stem}.wav"
            sf.write(orig_path, signal, sr)

            # Generate and save augmented variants
            variants = generate_augmentations(signal, sr, num_variants=num_variants)
            for i, variant in enumerate(variants):
                fixed = pad_or_crop(variant, sr=sr, target_duration=TARGET_DURATION_SEC)
                aug_path = label_out / f"{path.stem}_aug{i}.wav"
                sf.write(aug_path, fixed, sr)
            counts[label] += 1 + num_variants  # original + variants
    return counts["positive"], counts["negative"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment raw audio and save to processed.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory with positive/ and negative/ folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for augmented audio.",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="Number of augmented variants per file.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pos_count, neg_count = run_augmentation(
        input_dir, output_dir, num_variants=args.num_variants
    )
    total = pos_count + neg_count
    print(f"Augmentation complete: {pos_count} positive + {neg_count} negative = {total} files -> {output_dir}")


if __name__ == "__main__":
    main()
