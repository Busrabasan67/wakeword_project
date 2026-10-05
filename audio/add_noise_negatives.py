"""
MFCC CSV dosyasına sentetik gürültü (negatif) örnekleri ekle.
Sessizlik ve arka plan gürültüsünde tetiklemeyi azaltır.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import pad_or_crop, DEFAULT_SR, TARGET_DURATION_SEC
from .feature_extraction import extract_mfcc_features


def add_noise_rows(df: pd.DataFrame, n_samples: int = 200, n_mfcc: int | None = None) -> pd.DataFrame:
    """CSV'e rastgele gürültü örnekleri ekle.

    n_mfcc belirtilmezse CSV'deki feature sayısından otomatik belirlenir.
    (use_delta=True varsayımıyla: n_mfcc = len(feature_cols) // 6)
    """
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    if n_mfcc is None:
        n_mfcc = max(1, len(feature_cols) // 6)
    rng = np.random.default_rng(42)

    new_rows = []
    for i in range(n_samples):
        # Beyaz gürültü + hafif pembe (1/f) gürültü
        duration = TARGET_DURATION_SEC
        n_samp = int(DEFAULT_SR * duration)
        white = rng.standard_normal(n_samp).astype(np.float32) * 0.02
        pink = np.cumsum(rng.standard_normal(n_samp).astype(np.float32)) * 0.005
        signal = (white + pink) / (np.max(np.abs(white + pink)) + 1e-9)
        signal = np.clip(signal, -1.0, 1.0)
        signal = pad_or_crop(signal, sr=DEFAULT_SR, target_duration=TARGET_DURATION_SEC)

        feats = extract_mfcc_features(signal, sr=DEFAULT_SR, n_mfcc=n_mfcc)
        row = {"label": "negative", "filepath": f"__noise_{i}"}
        for j, v in enumerate(feats):
            row[f"f_{j}"] = float(v)
        new_rows.append(row)

    extra = pd.DataFrame(new_rows)
    return pd.concat([df, extra], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MFCC CSV'e sentetik gürültü negatifleri ekle."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/csv/mfcc_features.csv",
        help="Mevcut CSV dosyası.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Çıktı CSV (varsayılan: --csv üzerine yaz)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Eklenecek gürültü örnek sayısı.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.output) if args.output else csv_path

    df = pd.read_csv(csv_path)
    df_new = add_noise_rows(df, n_samples=args.n_samples, n_mfcc=None)  # auto-detect
    df_new.to_csv(out_path, index=False)
    print(f"Eklendi: {args.n_samples} gürültü negatifi. Toplam: {len(df_new)} satır -> {out_path}")


if __name__ == "__main__":
    main()
