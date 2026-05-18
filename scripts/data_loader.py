import os
import pandas as pd

from scripts.config import (
    DATA_PATH,
    MIN_DATA
)


def load_dataset():
    if not os.path.exists(DATA_PATH):
        print("dataset tidak ditemukan:", DATA_PATH)
        return None

    df = pd.read_csv(DATA_PATH)

    required = ["questions", "answers", "scores"]

    if not all(col in df.columns for col in required):
        print("kolom dataset tidak sesuai")
        return None

    df = df.dropna(subset=required)

    df["scores"] = pd.to_numeric(
        df["scores"],
        errors="coerce"
    )

    df = df.dropna(subset=["scores"])

    if len(df) < MIN_DATA:
        print(f"dataset terlalu kecil: {len(df)}")
        return None

    return df