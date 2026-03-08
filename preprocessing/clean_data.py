from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.helpers import CLEAN_DATASET_PATH, DEBATE_DATASET_PATH, clean_text, ensure_processed_dir


def clean_dataset(input_path=DEBATE_DATASET_PATH, output_path=CLEAN_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    for column in ("topic", "claim", "evidence_text", "evidence_type"):
        if column in df.columns:
            df[column] = df[column].map(clean_text)

    df = df.dropna(subset=["topic", "claim", "evidence_text"]).drop_duplicates().reset_index(drop=True)
    ensure_processed_dir()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned_df = clean_dataset()
    print(f"Cleaned dataset rows: {len(cleaned_df)}")
    print(f"Saved cleaned dataset to: {CLEAN_DATASET_PATH}")
