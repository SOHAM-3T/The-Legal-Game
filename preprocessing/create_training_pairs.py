from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.helpers import (
    TRAINING_PAIRS_PATH,
    clean_text,
    cosine_text_similarity,
    estimate_stance_signal,
    ensure_processed_dir,
    load_debate_dataset,
)


def infer_support_bucket(topic_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in topic_df.itertuples(index=False):
        claim = clean_text(row.claim)
        evidence = clean_text(row.evidence_text)
        similarity_score = cosine_text_similarity(claim, evidence)
        stance_signal = estimate_stance_signal(row.topic, claim, evidence)
        stance = "support" if stance_signal > 0 or (stance_signal == 0 and similarity_score >= 0.2) else "oppose"
        records.append(
            {
                "topic": clean_text(row.topic),
                "claim": claim,
                "evidence_text": evidence,
                "evidence_type": clean_text(row.evidence_type),
                "stance": stance,
                "claim_evidence_similarity": round(similarity_score, 4),
                "stance_signal": round(stance_signal, 4),
                "generator_input": (
                    f"Topic: {clean_text(row.topic)} "
                    f"Stance: {stance} "
                    f"Evidence: {evidence} "
                    f"Generate a {stance} argument."
                ),
                "generator_target": claim,
            }
        )
    return pd.DataFrame.from_records(records)


def create_training_pairs() -> pd.DataFrame:
    df = load_debate_dataset()
    grouped = [infer_support_bucket(topic_df) for _, topic_df in df.groupby("topic", sort=False)]
    pairs = pd.concat(grouped, ignore_index=True)

    ensure_processed_dir()
    pairs.to_csv(TRAINING_PAIRS_PATH, index=False)
    return pairs


if __name__ == "__main__":
    pairs_df = create_training_pairs()
    print(f"Training pairs created: {len(pairs_df)}")
    print(f"Saved to: {TRAINING_PAIRS_PATH}")
