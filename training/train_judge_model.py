from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.argument_quality import score_argument
from preprocessing.create_training_pairs import create_training_pairs

MODEL_DIR = BASE_DIR / "models" / "judge_model"


def build_training_frame() -> pd.DataFrame:
    pairs = create_training_pairs()
    support = pairs[pairs["stance"] == "support"].reset_index(drop=True)
    oppose = pairs[pairs["stance"] == "oppose"].reset_index(drop=True)

    if support.empty or oppose.empty:
        raise ValueError("Judge training requires both support and oppose rows in training_pairs.csv.")

    rows = []
    limit = min(len(support), len(oppose))

    for idx in range(limit):
        pro = support.iloc[idx]
        con = oppose.iloc[idx]

        prosecutor_argument = f"Support the motion because {pro['claim']}. Evidence: {pro['evidence_text']}"
        defense_argument = f"Oppose the motion because {con['claim']}. Evidence: {con['evidence_text']}"

        prosecutor_scores = score_argument(pro["topic"], prosecutor_argument, defense_argument)
        defense_scores = score_argument(pro["topic"], defense_argument, prosecutor_argument)

        rows.append(
            {
                "topic": pro["topic"],
                "pro_total": prosecutor_scores["total"],
                "def_total": defense_scores["total"],
                "pro_relevance": prosecutor_scores["relevance"],
                "def_relevance": defense_scores["relevance"],
                "pro_evidence": prosecutor_scores["evidence"],
                "def_evidence": defense_scores["evidence"],
                "label": int(prosecutor_scores["total"] >= defense_scores["total"]),
            }
        )

    return pd.DataFrame(rows)


def train_judge_model() -> Path:
    df = build_training_frame()
    features = df.drop(columns=["topic", "label"])
    labels = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "judge_model.joblib")

    metadata = {
        "features": list(features.columns),
        "validation_accuracy": round(float(accuracy), 4),
        "training_rows": int(len(df)),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return MODEL_DIR


if __name__ == "__main__":
    model_path = train_judge_model()
    print(f"Saved judge model to: {model_path}")
