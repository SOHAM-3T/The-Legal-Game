from pathlib import Path
import json
import joblib
from evaluation.argument_quality import score_argument

class JudgeAgent:
    def __init__(self):
        model_path = Path("models/judge_model/judge_model.joblib")
        metadata_path = Path("models/judge_model/metadata.json")
        self.model = joblib.load(model_path) if model_path.exists() else None
        self.feature_names = []

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.feature_names = list(metadata.get("features", []))

    def evaluate(self, topic, prosecutor_argument, defense_argument):
        prosecutor_scores = score_argument(topic, prosecutor_argument, defense_argument)
        defense_scores = score_argument(topic, defense_argument, prosecutor_argument)

        prosecutor_total = prosecutor_scores["total"]
        defense_total = defense_scores["total"]

        if self.model and self.feature_names:
            import pandas as pd

            feature_row = pd.DataFrame([{
                "pro_total": prosecutor_total,
                "def_total": defense_total,
                "pro_relevance": prosecutor_scores["relevance"],
                "def_relevance": defense_scores["relevance"],
                "pro_evidence": prosecutor_scores["evidence"],
                "def_evidence": defense_scores["evidence"],
            }]).reindex(columns=self.feature_names, fill_value=0.0)
            prediction = int(self.model.predict(feature_row)[0])
            winner = "Prosecutor" if prediction == 1 else "Defense"
        else:
            if prosecutor_total > defense_total:
                winner = "Prosecutor"
            elif defense_total > prosecutor_total:
                winner = "Defense"
            else:
                winner = "Tie"

        return {
            "prosecutor": prosecutor_scores,
            "defense": defense_scores,
            "winner": winner,
        }
