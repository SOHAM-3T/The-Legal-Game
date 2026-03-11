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
