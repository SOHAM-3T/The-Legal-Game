from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from debate_engine.debate_loop import run_debate
from utils.helpers import load_debate_dataset, load_training_pairs


def resolve_example(topic_query: str = "") -> tuple[str, str]:
    df = load_training_pairs()
    if not df.empty and "stance" in df.columns:
        df = df[df["stance"] == "support"].reset_index(drop=True)
    if df.empty:
        df = load_debate_dataset()
    if topic_query:
        mask = df["topic"].str.contains(topic_query, case=False, regex=False)
        if mask.any():
            row = df[mask].iloc[0]
            return row["topic"], row["evidence_text"]

    if df.empty:
        raise ValueError("No processed dataset available. Run preprocessing/load_data.py and preprocessing/create_training_pairs.py first.")

    row = df.iloc[0]
    return row["topic"], row["evidence_text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a retrieval-heavy legal debate simulation.")
    parser.add_argument("--topic", default="", help="Full topic or partial topic text.")
    parser.add_argument("--evidence", default="", help="Override evidence text.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds.")
    args = parser.parse_args()

    topic, default_evidence = resolve_example(args.topic)
    evidence = args.evidence or default_evidence

    debate = run_debate(topic=topic, evidence=evidence, rounds=max(1, args.rounds))

    print(f"Topic: {debate['topic']}")
    print(f"Overall winner: {debate['overall_winner']}")
    print(
        "Round wins: "
        f"Prosecutor={debate['scoreboard']['prosecutor_round_wins']}, "
        f"Defense={debate['scoreboard']['defense_round_wins']}"
    )

    for round_data in debate["rounds"]:
        print(f"\nRound {round_data.round_index}")
        print(f"Prosecutor Argument: {round_data.prosecutor_argument}")
        print(f"Defense Argument: {round_data.defense_argument}")
        print(f"Winner: {round_data.winner}")
        print(
            f"Scores -> Prosecutor={round_data.prosecutor_score:.4f}, "
            f"Defense={round_data.defense_score:.4f}"
        )


if __name__ == "__main__":
    main()
