from __future__ import annotations

from agents.defense import DefenseAgent
from agents.judge import JudgeAgent
from agents.prosecutor import ProsecutorAgent
from debate_engine.debate_round import DebateRound
from utils.helpers import clean_text, summarize_evidence


def build_rebuttal_evidence(base_evidence: str, prior_argument: str, round_index: int) -> str:
    summary = summarize_evidence(base_evidence, max_sentences=1)
    prior_summary = clean_text(prior_argument)
    return (
        f"{summary} Round {round_index} focus: respond directly to this prior argument: "
        f"{prior_summary}"
    )


def run_debate(topic, evidence, rounds=2):
    prosecutor = ProsecutorAgent()
    defense = DefenseAgent()
    judge = JudgeAgent()

    debate_rounds = []
    current_evidence = evidence
    prior_defense_argument = ""

    for round_index in range(1, rounds + 1):
        if prior_defense_argument:
            current_evidence = build_rebuttal_evidence(evidence, prior_defense_argument, round_index)

        prosecutor_argument = prosecutor.generate_argument(topic, current_evidence)
        defense_argument = defense.generate_counter_argument(topic, prosecutor_argument)

        result = judge.evaluate(topic, prosecutor_argument, defense_argument)
        debate_rounds.append(
            DebateRound(
                round_index=round_index,
                prosecutor_argument=prosecutor_argument,
                defense_argument=defense_argument,
                winner=result["winner"],
                prosecutor_score=result["prosecutor"]["total"],
                defense_score=result["defense"]["total"],
            )
        )

        prior_defense_argument = defense_argument

    prosecutor_wins = sum(1 for round in debate_rounds if round.winner == "Prosecutor")
    defense_wins = sum(1 for round in debate_rounds if round.winner == "Defense")

    if prosecutor_wins > defense_wins:
        overall_winner = "Prosecutor"
    elif defense_wins > prosecutor_wins:
        overall_winner = "Defense"
    else:
        overall_winner = "Tie"

    return {
        "topic": clean_text(topic),
        "rounds": debate_rounds,
        "overall_winner": overall_winner,
        "scoreboard": {
            "prosecutor_round_wins": prosecutor_wins,
            "defense_round_wins": defense_wins,
        },
    }


if __name__ == "__main__":
    debate = run_debate(
        topic="violent video games should be banned",
        evidence="Studies show violent games increase aggression in youth populations.",
        rounds=2,
    )

    print(f"Topic: {debate['topic']}")
    for round_data in debate["rounds"]:
        print(f"\nRound {round_data.round_index}")
        print(f"Prosecutor: {round_data.prosecutor_argument}")
        print(f"Defense: {round_data.defense_argument}")
        print(f"Winner: {round_data.winner}")
        print(
            f"Scores -> Prosecutor: {round_data.prosecutor_score:.4f}, "
            f"Defense: {round_data.defense_score:.4f}"
        )
    print(f"\nOverall winner: {debate['overall_winner']}")
