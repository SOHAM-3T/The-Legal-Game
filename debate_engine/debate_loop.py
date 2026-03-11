from __future__ import annotations

from agents.defense import DefenseAgent
from agents.judge import JudgeAgent
from agents.prosecutor import ProsecutorAgent
from debate_engine.debate_round import DebateRound
from utils.helpers import clean_text


def serialize_round(round_data: DebateRound) -> dict:
    return {
        "round_index": round_data.round_index,
        "prosecutor_argument": round_data.prosecutor_argument,
        "defense_argument": round_data.defense_argument,
        "winner": round_data.winner,
        "prosecutor_score": round_data.prosecutor_score,
        "defense_score": round_data.defense_score,
    }


def run_debate(topic, evidence, rounds=2):
    prosecutor = ProsecutorAgent()
    defense = DefenseAgent()
    judge = JudgeAgent()

    debate_rounds = []
    prior_defense_argument = ""
    used_prosecutor_claims = set()
    used_defense_claims = set()
    selected_input_evidence = clean_text(evidence)

    for round_index in range(1, rounds + 1):
        prosecutor_result = prosecutor.generate_argument(
            topic,
            evidence=evidence,
            rebuttal_context=prior_defense_argument,
            exclude_claims=used_prosecutor_claims,
        )
        prosecutor_argument = prosecutor_result["argument"]
        used_prosecutor_claims.add(prosecutor_result["claim"])
        selected_input_evidence = prosecutor_result["evidence_text"] or selected_input_evidence

        defense_result = defense.generate_counter_argument(
            topic,
            prosecutor_argument,
            exclude_claims=used_defense_claims | used_prosecutor_claims,
        )
        defense_argument = defense_result["argument"]
        used_defense_claims.add(defense_result["claim"])

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
        "input_evidence": clean_text(selected_input_evidence),
        "rounds": debate_rounds,
        "rounds_data": [serialize_round(round_data) for round_data in debate_rounds],
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
