from __future__ import annotations

from evaluation.similarity_scoring import evidence_coverage, novelty_score, topical_relevance
from utils.helpers import split_sentences, tokenize_words


def coherence_score(argument: str) -> float:
    sentences = split_sentences(argument)
    if not sentences:
        return 0.0
    return min(1.0, len(sentences) / 3.0)


def specificity_score(argument: str) -> float:
    tokens = tokenize_words(argument)
    if not tokens:
        return 0.0

    long_tokens = sum(1 for token in tokens if len(token) >= 6)
    return min(1.0, long_tokens / max(8, len(tokens)))


def score_argument(topic: str, argument: str, opponent_argument: str = "") -> dict[str, float]:
    scores = {
        "relevance": round(topical_relevance(topic, argument), 4),
        "coherence": round(coherence_score(argument), 4),
        "specificity": round(specificity_score(argument), 4),
        "evidence": round(evidence_coverage(argument), 4),
    }

    if opponent_argument:
        scores["novelty"] = round(novelty_score(opponent_argument, argument), 4)
    else:
        scores["novelty"] = 0.5

    scores["total"] = round(
        (scores["relevance"] * 0.35)
        + (scores["evidence"] * 0.25)
        + (scores["coherence"] * 0.15)
        + (scores["specificity"] * 0.10)
        + (scores["novelty"] * 0.15),
        4,
    )

    return scores
