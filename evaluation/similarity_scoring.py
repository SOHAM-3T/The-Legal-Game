from __future__ import annotations

from utils.helpers import lexical_jaccard, tokenize_words


def topical_relevance(topic: str, argument: str) -> float:
    return lexical_jaccard(topic, argument)


def novelty_score(reference_argument: str, candidate_argument: str) -> float:
    overlap = lexical_jaccard(reference_argument, candidate_argument)
    return max(0.0, 1.0 - overlap)


def evidence_coverage(argument: str) -> float:
    tokens = tokenize_words(argument)
    if not tokens:
        return 0.0

    cue_terms = {"evidence", "study", "studies", "research", "report", "expert", "analysis", "data"}
    covered = sum(1 for token in tokens if token in cue_terms)
    return min(1.0, covered / 3.0)
