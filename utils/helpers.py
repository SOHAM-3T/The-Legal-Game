from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DEBATE_DATASET_PATH = PROCESSED_DATA_DIR / "debate_dataset.csv"
CLEAN_DATASET_PATH = PROCESSED_DATA_DIR / "debate_dataset_clean.csv"
TRAINING_PAIRS_PATH = PROCESSED_DATA_DIR / "training_pairs.csv"


EVIDENCE_TYPE_WEIGHTS = {
    "[STUDY]": 1.0,
    "[EXPERT]": 0.9,
    "[ANALYSIS]": 0.85,
    "[STATISTICS]": 0.95,
}


@dataclass
class RetrievedArgument:
    topic: str
    claim: str
    evidence_text: str
    evidence_type: str
    score: float


def ensure_processed_dir() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("â€¦", "...")
    text = text.replace("â€™", "'")
    text = re.sub(r"\[REF[^\]]*\]", "", text)
    text = re.sub(r"\[[A-Z]+\]$", "", text).strip()
    return normalize_whitespace(text)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", clean_text(text).lower())


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize_words(left))
    right_tokens = set(tokenize_words(right))

    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def summarize_evidence(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return clean_text(text)
    return " ".join(sentences[:max_sentences])


def evidence_weight(evidence_type: str) -> float:
    return EVIDENCE_TYPE_WEIGHTS.get(str(evidence_type).strip(), 0.8)


def format_argument(claim: str, evidence_text: str, stance: str) -> str:
    evidence_summary = summarize_evidence(evidence_text)
    lead = "Support" if stance == "support" else "Oppose"
    return (
        f"{lead} the motion because {clean_text(claim)}. "
        f"Evidence: {evidence_summary}"
    )


def load_debate_dataset(prefer_clean: bool = True) -> pd.DataFrame:
    path = CLEAN_DATASET_PATH if prefer_clean and CLEAN_DATASET_PATH.exists() else DEBATE_DATASET_PATH
    df = pd.read_csv(path)

    for column in ("topic", "claim", "evidence_text", "evidence_type"):
        if column in df.columns:
            df[column] = df[column].map(clean_text)

    return df.dropna(subset=["topic", "claim", "evidence_text"]).reset_index(drop=True)


def rank_records(
    topic: str,
    records: pd.DataFrame,
    query_text: str = "",
    exclude_claims: Iterable[str] | None = None,
) -> list[RetrievedArgument]:
    exclude = {clean_text(claim) for claim in (exclude_claims or [])}
    ranked: list[RetrievedArgument] = []

    for row in records.itertuples(index=False):
        claim = clean_text(row.claim)
        if claim in exclude:
            continue

        topic_overlap = lexical_jaccard(topic, row.topic)
        query_overlap = lexical_jaccard(query_text, claim) if query_text else 0.0
        score = (topic_overlap * 0.45) + ((1.0 - query_overlap) * 0.35) + (evidence_weight(row.evidence_type) * 0.20)
        ranked.append(
            RetrievedArgument(
                topic=clean_text(row.topic),
                claim=claim,
                evidence_text=clean_text(row.evidence_text),
                evidence_type=clean_text(row.evidence_type),
                score=round(score, 4),
            )
        )

    return sorted(ranked, key=lambda item: item.score, reverse=True)
