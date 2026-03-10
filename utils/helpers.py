from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

RESTRICT_POLICY_CUES = {
    "ban", "banned", "prohibit", "prohibited", "abolish", "criminalize", "criminalise",
    "restrict", "restricted", "limit", "limited", "outlaw",
}
ENABLE_POLICY_CUES = {
    "support", "permit", "allow", "allowed", "legalize", "legalise", "compulsory",
    "subsidize", "subsidise", "reintroduce", "require", "required",
}
RESTRICT_SUPPORT_CUES = {
    "harm", "harmful", "risk", "danger", "dangerous", "aggression",
    "abuse", "injury", "damage", "threat", "addiction", "increase", "increases", "increased",
    "correlates", "negative", "crime", "criminal",
}
RESTRICT_OPPOSE_CUES = {
    "safe", "benefit", "beneficial", "help", "helps", "skills", "prosocial", "outlet",
    "positive", "unrelated", "unfair", "invalid", "rights", "freedom",
}
ENABLE_SUPPORT_CUES = {
    "benefit", "beneficial", "improve", "improves", "effective", "rights", "freedom",
    "growth", "protect", "supports", "necessary", "increase", "health", "education",
}
ENABLE_OPPOSE_CUES = {
    "harm", "harmful", "risk", "danger", "cost", "costly", "burden", "burdens",
    "unfair", "abuse", "damage", "ineffective", "invalid", "restrict",
}
NEGATION_CUES = {"not", "no", "never", "none", "without"}


@dataclass
class RetrievedArgument:
    topic: str
    claim: str
    evidence_text: str
    evidence_type: str
    score: float


@dataclass
class SimilarityIndex:
    vectorizer: TfidfVectorizer
    topic_matrix: object
    claim_matrix: object


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


def count_matching_terms(tokens: Iterable[str], lexicon: set[str]) -> int:
    return sum(1 for token in tokens if token in lexicon)


def infer_policy_direction(topic: str) -> str:
    tokens = tokenize_words(topic)
    restrict_hits = count_matching_terms(tokens, RESTRICT_POLICY_CUES)
    enable_hits = count_matching_terms(tokens, ENABLE_POLICY_CUES)
    return "restrict" if restrict_hits >= enable_hits else "enable"


def estimate_stance_signal(topic: str, claim: str, evidence_text: str = "") -> float:
    direction = infer_policy_direction(topic)
    claim_text = clean_text(claim).lower()
    evidence_text_clean = clean_text(evidence_text).lower()
    claim_tokens = tokenize_words(claim)
    evidence_tokens = tokenize_words(evidence_text)

    if direction == "restrict":
        hard_oppose_phrases = (
            "not related",
            "no conclusive link",
            "prosocial effect",
            "help students",
            "positively and not negatively",
            "practical and intellectual skills",
            "safe outlet",
        )
        if any(phrase in claim_text for phrase in hard_oppose_phrases):
            return -6.0
        if any(phrase in evidence_text_clean for phrase in hard_oppose_phrases):
            return -4.0

        support_cues = RESTRICT_SUPPORT_CUES
        oppose_cues = RESTRICT_OPPOSE_CUES
    else:
        support_cues = ENABLE_SUPPORT_CUES
        oppose_cues = ENABLE_OPPOSE_CUES

    claim_signal = count_matching_terms(claim_tokens, support_cues) - count_matching_terms(claim_tokens, oppose_cues)
    evidence_signal = count_matching_terms(evidence_tokens, support_cues) - count_matching_terms(evidence_tokens, oppose_cues)

    negation_penalty = 0
    if any(token in NEGATION_CUES for token in claim_tokens):
        negation_penalty += 2
    if "not related" in claim_text:
        negation_penalty += 6
    if "no conclusive link" in evidence_text_clean:
        negation_penalty += 4

    return float((claim_signal * 1.5) + evidence_signal - negation_penalty)


def fit_tfidf_vectorizer(texts: Iterable[str]) -> TfidfVectorizer:
    cleaned_texts = [clean_text(text) for text in texts if clean_text(text)]
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(cleaned_texts or ["empty"])


def cosine_text_similarity(left: str, right: str, vectorizer: TfidfVectorizer | None = None) -> float:
    left_clean = clean_text(left)
    right_clean = clean_text(right)

    if not left_clean or not right_clean:
        return 0.0

    active_vectorizer = vectorizer or fit_tfidf_vectorizer([left_clean, right_clean])
    matrix = active_vectorizer.transform([left_clean, right_clean])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def build_similarity_index(records: pd.DataFrame) -> SimilarityIndex:
    vectorizer = fit_tfidf_vectorizer(
        list(records["topic"].fillna("")) + list(records["claim"].fillna("")) + list(records["evidence_text"].fillna(""))
    )
    topic_matrix = vectorizer.transform(records["topic"].map(clean_text))
    claim_matrix = vectorizer.transform(records["claim"].map(clean_text))
    return SimilarityIndex(vectorizer=vectorizer, topic_matrix=topic_matrix, claim_matrix=claim_matrix)


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


def load_training_pairs() -> pd.DataFrame:
    if not TRAINING_PAIRS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(TRAINING_PAIRS_PATH)
    for column in ("topic", "claim", "evidence_text", "evidence_type", "stance"):
        if column in df.columns:
            df[column] = df[column].map(clean_text)

    required = ["topic", "claim", "evidence_text"]
    return df.dropna(subset=required).reset_index(drop=True)


def rank_records(
    topic: str,
    records: pd.DataFrame,
    query_text: str = "",
    exclude_claims: Iterable[str] | None = None,
    similarity_index: SimilarityIndex | None = None,
) -> list[RetrievedArgument]:
    exclude = {clean_text(claim) for claim in (exclude_claims or [])}
    ranked: list[RetrievedArgument] = []
    index = similarity_index or build_similarity_index(records)
    topic_vector = index.vectorizer.transform([clean_text(topic)])
    query_vector = index.vectorizer.transform([clean_text(query_text)]) if clean_text(query_text) else None
    topic_scores = cosine_similarity(topic_vector, index.topic_matrix)[0]
    query_scores = cosine_similarity(query_vector, index.claim_matrix)[0] if query_vector is not None else None

    for row_idx, row in enumerate(records.itertuples(index=False)):
        claim = clean_text(row.claim)
        if claim in exclude:
            continue

        topic_overlap = float(topic_scores[row_idx])
        query_overlap = float(query_scores[row_idx]) if query_scores is not None else 0.0
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
