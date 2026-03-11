from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from debate_engine.debate_loop import run_debate
from utils.helpers import load_debate_dataset, load_training_pairs


st.set_page_config(
    page_title="The Legal Game",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(163, 132, 74, 0.14), transparent 28%),
            linear-gradient(180deg, #f6f2ea 0%, #f3efe7 48%, #ece6db 100%);
        color: #18202b;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .hero-card, .panel-card, .winner-card, .metric-card {
        border-radius: 22px;
        border: 1px solid rgba(24, 32, 43, 0.09);
        background: rgba(255, 252, 247, 0.86);
        box-shadow: 0 18px 45px rgba(35, 40, 48, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-card {
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
    }
    .hero-eyebrow {
        font-size: 0.86rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8d6a2f;
        font-weight: 700;
    }
    .hero-title {
        font-size: 3rem;
        line-height: 1;
        margin: 0.4rem 0 0.85rem 0;
        font-weight: 800;
        color: #18202b;
    }
    .hero-copy {
        font-size: 1.06rem;
        line-height: 1.75;
        color: #334155;
        max-width: 52rem;
    }
    .hero-strip {
        margin-top: 1.35rem;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.9rem;
    }
    .role-chip {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(250,245,238,0.96), rgba(244,237,227,0.9));
        border: 1px solid rgba(141, 106, 47, 0.18);
    }
    .role-chip strong {
        display: block;
        color: #18202b;
        margin-bottom: 0.25rem;
    }
    .section-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        color: #8d6a2f;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .panel-card {
        padding: 1.35rem 1.4rem 1.45rem;
        margin-bottom: 1.2rem;
    }
    .winner-card {
        padding: 1.35rem 1.4rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(30,58,95,0.96), rgba(82, 104, 132, 0.92));
        color: #f8fafc;
        border: none;
    }
    .winner-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.85;
        margin-bottom: 0.45rem;
        font-weight: 700;
    }
    .winner-name {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    .winner-sub {
        opacity: 0.92;
        font-size: 1rem;
    }
    .metric-card {
        padding: 1rem 1.05rem;
        min-height: 118px;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8d6a2f;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.9rem;
        line-height: 1.1;
        font-weight: 800;
        color: #18202b;
        margin: 0.35rem 0 0.25rem;
    }
    .metric-copy {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .round-score {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 1rem;
        font-size: 0.95rem;
        color: #334155;
    }
    .score-pill {
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        background: rgba(141, 106, 47, 0.08);
        border: 1px solid rgba(141, 106, 47, 0.12);
    }
    .topic-note {
        padding: 0.8rem 0.95rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.04);
        color: #334155;
        line-height: 1.6;
        margin-top: 0.75rem;
    }
    @media (max-width: 900px) {
        .hero-strip {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.35rem;
        }
    }
</style>
"""


def render_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_frontend_dataset() -> tuple[pd.DataFrame, str]:
    pairs = load_training_pairs()
    if not pairs.empty:
        if "stance" in pairs.columns:
            support_df = pairs[pairs["stance"] == "support"].reset_index(drop=True)
            if not support_df.empty:
                return support_df, "training_pairs"
        return pairs.reset_index(drop=True), "training_pairs"

    dataset = load_debate_dataset()
    return dataset.reset_index(drop=True), "debate_dataset"


@st.cache_data(show_spinner=False)
def load_judge_metadata() -> dict:
    metadata_path = BASE_DIR / "models" / "judge_model" / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def dataset_status() -> tuple[bool, str]:
    processed_files = [
        BASE_DIR / "data" / "processed" / "debate_dataset.csv",
        BASE_DIR / "data" / "processed" / "debate_dataset_clean.csv",
        BASE_DIR / "data" / "processed" / "training_pairs.csv",
    ]
    if any(path.exists() for path in processed_files):
        return True, "Processed data available"
    return False, "Processed data missing. Run preprocessing before using the frontend."


def model_status() -> list[str]:
    notes = []
    prosecutor_path = BASE_DIR / "models" / "argument_generator"
    judge_path = BASE_DIR / "models" / "judge_model" / "judge_model.joblib"

    if not prosecutor_path.exists():
        notes.append("Prosecutor model not found. The app will use the template-based fallback.")
    if not judge_path.exists():
        notes.append("Judge model metadata/model not found. The heuristic judge will be used.")
    return notes


def resolve_sample(df: pd.DataFrame, selected_topic: str) -> tuple[str, str]:
    if df.empty:
        return "", ""
    matches = df[df["topic"] == selected_topic]
    if matches.empty:
        row = df.iloc[0]
    else:
        row = matches.iloc[0]
    return str(row["topic"]), str(row["evidence_text"])


def render_hero() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Legal NLP Demo</div>
            <div class="hero-title">The Legal Game</div>
            <div class="hero-copy">
                A multi-agent legal debate simulator where a prosecutor advances a motion,
                a defense agent rebuts it with opposing evidence, and a judge scores the exchange.
                This interface is tuned for live demos: select a motion, launch the debate,
                and inspect each round in a clear courtroom-style presentation.
            </div>
            <div class="hero-strip">
                <div class="role-chip">
                    <strong>Prosecutor</strong>
                    Generates a support-side argument using the current topic, evidence, and the trained generator when available.
                </div>
                <div class="role-chip">
                    <strong>Defense</strong>
                    Retrieves oppose-side evidence and composes a rebuttal grounded in the debate corpus.
                </div>
                <div class="role-chip">
                    <strong>Judge</strong>
                    Scores both sides on relevance, evidence use, coherence, specificity, and novelty to declare a winner.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame, source_name: str) -> None:
    metadata = load_judge_metadata()

    st.sidebar.markdown("## Debate Control")
    st.sidebar.caption("Demo-first controls and project health summary")

    st.sidebar.markdown("### Corpus Snapshot")
    st.sidebar.metric("Rows", f"{len(df):,}")
    st.sidebar.metric("Topics", f"{df['topic'].nunique():,}")
    st.sidebar.metric("Source", source_name.replace("_", " ").title())

    if "stance" in df.columns:
        support_rows = int((df["stance"] == "support").sum())
        oppose_rows = int((df["stance"] == "oppose").sum())
        st.sidebar.metric("Support / Oppose", f"{support_rows} / {oppose_rows}")

    if metadata:
        st.sidebar.markdown("### Judge Model")
        st.sidebar.metric("Validation Accuracy", f"{metadata.get('validation_accuracy', 0) * 100:.2f}%")
        st.sidebar.metric("Training Rows", f"{metadata.get('training_rows', 0):,}")

    st.sidebar.markdown("### Judge Logic")
    st.sidebar.caption(
        "Winner selection combines relevance, evidence usage, coherence, specificity, and novelty."
    )


def render_round(round_data: dict) -> None:
    with st.expander(f"Round {round_data['round_index']} | Winner: {round_data['winner']}", expanded=True):
        st.markdown(
            f"""
            <div class="round-score">
                <div class="score-pill"><strong>Prosecutor Score</strong> {round_data['prosecutor_score']:.4f}</div>
                <div class="score-pill"><strong>Defense Score</strong> {round_data['defense_score']:.4f}</div>
                <div class="score-pill"><strong>Round Winner</strong> {round_data['winner']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Prosecutor")
            st.write(round_data["prosecutor_argument"])
            st.progress(max(0.0, min(1.0, round_data["prosecutor_score"])))
        with right:
            st.markdown("#### Defense")
            st.write(round_data["defense_argument"])
            st.progress(max(0.0, min(1.0, round_data["defense_score"])))


def main() -> None:
    render_hero()

    data_ready, data_message = dataset_status()
    if not data_ready:
        st.error(data_message)
        st.stop()

    for note in model_status():
        st.warning(note)

    df, source_name = load_frontend_dataset()
    if df.empty:
        st.error("No rows are available in the processed dataset. Re-run preprocessing before launching the demo.")
        st.stop()

    render_sidebar(df, source_name)

    st.markdown('<div class="section-label">Debate Setup</div>', unsafe_allow_html=True)

    topics = sorted(df["topic"].dropna().unique().tolist())
    default_topic = topics[0]

    select_col, button_col = st.columns([4, 1], gap="medium")
    with select_col:
        selected_topic = st.selectbox("Sample topic", topics, index=0 if topics else None, label_visibility="collapsed")
    with button_col:
        use_sample = st.button("Use sample", use_container_width=True)

    sample_topic, sample_evidence = resolve_sample(df, selected_topic or default_topic)

    topic_default = sample_topic if use_sample or "topic_input" not in st.session_state else st.session_state["topic_input"]
    evidence_default = sample_evidence if use_sample or "evidence_input" not in st.session_state else st.session_state["evidence_input"]

    topic_input = st.text_input("Debate topic", value=topic_default, key="topic_input")
    evidence_input = st.text_area(
        "Evidence",
        value=evidence_default,
        height=140,
        key="evidence_input",
        help="You can use the sample evidence or replace it with your own supporting evidence.",
    )

    controls_left, controls_right = st.columns([1, 1], gap="large")
    with controls_left:
        rounds = st.slider("Number of rounds", min_value=1, max_value=4, value=2)
    with controls_right:
        st.markdown(
            """
            <div class="topic-note">
                <strong>How to use this demo</strong><br>
                Pick a sample motion, optionally edit the evidence, and run the debate.
                The backend will preserve the current legal-debate pipeline and render the result here.
            </div>
            """,
            unsafe_allow_html=True,
        )

    run_clicked = st.button("Run Debate", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Ready for a live demo. Select a topic and click `Run Debate`.")
        return

    if not topic_input.strip():
        st.error("Enter a debate topic before running the app.")
        return

    active_evidence = evidence_input.strip() or sample_evidence
    if not active_evidence:
        st.error("Evidence is required to launch the debate.")
        return

    with st.spinner("Running prosecutor, defense, and judge agents..."):
        debate = run_debate(topic=topic_input.strip(), evidence=active_evidence, rounds=rounds)

    st.markdown('<div class="section-label">Debate Outcome</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-title">Overall Winner</div>
            <div class="winner-name">{debate['overall_winner']}</div>
            <div class="winner-sub">
                Prosecutor round wins: {debate['scoreboard']['prosecutor_round_wins']}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Defense round wins: {debate['scoreboard']['defense_round_wins']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_col1, card_col2, card_col3 = st.columns(3, gap="large")
    with card_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Selected Topic</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-copy">{debate["topic"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with card_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Input Evidence</div>
                <div class="metric-copy">{debate['input_evidence']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_col3:
        render_card(
            "Judge Notes",
            "Scores combine topical relevance, evidence cues, specificity, coherence, and novelty across rounds.",
        )

    st.markdown('<div class="section-label">Round Breakdown</div>', unsafe_allow_html=True)
    for round_data in debate["rounds_data"]:
        render_round(round_data)

    st.markdown('<div class="section-label">Model Basis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
            <strong>Current demo stack</strong><br>
            Prosecutor generation uses the fine-tuned FLAN-T5 prosecutor when present, the defense uses
            retrieval-backed counter-argument selection, and the judge combines heuristic scoring with an
            optional weakly trained classifier. This interface is a presentation wrapper around the same
            backend used by the CLI pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
