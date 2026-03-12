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
            radial-gradient(circle at 12% 0%, rgba(175, 118, 31, 0.18), transparent 22%),
            radial-gradient(circle at 100% 20%, rgba(28, 47, 76, 0.10), transparent 25%),
            linear-gradient(180deg, #f3ede2 0%, #ece4d6 52%, #e4dbc9 100%);
        color: #17212b;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 {
        color: #17212b;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #152033 0%, #1c2a41 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f7f5ef;
    }
    .hero-card, .surface-card, .winner-card, .summary-card, .round-card {
        border-radius: 24px;
        border: 1px solid rgba(23, 33, 43, 0.08);
        box-shadow: 0 18px 48px rgba(32, 27, 14, 0.08);
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(255,250,243,0.96), rgba(246,240,230,0.96));
        padding: 1.6rem 1.8rem 1.4rem;
        margin-bottom: 1.25rem;
    }
    .hero-eyebrow {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #966a1d;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-size: 2.9rem;
        line-height: 0.98;
        margin: 0;
        font-weight: 900;
        color: #17212b;
    }
    .hero-copy {
        margin-top: 0.8rem;
        font-size: 1.01rem;
        line-height: 1.7;
        color: #4a5563;
        max-width: 53rem;
    }
    .hero-grid {
        margin-top: 1.1rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
    }
    .role-panel {
        border-radius: 18px;
        border: 1px solid rgba(150, 106, 29, 0.18);
        background: rgba(255, 250, 242, 0.78);
        padding: 1rem 1rem 0.95rem;
    }
    .role-title {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.28rem;
        color: #17212b;
    }
    .role-copy {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #4a5563;
    }
    .section-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #966a1d;
        font-weight: 800;
        margin: 0.55rem 0 0.7rem;
    }
    .surface-card {
        background: rgba(255, 252, 247, 0.88);
        padding: 1.15rem;
        margin-bottom: 1rem;
    }
    .field-note {
        font-size: 0.92rem;
        line-height: 1.55;
        color: #566172;
        margin-top: -0.15rem;
        margin-bottom: 0.75rem;
    }
    .winner-card {
        background: linear-gradient(135deg, #162335 0%, #253851 100%);
        color: #f8fafc;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1rem;
        border: none;
    }
    .winner-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 800;
        opacity: 0.82;
    }
    .winner-name {
        font-size: 2.15rem;
        line-height: 1.05;
        font-weight: 900;
        margin-top: 0.35rem;
    }
    .winner-copy {
        margin-top: 0.55rem;
        font-size: 0.98rem;
        opacity: 0.92;
    }
    .summary-card {
        background: rgba(255, 252, 247, 0.88);
        padding: 1rem 1.05rem;
        min-height: 132px;
    }
    .summary-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #966a1d;
        font-weight: 800;
    }
    .summary-value {
        margin-top: 0.35rem;
        font-size: 1.8rem;
        line-height: 1.05;
        font-weight: 900;
        color: #17212b;
    }
    .summary-copy {
        margin-top: 0.4rem;
        color: #4a5563;
        font-size: 0.95rem;
        line-height: 1.55;
        word-break: break-word;
    }
    .round-shell {
        margin-bottom: 0.8rem;
    }
    .score-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-bottom: 0.9rem;
    }
    .score-badge {
        border-radius: 999px;
        padding: 0.48rem 0.82rem;
        border: 1px solid rgba(150, 106, 29, 0.14);
        background: rgba(150, 106, 29, 0.08);
        font-size: 0.92rem;
        color: #35404f;
    }
    .mini-note {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(22,35,53,0.95), rgba(35,56,81,0.95));
        color: #f7f8fb;
        line-height: 1.6;
    }
    .mini-note strong {
        display: block;
        margin-bottom: 0.25rem;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label {
        font-weight: 700 !important;
        color: #17212b !important;
    }
    .stButton > button {
        border-radius: 14px;
        min-height: 3rem;
        font-weight: 800;
        border: 1px solid rgba(23, 33, 43, 0.08);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #162335 0%, #22354d 100%);
        color: #f8fafc;
        border: none;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1d3550 0%, #b07a26 100%);
    }
    @media (max-width: 900px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.25rem;
        }
    }
</style>
"""


def render_metric_card(title: str, value: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{title}</div>
            <div class="summary-value">{value}</div>
            <div class="summary-copy">{body}</div>
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


def initialize_form_state(topics: list[str], df: pd.DataFrame) -> None:
    if "selected_topic" not in st.session_state:
        st.session_state["selected_topic"] = topics[0]
    if "last_selected_topic" not in st.session_state:
        st.session_state["last_selected_topic"] = st.session_state["selected_topic"]
    if "topic_input" not in st.session_state or "evidence_input" not in st.session_state:
        sample_topic, sample_evidence = resolve_sample(df, st.session_state["selected_topic"])
        st.session_state["topic_input"] = sample_topic
        st.session_state["evidence_input"] = sample_evidence


def sync_selected_sample(df: pd.DataFrame) -> None:
    selected_topic = st.session_state["selected_topic"]
    sample_topic, sample_evidence = resolve_sample(df, selected_topic)
    st.session_state["topic_input"] = sample_topic
    st.session_state["evidence_input"] = sample_evidence
    st.session_state["last_selected_topic"] = selected_topic


def render_hero() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Legal NLP Presentation Demo</div>
            <div class="hero-title">The Legal Game</div>
            <div class="hero-copy">
                Run a legal-style AI debate around a motion, inspect how each side argues, and review
                round-by-round judging in a format built for presentations and live demos.
            </div>
            <div class="hero-grid">
                <div class="role-panel">
                    <div class="role-title">Prosecutor</div>
                    <div class="role-copy">Builds a support-side case from the selected motion and supporting evidence.</div>
                </div>
                <div class="role-panel">
                    <div class="role-title">Defense</div>
                    <div class="role-copy">Retrieves opposing evidence and composes a structured counter-argument.</div>
                </div>
                <div class="role-panel">
                    <div class="role-title">Judge</div>
                    <div class="role-copy">Scores relevance, evidence use, coherence, specificity, and novelty to decide each round.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame, source_name: str) -> None:
    metadata = load_judge_metadata()

    st.sidebar.markdown("## Demo Snapshot")
    st.sidebar.caption("Current corpus and judge metadata")

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

    st.sidebar.markdown("### Decision Logic")
    st.sidebar.caption("Displayed totals and winner use the same heuristic score path.")


def render_round(round_data: dict) -> None:
    with st.expander(f"Round {round_data['round_index']} | Winner: {round_data['winner']}", expanded=True):
        st.markdown('<div class="round-shell">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="score-strip">
                <div class="score-badge"><strong>Prosecutor</strong> {round_data['prosecutor_score']:.4f}</div>
                <div class="score-badge"><strong>Defense</strong> {round_data['defense_score']:.4f}</div>
                <div class="score-badge"><strong>Winner</strong> {round_data['winner']}</div>
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
        st.markdown("</div>", unsafe_allow_html=True)


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

    topics = sorted(df["topic"].dropna().unique().tolist())
    initialize_form_state(topics, df)

    st.markdown('<div class="section-label">Debate Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    with left_col:
        st.selectbox(
            "Sample topic",
            topics,
            key="selected_topic",
            on_change=sync_selected_sample,
            args=(df,),
            help="Selecting a sample topic immediately updates the debate topic and matching evidence.",
        )
        st.caption("Sample selection auto-fills the active debate topic. You can still edit the topic manually before running.")
        topic_input = st.text_area(
            "Debate topic",
            key="topic_input",
            height=110,
            help="Edit the motion if you want to refine the selected sample before running the debate.",
        )

    with right_col:
        evidence_input = st.text_area(
            "Supporting evidence",
            key="evidence_input",
            height=185,
            help="This field is auto-filled from the selected sample topic and can be edited manually.",
        )
        controls_left, controls_right = st.columns([1, 1], gap="medium")
        with controls_left:
            rounds = st.slider("Rounds", min_value=1, max_value=4, value=2)
        with controls_right:
            st.markdown(
                """
                <div class="mini-note">
                    <strong>How this works</strong>
                    Pick a sample motion, optionally edit the topic or evidence, and then run the debate.
                </div>
                """,
                unsafe_allow_html=True,
            )
        run_clicked = st.button("Run Debate", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not run_clicked:
        st.info("Select a sample topic or edit the inputs manually, then click `Run Debate`.")
        return

    if not topic_input.strip():
        st.error("Enter a debate topic before running the app.")
        return

    if not evidence_input.strip():
        st.error("Evidence is required to launch the debate.")
        return

    with st.spinner("Running prosecutor, defense, and judge agents..."):
        debate = run_debate(topic=topic_input.strip(), evidence=evidence_input.strip(), rounds=rounds)

    st.markdown('<div class="section-label">Debate Outcome</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="winner-card">
            <div class="winner-label">Overall Winner</div>
            <div class="winner-name">{debate['overall_winner']}</div>
            <div class="winner-copy">
                Prosecutor round wins: {debate['scoreboard']['prosecutor_round_wins']}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Defense round wins: {debate['scoreboard']['defense_round_wins']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4, gap="large")
    with summary_col1:
        render_metric_card("Winner", debate["overall_winner"], "Final outcome across all completed rounds.")
    with summary_col2:
        render_metric_card(
            "Round Split",
            f"{debate['scoreboard']['prosecutor_round_wins']} / {debate['scoreboard']['defense_round_wins']}",
            "Displayed as prosecutor wins versus defense wins.",
        )
    with summary_col3:
        render_metric_card("Rounds", str(len(debate["rounds_data"])), "Total rounds executed in this debate run.")
    with summary_col4:
        render_metric_card("Judge Basis", "Heuristic", "Winner matches the displayed total scores in each round.")

    detail_col1, detail_col2 = st.columns([1.05, 0.95], gap="large")
    with detail_col1:
        render_metric_card("Selected Topic", debate["topic"], "The active motion sent to the debate engine.")
    with detail_col2:
        render_metric_card("Input Evidence", debate["input_evidence"], "The supporting evidence used to initialize the prosecutor.")

    st.markdown('<div class="section-label">Round Breakdown</div>', unsafe_allow_html=True)
    for round_data in debate["rounds_data"]:
        render_round(round_data)

    st.markdown('<div class="section-label">Model Basis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="surface-card">
            <strong>Current demo stack</strong><br>
            Prosecutor generation uses the fine-tuned FLAN-T5 prosecutor when present, the defense uses
            retrieval-backed counter-argument selection, and the judge uses the current heuristic scoring path
            so displayed totals and declared winners remain consistent during the demo.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
