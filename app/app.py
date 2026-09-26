"""University defense dashboard for the final, locally trained comparison."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from final_project import CATEGORIES, RESULT_PATHS
from predict import (
    MODEL_NAMES, MAX_INPUT_CHARACTERS, TRANSFORMER_MAX_LENGTH, CHUNK_OVERLAP_TOKENS,
    artifact_fingerprint, get_model_metadata, load_classifier, model_agreement,
    predict_sentiment, predict_text,
)

APP_NAME = "Fake News & Content Classification System"
st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root { --navy:#0b1739; --blue:#315efb; --cyan:#19b8d4; --violet:#7557e8; --ink:#13213c; --muted:#66758d; --panel:#ffffff; --line:#dce4ef; }
.stApp { background: radial-gradient(circle at 85% 5%, #e8eeff 0, transparent 25%), #f5f7fb; color: var(--ink); }
.block-container { max-width: 1380px; padding-top: 1.15rem; padding-bottom: 2.5rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0b1739; }
[data-testid="stSidebar"] * { color: #eef4ff !important; }
h1,h2,h3 { color: var(--ink); letter-spacing:-.025em; }
h2 { margin-top:.8rem !important; }
.hero { background: linear-gradient(120deg,#0b1739 0%,#172d65 58%,#314ec9 100%); border-radius:24px; padding:30px 34px; color:white; box-shadow:0 18px 45px rgba(26,50,105,.18); margin-bottom:18px; }
.hero .eyebrow { font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; opacity:.78; font-weight:700; }
.hero h1 { color:white !important; font-size:clamp(2rem,4vw,3.25rem) !important; margin:.25rem 0 .35rem; }
.hero p { margin:0; color:#dce6ff; font-size:1.05rem; max-width:850px; }
.model-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0 22px; }
.model-chip { background:white; border:1px solid var(--line); border-radius:16px; padding:15px 17px; box-shadow:0 7px 20px rgba(31,48,84,.06); }
.model-chip b { display:block; color:var(--ink); font-size:1.02rem; }
.model-chip span { color:var(--muted); font-size:.82rem; }
.model-chip strong { float:right; color:#315efb; font-size:.95rem; }
.section-kicker { color:#315efb; font-weight:800; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:-.25rem; }
.input-shell { background:white; border:1px solid var(--line); border-radius:20px; padding:8px 14px 14px; box-shadow:0 10px 28px rgba(31,48,84,.07); }
.stTextArea textarea { border-radius:14px !important; background:#fbfcff !important; border:1px solid #d9e2f0 !important; font-size:1rem !important; }
.stButton button { border-radius:12px !important; font-weight:700 !important; min-height:44px; }
.stButton button[kind="primary"] { background:linear-gradient(90deg,#315efb,#6048e8) !important; border:0 !important; color:white !important; box-shadow:0 8px 18px rgba(49,94,251,.2); }
[data-testid="stVerticalBlockBorderWrapper"] { background:white; border:1px solid var(--line); border-radius:18px; box-shadow:0 8px 24px rgba(31,48,84,.06); }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-weight:800; }
[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { font-weight:700; border-radius:10px; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
.project-footer { color:#77859a; font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:1.5rem; text-align:center; }
@media(max-width:900px){ .model-strip{grid-template-columns:repeat(2,1fr)} .hero{padding:24px} }
@media(max-width:600px){ .model-strip{grid-template-columns:1fr} .block-container{padding:1rem} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">University Final Project · NLP & Machine Learning</div>
  <h1>Fake News & Content Classification System</h1>
  <p>Multi-model analysis across six content categories using BERT, DistilBERT and TF-IDF + Logistic Regression, with independent VADER sentiment analysis.</p>
</div>
<div class="model-strip">
  <div class="model-chip"><strong>97.48%</strong><b>BERT</b><span>Transformer · Test accuracy</span></div>
  <div class="model-chip"><strong>97.87%</strong><b>DistilBERT</b><span>Transformer · Test accuracy</span></div>
  <div class="model-chip"><strong>84.16%</strong><b>Logistic Regression</b><span>TF-IDF baseline · Test accuracy</span></div>
  <div class="model-chip"><strong>Sentiment</strong><b>VADER</b><span>Independent emotional-tone analysis</span></div>
</div>
""", unsafe_allow_html=True)

LABELS = list(CATEGORIES.values())
signature = artifact_fingerprint()

with st.sidebar:
    st.markdown("### Project categories")
    for project_id, label in CATEGORIES.items():
        st.caption(f"{project_id}. {label}")
    st.divider()
    st.caption("Three classifiers · one frozen dataset\n\nVADER provides separate sentiment analysis.\n\nLocal inference · no external API")

@st.cache_resource(show_spinner=False)
def cached_classifiers(artifact_version):
    """Streamlit caches resources by the actual saved-artifact version."""
    return {name: load_classifier(name) for name in MODEL_NAMES}


def reset_results():
    st.session_state.pop("analysis", None)


def clear_analysis():
    st.session_state["input_text"] = ""
    reset_results()


def show_results(results):
    st.markdown('<div class="section-kicker">Multi-model inference</div>', unsafe_allow_html=True)
    st.subheader("Classification Results")
    for column, name in zip(st.columns(3), MODEL_NAMES):
        result = results[name]
        with column, st.container(border=True):
            st.markdown(f"### {name}")
            st.metric("Predicted category", result["predicted_label"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.caption(
                "TF-IDF features · full text" if name == "Logistic Regression"
                else f"{result['number_of_chunks']} text section{'s' if result['number_of_chunks'] != 1 else ''}"
            )

    agreement = model_agreement(results)
    st.markdown('<div class="section-kicker">Consensus</div>', unsafe_allow_html=True)
    st.info("🧠 " + agreement["message"])
    st.caption("Agreement describes consistency between classifiers; it does not establish factual truth.")
    with st.expander("Prediction comparison"):
        st.dataframe(pd.DataFrame([
            {"Model": name, "Category": result["predicted_label"], "Confidence": f"{result['confidence']:.2%}"}
            for name, result in results.items()
        ]), hide_index=True, width="stretch")

    st.markdown('<div class="section-kicker">Explainability</div>', unsafe_allow_html=True)
    st.subheader("Category Probability Explorer")
    for tab, name in zip(st.tabs(list(MODEL_NAMES)), MODEL_NAMES):
        result = results[name]
        with tab:
            chart_column, table_column = st.columns([3, 2])
            frame = pd.DataFrame({"Category": LABELS,
                                  "Probability": [result["probability_by_label"][label] * 100 for label in LABELS]})
            with chart_column:
                st.bar_chart(frame, x="Category", y="Probability", horizontal=True, sort=False,
                             color="#35649b", height=245, x_label="Probability (%)", y_label="")
            with table_column:
                table = frame.copy()
                table["Probability"] = table["Probability"].map(lambda value: f"{value:.2f}%")
                st.dataframe(table, hide_index=True, width="stretch", height=245)
            if name != "Logistic Regression" and result["number_of_chunks"] > 1:
                st.caption("Overlapping sections cover the complete text. Document probabilities are their arithmetic mean.")
            with st.expander(f"{name} model details"):
                st.caption(f"Model: {result['model_path']}")
                if result["tokenizer_path"]:
                    st.caption(f"Tokenizer: {result['tokenizer_path']}")
                st.caption(f"Loaded artifact SHA-256: {result['weights_sha256']}")
                st.caption("Project labels are read from the saved final mapping.")


st.markdown('<div class="section-kicker">Analyze content</div>', unsafe_allow_html=True)
st.subheader("Paste a headline, post or article")
text = st.text_area("Text to analyze", key="input_text", height=190,
                    placeholder="Paste a headline, article or passage…",
                    max_chars=MAX_INPUT_CHARACTERS, on_change=reset_results, label_visibility="collapsed")
analyze_column, clear_column, _ = st.columns([1, 1, 6])
analyze = analyze_column.button("⚡ Analyze", type="primary", width="stretch")
clear_column.button("↺ Clear", on_click=clear_analysis, width="stretch")

if analyze:
    reset_results()
    if not text.strip():
        st.warning("Please enter text before running the analysis.")
    else:
        try:
            with st.spinner("Loading local models and analyzing text…"):
                cached_classifiers(signature)
                results = predict_text(text)
            analysis = {"text": text, "signature": signature, "classification": results}
            try:
                analysis["sentiment"] = predict_sentiment(text)
            except Exception:
                analysis["sentiment_error"] = True
            st.session_state["analysis"] = analysis
        except Exception as error:
            st.error("The analysis could not be completed. All three final local classifiers are required.")
            with st.expander("Technical details"):
                st.exception(error)

analysis = st.session_state.get("analysis")
if analysis and analysis["text"] == text and analysis["signature"] == signature:
    show_results(analysis["classification"])
    st.markdown('<div class="section-kicker">Independent signal</div>', unsafe_allow_html=True)
    st.subheader("VADER Sentiment Analysis")
    st.caption("VADER measures emotional tone independently. Its scores do not determine any classifier's category.")
    if "sentiment" in analysis:
        sentiment = analysis["sentiment"]
        with st.container(border=True):
            columns = st.columns(5)
            columns[0].metric("Overall Sentiment", sentiment["sentiment_label"])
            columns[1].metric("Compound Score", f"{sentiment['compound']:.3f}")
            for column, key in zip(columns[2:], ("positive", "neutral", "negative")):
                column.metric(key.capitalize(), f"{sentiment[key]:.1%}")
        st.caption("Compound score: −1 (negative) to +1 (positive).")
    else:
        st.warning("Sentiment is unavailable; the three classification results above are complete.")

with st.expander("📊 Research performance & model information"):
    st.markdown(
        "**BERT:** a full-size bidirectional Transformer. "
        "**DistilBERT:** a smaller distilled Transformer using the preserved corrected experiment. "
        "**Logistic Regression:** a classical classifier using TF-IDF word and phrase features. "
        "**VADER:** a separate rule-based sentiment analyzer."
    )
    st.write(
        f"Both Transformers use a {TRANSFORMER_MAX_LENGTH}-token limit including special tokens. "
        f"Short input is classified directly. Long input uses a {CHUNK_OVERLAP_TOKENS}-token overlap, "
        "with mean section probabilities. Logistic Regression processes the full text."
    )
    comparison_path = ROOT / "results/final_model_comparison/test_metrics.csv"
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption("All three models use the same frozen split: 5,921 training, 1,269 validation and 1,269 test records.")
    st.caption(
        "The dataset combines source-supervised and heuristic labels. Source/style cues and overlap in concepts "
        "limit generalization; these results are not independent fact-checking or unseen-source validation."
    )

st.caption(
    "Model confidence is the classifier's probability output and should not be interpreted "
    "as independent factual verification of a news claim. The system chooses among six supported categories."
)
st.markdown('<div class="project-footer">University Final Project · Local NLP Model Comparison</div>', unsafe_allow_html=True)
