import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_sentiment, predict_text


# ============================================================
# Page configuration
# ============================================================

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Constants
# ============================================================

LABELS = [
    "True",
    "Satire",
    "False Connection",
    "Imposter Content",
    "Manipulated Content",
    "Misleading Content"
]

ALLOWED_COMPARISON_MODELS = {
    "BERT",
    "DistilBERT",
    "TF-IDF + Logistic Regression",
    "Logistic Regression"
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%,
            rgba(37,99,235,0.10), transparent 25%),
            radial-gradient(circle at 90% 15%,
            rgba(124,58,237,0.08), transparent 25%),
            #f7f9fc;
    }

    .block-container {
        max-width: 1300px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    .hero {
        padding: 36px 40px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 45%,
                #312e81 100%
            );
        box-shadow:
            0 18px 50px rgba(15,23,42,0.18);
        margin-bottom: 28px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        color: #dbeafe;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 13px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero h1 {
        color: white;
        font-size: 46px;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 17px;
        margin-top: 12px;
        margin-bottom: 0;
        max-width: 950px;
        line-height: 1.6;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow:
            0 5px 22px rgba(15,23,42,0.05);
        height: 100%;
    }

    .model-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        box-shadow:
            0 7px 22px rgba(15,23,42,0.05);
        margin-bottom: 12px;
    }

    .model-name {
        font-size: 21px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .model-label {
        color: #475569;
        font-size: 14px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #64748b;
        font-size: 14px;
        line-height: 1.6;
    }

    .chunk-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: 12px;
        margin-bottom: 20px;
        color: #1e3a8a;
    }

    .stTextArea textarea {
        border-radius: 15px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: white !important;
        padding: 16px !important;
        font-size: 16px !important;
    }

    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow:
            0 0 0 2px rgba(37,99,235,0.12)
            !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: none;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        background:
            linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );
        color: white;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 8px 18px rgba(37,99,235,0.22);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 16px;
        box-shadow:
            0 5px 18px rgba(15,23,42,0.04);
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    .custom-footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 22px;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Helper functions
# ============================================================

def probability_dataframe(result):
    """Create a display dataframe from model probabilities."""

    probability_map = result["probability_by_label"]

    df = pd.DataFrame(
        {
            "Category": list(probability_map.keys()),
            "Probability": [
                value * 100
                for value in probability_map.values()
            ]
        }
    )

    return (
        df
        .sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )


def show_model_result(model_name, result):
    """Display the result of one classifier."""

    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">
                {model_name}
            </div>
            <div class="model-label">
                Independent model prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Category",
            result["predicted_label"]
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

    with col3:

        if "number_of_chunks" in result:
            st.metric(
                "Chunks Analyzed",
                result["number_of_chunks"]
            )
        else:
            st.metric(
                "Processing",
                "Full Text"
            )

    probability_df = probability_dataframe(
        result
    )

    st.markdown("#### Class Probabilities")

    st.bar_chart(
        probability_df.set_index(
            "Category"
        )
    )

    display_df = probability_df.copy()

    display_df["Probability"] = (
        display_df["Probability"].map(
            lambda value:
            f"{value:.2f}%"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    if len(probability_df) >= 2:

        first = probability_df.iloc[0]
        second = probability_df.iloc[1]

        st.caption(
            f"Highest probability: "
            f"{first['Category']} "
            f"({first['Probability']:.2f}%). "
            f"Second highest: "
            f"{second['Category']} "
            f"({second['Probability']:.2f}%)."
        )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 TruthLens AI")

    st.caption(
        "Multi-Model Fake News Classification"
    )

    st.divider()

    st.markdown("### Classification Models")

    st.write("**1. BERT**")
    st.write("**2. DistilBERT**")
    st.write("**3. Logistic Regression**")

    st.divider()

    st.markdown("### System")

    st.write("**Classes:** 6")
    st.write("**Transformer input:** 128 tokens")
    st.write("**Long text:** Automatic chunking")
    st.write("**Comparison:** Same input, 3 models")

    st.divider()

    st.markdown("### Classification Labels")

    sidebar_labels = [
        "✅ True",
        "🎭 Satire",
        "🔗 False Connection",
        "👤 Imposter Content",
        "🖼️ Manipulated Content",
        "⚠️ Misleading Content"
    ]

    for label in sidebar_labels:
        st.write(label)

    st.divider()

    st.caption(
        "Ruppin Academic Center\n\n"
        "Final Engineering Project"
    )


# ============================================================
# Hero
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-badge">AI • NLP • MODEL COMPARISON</div>
<h1>TruthLens AI</h1>
<p>A multi-model fake news classification system comparing BERT, DistilBERT and Logistic Regression on the same input. Long texts are automatically processed using token-based chunking.</p>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# Model information
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 BERT</h3>
            <p class="small-muted">
                A fine-tuned Transformer model that uses
                bidirectional contextual representations
                for six-class text classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        """
        <div class="info-card">
            <h3>⚡ DistilBERT</h3>
            <p class="small-muted">
                A lighter Transformer architecture used
                to compare classification performance and
                computational efficiency with BERT.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info3:

    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Logistic Regression</h3>
            <p class="small-muted">
                A TF-IDF based traditional machine-learning
                baseline used to measure the benefit of
                Transformer-based classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Input
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Analyze News Content'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Paste a headline, post or full article. "
    "The same text will be analyzed by all three models."
)

text = st.text_area(
    "News text",
    height=230,
    placeholder=(
        "Paste the text you want to classify here..."
    ),
    label_visibility="collapsed"
)

button_col, clear_col, space_col = st.columns(
    [1, 1, 5]
)

with button_col:

    analyze_button = st.button(
        "🔍 Analyze",
        use_container_width=True
    )

with clear_col:

    clear_button = st.button(
        "Clear",
        use_container_width=True
    )

if clear_button:
    st.rerun()


# ============================================================
# Prediction
# ============================================================

if analyze_button:

    if not text.strip():

        st.warning(
            "Please enter text before running the analysis."
        )

    else:

        try:

            with st.spinner(
                "Running BERT, DistilBERT and "
                "Logistic Regression..."
            ):

                results = predict_text(text)

            st.markdown(
                '<div class="section-title">'
                'Multi-Model Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            # --------------------------------------------
            # Quick comparison
            # --------------------------------------------

            quick1, quick2, quick3 = st.columns(3)

            model_order = [
                "BERT",
                "DistilBERT",
                "Logistic Regression"
            ]

            quick_columns = [
                quick1,
                quick2,
                quick3
            ]

            for column, model_name in zip(
                quick_columns,
                model_order
            ):

                result = results[model_name]

                with column:

                    st.markdown(
                        f"### {model_name}"
                    )

                    st.metric(
                        "Prediction",
                        result["predicted_label"]
                    )

                    st.metric(
                        "Confidence",
                        (
                            f"{result['confidence'] * 100:.2f}%"
                        )
                    )

            # --------------------------------------------
            # Agreement summary
            # --------------------------------------------

            predictions = [
                results[name]["predicted_label"]
                for name in model_order
            ]

            unique_predictions = set(
                predictions
            )

            if len(unique_predictions) == 1:

                st.success(
                    "All three models produced the same "
                    f"classification: **{predictions[0]}**."
                )

            else:

                st.info(
                    "The models produced different predictions. "
                    "The detailed tabs below show each model's "
                    "probability distribution."
                )

            # --------------------------------------------
            # Chunk information
            # --------------------------------------------

            bert_chunks = results[
                "BERT"
            ].get(
                "number_of_chunks",
                1
            )

            distil_chunks = results[
                "DistilBERT"
            ].get(
                "number_of_chunks",
                1
            )

            max_chunks = max(
                bert_chunks,
                distil_chunks
            )

            if max_chunks > 1:

                st.markdown(
                    f"""
                    <div class="chunk-box">
                        <strong>
                            Long-text processing active
                        </strong><br>
                        This input required multiple Transformer
                        chunks. BERT analyzed
                        <strong>{bert_chunks}</strong> chunks and
                        DistilBERT analyzed
                        <strong>{distil_chunks}</strong> chunks.
                        Chunk-level class probabilities were
                        averaged to produce each model's final
                        document-level prediction.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.caption(
                    "The input fits within one Transformer "
                    "chunk; long-text splitting was not required."
                )

            sentiment_result = predict_sentiment(text)

            st.markdown(
                "## Sentiment Analysis (VADER)",
                unsafe_allow_html=False
            )
            st.caption(
                "VADER analyzes the emotional tone of the input independently from the classification models."
            )

            sentiment_cols = st.columns(5)
            with sentiment_cols[0]:
                st.metric("Overall Sentiment", sentiment_result["sentiment_label"])
            with sentiment_cols[1]:
                st.metric("Compound Score", f"{sentiment_result['compound']:.3f}")
            with sentiment_cols[2]:
                st.metric("Positive", f"{sentiment_result['positive']:.3f}")
            with sentiment_cols[3]:
                st.metric("Neutral", f"{sentiment_result['neutral']:.3f}")
            with sentiment_cols[4]:
                st.metric("Negative", f"{sentiment_result['negative']:.3f}")

            # --------------------------------------------
            # Model tabs
            # --------------------------------------------

            bert_tab, distil_tab, logistic_tab, tech_tab = (
                st.tabs(
                    [
                        "🧠 BERT",
                        "⚡ DistilBERT",
                        "📊 Logistic Regression",
                        "🔬 Technical Details"
                    ]
                )
            )

            with bert_tab:

                show_model_result(
                    "BERT",
                    results["BERT"]
                )

            with distil_tab:

                show_model_result(
                    "DistilBERT",
                    results["DistilBERT"]
                )

            with logistic_tab:

                show_model_result(
                    "Logistic Regression",
                    results[
                        "Logistic Regression"
                    ]
                )

            with tech_tab:

                st.markdown(
                    "### Classification Pipeline"
                )

                st.write(
                    "**BERT:** Fine-tuned Transformer "
                    "classifier."
                )

                st.write(
                    "**DistilBERT:** Fine-tuned lightweight "
                    "Transformer classifier."
                )

                st.write(
                    "**Logistic Regression:** TF-IDF "
                    "baseline classifier."
                )

                st.write(
                    "**Number of classes:** 6"
                )

                st.write(
                    "**Transformer chunk size:** "
                    "128 tokens"
                )

                st.write(
                    "**Long-text aggregation:** "
                    "Mean of chunk-level class "
                    "probabilities"
                )

                st.markdown(
                    "### Current Input"
                )

                st.code(
                    text,
                    language=None
                )

                st.caption(
                    "Model confidence represents the "
                    "classifier's probability output. "
                    "It should not be interpreted as "
                    "independent factual verification."
                )

        except Exception as error:

            st.error(
                "The analysis could not be completed."
            )

            with st.expander(
                "Show technical error"
            ):

                st.exception(error)


# ============================================================
# Research model comparison
# ============================================================

comparison_file = (
    RESULTS_DIR /
    "model_comparison.csv"
)

if comparison_file.exists():

    st.markdown(
        '<div class="section-title">'
        'Experimental Model Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Evaluation results from the controlled "
        "project experiments."
    )

    try:

        comparison = pd.read_csv(
            comparison_file
        )

        # Identify model-name column.
        model_column = None

        for candidate in [
            "Model",
            "model",
            "Model Name",
            "model_name"
        ]:

            if candidate in comparison.columns:
                model_column = candidate
                break

        if model_column is not None:

            comparison = comparison[
                comparison[
                    model_column
                ].astype(str).isin(
                    ALLOWED_COMPARISON_MODELS
                )
            ].copy()

        comparison_display = (
            comparison.copy()
        )

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]:

            if column in comparison_display.columns:

                comparison_display[
                    column
                ] = comparison_display[
                    column
                ].map(
                    lambda value:
                    f"{float(value):.4f}"
                )

        st.dataframe(
            comparison_display,
            use_container_width=True,
            hide_index=True
        )

        if (
            model_column is not None
            and "Accuracy" in comparison.columns
            and not comparison.empty
        ):

            accuracy_chart = comparison[
                [
                    model_column,
                    "Accuracy"
                ]
            ].copy()

            accuracy_chart[
                "Accuracy"
            ] = (
                pd.to_numeric(
                    accuracy_chart["Accuracy"],
                    errors="coerce"
                ) * 100
            )

            accuracy_chart = (
                accuracy_chart.dropna()
            )

            if not accuracy_chart.empty:

                st.markdown(
                    "#### Accuracy Comparison"
                )

                st.bar_chart(
                    accuracy_chart.set_index(
                        model_column
                    )
                )

        st.caption(
            "The comparison is based on the project's "
            "stored evaluation results. Live confidence "
            "scores above are input-specific and are not "
            "the same as test-set accuracy."
        )

    except Exception as error:

        st.warning(
            "Stored comparison results could not "
            "be displayed."
        )

        with st.expander(
            "Show comparison error"
        ):
            st.exception(error)


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        <strong>TruthLens AI</strong><br>
        Final Engineering Project •
        Ruppin Academic Center •
        Department of Electrical & Computer Engineering
    </div>
    """,
    unsafe_allow_html=True
)