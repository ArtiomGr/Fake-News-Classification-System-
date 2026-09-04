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

from predict import predict_text


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
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(37,99,235,0.10), transparent 25%),
            radial-gradient(circle at 90% 15%, rgba(124,58,237,0.08), transparent 25%),
            #f7f9fc;
    }

    /* Main container */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Hero */
    .hero {
        padding: 35px 38px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 45%,
                #312e81 100%
            );
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
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
        max-width: 850px;
        line-height: 1.6;
    }

    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 5px 22px rgba(15,23,42,0.05);
        height: 100%;
    }

    .result-card {
        background: white;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.06);
        margin-bottom: 14px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #64748b;
        font-size: 14px;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: white !important;
        padding: 16px !important;
        font-size: 16px !important;
    }

    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.12) !important;
    }

    /* Primary button */
    .stButton > button {
        border-radius: 12px;
        border: none;
        padding: 0.65rem 1.5rem;
        font-weight: 700;
        background: linear-gradient(
            90deg,
            #2563eb,
            #4f46e5
        );
        color: white;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(37,99,235,0.22);
        color: white;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.04);
    }

    /* Sidebar */
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

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    /* Footer */
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
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 TruthLens AI")

    st.caption(
        "AI-Based Fake News Classification"
    )

    st.divider()

    st.markdown("### Model")

    st.write("**Architecture:** DistilBERT")
    st.write("**Classes:** 6")
    st.write("**Sentiment:** VADER")
    st.write("**Max tokens:** 128")

    st.divider()

    st.markdown("### Classification Labels")

    labels = [
        "✅ True",
        "🎭 Satire",
        "🔗 False Connection",
        "👤 Imposter Content",
        "🖼️ Manipulated Content",
        "⚠️ Misleading Content"
    ]

    for label in labels:
        st.write(label)

    st.divider()

    # Show final test performance if results file exists
    metrics_file = (
        RESULTS_DIR /
        "distilbert_metrics.csv"
    )

    if metrics_file.exists():

        try:

            metrics_df = pd.read_csv(
                metrics_file
            )

            row = metrics_df.iloc[0]

            st.markdown(
                "### Test Performance"
            )

            st.metric(
                "Accuracy",
                f"{row['Accuracy'] * 100:.2f}%"
            )

            st.metric(
                "Weighted F1",
                f"{row['F1']:.4f}"
            )

        except Exception:
            pass

    st.divider()

    st.caption(
        "Ruppin Academic Center\n\n"
        "Final Engineering Project"
    )


# ============================================================
# Hero section
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-badge">AI • NLP • Fake News Detection</div>
<h1>TruthLens AI</h1>
<p>An intelligent text-analysis system that uses a fine-tuned DistilBERT model to classify online content into six information categories, while independently analyzing its emotional sentiment.</p>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# Information cards
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 Transformer AI</h3>
            <p class="small-muted">
                Fine-tuned DistilBERT performs semantic text
                classification using contextual language understanding.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Six Categories</h3>
            <p class="small-muted">
                The system provides a probability distribution across
                all six classification labels, not only the final class.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info3:

    st.markdown(
        """
        <div class="info-card">
            <h3>💬 Sentiment Analysis</h3>
            <p class="small-muted">
                VADER independently evaluates the emotional polarity of
                the submitted text as positive, neutral or negative.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Input section
# ============================================================

st.markdown(
    '<div class="section-title">Analyze News Content</div>',
    unsafe_allow_html=True
)

st.caption(
    "Paste a headline, social-media post or short news article below."
)

text = st.text_area(
    "News text",
    height=190,
    placeholder=(
        "Example: Scientists announced a major medical "
        "breakthrough following a five-year international study..."
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
            "Please enter some text before running the analysis."
        )

    else:

        try:

            with st.spinner(
                "TruthLens AI is analyzing the text..."
            ):

                result = predict_text(
                    text
                )

            st.markdown(
                '<div class="section-title">Analysis Result</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Main metrics
            # ------------------------------------------------

            metric1, metric2, metric3 = st.columns(
                3
            )

            with metric1:

                st.metric(
                    "Predicted Category",
                    result[
                        "predicted_label"
                    ]
                )

            with metric2:

                st.metric(
                    "Model Confidence",
                    (
                        f"{result['confidence'] * 100:.2f}%"
                    )
                )

            with metric3:

                sentiment_label = (
                    result["sentiment"]["label"]
                )

                st.metric(
                    "Sentiment",
                    sentiment_label
                )

            st.markdown("")

            # ------------------------------------------------
            # Tabs
            # ------------------------------------------------

            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 Classification",
                    "💬 Sentiment",
                    "🔬 Technical Details"
                ]
            )


            # =================================================
            # Classification tab
            # =================================================

            with tab1:

                st.markdown(
                    "### Category Probabilities"
                )

                if (
                    "probability_by_label"
                    in result
                ):

                    probability_df = pd.DataFrame(
                        {
                            "Category": list(
                                result[
                                    "probability_by_label"
                                ].keys()
                            ),
                            "Probability": [
                                value * 100
                                for value in result[
                                    "probability_by_label"
                                ].values()
                            ]
                        }
                    )

                else:

                    fallback_labels = [
                        "True",
                        "Satire",
                        "False Connection",
                        "Imposter Content",
                        "Manipulated Content",
                        "Misleading Content"
                    ]

                    probability_df = pd.DataFrame(
                        {
                            "Category":
                                fallback_labels,

                            "Probability": [
                                float(value) * 100
                                for value
                                in result[
                                    "probabilities"
                                ]
                            ]
                        }
                    )

                probability_df = (
                    probability_df
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                st.bar_chart(
                    probability_df.set_index(
                        "Category"
                    )
                )

                display_df = (
                    probability_df.copy()
                )

                display_df[
                    "Probability"
                ] = display_df[
                    "Probability"
                ].map(
                    lambda x: f"{x:.2f}%"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                top_category = (
                    probability_df.iloc[0]
                )

                second_category = (
                    probability_df.iloc[1]
                )

                st.info(
                    f"Highest probability: "
                    f"**{top_category['Category']}** "
                    f"({top_category['Probability']:.2f}%). "
                    f"The second-highest category is "
                    f"**{second_category['Category']}** "
                    f"({second_category['Probability']:.2f}%)."
                )


            # =================================================
            # Sentiment tab
            # =================================================

            with tab2:

                sentiment = result[
                    "sentiment"
                ]

                sent1, sent2 = st.columns(
                    2
                )

                with sent1:

                    st.metric(
                        "Overall Sentiment",
                        sentiment["label"]
                    )

                with sent2:

                    st.metric(
                        "Compound Score",
                        (
                            f"{sentiment['compound']:.4f}"
                        )
                    )

                st.markdown(
                    "### Sentiment Distribution"
                )

                sentiment_df = pd.DataFrame(
                    {
                        "Sentiment": [
                            "Negative",
                            "Neutral",
                            "Positive"
                        ],

                        "Score": [
                            sentiment[
                                "negative"
                            ] * 100,

                            sentiment[
                                "neutral"
                            ] * 100,

                            sentiment[
                                "positive"
                            ] * 100
                        ]
                    }
                )

                st.bar_chart(
                    sentiment_df.set_index(
                        "Sentiment"
                    )
                )

                sentiment_display = (
                    sentiment_df.copy()
                )

                sentiment_display[
                    "Score"
                ] = sentiment_display[
                    "Score"
                ].map(
                    lambda x:
                    f"{x:.2f}%"
                )

                st.dataframe(
                    sentiment_display,
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # Technical details tab
            # =================================================

            with tab3:

                detail1, detail2 = st.columns(
                    2
                )

                with detail1:

                    st.markdown(
                        "### NLP Classifier"
                    )

                    st.write(
                        "**Model:** DistilBERT"
                    )

                    st.write(
                        "**Task:** "
                        "6-class text classification"
                    )

                    st.write(
                        "**Maximum token length:** 128"
                    )

                    st.write(
                        "**Output:** "
                        "Softmax class probabilities"
                    )

                with detail2:

                    st.markdown(
                        "### Sentiment Module"
                    )

                    st.write(
                        "**Model:** VADER"
                    )

                    st.write(
                        "**Output:** "
                        "Positive / Neutral / Negative"
                    )

                    st.write(
                        "**Compound score range:** "
                        "-1 to +1"
                    )

                st.markdown(
                    "### Current Input"
                )

                st.code(
                    text,
                    language=None
                )

                st.caption(
                    "The prediction represents the output "
                    "of the trained research model and should "
                    "not be treated as independent verification "
                    "of factual truth."
                )

        except Exception as error:

            st.error(
                "The analysis could not be completed."
            )

            with st.expander(
                "Show technical error"
            ):

                st.exception(
                    error
                )


# ============================================================
# Project performance section
# ============================================================

comparison_file = (
    RESULTS_DIR /
    "model_comparison.csv"
)

if comparison_file.exists():

    st.markdown(
        '<div class="section-title">Project Model Comparison</div>',
        unsafe_allow_html=True
    )

    try:

        comparison = pd.read_csv(
            comparison_file
        )

        comparison_display = (
            comparison.copy()
        )

        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]:

            if column in (
                comparison_display.columns
            ):

                comparison_display[
                    column
                ] = comparison_display[
                    column
                ].map(
                    lambda value:
                    f"{value:.4f}"
                )

        st.dataframe(
            comparison_display,
            use_container_width=True,
            hide_index=True
        )

    except Exception:
        pass


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