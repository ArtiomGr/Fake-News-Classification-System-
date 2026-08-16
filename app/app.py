import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_text


st.set_page_config(
    page_title="Fake News Classification System",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Fake News Classification System")

st.write(
    "Enter a news headline or short text and the system will classify it "
    "using the trained DistilBERT model and perform sentiment analysis."
)

st.divider()

text = st.text_area(
    "Enter news text:",
    height=180,
    placeholder="Paste a news headline or short article here..."
)

analyze_button = st.button(
    "Analyze Text",
    type="primary"
)

if analyze_button:

    if not text.strip():

        st.warning(
            "Please enter some text before analysis."
        )

    else:

        with st.spinner(
            "Analyzing text..."
        ):

            result = predict_text(
                text
            )

        st.success(
            "Analysis completed."
        )

        st.subheader(
            "Classification Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Class",
                result["predicted_class"]
            )

        with col2:

            st.metric(
                "Predicted Label",
                result["predicted_label"]
            )

        with col3:

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

        st.subheader(
            "Class Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    f"Class {i}"
                    for i in range(
                        len(
                            result["probabilities"]
                        )
                    )
                ],

                "Probability": [
                    float(p) * 100
                    for p in result[
                        "probabilities"
                    ]
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Sentiment Analysis"
        )

        sentiment = result[
            "sentiment"
        ]

        sent_col1, sent_col2 = st.columns(2)

        with sent_col1:

            st.metric(
                "Sentiment",
                sentiment["label"]
            )

        with sent_col2:

            st.metric(
                "Compound Score",
                f"{sentiment['compound']:.4f}"
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
                    ],
                    sentiment[
                        "neutral"
                    ],
                    sentiment[
                        "positive"
                    ]
                ]
            }
        )

        st.bar_chart(
            sentiment_df.set_index(
                "Sentiment"
            )
        )

        st.subheader(
            "Detailed Output"
        )

        with st.expander(
            "Show model details"
        ):

            st.write(
                "Model: DistilBERT"
            )

            st.write(
                "Number of classes: 6"
            )

            st.write(
                "Maximum token length: 128"
            )

            st.write(
                "Sentiment model: VADER"
            )

st.divider()

st.caption(
    "Final Project - Ruppin Academic Center | "
    "AI-Based Fake News Classification"
)