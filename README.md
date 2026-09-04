# TruthLens AI
## AI-Based Fake News Classification System

Final Engineering Project  
Ruppin Academic Center  
Department of Electrical & Computer Engineering
Muhammed Zedan
Artium Grenberg

---

## Project Overview

TruthLens AI is an artificial-intelligence-based system for classifying online text into six information categories using Natural Language Processing.

The final system uses a fine-tuned DistilBERT model for text classification and VADER for sentiment analysis.

The application provides:

- Predicted category
- Prediction confidence
- Probability distribution across all six classes
- Sentiment classification
- Sentiment score
- Model comparison results
- Interactive Streamlit user interface

---

## Classification Categories

The system classifies text into six categories:

1. True
2. Satire
3. False Connection
4. Imposter Content
5. Manipulated Content
6. Misleading Content

---

## Dataset

The project uses the Fakeddit dataset.

The model uses the textual field:

```text
clean_title