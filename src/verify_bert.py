import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "BERT_Final"
    / "BERT_model"
    / "models"
    / "bert"
)

LABELS = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}

print("=" * 70)
print("VERIFYING BERT MODEL")
print("=" * 70)

print("\nModel path:")
print(MODEL_DIR)

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"BERT model folder not found: {MODEL_DIR}"
    )

print("\n[1/3] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("Tokenizer loaded successfully.")

print("\n[2/3] Loading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("Device:", device)

print("\n[3/3] Running test prediction...")

text = (
    "Scientists announced a new medical breakthrough "
    "after a five-year international study."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128,
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)[0]

predicted_class = torch.argmax(
    probabilities
).item()

confidence = probabilities[
    predicted_class
].item()

print("\nText:")
print(text)

print("\nPredicted class:")
print(predicted_class)

print("\nPredicted label:")
print(LABELS[predicted_class])

print("\nConfidence:")
print(f"{confidence * 100:.2f}%")

print("\nAll probabilities:")

for index, probability in enumerate(probabilities):
    print(
        f"{LABELS[index]:25s}: "
        f"{probability.item() * 100:.2f}%"
    )

print("\n" + "=" * 70)
print("BERT MODEL VERIFICATION COMPLETED SUCCESSFULLY")
print("=" * 70)