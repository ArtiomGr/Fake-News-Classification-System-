import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

train_file = DATA_DIR / "all_train.tsv"
val_file = DATA_DIR / "all_validate.tsv"
test_file = DATA_DIR / "all_test_public.tsv"

print("=" * 60)
print("FAKEDDIT DATASET INSPECTION")
print("=" * 60)

train_df = pd.read_csv(train_file, sep="\t")
val_df = pd.read_csv(val_file, sep="\t")
test_df = pd.read_csv(test_file, sep="\t")

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst 5 rows:")
print(train_df.head())

print("\nMissing clean_title:")
print(train_df["clean_title"].isna().sum())

print("\n6-way label distribution:")
print(train_df["6_way_label"].value_counts().sort_index())

print("\nExample texts:")
for i in range(5):
    print("\nExample", i + 1)
    print("Text:", train_df.iloc[i]["clean_title"])
    print("Label:", train_df.iloc[i]["6_way_label"])

print("\nDataset loaded successfully.")