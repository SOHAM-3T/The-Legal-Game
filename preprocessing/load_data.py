import pandas as pd
from pathlib import Path

# Locate project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw"

motions_file = DATA_PATH / "motions.txt"
claims_file = DATA_PATH / "claims.txt"
evidence_file = DATA_PATH / "evidence.txt"

print("Loading dataset...")
print("Dataset path:", DATA_PATH)

# Load raw datasets
motions = pd.read_csv(motions_file, sep="\t")
claims = pd.read_csv(claims_file, sep="\t")
evidence = pd.read_csv(evidence_file, sep="\t", header=None)

print("\nDataset loaded successfully\n")

print("Motions shape:", motions.shape)
print("Claims shape:", claims.shape)
print("Evidence shape:", evidence.shape)

# Rename columns
motions = motions.rename(columns={
    "Topic id": "topic_id",
    "Topic": "topic"
})

claims = claims.rename(columns={
    "Topic": "topic",
    "Claim corrected version": "claim"
})

claims = claims[["topic", "claim"]]

# Fix evidence column names
evidence.columns = [
    "topic",
    "claim",
    "evidence_text",
    "evidence_type"
]

print("\nColumns after cleaning:")

print("\nMotions columns:", motions.columns)
print("Claims columns:", claims.columns)
print("Evidence columns:", evidence.columns)

# Merge claims with evidence
print("\nMerging claims and evidence...")

dataset = claims.merge(
    evidence,
    on=["topic", "claim"],
    how="left"
)

print("Merged dataset shape (before cleaning):", dataset.shape)

# Remove rows without evidence
before_rows = len(dataset)

dataset = dataset.dropna(subset=["evidence_text"])

after_rows = len(dataset)

print("Removed rows with missing evidence:", before_rows - after_rows)
print("Dataset shape (after cleaning):", dataset.shape)

# Show sample
print("\nSample merged data:")
print(dataset.head())

# Save processed dataset
processed_dir = BASE_DIR / "data" / "processed"
processed_dir.mkdir(exist_ok=True)

output_file = processed_dir / "debate_dataset.csv"

dataset.to_csv(output_file, index=False)

print("\nProcessed dataset saved to:")
print(output_file)

print("\nPreprocessing complete!")