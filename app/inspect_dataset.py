import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

df = pd.read_csv("data/processed/debate_dataset.csv")

print("Number of unique topics:", df["topic"].nunique())
print()

print("Top topics by number of claims:")
print(df["topic"].value_counts().head(10))
