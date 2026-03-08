import pandas as pd

df = pd.read_csv("data/processed/debate_dataset.csv")

print("Number of unique topics:", df["topic"].nunique())
print()

print("Top topics by number of claims:")
print(df["topic"].value_counts().head(10))