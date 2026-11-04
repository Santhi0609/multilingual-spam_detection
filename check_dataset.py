
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/combined_dataset.csv")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLanguage distribution:")
print(df["language"].value_counts())
