
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Inspect French JSONL
french_path = RAW_DIR / "french.jsonl"
french = pd.read_json(french_path, lines=True)

# Inspect English CSV
english_path = RAW_DIR / "enron_spam_data.csv"
english = pd.read_csv(english_path, nrows=5)

# Inspect Hindi CSV
hindi_path = RAW_DIR / "hindi_sms_data.csv"
hindi = pd.read_csv(hindi_path, nrows=5)

print("\n--- FRENCH DATASET ---")
print("Columns:", french.columns.tolist())
print(french.head(3).to_string())

print("\n--- ENGLISH DATASET ---")
print("Columns:", english.columns.tolist())
print(english.head(3).to_string())

print("\n--- HINDI DATASET ---")
print("Columns:", hindi.columns.tolist())
print(hindi.head(3).to_string())
