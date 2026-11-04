
from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize(df, text_col, label_col, language):
    """Convert each dataset to text, label, and language columns."""

    df.columns = df.columns.astype(str).str.strip()

    result = pd.DataFrame({
        "text": df[text_col].fillna("").astype(str).str.strip(),
        "label": df[label_col].fillna("").astype(str).str.strip().str.lower(),
        "language": language
    })

    # Keep only valid spam/ham records with non-empty messages.
    result = result[
        result["label"].isin(["spam", "ham"])
        & result["text"].ne("")
    ].copy()

    return result


# 1. Load French dataset
french_df = pd.read_json(
    RAW_DIR / "french.jsonl",
    lines=True
)

french = standardize(
    french_df, "text", "label", "French"
)


# 2. Load English dataset
english_df = pd.read_csv(
    RAW_DIR / "enron_spam_data.csv",
    encoding="utf-8",
    encoding_errors="replace"
)

# Combine subject and email body.
subject = english_df["Subject"].fillna("").astype(str).str.strip()
message = english_df["Message"].fillna("").astype(str).str.strip()

english_df["combined_text"] = (
    subject + " " + message
).str.strip()

english = standardize(
    english_df, "combined_text", "Spam/Ham", "English"
)


# 3. Load Hindi dataset
hindi_df = pd.read_csv(
    RAW_DIR / "hindi_sms_data.csv",
    encoding="utf-8",
    encoding_errors="replace"
)

hindi = standardize(
    hindi_df, "Message", "Label", "Hindi"
)


# 4. Combine all three datasets
combined = pd.concat(
    [french, english, hindi],
    ignore_index=True
)

# Remove exact duplicate records within each language.
combined = combined.drop_duplicates(
    subset=["text", "label", "language"]
).reset_index(drop=True)


# 5. Save the combined dataset
output_file = OUTPUT_DIR / "combined_dataset.csv"

combined.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# 6. Display results
print("\nDataset combination completed!")
print("\nRecords retained:")
print("French:", len(french))
print("English:", len(english))
print("Hindi:", len(hindi))
print("Total:", len(combined))

print("\nLanguage-wise spam/ham counts:")
print(pd.crosstab(combined["language"], combined["label"]))

print("\nOutput saved to:")
print(output_file)
