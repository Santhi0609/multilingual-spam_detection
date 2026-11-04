
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "processed" / "combined_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df[df["text"] != ""]
df = df[df["label"].isin(["spam", "ham"])]

X = df["text"]
y = df["label"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Create TF-IDF + Linear SVM pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(class_weight="balanced", random_state=42)
    )
])

# Train model
print("\nTraining model... Please wait.")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

# Save trained model
model_path = MODEL_DIR / "multilingual_spam_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved successfully at:")
print(model_path)
