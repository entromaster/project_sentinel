"""
Core C — Category Classifier

Trains a TfidfVectorizer + LogisticRegression pipeline to predict book category
from title + description. Reports accuracy and per-class F1 score.
Saves the trained model to models/classifier.pkl.
"""

import os
import sqlite3
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DB_FILE = "books.db"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "classifier.pkl")


def load_training_data(db_path: str) -> tuple[list[str], list[str]]:
    """Load text features and labels from SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT b.title, b.description, c.name
        FROM books b
        JOIN categories c ON b.category_id = c.id
    """)
    rows = cursor.fetchall()
    conn.close()

    texts = []
    labels = []
    for title, description, category in rows:
        # Combine title and description as features
        text = f"{title} {description}" if description else title
        texts.append(text)
        labels.append(category)

    return texts, labels


def main():
    print("=" * 60)
    print("Project SENTINEL — Category Classifier Training")
    print("=" * 60)

    if not os.path.exists(DB_FILE):
        print(f"ERROR: {DB_FILE} not found. Run load_db.py first.")
        return

    # Load data
    texts, labels = load_training_data(DB_FILE)
    print(f"Loaded {len(texts)} samples across {len(set(labels))} categories")
    print(f"Categories: {sorted(set(labels))}")

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Build pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"  # handle imbalanced categories
        )),
    ])

    # Train
    print("\nTraining TF-IDF + LogisticRegression pipeline...")
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nPer-class metrics (precision / recall / F1):")
    print(report)

    # Honest note about small dataset
    print("NOTE: This dataset is small (~60-100 books). Metrics should be")
    print("interpreted with caution. With more data, performance would improve.")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
