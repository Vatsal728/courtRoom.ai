import json
import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load training data
with open("data/training/legal_complaints.json", "r") as f:
    data = json.load(f)

# Prepare X, y
texts = []
labels = []

for domain, examples in data.items():
    texts.extend(examples)
    labels.extend([domain] * len(examples))

print(f"Total training samples: {len(texts)}")
print(f"Domains: {set(labels)}")

# Train-test split (for evaluation only)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorize (evaluation)
eval_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
X_train_vec = eval_vectorizer.fit_transform(X_train)
X_test_vec = eval_vectorizer.transform(X_test)

# Train classifier (evaluation)
eval_clf = MultinomialNB()
eval_clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = eval_clf.predict(X_test_vec)
accuracy = (y_pred == y_test).mean()

print(f"\nEvaluation Results (80/20 Split):")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Accuracy: {accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Train final model on 100% of the data
print("\nTraining final model on 100% of the data...")
final_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
X_all_vec = final_vectorizer.fit_transform(texts)

final_clf = MultinomialNB()
final_clf.fit(X_all_vec, labels)

# Save final models
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(final_clf, models_dir / "nb_classifier.pkl")
joblib.dump(final_vectorizer, models_dir / "tfidf_vectorizer.pkl")

print(f"\n[OK] Final models saved:")
print(f"   - models/nb_classifier.pkl")
print(f"   - models/tfidf_vectorizer.pkl")
