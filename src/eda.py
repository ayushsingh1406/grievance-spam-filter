# src/eda.py
"""
Standalone EDA & evaluation script for the grievance spam filter project.
Saves plots to results/ and prints summary statistics + classification report.
Run: python src/eda.py
"""

import os
import sys
from collections import Counter

# Ensure project root is on sys.path so we can import src modules reliably.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.data_fetch import load_sms
from src.preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def top_tokens(texts, n=20):
    c = Counter()
    for t in texts:
        c.update(str(t).split())
    return c.most_common(n)

def save_histogram(df):
    plt.figure(figsize=(8,4))
    plt.hist(df[df.label==0]['word_count'], bins=15, alpha=0.6, label='ham')
    plt.hist(df[df.label==1]['word_count'], bins=15, alpha=0.6, label='spam')
    plt.legend()
    plt.title("Word Count Distribution by Class")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    out = os.path.join(RESULTS_DIR, "word_count_distribution.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved histogram to:", out)

def save_confusion_matrix(cm, labels=("ham","spam")):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           ylabel="True label",
           xlabel="Predicted label",
           title="Confusion matrix")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out)
    plt.close()
    print("Saved confusion matrix to:", out)

def main():
    csv_path = os.path.join(project_root, "data", "raw", "sms_spam_collection.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected dataset at {csv_path}. Create it or update path.")

    print("Loading dataset:", csv_path)
    df = load_sms(csv_path)
    print("Raw counts:\n", df.label.value_counts())
    df['clean_text'] = df['text'].apply(preprocess)

    # basic length metrics
    df['char_len'] = df['text'].str.len()
    df['word_count'] = df['clean_text'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    print("\nCharacter length describe:\n", df['char_len'].describe())
    print("\nWord count describe:\n", df['word_count'].describe())

    # save histogram
    save_histogram(df)

    # top tokens
    print("\nTop tokens (ham):", top_tokens(df[df.label==0]['clean_text'], n=30))
    print("\nTop tokens (spam):", top_tokens(df[df.label==1]['clean_text'], n=30))

    # load trained model
    model_path = os.path.join(project_root, "models", "nb_pipeline.joblib")
    if not os.path.exists(model_path):
        print("\nModel not found at", model_path)
        print("If you haven't trained a model yet, run: python src/train.py")
        return

    print("\nLoading model:", model_path)
    model = joblib.load(model_path)

    # evaluate on holdout split (same splitting strategy we used in training)
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    preds = model.predict(X_test)

    print("\nClassification report (on holdout test):\n")
    print(classification_report(y_test, preds, digits=4))
    cm = confusion_matrix(y_test, preds)
    save_confusion_matrix(cm)

    # show top features for spam class (explainability)
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        feature_names = tfidf.get_feature_names_out()
        spam_log_prob = clf.feature_log_prob_[1]
        top_idx = np.argsort(spam_log_prob)[-30:][::-1]
        top_features = [(feature_names[i], float(spam_log_prob[i])) for i in top_idx]
        print("\nTop TF-IDF features for spam (feature, log-prob):")
        for feat, val in top_features:
            print(feat, f"{val:.4f}")
        # optionally save to CSV
        feats_out = os.path.join(RESULTS_DIR, "top_spam_features.csv")
        pd.DataFrame(top_features, columns=["feature","log_prob"]).to_csv(feats_out, index=False)
        print("Saved top spam features to:", feats_out)
    except Exception as e:
        print("Could not extract top features from the model:", e)

if __name__ == "__main__":
    main()
