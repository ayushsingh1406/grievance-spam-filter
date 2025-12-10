import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.data_fetch import load_sms
from src.preprocess import preprocess

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def analyze(model_path="models/nb_pipeline.joblib",
            data_path="data/processed/merged_dataset.csv"):
    
    print("Loading model:", model_path)
    model = joblib.load(model_path)

    print("Loading dataset:", data_path)
    df = load_sms(data_path)
    df["clean_text"] = df["text"].apply(preprocess)

    X = df["clean_text"]
    y = df["label"]

    preds = model.predict(X)

    print("\n=== CLASSIFICATION REPORT ===\n")
    print(classification_report(y, preds, digits=4))

    print("\n=== CONFUSION MATRIX ===\n")
    cm = confusion_matrix(y, preds)
    print(cm)

    # Save confusion matrix image
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print("Saved:", cm_path)

    # Top spam features
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    feature_names = vectorizer.get_feature_names_out()
    spam_scores = clf.feature_log_prob_[1]

    top = sorted(zip(feature_names, spam_scores),
                 key=lambda x: x[1], reverse=True)[:30]

    # Save as image
    words, scores = zip(*top)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(scores), y=list(words))
    plt.title("Top Spam Features (Log Probabilities)")
    plt.tight_layout()
    features_path = os.path.join(RESULTS_DIR, "top_spam_features.png")
    plt.savefig(features_path)
    plt.close()
    print("Saved:", features_path)

if __name__ == "__main__":
    analyze()
