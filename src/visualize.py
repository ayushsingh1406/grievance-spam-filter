import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import joblib
from src.data_fetch import load_sms
from src.preprocess import preprocess

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Ham", "Spam"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

def plot_top_features(model, n=20):
    vectorizer = model.named_steps['tfidf']
    clf = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names_out()
    top_spam = clf.feature_log_prob_[1].argsort()[-n:][::-1]

    df = pd.DataFrame({
        "feature": feature_names[top_spam],
        "log_prob": clf.feature_log_prob_[1][top_spam]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(y=df["feature"], x=df["log_prob"])
    plt.title("Top Spam Features")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "top_spam_features.png"))
    plt.close()

def main():
    df = load_sms("data/processed/merged_dataset.csv")
    df["clean"] = df["text"].apply(preprocess)

    model = joblib.load("models/nb_pipeline.joblib")
    preds = model.predict(df["clean"])

    plot_confusion_matrix(df["label"], preds)
    plot_top_features(model)

    print("Saved confusion matrix and feature importance plots.")

if __name__ == "__main__":
    main()
