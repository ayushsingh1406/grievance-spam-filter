import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

from src.data_fetch import load_sms
from src.preprocess import preprocess

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_series(series):
    return series.apply(preprocess)

def train(path="data/processed/merged_dataset.csv"):
    print("Loading dataset:", path)
    df = load_sms(path)
    print("Original dataset shape:", df.shape)
    print(df.label.value_counts())

    # -----------------------------------------
    # UPSAMPLING SPAM to learn outgoing-style spam better
    # -----------------------------------------
    df_ham = df[df.label == 0]
    df_spam = df[df.label == 1]

    # Increase spam representation (2X is ideal for 1200 extra spam)
    df_spam_upsampled = resample(
        df_spam,
        replace=True,
        n_samples=int(len(df_spam) * 2),  # 2X oversampling
        random_state=42
    )

    df = pd.concat([df_ham, df_spam_upsampled], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nAfter upsampling:")
    print(df.label.value_counts())
    print("New dataset shape:", df.shape)

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------
    print("\nPreprocessing text...")
    df['clean_text'] = preprocess_series(df['text'])

    X = df['clean_text']
    y = df['label']

    # -----------------------------------------
    # Train-test split
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------------------
    # Model pipeline
    # -----------------------------------------
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', MultinomialNB())
    ])

    params = {
        'clf__alpha': [0.1, 0.3, 0.5, 1.0]
    }

    print("\nTraining model with GridSearchCV...")
    grid = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------
    print("\nBest params:", grid.best_params_)
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

    # -----------------------------------------
    # Save final model
    # -----------------------------------------
    output_model = os.path.join(MODEL_DIR, "nb_pipeline.joblib")
    joblib.dump(best, output_model)
    print("\nSaved final model to:", output_model)

if __name__ == "__main__":
    train()
