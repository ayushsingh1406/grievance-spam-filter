import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_fetch import load_sms
from src.preprocess import preprocess

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_series(series):
    return series.apply(preprocess)

def train(path="data/processed/merged_dataset.csv"):
    print("Loading dataset:", path)
    df = load_sms(path)
    print("Dataset shape:", df.shape)
    print(df.label.value_counts())

    # Preprocess
    print("Preprocessing text...")
    df['clean_text'] = preprocess_series(df['text'])

    X = df['clean_text']
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', MultinomialNB())
    ])

    # Hyperparameters
    params = {
        'clf__alpha': [0.1, 0.3, 0.5, 1.0]
    }

    print("Training model with GridSearchCV...")
    grid = GridSearchCV(pipeline, params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    print("\nBest params:", grid.best_params_)
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

    # Save final model
    output_model = os.path.join(MODEL_DIR, "nb_pipeline.joblib")
    joblib.dump(best, output_model)
    print("\nSaved final model to:", output_model)

if __name__ == "__main__":
    train()
