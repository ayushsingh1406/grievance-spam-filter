Naïve Bayes Spam Filter for Citizen Grievances (NLP Project)
1. Overview

This project builds a machine learning model that classifies citizen grievance messages as ham (genuine issue) or spam (fraud, scam, fake notices).
It uses NLP techniques, TF-IDF vectorization, and Naïve Bayes classification.

The model is trained on a hybrid dataset:

SMS Spam Collection dataset (benchmark)

Custom 500-row Citizen Grievance dataset

Small SMS dataset

Total samples after merging: 6,336

2. Project Structure
grievance-spam-filter/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   └── nb_pipeline.joblib
│
├── results/
│   ├── confusion_matrix.png
│   ├── top_spam_features.png
│   └── evaluation_report.md
│
├── src/
│   ├── data_fetch.py
│   ├── preprocess.py
│   ├── merge_datasets.py
│   ├── train.py
│   ├── visualize.py
│   └── app.py     (API)
│
└── README.md

3. Installation
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

4. Training the Model
python -m src.merge_datasets
python -m src.train

5. Running the API
python -m src.app


Test:

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"text":"Road flooded near sector 5"}'

6. Results Summary
Metric	Score
Accuracy	97.71%
Spam Precision	95.21%
Spam Recall	89.95%
Ham Precision	98.15%

This performance is excellent for a Naïve Bayes model.

7. Visualizations

Confusion Matrix

Top Spam Features

Token Statistics

Located in: results/

8. Future Improvements

Switch to Logistic Regression or Linear SVM

Add Hinglish / Roman Hindi examples

Use FastText/BERT embeddings

Improve spam recall using class-weighting