1. Introduction

This report evaluates a Naïve Bayes spam classifier trained on a hybrid dataset combining:

SMS Spam Collection dataset (5,574 rows)

Custom 500-row Citizen Grievance dataset

300-row small SMS dataset

Final merged dataset size: 6,336 rows

This hybrid dataset helps the model generalize to both traditional SMS spam and domain-specific municipal complaint spam.

2. Model Details

Vectorizer: TF-IDF (1–2 grams)

Classifier: Multinomial Naïve Bayes

Hyperparameters tuned: alpha=[0.1, 0.3, 0.5, 1.0]

Best alpha: 0.1

Train/Test split: 80/20 stratified

3. Final Metrics (From Test Set)
Overall Accuracy: 97.71%
Class-wise performance:
Class	Precision	Recall	F1-score	Support
Ham (0)	0.9815	0.9916	0.9865	1069
Spam (1)	0.9521	0.8995	0.9251	199
Interpretation

The model is highly accurate for ham messages, meaning real grievances are very unlikely to be mislabeled as spam (important for real deployments).

Spam detection recall is ~89%, which is excellent for Naïve Bayes.

Precision for spam is 95%, meaning very few false alarms.

4. Confusion Matrix
            Predicted
            Ham   Spam
Actual Ham  1060    9
Actual Spam   20   179

Key Observations:

Only 9 ham → spam misclassifications (low false positives).

20 spam → ham were missed (false negatives).

Overall, the classifier is well-balanced.

5. Strengths of This Model

Learns both general spam patterns and domain-specific fraud messages.

High accuracy with low computational cost.

Works extremely well on short complaint texts.

Very low false positives — critical for citizen grievance systems.

6. Limitations

Naïve Bayes assumes feature independence; deep learning models can outperform it on complex patterns.

The model may miss extremely novel or sophisticated scam messages.

Large grammatical variations or code-mixed messages may reduce accuracy (solved by dataset expansion).

7. Future Improvements

Use Logistic Regression or Linear SVM for better spam recall.

Add domain-specific embeddings (FastText / BERT).

Apply oversampling (SMOTE) to improve minority-class recall.

Add Hinglish/Roman Hindi examples for Indian context (optional).