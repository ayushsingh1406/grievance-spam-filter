import sys, os
import joblib
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import preprocess

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nb_pipeline.joblib")

model = joblib.load(MODEL_PATH)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Citizen Grievance Spam Classifier",
    page_icon="⚡",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio("Go to:", ["Predict", "Model Insights", "Examples", "About Project"])

st.markdown("""
<style>
/* container */
.result-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f0f2f6;  /* light background */
    border: 1px solid #d1d1d1;
    color: #0b0b0b;               /* dark text for readability */
}

/* headings and paragraphs inside the box */
.result-box h3,
.result-box p,
.result-box strong {
    color: #0b0b0b !important;    /* force dark text */
    margin: 0;
}

/* success (green) box */
.success-box {
    background-color: #dff7df;
    border-left: 8px solid #00a000;
    color: #0b0b0b;
}

/* error (red) box */
.error-box {
    background-color: #ffd6d6;
    border-left: 8px solid #d10000;
    color: #0b0b0b;
}

/* make the Streamlit textarea text dark as well (best-effort selector) */
stTextArea textarea {
    color: white !important;
    background-color: #333333 !important; 
}
            
/* tokens text / small inline outputs */
.streamlit-token-list, .stText {
    color: #0b0b0b !important;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# PAGE 1: PREDICTION UI
# ───────────────────────────────────────────────
if page == "Predict":
    st.title("⚡ Citizen Grievance Spam Classification")
    st.write("Enter a message to classify it as a genuine grievance or spam/fraud message.")

    user_input = st.text_area("Enter message:", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message before predicting.")
        else:
            clean_text = preprocess(user_input)
            pred = model.predict([clean_text])[0]
            proba = float(model.predict_proba([clean_text])[0][1])

            label = "Spam / Fraud" if pred == 1 else "Genuine Grievance"

            # Result box
            if pred == 1:
                st.markdown(f"""
                    <div class="result-box error-box">
                        <h3>🔴 Prediction: {label}</h3>
                        <p><strong>Spam Probability:</strong> {proba:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-box success-box">
                        <h3>🟢 Prediction: {label}</h3>
                        <p><strong>Spam Probability:</strong> {proba:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)

            # Probability bar
            st.write("### Spam Probability Meter")
            st.progress(proba)

            # Token importance explanation
            st.write("### Key Spam Indicator Words (Model Learned)")
            vectorizer = model.named_steps['tfidf']
            clf = model.named_steps['clf']
            feature_names = vectorizer.get_feature_names_out()

            important_tokens = sorted(
                zip(feature_names, clf.feature_log_prob_[1]),
                key=lambda x: x[1], reverse=True
            )[:10]

            st.write(", ".join([tok for tok, _ in important_tokens]))


# ───────────────────────────────────────────────
# PAGE 2: MODEL INSIGHTS
# ───────────────────────────────────────────────
elif page == "Model Insights":
    st.title("📊 Model Insights & Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        st.image("results/confusion_matrix.png")

    with col2:
        st.subheader("Top Spam Features")
        st.image("results/top_spam_features.png")

    st.subheader("📘 Evaluation Summary")
    st.write("""
    - **Accuracy:** 97.71%
    - **Ham Precision:** 98.15%
    - **Spam Precision:** 95.21%
    - **Spam Recall:** 89.95%
    """)


# ───────────────────────────────────────────────
# PAGE 3: EXAMPLES
# ───────────────────────────────────────────────
elif page == "Examples":
    st.title("📝 Example Messages")
    st.write("Test the classifier with sample messages.")

    ham_example = "Streetlight not working near sector 9 for two days."
    spam_example = "Your complaint has been selected for ₹10,000 compensation. Click now to claim."

    if st.button("Test Genuine Grievance Example"):
        clean = preprocess(ham_example)
        pred = model.predict([clean])[0]
        st.success(f"Genuine Grievance Prediction: {pred}")
        st.write(ham_example)

    if st.button("Test Spam Example"):
        clean = preprocess(spam_example)
        pred = model.predict([clean])[0]
        st.error(f"Spam Prediction: {pred}")
        st.write(spam_example)


# ───────────────────────────────────────────────
# PAGE 4: ABOUT PROJECT
# ───────────────────────────────────────────────
elif page == "About Project":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 📌 Overview
    This project builds a Naïve Bayes classifier to detect spam within citizen grievance systems.

    ### 🚀 Features
    - Hybrid dataset (SMS + 500 grievance messages)
    - Clean preprocessing pipeline
    - TF-IDF bigram features
    - Grid-search optimized Naïve Bayes model
    - API + Streamlit user interface
    - 97.7% accuracy on test data

    ### 🛠 Technologies Used
    - Python  
    - Scikit-learn  
    - Streamlit  
    - Flask API  
    - Pandas / NumPy / Matplotlib / Seaborn  

    ### 👨‍💻 Authors
    **Ayush Singh**, CSE Student     
    """)
