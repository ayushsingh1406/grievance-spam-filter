from flask import Flask, request, jsonify
import joblib
from src.preprocess import preprocess
import os

MODEL_PATH = os.path.join("models", "nb_pipeline.joblib")
app = Flask(__name__)

# load model once
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    text = body.get("text", "")
    cleaned = preprocess(text)
    label = int(model.predict([cleaned])[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([cleaned])[0][1])
    return jsonify({"label": label, "prob_spam": prob})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
