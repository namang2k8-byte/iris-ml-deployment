from pathlib import Path
import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
app = Flask(__name__)

# Train the model first with: python train_model.py
model_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

def predict_one(values):
    if model_bundle is None:
        raise RuntimeError("Model file not found. Run `python train_model.py` first.")
    array = np.asarray(values, dtype=float).reshape(1, -1)
    prediction = int(model_bundle["model"].predict(array)[0])
    probabilities = model_bundle["model"].predict_proba(array)[0]
    return {
        "prediction": TARGET_NAMES[prediction],
        "class_id": prediction,
        "probabilities": {
            TARGET_NAMES[i]: round(float(probabilities[i]), 6)
            for i in range(len(TARGET_NAMES))
        },
    }

@app.get("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_bundle is not None})

@app.get("/model-info")
def model_info():
    return jsonify({
        "model": "StandardScaler + LogisticRegression",
        "dataset": "Iris flower dataset (scikit-learn)",
        "features": FEATURES,
        "classes": TARGET_NAMES,
        "model_loaded": model_bundle is not None,
    })

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with a 'features' array."}), 400
    values = payload.get("features")
    if not isinstance(values, list) or len(values) != 4:
        return jsonify({"error": "'features' must be an array of 4 numeric values."}), 400
    try:
        numeric_values = [float(value) for value in values]
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError
        result = predict_one(numeric_values)
        return jsonify(result)
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be finite numbers."}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
