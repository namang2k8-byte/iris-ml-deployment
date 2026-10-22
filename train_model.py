from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    bundle = {
        "model": pipeline,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "dataset": "scikit-learn Iris",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "dataset": "scikit-learn Iris",
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=iris.target_names, output_dict=True
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

if __name__ == "__main__":
    main()
