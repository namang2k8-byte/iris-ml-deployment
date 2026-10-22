# End-to-End Machine Learning Deployment: Iris Predictor

A beginner-friendly machine learning application that trains a classifier, serializes it with Joblib, exposes a Flask prediction API, provides a small web interface, and includes tests and Docker support.

## Project overview

- **Problem:** Predict Iris flower species from four measurements.
- **Dataset:** Built-in scikit-learn Iris dataset (150 samples, 4 numeric features, 3 classes).
- **Preprocessing:** `StandardScaler`, included inside a scikit-learn pipeline to prevent train/test leakage.
- **Model:** Logistic Regression.
- **Evaluation:** Stratified 80/20 train-test split, accuracy, confusion matrix, and classification report.
- **Serialization:** `joblib.dump()` and `joblib.load()`.
- **API:** Flask JSON endpoints.
- **Interface:** Browser-based form.
- **Testing:** Pytest.
- **Deployment option:** Docker.

## Project structure

```text
ml_deployment_project/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── templates/
│   └── index.html
├── tests/
│   └── test_app.py
└── README.md
```

After training, `model.joblib` and `metrics.json` are generated locally.

## Run locally

Use Python 3.10 or newer.

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies and train

```bash
pip install -r requirements.txt
python train_model.py
```

The script saves `model.joblib` and `metrics.json`. It prints test accuracy and a confusion matrix. Report the actual result produced by your run; do not assume an accuracy value before running the experiment.

### 3. Start the API and web interface

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## API reference

### Health check

```bash
curl http://127.0.0.1:5000/health
```

### Model metadata

```bash
curl http://127.0.0.1:5000/model-info
```

### Prediction

Feature order is **sepal length, sepal width, petal length, petal width**, all in centimetres.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Example response shape:

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.0
  }
}
```

Probability values shown above are illustrative; exact values depend on the trained model and library versions.

## Test

Train the model first, then run:

```bash
pytest -q
```

## Run with Docker

Build and run:

```bash
docker build -t iris-predictor .
docker run --rm -p 5000:5000 iris-predictor
```

Then visit http://127.0.0.1:5000.

## Deploy to a hosting platform

The app can be deployed to a Python-capable host or container platform. A generic production start command is:

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

Add `gunicorn` to `requirements.txt` before using that command. Ensure `model.joblib` exists in the deployed image or build process. Configure the platform's health check to use `/health`.

For a public deployment:
1. Push the project to a Git repository.
2. Select a Python web service or Docker deployment.
3. Set the build/install command to `pip install -r requirements.txt && python train_model.py` (or build the supplied Docker image).
4. Set the start command to `gunicorn --bind 0.0.0.0:$PORT app:app` after adding Gunicorn.
5. Deploy and test `/health` and `/predict`.

## Design and security notes

- Only load Joblib files from trusted sources. Pickle-based formats can execute code during deserialization.
- The example is educational and does not include authentication, rate limiting, monitoring, or production secrets management.
- Validate request inputs and keep model and preprocessing steps in the same pipeline.
- For a real production project, add CI, structured logs, monitoring, versioned model artifacts, and drift checks.

## Final submission checklist

- [ ] Run `python train_model.py` and retain `metrics.json`.
- [ ] Run `pytest -q` and record the result.
- [ ] Test `/predict` with valid and invalid inputs.
- [ ] Build and run Docker image, if Docker is available.
- [ ] Deploy to your selected host and add the public URL here: **Deployment URL: TBD**
- [ ] Add screenshots of the running interface and successful API request.
- [ ] Include the generated presentation in your submission.
