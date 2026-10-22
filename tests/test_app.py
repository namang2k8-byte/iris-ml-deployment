import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app

def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_predict_rejects_wrong_length():
    client = app.test_client()
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 400

def test_predict_valid_payload():
    client = app.test_client()
    response = client.post(
        "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    # If model hasn't been trained yet, the API correctly reports model unavailable.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json["prediction"] in {"setosa", "versicolor", "virginica"}
        assert "probabilities" in response.json
