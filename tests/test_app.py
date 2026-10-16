from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
SAMPLE = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
          1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
          25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_features_endpoint_lists_30_features():
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json()["count"] == 30


def test_predict_endpoint_returns_scores():
    response = client.post("/predict", json={"features": SAMPLE})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_benchmark_class"] in {"malignant", "benign"}
    scores = body["model_probability_scores"]
    assert set(scores) == {"malignant", "benign"}
    assert abs(sum(scores.values()) - 1.0) < 0.00001


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422
