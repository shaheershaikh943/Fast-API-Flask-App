from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
SAMPLES = client.get("/demo-samples").json()["samples"]
SAMPLE_A = SAMPLES[0]["features"]
SAMPLE_B = SAMPLES[1]["features"]


def test_prediction_page_is_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "prediction playground" in response.text.lower()
    assert "Run prediction" in response.text


def test_health_endpoint_reports_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_features_endpoint_lists_30_features():
    response = client.get("/features")
    assert response.status_code == 200
    assert response.json()["count"] == 30
    assert len(response.json()["features_in_required_order"]) == 30


def test_predict_returns_valid_scores_for_malignant_benchmark_example():
    response = client.post("/predict", json={"features": SAMPLE_A})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_benchmark_class"] == "malignant"
    scores = body["model_probability_scores"]
    assert set(scores) == {"malignant", "benign"}
    assert abs(sum(scores.values()) - 1.0) < 1e-9
    assert scores["malignant"] > 0.99


def test_predict_returns_valid_scores_for_benign_benchmark_example():
    response = client.post("/predict", json={"features": SAMPLE_B})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_benchmark_class"] == "benign"
    assert body["model_probability_scores"]["benign"] > 0.8


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_predict_rejects_nonpositive_measurements():
    values = list(SAMPLE_A)
    values[0] = 0
    response = client.post("/predict", json={"features": values})
    assert response.status_code == 422


def test_demo_samples_are_valid_30_feature_inputs():
    response = client.get("/demo-samples")
    assert response.status_code == 200
    samples = response.json()["samples"]
    assert len(samples) >= 2
    assert all(len(item["features"]) == 30 for item in samples)
