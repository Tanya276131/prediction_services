from fastapi.testclient import TestClient

from src.api.main import app, load_artifacts

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def get_client() -> TestClient:
    load_artifacts()  # TestClient doesn't always fire startup events reliably in older FastAPI
    return TestClient(app)


def test_health_ok():
    client = get_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response():
    client = get_client()
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_name"]
    assert body["model_version_run_id"]


def test_predict_rejects_bad_enum():
    client = get_client()
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Contract"] = "Three year"  # not a valid Literal
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_tenure():
    client = get_client()
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["tenure"] = -5
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_model_info_endpoint():
    client = get_client()
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_name" in body
    assert "run_id" in body
