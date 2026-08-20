from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root_redirect():
    # Test that the root URL redirects to /docs
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

def test_health_check_uninitialized():
    # Because we don't trigger the @app.on_event("startup") in the test environment,
    # the services (llm, collection) are not initialized.
    # The health check should correctly report 503.
    response = client.get("/health")
    assert response.status_code == 503
    assert "Services not fully initialized" in response.json()["detail"]

def test_predict_validation_error():
    # Test that invalid payloads are caught by Pydantic before reaching inference
    response = client.post("/predict", json={"wrong_field": "test"})
    assert response.status_code == 422 # Unprocessable Entity
