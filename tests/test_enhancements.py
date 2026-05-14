import pytest
import json
from web_app import app, load_data


@pytest.fixture
def client():
    app.config["TESTING"] = True
    load_data()
    with app.test_client() as client:
        yield client


def test_prediction_issues(client):
    """Test getting list of issues for prediction"""
    response = client.get("/api/prediction/issues")
    assert response.status_code == 200
    data = response.get_json()
    assert "issues" in data
    assert isinstance(data["issues"], list)
    assert len(data["issues"]) > 0


def test_train_model(client):
    """Test model training endpoint"""
    # Use a small range for speed
    payload = {"train_end": 2015, "test_start": 2016}
    response = client.post(
        "/api/prediction/train",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "accuracy" in data
    assert "report" in data


def test_predict_vote(client):
    """Test vote prediction endpoint"""
    # First ensure model is trained
    client.post(
        "/api/prediction/train",
        data=json.dumps({"train_end": 2015, "test_start": 2016}),
        content_type="application/json",
    )

    payload = {"issue": "Nuclear weapons", "countries": ["USA", "RUS", "CHN"]}
    response = client.post(
        "/api/prediction/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "summary" in data
    assert "details" in data
    assert len(data["details"]) == 3


def test_compare_countries(client):
    """Test country comparison endpoint"""
    payload = {
        "country_a": "USA",
        "country_b": "RUS",
        "start_year": 2010,
        "end_year": 2020,
    }
    response = client.post(
        "/api/analysis/compare",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "similarity" in data
    assert "anomalies" in data
    assert isinstance(data["anomalies"], list)


def test_divergence_report(client):
    """Test divergence report endpoint"""
    payload = {"countries": ["USA", "GBR", "FRA"], "year": 2015, "window": 2}
    response = client.post(
        "/api/analysis/divergence-report",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "cluster_size" in data
    assert "top_divisive_issues" in data
    assert "potential_leavers" in data
