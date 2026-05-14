import pytest
import json
from web_app import app, load_data


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Ensure data is loaded
    load_data()
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "data_loaded" in data


def test_data_summary_endpoint(client):
    """Test the data summary endpoint"""
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_votes" in data
    assert "countries" in data


def test_clustering_endpoint(client):
    """Test the clustering analysis endpoint"""
    payload = {"start_year": 2015, "end_year": 2020, "num_clusters": 5}
    response = client.post(
        "/api/analysis/clustering",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "clusters" in data
    assert "num_clusters" in data


def test_soft_power_endpoint(client):
    """Test the soft power analysis endpoint"""
    payload = {"start_year": 2015, "end_year": 2020}
    response = client.post(
        "/api/analysis/soft-power",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "scores" in data
    assert "centrality" in data


def test_network_graph_endpoint(client):
    """Test the network graph visualization endpoint"""
    payload = {
        "start_year": 2018,
        "end_year": 2020,
        "layout": "force",
        "threshold": 0.7,
    }
    response = client.post(
        "/api/visualization/network",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data
    assert "layout" in data


def test_pca_endpoint(client):
    """Test the PCA visualization endpoint"""
    payload = {"start_year": 2015, "end_year": 2020}
    response = client.post(
        "/api/visualization/pca",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data
    assert "layout" in data


def test_invalid_input(client):
    """Test error handling for invalid input"""
    payload = {"start_year": 2025, "end_year": 2020}  # Invalid: start > end
    response = client.post(
        "/api/analysis/clustering",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_bloc_timeline_endpoint(client):
    payload = {
        "start_year": 2000,
        "end_year": 2020,
        "window": 5,
        "num_clusters": 4,
    }
    response = client.post(
        "/api/analysis/bloc-timeline",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "data" in data
    assert "layout" in data


def test_methods_endpoint(client):
    """Methods metadata endpoint returns assumptions and caveats"""
    response = client.get("/api/methods")
    assert response.status_code == 200
    data = response.get_json()
    assert "methods" in data
    assert "caveats" in data


def test_insights_filtered_endpoint(client):
    """Insights should support filtered date windows via POST"""
    payload = {"start_year": 2015, "end_year": 2020}
    response = client.post(
        "/api/insights",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "insights" in data
    assert "meta" in data


def test_compare_rejects_unknown_country_code(client):
    payload = {
        "country_a": "ZZZ",
        "country_b": "RUS",
        "start_year": 2010,
        "end_year": 2020,
    }
    response = client.post(
        "/api/analysis/compare",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_report_endpoint_markdown(client):
    payload = {"start_year": 2015, "end_year": 2020, "format": "markdown"}
    response = client.post(
        "/api/report",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "UN Voting Intelligence Report" in body
    assert response.headers.get("Content-Type", "").startswith("text/markdown")


def test_report_endpoint_pdf(client):
    payload = {"start_year": 2015, "end_year": 2020, "format": "pdf"}
    response = client.post(
        "/api/report",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.headers.get("Content-Type", "").startswith("application/pdf")
    assert len(response.get_data()) > 1000
