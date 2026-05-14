import pytest
import pandas as pd
from src.data_processing import (
    _load_and_preprocess_data_impl as load_and_preprocess_data,
)
from src.network_analysis import VotingNetwork
from src.soft_power import SoftPowerCalculator


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "undl_id": ["1001", "1001", "1001", "1002", "1002", "1002"],
            "ms_code": ["US", "UK", "FR", "US", "UK", "FR"],
            "ms_vote": ["Y", "Y", "N", "Y", "Y", "N"],
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-02-01",
                "2020-02-01",
                "2020-02-01",
            ],
            "title": ["Res 1", "Res 1", "Res 1", "Res 2", "Res 2", "Res 2"],
        }
    )
    filepath = tmp_path / "test_votes.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


def test_full_pipeline(sample_csv):
    """Test the full pipeline from data loading to soft power calculation."""
    # 1. Load Data
    # Integration tests call the same loading path used by the web API.

    df, issues = load_and_preprocess_data(sample_csv, use_cache=False)

    assert not df.empty
    assert len(issues) > 0
    assert "vote" in df.columns
    assert "year" in df.columns

    print("\nDEBUG: DF Head:\n", df.head())
    print("\nDEBUG: Unique votes:", df["vote"].unique())
    print("\nDEBUG: Vote dtype:", df["vote"].dtype)

    # 2. Create Vote Matrix (Manual step in app, but part of logic)
    # The app calls preprocess_for_similarity which calls create_vote_matrix
    from src.data_processing import create_vote_matrix

    vote_matrix, countries, filtered_df = create_vote_matrix(df, 2020, 2020)

    assert vote_matrix is not None
    assert len(countries) == 3
    assert vote_matrix.shape == (3, 2)  # 3 countries, 2 resolutions

    # 3. Build Network
    network = VotingNetwork(vote_matrix, similarity_threshold=0.5)
    graph = network.build_graph()

    assert graph.number_of_nodes() == 3
    # US and UK should be connected
    assert graph.has_edge("US", "UK")

    # 4. Calculate Centrality
    metrics = network.calculate_centrality_metrics()
    assert "pagerank" in metrics

    # 5. Detect Communities
    communities = network.detect_communities()
    assert len(communities) >= 1

    # 6. Calculate Soft Power
    calculator = SoftPowerCalculator(network, filtered_df, centrality_metrics=metrics)
    soft_power_scores = calculator.aggregate_soft_power_score()

    assert not soft_power_scores.empty
    assert "US" in soft_power_scores
    assert "UK" in soft_power_scores
    assert "FR" in soft_power_scores

    # US and UK should have higher scores than FR (who is isolated/disagreeing)
    assert soft_power_scores["US"] > soft_power_scores["FR"]


def test_pipeline_no_data():
    """Test pipeline behavior with empty data."""
    # Should handle gracefully
