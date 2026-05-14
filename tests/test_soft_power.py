import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.soft_power import SoftPowerCalculator, track_soft_power_over_time


@pytest.fixture
def mock_network():
    """Create a mock VotingNetwork."""
    network = MagicMock()
    network.graph.nodes.return_value = ["US", "UK", "FR"]

    # Mock centrality metrics
    network.calculate_centrality_metrics.return_value = {
        "pagerank": pd.Series({"US": 0.5, "UK": 0.3, "FR": 0.2}),
        "betweenness": pd.Series({"US": 0.4, "UK": 0.4, "FR": 0.2}),
        "eigenvector": pd.Series({"US": 0.6, "UK": 0.3, "FR": 0.1}),
        "degree": pd.Series({"US": 0.5, "UK": 0.5, "FR": 0.2}),
        "closeness": pd.Series({"US": 0.6, "UK": 0.5, "FR": 0.4}),
    }
    return network


@pytest.fixture
def sample_vote_data():
    """Create sample voting data."""
    # US and UK vote together often
    # FR votes differently
    data = {
        "rcid": ["R1", "R1", "R1", "R2", "R2", "R2", "R3", "R3", "R3"],
        "country_identifier": ["US", "UK", "FR", "US", "UK", "FR", "US", "UK", "FR"],
        "vote": [1, 1, -1, 1, 1, -1, -1, -1, 1],
        "date": pd.to_datetime(
            ["2020-01-01"] * 3 + ["2020-02-01"] * 3 + ["2020-03-01"] * 3
        ),
        "year": [2020] * 9,
    }
    return pd.DataFrame(data)


class TestSoftPowerCalculator:

    def test_initialization(self, mock_network, sample_vote_data):
        """Test initialization logic."""
        # With pre-calculated metrics
        metrics = mock_network.calculate_centrality_metrics()
        calc = SoftPowerCalculator(
            mock_network, sample_vote_data, centrality_metrics=metrics
        )
        assert calc.centrality_metrics == metrics

        # Without pre-calculated metrics (should call calculate_centrality_metrics)
        mock_network.reset_mock()
        mock_network.calculate_centrality_metrics.return_value = metrics
        calc2 = SoftPowerCalculator(mock_network, sample_vote_data)
        mock_network.calculate_centrality_metrics.assert_called_once()
        assert calc2.centrality_metrics == metrics

    def test_calculate_vote_swaying_rate(self, mock_network, sample_vote_data):
        """Test vote swaying rate calculation."""
        calc = SoftPowerCalculator(mock_network, sample_vote_data)

        # US votes: R1(1), R2(1), R3(-1)
        # UK votes: R1(1), R2(1), R3(-1) -> Matches US 3/3 times
        # FR votes: R1(-1), R2(-1), R3(1) -> Matches US 0/3 times

        # Note: The implementation calculates how often OTHERS follow THIS country
        # For US:
        # R1: UK(1) match, FR(-1) mismatch. 1/2
        # R2: UK(1) match, FR(-1) mismatch. 1/2
        # R3: UK(-1) match, FR(1) mismatch. 1/2
        # Avg: 0.5

        # We need to lower min_votes because sample only has 3 votes per country
        swaying = calc.calculate_vote_swaying_rate(min_votes=1)

        assert isinstance(swaying, pd.Series)
        assert "US" in swaying
        assert np.isclose(swaying["US"], 0.5)
        assert np.isclose(swaying["UK"], 0.5)  # Symmetric in this case

    def test_aggregate_soft_power_score(self, mock_network, sample_vote_data):
        """Test score aggregation and normalization."""
        calc = SoftPowerCalculator(mock_network, sample_vote_data)

        # Mock swaying rate to avoid recalculation and control values
        with patch.object(calc, "calculate_vote_swaying_rate") as mock_sway:
            mock_sway.return_value = pd.Series({"US": 1.0, "UK": 0.5, "FR": 0.0})

            # Use custom weights for easier verification
            weights = {
                "pagerank": 0.0,
                "betweenness": 0.0,
                "eigenvector": 0.0,
                "vote_swaying": 1.0,
            }

            scores = calc.aggregate_soft_power_score(custom_weights=weights)

            # Since swaying is 1.0, 0.5, 0.0 and weight is 1.0 for swaying
            # Normalization of swaying: (x - 0) / (1 - 0) -> 1.0, 0.5, 0.0
            # Final normalization: 1.0, 0.5, 0.0
            assert np.isclose(scores["US"], 1.0)
            assert np.isclose(scores["UK"], 0.5)
            assert np.isclose(scores["FR"], 0.0)

    def test_get_metric_breakdown(self, mock_network, sample_vote_data):
        """Test metric breakdown retrieval."""
        calc = SoftPowerCalculator(mock_network, sample_vote_data)

        with patch.object(calc, "calculate_vote_swaying_rate") as mock_sway:
            mock_sway.return_value = pd.Series({"US": 0.8})

            breakdown = calc.get_metric_breakdown("US")

            assert isinstance(breakdown, dict)
            assert breakdown["pagerank"] == 0.5
            assert breakdown["vote_swaying"] == 0.8
            assert "betweenness" in breakdown
            assert "eigenvector" in breakdown


def test_track_soft_power_over_time(sample_vote_data):
    """Test tracking soft power over time."""
    # Mock VotingNetwork to avoid complex graph building
    with patch("src.network_analysis.VotingNetwork") as MockNetworkClass:
        mock_net_instance = MockNetworkClass.return_value
        mock_net_instance.graph.nodes.return_value = ["US", "UK", "FR"]
        mock_net_instance.calculate_centrality_metrics.return_value = {
            "pagerank": {"US": 0.5},
            "betweenness": {"US": 0.5},
            "eigenvector": {"US": 0.5},
        }

        # Mock SoftPowerCalculator to avoid complex calculations
        with patch("src.soft_power.SoftPowerCalculator") as MockCalcClass:
            mock_calc_instance = MockCalcClass.return_value
            mock_calc_instance.centrality_metrics = {
                "pagerank": {"US": 0.5},
                "betweenness": {"US": 0.5},
                "eigenvector": {"US": 0.5},
            }
            mock_calc_instance.aggregate_soft_power_score.return_value = pd.Series(
                {"US": 0.9, "UK": 0.5, "FR": 0.1}
            )

            df_result = track_soft_power_over_time(
                sample_vote_data, start_year=2020, end_year=2020, frequency="1Y"
            )

            assert isinstance(df_result, pd.DataFrame)
            assert not df_result.empty
            assert "year" in df_result.columns
            assert "soft_power_score" in df_result.columns
            assert len(df_result) == 3  # 3 countries


def test_track_soft_power_invalid_frequency(sample_vote_data):
    """Test invalid frequency parameter."""
    with pytest.raises(ValueError, match="Invalid frequency"):
        track_soft_power_over_time(sample_vote_data, 2020, 2020, frequency="10Y")
