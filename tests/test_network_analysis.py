import pytest
import pandas as pd
import numpy as np
import networkx as nx
from unittest.mock import patch
from src.network_analysis import VotingNetwork, build_network_over_time


@pytest.fixture
def sample_vote_matrix():
    """Create a sample vote matrix for testing."""
    # 3 countries, 3 resolutions
    # US and UK similar, FR different
    data = {"R1": [1, 1, -1], "R2": [1, 1, -1], "R3": [-1, -1, 1]}
    return pd.DataFrame(data, index=["US", "UK", "FR"])


@pytest.fixture
def sample_voting_df():
    """Create a sample DataFrame for build_network_over_time."""
    data = {
        "country_identifier": ["US", "UK", "FR", "US", "UK", "FR"],
        "rcid": ["R1", "R1", "R1", "R2", "R2", "R2"],
        "vote": [1, 1, -1, 1, 1, -1],
        "year": [2020, 2020, 2020, 2021, 2021, 2021],
    }
    return pd.DataFrame(data)


class TestVotingNetwork:

    def test_initialization(self, sample_vote_matrix):
        """Test initialization of VotingNetwork."""
        network = VotingNetwork(sample_vote_matrix)
        assert network.vote_matrix.equals(sample_vote_matrix)
        assert network.similarity_threshold == 0.7  # Default
        assert network.graph is None

        network_custom = VotingNetwork(
            sample_vote_matrix, similarity_threshold=0.5, apply_temporal_weighting=True
        )
        assert network_custom.similarity_threshold == 0.5
        assert network_custom.apply_temporal_weighting is True

    def test_build_graph(self, sample_vote_matrix):
        """Test building the network graph."""
        network = VotingNetwork(sample_vote_matrix, similarity_threshold=0.9)
        graph = network.build_graph()

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3
        # US and UK should be connected (identical votes, sim=1.0)
        # FR should not be connected to US or UK (opposite votes, sim=-1.0)
        assert graph.has_edge("US", "UK")
        assert not graph.has_edge("US", "FR")
        assert not graph.has_edge("UK", "FR")

        # Check edge weight
        edge_data = graph.get_edge_data("US", "UK")
        assert "weight" in edge_data
        assert np.isclose(edge_data["weight"], 1.0)

    def test_build_graph_threshold(self, sample_vote_matrix):
        """Test graph building with different thresholds."""
        # Low threshold, everyone connected?
        # US-FR sim is -1, so even with low threshold they might not connect if threshold > -1
        # Let's make FR slightly similar
        matrix = pd.DataFrame(
            {"R1": [1, 1, 0.5], "R2": [1, 1, 0.5]}, index=["US", "UK", "FR"]
        )

        network = VotingNetwork(matrix, similarity_threshold=0.1)
        graph = network.build_graph()
        assert graph.has_edge("US", "FR")  # Should connect now

    def test_calculate_centrality_metrics(self, sample_vote_matrix):
        """Test centrality metric calculations."""
        network = VotingNetwork(sample_vote_matrix)
        network.build_graph()

        metrics = network.calculate_centrality_metrics()

        expected_metrics = [
            "pagerank",
            "betweenness",
            "eigenvector",
            "degree",
            "closeness",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], pd.Series)
            assert len(metrics[metric]) == 3

    def test_calculate_centrality_metrics_no_graph(self, sample_vote_matrix):
        """Test error when calculating metrics without building graph."""
        network = VotingNetwork(sample_vote_matrix)
        with pytest.raises(ValueError, match="Graph not built"):
            network.calculate_centrality_metrics()

    def test_detect_communities_louvain(self, sample_vote_matrix):
        """Test Louvain community detection."""
        network = VotingNetwork(sample_vote_matrix)
        network.build_graph()

        communities = network.detect_communities(algorithm="louvain")
        assert isinstance(communities, dict)
        # US and UK should likely be in the same community
        # We need to find which community US is in
        us_comm = None
        uk_comm = None
        for comm_id, members in communities.items():
            if "US" in members:
                us_comm = comm_id
            if "UK" in members:
                uk_comm = comm_id

        assert us_comm is not None
        assert us_comm == uk_comm

    def test_detect_communities_algorithms(self, sample_vote_matrix):
        """Test different community detection algorithms."""
        network = VotingNetwork(sample_vote_matrix)
        network.build_graph()

        # Label Propagation
        comm_lp = network.detect_communities(algorithm="label_propagation")
        assert isinstance(comm_lp, dict)

        # Girvan Newman
        comm_gn = network.detect_communities(algorithm="girvan_newman")
        assert isinstance(comm_gn, dict)

        # Invalid algorithm
        with pytest.raises(ValueError, match="Unknown algorithm"):
            network.detect_communities(algorithm="invalid_algo")

    def test_detect_communities_no_graph(self, sample_vote_matrix):
        """Test error when detecting communities without building graph."""
        network = VotingNetwork(sample_vote_matrix)
        with pytest.raises(ValueError, match="Graph not built"):
            network.detect_communities()

    def test_get_network_stats(self, sample_vote_matrix):
        """Test network statistics calculation."""
        network = VotingNetwork(sample_vote_matrix)
        network.build_graph()

        stats = network.get_network_stats()
        assert isinstance(stats, dict)
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "density" in stats
        assert stats["num_nodes"] == 3

    def test_get_network_stats_no_graph(self, sample_vote_matrix):
        """Test error when getting stats without building graph."""
        network = VotingNetwork(sample_vote_matrix)
        with pytest.raises(ValueError, match="Graph not built"):
            network.get_network_stats()

    def test_temporal_weighting(self):
        """Test temporal weighting logic."""
        # Create matrix with years
        matrix = pd.DataFrame({"R1": [1, 1], "R2": [1, 1]}, index=["A", "B"])
        years = np.array([2020, 2021])

        network = VotingNetwork(matrix, apply_temporal_weighting=True, years=years)

        # Mock datetime to have a fixed current year
        with patch("src.network_analysis.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2022

            # Access the private method directly to test logic
            weighted = network._apply_temporal_weights(matrix, years)

            # 2021 should have higher weight than 2020
            # weight = decay ^ (current - year)
            # 2021: decay ^ 1
            # 2020: decay ^ 2
            # Since decay < 1, decay^1 > decay^2
            assert weighted.iloc[0, 1] > weighted.iloc[0, 0]  # R2 > R1


def test_build_network_over_time(sample_voting_df):
    """Test building networks over time windows."""
    networks = build_network_over_time(
        sample_voting_df, start_year=2020, end_year=2021, window="1Y"
    )

    assert isinstance(networks, list)
    assert len(networks) == 2

    label1, net1 = networks[0]
    assert label1 == "2020"
    assert isinstance(net1, VotingNetwork)
    assert net1.graph is not None

    label2, net2 = networks[1]
    assert label2 == "2021"
    assert isinstance(net2, VotingNetwork)


def test_build_network_over_time_invalid_window(sample_voting_df):
    """Test invalid window parameter."""
    with pytest.raises(ValueError, match="Invalid window"):
        build_network_over_time(sample_voting_df, 2020, 2021, window="invalid")
