from src import main
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.main import (
    load_un_votes_data,
    preprocess_for_similarity,
    calculate_similarity,
    perform_clustering,
    calculate_vote_entropy,
    train_vote_predictor
)
from src.config import UN_VOTES_CSV_PATH, VOTE_MAP

def test_load_un_votes_data():
    """Test the UN votes data loading function."""
    # Test with invalid file path
    result, issues = load_un_votes_data("nonexistent_file.csv")
    assert result is None
    assert issues is None

    # Test with sample data
    sample_data = pd.DataFrame({
        'undl_id': [1, 2, 3],
        'ms_code': ['US', 'UK', 'FR'],
        'ms_name': ['United States', 'United Kingdom', 'France'],
        'ms_vote': ['Y', 'N', 'A'],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'session': [1, 1, 1],
        'title': ['Issue 1', 'Issue 2', 'Issue 3'],
        'subjects': ['Subject 1', 'Subject 2', 'Subject 3'],
        'resolution': ['R1', 'R2', 'R3'],
        'agenda_title': ['A1', 'A2', 'A3'],
        'undl_link': ['L1', 'L2', 'L3']
    })
    
    # Save sample data to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        result, issues = load_un_votes_data(temp_file.name)
        
    assert isinstance(result, pd.DataFrame)
    assert isinstance(issues, list)
    assert len(result) > 0
    assert len(issues) > 0
    assert 'country_identifier' in result.columns
    assert 'vote' in result.columns
    assert 'date' in result.columns
    assert 'year' in result.columns
    assert 'issue' in result.columns

def test_preprocess_for_similarity():
    """Test the preprocessing function for similarity analysis."""
    # Create sample data
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK', 'FR', 'FR'],
        'rcid': [1, 2, 1, 2, 1, 2],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'],
        'date': pd.date_range(start='2020-01-01', periods=6),
        'year': [2020, 2020, 2020, 2020, 2020, 2020],
        'issue': ['Issue 1', 'Issue 2', 'Issue 1', 'Issue 2', 'Issue 1', 'Issue 2']
    })
    
    vote_matrix, country_list, df_filtered = preprocess_for_similarity(
        sample_data, 2020, 2020
    )
    
    assert isinstance(vote_matrix, pd.DataFrame)
    assert isinstance(country_list, list)
    assert isinstance(df_filtered, pd.DataFrame)
    assert len(country_list) == 3  # US, UK, FR
    assert vote_matrix.shape[0] == 3  # Number of countries
    assert vote_matrix.shape[1] == 2  # Number of votes

def test_calculate_similarity():
    """Test the similarity calculation function."""
    # Create sample vote matrix
    vote_matrix = pd.DataFrame({
        1: [1, 0, -1],
        2: [0, 1, 0]
    }, index=['US', 'UK', 'FR'])
    
    similarity_matrix = calculate_similarity(vote_matrix)
    
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (3, 3)  # 3x3 similarity matrix
    assert np.all(np.diag(similarity_matrix) == 1)  # Diagonal should be 1 (self-similarity)

def test_perform_clustering():
    """Test the clustering function."""
    # Create sample similarity matrix
    similarity_matrix = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    country_list = ['US', 'UK', 'FR']
    
    clusters, n_clusters, labels = perform_clustering(
        similarity_matrix, 2, country_list
    )
    
    assert isinstance(clusters, dict)
    assert isinstance(n_clusters, int)
    assert isinstance(labels, np.ndarray)
    assert n_clusters == 2
    assert len(clusters) == 2
    assert len(labels) == 3

def test_calculate_vote_entropy():
    """Test the vote entropy calculation function."""
    # Create sample vote series
    vote_series = pd.Series(['Y', 'Y', 'Y', 'N', 'A'])
    
    entropy = calculate_vote_entropy(vote_series)
    
    assert isinstance(entropy, float)
    assert entropy >= 0  # Entropy should be non-negative

def test_train_vote_predictor():
    """Test the vote predictor training function."""
    # Create sample data
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK', 'FR', 'FR'],
        'issue': ['Issue 1', 'Issue 2', 'Issue 1', 'Issue 2', 'Issue 1', 'Issue 2'],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'],
        'date': pd.date_range(start='2020-01-01', periods=6),
        'year': [2020, 2020, 2020, 2020, 2020, 2020]
    })
    
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        sample_data, 2020, 2021
    )
    
    assert model is not None
    assert isinstance(accuracy, (int, float))  # Allow both int and float
    assert isinstance(report, dict)
    assert isinstance(countries, np.ndarray)
    assert isinstance(train_size, int)
    assert isinstance(test_size, int)

def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    vote_matrix, country_list, df_filtered = preprocess_for_similarity(empty_df, 2020, 2020)
    assert vote_matrix is None
    assert country_list is None
    assert df_filtered is None
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
    vote_matrix, country_list, df_filtered = preprocess_for_similarity(invalid_df, 2020, 2020)
    assert vote_matrix is None
    assert country_list is None
    assert df_filtered is None
    
    # Test with invalid similarity matrix
    invalid_similarity = np.array([[1, 2], [3, 4]])  # Not a proper similarity matrix
    clusters, n_clusters, labels = perform_clustering(invalid_similarity, 2, ['US', 'UK'])
    # The clustering function will still return clusters even for invalid similarity matrix
    assert isinstance(clusters, dict)
    assert n_clusters == 2
    assert isinstance(labels, np.ndarray)

def test_train_vote_predictor_advanced():
    """Test more advanced scenarios for vote predictor training."""
    # Test with data spanning multiple years
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK', 'FR', 'FR'] * 2,
        'issue': ['Issue 1', 'Issue 2'] * 6,
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'] * 2,
        'date': pd.date_range(start='2020-01-01', periods=12),
        'year': [2020] * 6 + [2021] * 6
    })
    
    # Test training with data split across years
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        sample_data, 2020, 2021
    )
    assert model is not None
    assert accuracy > 0  # Should have some accuracy with test data
    assert len(report) > 0
    assert len(countries) == 3
    assert train_size > 0
    assert test_size > 0
    
    # Test with single year data
    single_year_data = sample_data[sample_data['year'] == 2020].copy()
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        single_year_data, 2020, 2021
    )
    assert model is not None
    assert train_size == len(single_year_data)
    assert test_size == 0  # No test data for 2021
    
    # Test with future test year
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        sample_data, 2020, 2022
    )
    assert model is not None
    assert accuracy == 0  # No test data for 2022
    assert report == {}
    assert train_size > 0
    assert test_size == 0

def test_calculate_similarity_advanced():
    """Test more advanced scenarios for similarity calculation."""
    # Test with identical voting patterns
    vote_matrix = pd.DataFrame({
        1: [1, 1, 1],
        2: [0, 0, 0]
    }, index=['US', 'UK', 'FR'])
    
    similarity_matrix = calculate_similarity(vote_matrix)
    assert np.allclose(similarity_matrix, np.ones((3, 3)))  # All countries should be perfectly similar
    
    # Test with opposite voting patterns
    vote_matrix = pd.DataFrame({
        1: [1, -1],
        2: [1, -1]
    }, index=['US', 'UK'])
    
    similarity_matrix = calculate_similarity(vote_matrix)
    assert np.allclose(similarity_matrix[0, 1], -1)  # Countries should be perfectly dissimilar
    
    # Test with missing votes
    vote_matrix = pd.DataFrame({
        1: [1, 0, -1],
        2: [0, 0, 0]
    }, index=['US', 'UK', 'FR'])
    
    similarity_matrix = calculate_similarity(vote_matrix)
    assert similarity_matrix is not None
    assert similarity_matrix.shape == (3, 3)

def test_perform_clustering_advanced():
    """Test more advanced scenarios for clustering."""
    # Test with perfect clusters
    similarity_matrix = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ])
    country_list = ['US', 'UK', 'FR', 'DE']
    
    clusters, n_clusters, labels = perform_clustering(
        similarity_matrix, 2, country_list
    )
    assert n_clusters == 2
    assert len(clusters) == 2
    assert len(labels) == 4
    # Check that US and UK are in the same cluster
    assert any(all(c in cluster for c in ['US', 'UK']) for cluster in clusters.values())
    # Check that FR and DE are in the same cluster
    assert any(all(c in cluster for c in ['FR', 'DE']) for cluster in clusters.values())
    
    # Test with more clusters than countries
    clusters, n_clusters, labels = perform_clustering(
        similarity_matrix, 5, country_list
    )
    assert n_clusters == 4  # Should be limited to number of countries
    
    # Test with single cluster request
    clusters, n_clusters, labels = perform_clustering(
        similarity_matrix, 1, country_list
    )
    assert clusters is None  # Should return None for invalid cluster count
    assert n_clusters == 1  # Original request preserved
    assert labels is None  # No labels generated for invalid case

def test_preprocess_for_similarity_advanced():
    """Test more advanced scenarios for similarity preprocessing."""
    # Test with multiple votes per country-issue combination
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'US', 'UK', 'UK', 'FR'],
        'rcid': [1, 1, 2, 1, 2, 1],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'],
        'date': pd.date_range(start='2020-01-01', periods=6),
        'year': [2020] * 6
    })
    
    vote_matrix, country_list, df_filtered = preprocess_for_similarity(
        sample_data, 2020, 2020
    )
    assert vote_matrix is not None
    assert len(country_list) == 3
    # Should take the last vote for duplicate country-issue combinations
    assert vote_matrix.shape == (3, 2)
    
    # Test with invalid vote values
    invalid_votes_data = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'rcid': [1, 1],
        'vote': ['INVALID', 'X'],
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    
    vote_matrix, country_list, df_filtered = preprocess_for_similarity(
        invalid_votes_data, 2020, 2020
    )
    assert vote_matrix is not None
    # Invalid votes should be mapped to 0 or handled gracefully
    assert not vote_matrix.isnull().any().any()

if __name__ == '__main__':
    pytest.main([__file__])