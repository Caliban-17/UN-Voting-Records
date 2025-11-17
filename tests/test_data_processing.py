import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_processing import (
    load_and_preprocess_data,
    create_vote_matrix,
    calculate_country_entropy,
    get_issue_statistics
)
from src.config import VOTE_MAP

def test_load_and_preprocess_data():
    """Test the data loading and preprocessing function."""
    # Test with invalid file path
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("nonexistent_file.csv")

    # Test with sample data
    sample_data = pd.DataFrame({
        'undl_id': [1, 2, 3],
        'ms_code': ['US', 'UK', 'FR'],
        'ms_vote': ['Y', 'N', 'A'],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'title': ['Issue 1', 'Issue 2', 'Issue 3']
    })
    
    # Save sample data to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        result, issues = load_and_preprocess_data(temp_file.name)
        
    assert isinstance(result, pd.DataFrame)
    assert isinstance(issues, list)
    assert len(result) > 0
    assert len(issues) > 0
    assert 'country_identifier' in result.columns
    assert 'vote' in result.columns
    assert 'date' in result.columns
    assert 'year' in result.columns

    # Test with missing required columns
    invalid_data = pd.DataFrame({
        'random_col': [1, 2, 3],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03']
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        invalid_data.to_csv(temp_file.name, index=False)
        with pytest.raises(ValueError):
            load_and_preprocess_data(temp_file.name)

    # Test with invalid date format
    invalid_date_data = pd.DataFrame({
        'undl_id': [1, 2, 3],
        'ms_code': ['US', 'UK', 'FR'],
        'ms_vote': ['Y', 'N', 'A'],
        'date': ['invalid', '2020-01-02', '2020-01-03'],
        'title': ['Issue 1', 'Issue 2', 'Issue 3']
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        invalid_date_data.to_csv(temp_file.name, index=False)
        result, issues = load_and_preprocess_data(temp_file.name)
        assert len(result) == 3  # All rows should be kept, invalid date becomes NaT

def test_create_vote_matrix():
    """Test the vote matrix creation function."""
    # Create sample data
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK', 'FR', 'FR'],
        'rcid': [1, 2, 1, 2, 1, 2],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'],  # Using string values
        'date': pd.date_range(start='2020-01-01', periods=6),
        'year': [2020, 2020, 2020, 2020, 2020, 2020]
    })
    
    vote_matrix, country_list, df_filtered = create_vote_matrix(
        sample_data, 2020, 2020
    )
    
    assert isinstance(vote_matrix, pd.DataFrame)
    assert isinstance(country_list, list)
    assert isinstance(df_filtered, pd.DataFrame)
    assert len(country_list) == 3  # US, UK, FR
    assert vote_matrix.shape == (3, 2)  # 3 countries, 2 votes
    
    # Test with different year range
    vote_matrix, country_list, df_filtered = create_vote_matrix(
        sample_data, 2019, 2021
    )
    assert vote_matrix is not None
    assert len(country_list) == 3
    
    # Test with empty year range
    vote_matrix, country_list, df_filtered = create_vote_matrix(
        sample_data, 2021, 2022
    )
    assert vote_matrix is None
    assert country_list is None
    assert df_filtered is None

    # Test with duplicate votes for same country and resolution
    duplicate_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'US', 'UK', 'UK'],
        'rcid': [1, 1, 2, 1, 2],
        'vote': ['Y', 'N', 'Y', 'Y', 'N'],
        'date': pd.date_range(start='2020-01-01', periods=5),
        'year': [2020, 2020, 2020, 2020, 2020]
    })
    vote_matrix, country_list, df_filtered = create_vote_matrix(
        duplicate_data, 2020, 2020
    )
    assert vote_matrix is not None
    assert len(country_list) == 2  # Should handle duplicates correctly

    # Test with missing votes
    missing_vote_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK'],
        'rcid': [1, 2, 1, 2],
        'vote': ['Y', np.nan, 'N', 'A'],
        'date': pd.date_range(start='2020-01-01', periods=4),
        'year': [2020, 2020, 2020, 2020]
    })
    vote_matrix, country_list, df_filtered = create_vote_matrix(
        missing_vote_data, 2020, 2020
    )
    assert vote_matrix is not None
    assert vote_matrix.loc['US', 2] == 0.0  # Missing vote should be mapped to 0.0

def test_calculate_country_entropy():
    """Test the country entropy calculation function."""
    # Create sample data
    sample_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'US', 'UK', 'UK', 'UK'],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y']
    })
    
    entropy_scores = calculate_country_entropy(sample_data)
    
    assert isinstance(entropy_scores, pd.Series)
    assert len(entropy_scores) == 2  # US and UK
    assert all(score >= 0 for score in entropy_scores)  # Entropy should be non-negative
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    entropy_scores = calculate_country_entropy(empty_df)
    assert entropy_scores.empty
    
    # Test with single vote per country
    single_vote_df = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'vote': ['Y', 'N']
    })
    entropy_scores = calculate_country_entropy(single_vote_df)
    assert len(entropy_scores) == 2
    assert all(score == 0 for score in entropy_scores)  # Single vote should have 0 entropy

    # Test with all same votes
    same_votes_df = pd.DataFrame({
        'country_identifier': ['US', 'US', 'US', 'UK', 'UK', 'UK'],
        'vote': ['Y', 'Y', 'Y', 'N', 'N', 'N']
    })
    entropy_scores = calculate_country_entropy(same_votes_df)
    assert all(score == 0 for score in entropy_scores)  # All same votes should have 0 entropy

    # Test with maximum entropy (equal distribution of votes)
    max_entropy_df = pd.DataFrame({
        'country_identifier': ['US', 'US', 'US', 'US', 'US', 'US'],
        'vote': ['Y', 'N', 'A', 'Y', 'N', 'A']
    })
    entropy_scores = calculate_country_entropy(max_entropy_df)
    assert entropy_scores['US'] > 0  # Should have positive entropy

def test_get_issue_statistics():
    """Test the issue statistics calculation function."""
    # Create sample data
    sample_data = pd.DataFrame({
        'issue': ['Issue 1', 'Issue 1', 'Issue 2', 'Issue 2', 'Issue 2'],
        'rcid': [1, 2, 3, 4, 5],
        'vote': ['Y', 'N', 'Y', 'A', 'N']
    })
    
    stats = get_issue_statistics(sample_data)
    
    assert isinstance(stats, dict)
    assert 'vote_counts' in stats
    assert 'distributions' in stats
    assert 'entropy' in stats
    
    # Test vote counts
    assert len(stats['vote_counts']) == 2  # Two unique issues
    assert stats['vote_counts']['Issue 1'] == 2  # Two votes for Issue 1
    
    # Test distributions
    assert isinstance(stats['distributions'], pd.DataFrame)
    assert all(col in ['Y', 'N', 'A'] for col in stats['distributions'].columns)
    
    # Test entropy
    assert isinstance(stats['entropy'], pd.Series)
    assert len(stats['entropy']) == 2
    assert all(score >= 0 for score in stats['entropy'])
    
    # Test with empty DataFrame
    empty_stats = get_issue_statistics(pd.DataFrame())
    assert empty_stats == {}

    # Test with single issue
    single_issue_data = pd.DataFrame({
        'issue': ['Issue 1', 'Issue 1', 'Issue 1'],
        'rcid': [1, 2, 3],
        'vote': ['Y', 'N', 'A']
    })
    stats = get_issue_statistics(single_issue_data)
    assert len(stats['vote_counts']) == 1
    assert stats['vote_counts']['Issue 1'] == 3

    # Test with missing votes
    missing_vote_data = pd.DataFrame({
        'issue': ['Issue 1', 'Issue 1', 'Issue 2'],
        'rcid': [1, 2, 3],
        'vote': ['Y', np.nan, 'N']
    })
    stats = get_issue_statistics(missing_vote_data)
    assert len(stats['vote_counts']) == 2
    assert stats['vote_counts']['Issue 1'] == 2  # Both votes should be counted

def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    vote_matrix, country_list, df_filtered = create_vote_matrix(empty_df, 2020, 2020)
    assert vote_matrix is None
    assert country_list is None
    assert df_filtered is None
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
    vote_matrix, country_list, df_filtered = create_vote_matrix(invalid_df, 2020, 2020)
    assert vote_matrix is None
    assert country_list is None
    assert df_filtered is None
    
    # Test with valid vote values
    valid_votes_df = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'rcid': [1, 1],
        'vote': ['Y', 'N'],  # Using string values
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    vote_matrix, country_list, df_filtered = create_vote_matrix(valid_votes_df, 2020, 2020)
    assert isinstance(vote_matrix, pd.DataFrame)
    assert len(country_list) == 2
    assert isinstance(df_filtered, pd.DataFrame)

    # Test with invalid vote values
    invalid_votes_df = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'rcid': [1, 1],
        'vote': ['Invalid', 'Y'],  # Invalid vote value
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    vote_matrix, country_list, df_filtered = create_vote_matrix(invalid_votes_df, 2020, 2020)
    assert vote_matrix is not None
    assert len(country_list) == 1  # Only UK should be included (US has invalid vote)
    assert 'UK' in country_list  # Verify UK is in the list

    # Test with mixed data types
    mixed_df = pd.DataFrame({
        'country_identifier': ['US', 123],  # Mixed types
        'rcid': [1, 1],
        'vote': ['Y', 'N'],
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    vote_matrix, country_list, df_filtered = create_vote_matrix(mixed_df, 2020, 2020)
    assert vote_matrix is not None
    assert len(country_list) == 2  # Both identifiers should be included
    assert set(country_list) == {123, 'US'}  # Check exact values

if __name__ == '__main__':
    pytest.main([__file__]) 