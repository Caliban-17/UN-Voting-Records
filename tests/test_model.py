import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.model import (
    train_vote_predictor,
    predict_votes,
    save_model,
    load_model
)
import tempfile
import os

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
    
    # Test with valid data
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        sample_data, 2020, 2021
    )
    
    assert model is not None
    assert isinstance(accuracy, (int, float))  # Allow both int and float
    assert isinstance(report, dict)
    assert isinstance(countries, np.ndarray)
    assert isinstance(train_size, int)
    assert isinstance(test_size, int)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        empty_df, 2020, 2021
    )
    assert model is None
    assert accuracy == 0
    assert report == {}
    assert countries is None
    assert train_size == 0
    assert test_size == 0
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        invalid_df, 2020, 2021
    )
    assert model is None
    assert accuracy == 0
    assert report == {}
    assert countries is None
    assert train_size == 0
    assert test_size == 0

def test_predict_votes():
    """Test the vote prediction function."""
    # Create sample training data
    train_data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK'],
        'issue': ['Issue 1', 'Issue 2', 'Issue 1', 'Issue 2'],
        'vote': ['Y', 'N', 'Y', 'A'],
        'date': pd.date_range(start='2020-01-01', periods=4),
        'year': [2020, 2020, 2020, 2020]
    })
    
    # Train model
    model, _, _, countries, _, _ = train_vote_predictor(train_data, 2020, 2021)
    
    # Test prediction
    vote_counts, detailed_predictions = predict_votes(model, countries, 'Issue 1')
    
    assert isinstance(vote_counts, pd.DataFrame)
    assert isinstance(detailed_predictions, pd.DataFrame)
    assert len(detailed_predictions) == len(countries)
    assert 'Vote' in vote_counts.columns
    assert 'Count' in vote_counts.columns
    assert 'Country' in detailed_predictions.columns
    assert 'Predicted Vote' in detailed_predictions.columns

def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        empty_df, 2020, 2021
    )
    assert model is None
    assert accuracy == 0
    assert report == {}
    assert countries is None
    assert train_size == 0
    assert test_size == 0
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'random_col': [1, 2, 3]})
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        invalid_df, 2020, 2021
    )
    assert model is None
    assert accuracy == 0
    assert report == {}
    assert countries is None
    assert train_size == 0
    assert test_size == 0
    
    # Test predict_votes with invalid inputs
    assert predict_votes(None, None, 'Issue 1') == (None, None)
    assert predict_votes(None, [], 'Issue 1') == (None, None)

def test_model_persistence():
    """Test saving and loading of trained models."""
    # Create sample data
    data = pd.DataFrame({
        'country_identifier': ['US', 'US', 'UK', 'UK', 'FR', 'FR'],
        'issue': ['Issue 1', 'Issue 2', 'Issue 1', 'Issue 2', 'Issue 1', 'Issue 2'],
        'vote': ['Y', 'N', 'Y', 'A', 'N', 'Y'],
        'date': pd.date_range(start='2020-01-01', periods=6),
        'year': [2020, 2020, 2020, 2020, 2020, 2020]
    })
    
    # Train initial model
    model, accuracy, report, countries, train_size, test_size = train_vote_predictor(
        data, 2020, 2021
    )
    assert model is not None
    assert isinstance(accuracy, (int, float))
    assert isinstance(report, dict)
    assert isinstance(countries, np.ndarray)
    assert isinstance(train_size, int)
    assert isinstance(test_size, int)
    
    # Test saving and loading
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        # Save model
        save_model(model, tmp.name)
        
        # Load model
        loaded_model = load_model(tmp.name)
        
        # Verify loaded model works
        vote_counts, detailed_predictions = predict_votes(loaded_model, countries, 'Issue 1')
        
        assert loaded_model is not None
        assert isinstance(vote_counts, pd.DataFrame)
        assert isinstance(detailed_predictions, pd.DataFrame)
        assert len(detailed_predictions) == len(countries)
        assert 'Vote' in vote_counts.columns
        assert 'Count' in vote_counts.columns
        assert 'Country' in detailed_predictions.columns
        assert 'Predicted Vote' in detailed_predictions.columns
        
        # Verify predictions match original model
        original_vote_counts, original_detailed_predictions = predict_votes(model, countries, 'Issue 1')
        assert vote_counts.equals(original_vote_counts)
        assert detailed_predictions.equals(original_detailed_predictions)

def test_save_model():
    """Test saving a model to disk."""
    # Create a simple model
    data = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'issue': ['Issue 1', 'Issue 1'],
        'vote': ['Y', 'N'],
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    
    model, _, _, _, _, _ = train_vote_predictor(data, 2020, 2021)
    
    # Test saving
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        filepath = tmp.name
    
    try:
        save_model(model, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def test_load_model():
    """Test loading a model from disk."""
    # Create and save a model
    data = pd.DataFrame({
        'country_identifier': ['US', 'UK'],
        'issue': ['Issue 1', 'Issue 1'],
        'vote': ['Y', 'N'],
        'date': pd.date_range(start='2020-01-01', periods=2),
        'year': [2020, 2020]
    })
    
    model, _, _, countries, _, _ = train_vote_predictor(data, 2020, 2021)
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        filepath = tmp.name
    
    try:
        # Save the model
        save_model(model, filepath)
        
        # Load the model
        loaded_model = load_model(filepath)
        
        # Verify loaded model works
        assert loaded_model is not None
        vote_counts, detailed_predictions = predict_votes(loaded_model, countries, 'Issue 1')
        assert isinstance(vote_counts, pd.DataFrame)
        assert isinstance(detailed_predictions, pd.DataFrame)
        
        # Verify predictions match original model
        original_vote_counts, original_detailed_predictions = predict_votes(model, countries, 'Issue 1')
        assert vote_counts.equals(original_vote_counts)
        assert detailed_predictions.equals(original_detailed_predictions)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def test_model_persistence_error_handling():
    """Test error handling in model persistence."""
    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        load_model('non_existent_file.joblib')
    
    # Test saving with invalid model
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        filepath = tmp.name
    
    try:
        with pytest.raises(Exception):
            save_model(None, filepath)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    pytest.main([__file__]) 