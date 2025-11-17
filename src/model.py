import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, Optional, Dict, List, Union
import streamlit as st
import joblib

from src.config import VOTE_MAP

logger = logging.getLogger(__name__)

def train_vote_predictor(
    un_data_df: pd.DataFrame,
    train_yr_end: int,
    test_yr_start: int
) -> Tuple[Optional[Pipeline], float, Dict, Optional[np.ndarray], int, int]:
    """Train a vote predictor model."""
    if un_data_df is None or un_data_df.empty:
        return None, 0, {}, None, 0, 0
        
    features = ['country_identifier', 'issue']
    target = 'vote'
    required_cols = features + [target, 'date', 'year']
    
    if not all(col in un_data_df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in un_data_df.columns]
        logger.warning(f"ML Skip: Missing cols: {missing}")
        return None, 0, {}, None, 0, 0
        
    if not pd.api.types.is_datetime64_any_dtype(un_data_df['date']):
        un_data_df['date'] = pd.to_datetime(un_data_df['date'], errors='coerce')
        
    df_ml = un_data_df[required_cols].dropna(subset=features + [target]).copy()
    train_df = df_ml[df_ml['year'] <= train_yr_end]
    test_df = df_ml[df_ml['year'] >= test_yr_start]
    
    if train_df.empty:
        return None, 0, {}, None, 0, len(test_df)
        
    X_train = train_df[features]
    y_train = train_df[target]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, 0, {}, None, len(train_df), len(test_df)
        
    accuracy = 0
    report = {}
    
    if not test_df.empty:
        X_test = test_df[features]
        y_test = test_df[target]
        try:
            y_pred = model_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        except Exception as e:
            logger.warning(f"Error evaluating model: {e}")
            
    all_countries = un_data_df['country_identifier'].unique() if 'country_identifier' in un_data_df else None
    return model_pipeline, accuracy, report, all_countries, len(train_df), len(test_df)

def predict_votes(
    model: Pipeline,
    countries: List[str],
    issue: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Predict votes for a given issue."""
    if model is None or countries is None or len(countries) == 0:
        return None, None
        
    try:
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'country_identifier': countries,
            'issue': [issue] * len(countries)
        })
        
        # Make predictions
        predictions = model.predict(pred_df)
        
        # Create detailed predictions DataFrame
        detailed_predictions = pd.DataFrame({
            'Country': countries,
            'Predicted Vote': predictions
        })
        
        # Create summary of predictions
        vote_counts = pd.DataFrame(
            detailed_predictions['Predicted Vote'].value_counts()
        ).reset_index()
        vote_counts.columns = ['Vote', 'Count']
        
        return vote_counts, detailed_predictions
        
    except Exception as e:
        logger.error(f"Error in vote prediction: {e}")
        return None, None

def save_model(model: Pipeline, filepath: str) -> None:
    """Save a trained model to disk.
    
    Args:
        model: The trained model to save
        filepath: Path where to save the model
        
    Raises:
        ValueError: If model is None or invalid
        Exception: For other errors during saving
    """
    if model is None:
        logger.error("Cannot save None model")
        raise ValueError("Model cannot be None")
        
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str) -> Pipeline:
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        The loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise 