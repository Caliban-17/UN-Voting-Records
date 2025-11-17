# Single file: UN Voting Pattern Visualizer & Predictor (Streamlit App - CSV ONLY)
# RELIES ON LOCAL CSV FILE: data/2025_03_31_ga_voting_corr1.csv
# Added Issue Salience, Resolution Polarity, Vote Entropy visualizations. Removed Heatmap. Corrected syntax errors.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from scipy.stats import entropy
from typing import Tuple, Optional, Dict, List, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import psutil
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack

from src.config import (
    UN_VOTES_CSV_PATH,
    VOTE_MAP,
    COLUMN_RENAME_MAP,
    ESSENTIAL_COLUMNS,
    OPTIONAL_COLUMNS,
    DEFAULT_N_CLUSTERS,
    DEFAULT_TRAIN_TEST_SPLIT_YEAR,
    DEFAULT_N_TOP_ISSUES,
    DEFAULT_FIG_SIZE,
    DEFAULT_MARKER_SIZE,
    DEFAULT_ALPHA,
    PLOT_COLORS
)
from src.data_processing import (
    load_and_preprocess_data,
    create_vote_matrix,
    calculate_country_entropy,
    get_issue_statistics
)
from src.model import train_vote_predictor, predict_votes
from src.visualization import (
    plot_cluster_vote_distribution,
    plot_pca_projection,
    plot_issue_salience,
    plot_resolution_polarity,
    plot_entropy_distribution
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for shutdown management
shutdown_event = asyncio.Event()

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def create_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create minimal features to reduce memory usage."""
    if shutdown_event.is_set():
        logger.info("Shutdown requested during feature creation")
        return None
        
    df = df.copy()
    df['text_features'] = df['country_identifier'].astype(str) + ' ' + df['issue'].astype(str)
    gc.collect()
    return df

def train_vote_predictor_async(df: pd.DataFrame, batch_size: int = 5000) -> Tuple[LogisticRegression, CountVectorizer]:
    """Train a vote predictor model with minimal memory usage."""
    if shutdown_event.is_set():
        logger.info("Shutdown requested before starting model training")
        return None, None

    logger.info("Starting model training with minimal memory usage")
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    try:
        # Create minimal features
        df = create_minimal_features(df)
        if df is None:
            logger.warning("Feature creation aborted due to shutdown")
            return None, None
            
        gc.collect()
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.04, random_state=42)
        logger.info(f"Memory after data splitting: {get_memory_usage():.2f} MB")
        
        # Initialize vectorizer with minimal memory settings and better numerical stability
        vectorizer = CountVectorizer(
            max_features=3000,
            min_df=10,
            max_df=0.9,
            binary=True,
            dtype=np.float32
        )
        
        # Fit vectorizer on all text data
        logger.info("Fitting vectorizer on all text data...")
        vectorizer.fit(df['text_features'])
        logger.info(f"Memory after vectorizer fitting: {get_memory_usage():.2f} MB")
        
        # Initialize model with proper warm start and better numerical stability
        model = LogisticRegression(
            max_iter=200,
            tol=1e-4,
            C=0.1,
            solver='saga',
            warm_start=True,
            n_jobs=-1,
            random_state=42,
            penalty='l2',
            class_weight='balanced'
        )
        
        # Get unique classes from the entire dataset to ensure consistent shape
        all_classes = sorted(df['vote'].unique())
        n_classes = len(all_classes)
        n_features = len(vectorizer.get_feature_names_out())
        
        logger.info(f"Found {n_classes} unique vote classes: {all_classes}")
        
        # Initialize model with correct shape and small initial values
        model.classes_ = np.array(all_classes)
        model.coef_ = np.zeros((len(all_classes), n_features))
        model.intercept_ = np.zeros(len(all_classes))
        
        # Train in batches
        n_batches = len(train_df) // batch_size + (1 if len(train_df) % batch_size else 0)
        logger.info(f"Training model in {n_batches} batches of size {batch_size}")
        
        for batch_idx in range(n_batches):
            if shutdown_event.is_set():
                logger.info("Shutdown requested during batch training")
                return None, None
                
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_df))
            batch_df = train_df.iloc[start_idx:end_idx]
            
            # Transform text features with numerical stability checks
            X_batch = vectorizer.transform(batch_df['text_features'])
            y_batch = batch_df['vote']
            
            # Add small epsilon to prevent division by zero
            X_batch.data = np.clip(X_batch.data, 1e-10, None)
            
            # Train on batch
            try:
                model.fit(X_batch, y_batch)
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{n_batches}")
                    logger.info(f"Memory usage: {get_memory_usage():.2f} MB")
            except Exception as e:
                logger.error(f"Error in batch {batch_idx + 1}: {str(e)}")
                continue
            
            gc.collect()
        
        # Evaluate on test set in batches
        test_batches = len(test_df) // batch_size + (1 if len(test_df) % batch_size else 0)
        logger.info(f"Evaluating on {test_batches} test batches")
        
        all_predictions = []
        all_true_labels = []
        
        for batch_idx in range(test_batches):
            if shutdown_event.is_set():
                logger.info("Shutdown requested during evaluation")
                return model, vectorizer
                
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_df))
            test_batch = test_df.iloc[start_idx:end_idx]
            
            X_test = vectorizer.transform(test_batch['text_features'])
            X_test.data = np.clip(X_test.data, 1e-10, None)
            y_test = test_batch['vote']
            
            predictions = model.predict(X_test)
            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
            
            if batch_idx % 2 == 0:
                logger.info(f"Processed test batch {batch_idx + 1}/{test_batches}")
        
        # Calculate final metrics with zero_division handling
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
        
        logger.info(f"Final model accuracy: {accuracy:.2%}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1-score: {f1:.2%}")
        logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
        
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return None, None

async def main():
    """Main function to test core functionality."""
    try:
        # Load the data
        logger.info("Loading UN voting data...")
        df, issues = load_un_votes_data(UN_VOTES_CSV_PATH)
        
        if df is None or df.empty:
            logger.error("Failed to load data")
            return
            
        logger.info(f"Successfully loaded {len(df)} records with {len(issues)} unique issues")
        
        # Log available years
        available_years = sorted(df['year'].unique())
        logger.info(f"Available years in the data: {available_years}")
        
        # Test preprocessing
        logger.info("Testing preprocessing...")
        vote_matrix, countries, filtered_df = preprocess_for_similarity(df, 2020, 2023)
        
        if vote_matrix is not None:
            logger.info(f"Successfully created vote matrix with {len(countries)} countries")
            
            # Test similarity calculation
            logger.info("Testing similarity calculation...")
            similarity_matrix = calculate_similarity(vote_matrix)
            
            if similarity_matrix is not None:
                logger.info("Successfully calculated similarity matrix")
                
                # Test clustering
                logger.info("Testing clustering...")
                clusters, n_clusters, labels = perform_clustering(similarity_matrix, 5, countries)
                
                if clusters is not None:
                    logger.info(f"Successfully created {n_clusters} clusters")
                    
                    # Test model training
                    logger.info("Testing model training...")
                    model, vectorizer = train_vote_predictor_async(df, batch_size=5000)
                    
                    if model is not None:
                        logger.info(f"Successfully trained model with {len(df)} records")
                    else:
                        logger.warning("Model training failed")
                else:
                    logger.warning("Clustering failed")
            else:
                logger.warning("Similarity calculation failed")
        else:
            logger.warning("Preprocessing failed")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        if shutdown_event.is_set():
            logger.info("Graceful shutdown complete")
            logger.info("Cleaning up resources...")
            gc.collect()
            logger.info("Final memory usage: {:.2f} MB".format(get_memory_usage()))

def load_un_votes_data(filepath: str) -> tuple[pd.DataFrame | None, list | None]:
    """Loads UN voting data from the specified CSV, maps columns, preprocesses."""
    abs_filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        logger.error(f"File not found: '{filepath}' ({abs_filepath})")
        return None, None
    try:
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} rows initially.")
        
        column_rename_map = {
            'undl_id': 'rcid',
            'ms_code': 'country_code',
            'ms_name': 'country_name',
            'ms_vote': 'vote',
            'date': 'date',
            'session': 'session',
            'title': 'descr',
            'subjects': 'issue',
            'resolution': 'resolution',
            'agenda_title': 'agenda',
            'undl_link': 'undl_link'
        }
        
        source_columns_needed = list(column_rename_map.keys())
        missing_source_cols = [col for col in source_columns_needed if col not in df.columns]
        if missing_source_cols:
            raise ValueError(f"CSV missing source columns: {', '.join(missing_source_cols)}")
            
        df.rename(columns=column_rename_map, inplace=True)
        
        essential_internal_cols = ['rcid', 'country_code', 'vote', 'date']
        missing_internal = [col for col in essential_internal_cols if col not in df.columns]
        if missing_internal:
            raise ValueError(f"Essential columns missing after renaming: {', '.join(missing_internal)}")
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['date'], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} invalid date rows.")
        if df.empty:
            raise ValueError("No valid date entries found.")
            
        df['year'] = df['date'].dt.year
        df['country_identifier'] = df['country_code'].astype(str)
        df.dropna(subset=['country_identifier'], inplace=True)
        
        df['vote'] = df['vote'].astype(str).str.strip()
        df.dropna(subset=['vote'], inplace=True)
        
        # Map vote values to standardized categories
        vote_map = {
            'Y': 'Yes',
            'N': 'No',
            'A': 'Abstain',
            'Yes': 'Yes',
            'No': 'No',
            'Abstain': 'Abstain',
            ' ': 'Absent'  # Added mapping for absent votes
        }
        df['vote'] = df['vote'].map(vote_map).fillna('Other')
        
        if 'issue' not in df.columns:
            if 'descr' in df.columns:
                df['issue'] = df['descr'].astype(str).str.split().str[:5].str.join(' ').fillna('Unknown/No Issue')
            else:
                df['issue'] = 'Unknown/No Issue'
        else:
            df['issue'] = df['issue'].fillna('Unknown/No Issue')
            
        # Handle potential multi-valued subjects (split by '; ' if present) take first?
        if df['issue'].dtype == 'object':  # Check if it's string data
            df['issue'] = df['issue'].astype(str).str.split(';').str[0].str.strip()
        
        unique_issues = sorted(df['issue'].astype(str).unique().tolist())
        
        final_cols_needed = ['rcid', 'country_identifier', 'vote', 'date', 'year', 'issue']
        optional_context_cols = ['country_name', 'country_code', 'descr', 'session', 'resolution', 'agenda', 'undl_link']
        if 'importantvote' in df.columns:
            optional_context_cols.append('importantvote')
            
        final_cols = final_cols_needed + [col for col in optional_context_cols if col in df.columns]
        missing_final = [col for col in final_cols_needed if col not in df.columns]
        if missing_final:
            raise ValueError(f"Essential columns missing before final selection: {', '.join(missing_final)}")
            
        df_final = df[final_cols].copy()
        logger.info(f"Loaded and processed {len(df_final)} records.")
        return df_final, unique_issues
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None, None
    except ValueError as ve:
        logger.error(f"Data loading/validation error: {ve}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error loading/processing {filepath}: {e}")
        return None, None

def preprocess_for_similarity(df: pd.DataFrame, start_year: int, end_year: int) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Preprocess the data for similarity calculation.
    
    Args:
        df (pd.DataFrame): Input DataFrame with voting data
        start_year (int): Start year for filtering
        end_year (int): End year for filtering
        
    Returns:
        tuple: (vote_matrix, country_list, filtered_df) or (None, None, None) if error
    """
    try:
        from src.data_processing import create_vote_matrix
        
        vote_matrix, country_list, filtered_df = create_vote_matrix(df, start_year, end_year)
        
        if vote_matrix is None or country_list is None:
            logger.warning("Failed to create vote matrix")
            return None, None, None
            
        return vote_matrix, country_list, filtered_df
        
    except Exception as e:
        logger.error(f"Error in preprocess_for_similarity: {str(e)}")
        return None, None, None

def calculate_similarity(vote_matrix: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate similarity matrix between countries based on voting patterns."""
    try:
        # Add small epsilon to prevent division by zero and improve numerical stability
        vote_matrix = vote_matrix.replace(0, 1e-10)
        
        # Normalize the vote matrix to prevent overflow
        vote_matrix = vote_matrix / vote_matrix.max().max()
        
        # Calculate cosine similarity with additional numerical stability
        similarity_matrix = pd.DataFrame(
            cosine_similarity(vote_matrix),
            index=vote_matrix.index,
            columns=vote_matrix.index
        )
        
        # Clip values to prevent numerical instability
        similarity_matrix = similarity_matrix.clip(-1, 1)
        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {e}")
        return None

def perform_clustering(similarity_matrix: pd.DataFrame, n_clusters: int, country_list: list) -> tuple[dict | None, int | None, np.ndarray | None]:
    """Perform hierarchical clustering on the similarity matrix."""
    try:
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Create cluster dictionary
        clusters = {}
        for i, country in enumerate(country_list):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(country)
            
        return clusters, len(clusters), cluster_labels
        
    except Exception as e:
        logger.error(f"Error performing clustering: {e}")
        return None, None, None

if __name__ == "__main__":
    try:
        logger.info("Starting application...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating graceful shutdown...")
        shutdown_event.set()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)  