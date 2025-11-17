import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
import os
from scipy.stats import entropy
from src.config import (
    VOTE_MAP,
    COLUMN_RENAME_MAP,
    ESSENTIAL_COLUMNS,
    OPTIONAL_COLUMNS
)
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the UN voting data.
    
    Args:
        file_path (str): Path to the CSV file containing UN voting data
        
    Returns:
        tuple: (processed DataFrame, list of unique issues)
    """
    try:
        logging.info(f"Attempting to load data from: {file_path}")
        
        # First, read the file as text to fix formatting issues
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read header line and fix column names
            header = f.readline().strip()
            # Fix split column names
            header = header.replace('\n', '')
            # Remove extra spaces
            header = ','.join(col.strip() for col in header.split(','))
            
            # Read and fix data lines
            lines = []
            current_line = ''
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # If line starts with a number (new record), save previous line and start new one
                if line[0].isdigit():
                    if current_line:
                        lines.append(current_line)
                    current_line = line
                else:
                    # Continue the current line
                    current_line += ' ' + line
                    
            # Add the last line
            if current_line:
                lines.append(current_line)
                
        # Create a temporary file with fixed formatting
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_file.write(header + '\n')
            temp_file.write('\n'.join(lines))
            temp_path = temp_file.name
            
        # Now read the fixed CSV file
        df = pd.read_csv(
            temp_path,
            low_memory=False,
            skipinitialspace=True
        )
        
        # Clean up the temporary file
        import os
        os.unlink(temp_path)
        
        logging.info(f"Initial data shape: {df.shape}")
        logging.info(f"Columns in raw data: {list(df.columns)}")
        
        # Clean up column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Define column mapping
        column_mapping = {
            'undl_id': 'rcid',
            'ms_code': 'country_code',
            'ms_vote': 'vote',
            'date': 'date',
            'title': 'issue',
            'subjects': 'subjects'
        }
        
        # Check for required columns
        required_columns = ['undl_id', 'ms_code', 'ms_vote', 'date', 'title']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Rename columns according to mapping
        df = df.rename(columns=column_mapping)
        
        # Create country_identifier from country_code
        df['country_identifier'] = df['country_code'].astype(str).str.strip()
        
        # Drop rows with missing country_identifier
        df = df.dropna(subset=['country_identifier'])
        
        # Process dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Add year column
        df['year'] = df['date'].dt.year
        
        # Process votes - clean up vote values
        df['vote'] = df['vote'].astype(str).str.strip()
        vote_mapping = {
            'Y': 1,
            'N': 0,
            'A': 0.5,
            'X': np.nan,  # Not voting
            ' ': np.nan   # Absent
        }
        df['vote'] = df['vote'].map(vote_mapping)
        
        # Process issues
        df['issue'] = df['issue'].fillna('Unknown').astype(str).str.strip()
        
        # Log the shape after processing
        logging.info(f"Final data shape: {df.shape}")
        
        # Get unique issues
        unique_issues = df['issue'].unique().tolist()
        logging.info(f"Number of unique issues: {len(unique_issues)}")
        
        return df, unique_issues
        
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {str(e)}")
        raise

def create_vote_matrix(
    df: pd.DataFrame,
    start_year: int,
    end_year: int
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[pd.DataFrame]]:
    """Creates a vote matrix for the specified year range."""
    try:
        logger.info(f"Creating vote matrix for years {start_year}-{end_year}")
        logger.info(f"Input DataFrame shape: {df.shape}")
        
        # Filter by year range
        df_filtered = df[
            (df['year'] >= start_year) & 
            (df['year'] <= end_year)
        ].copy()
        
        logger.info(f"After year filtering: {df_filtered.shape} rows")
        
        if df_filtered.empty:
            logger.warning(f"No data found for years {start_year}-{end_year}")
            return None, None, None

        # Map votes to numeric values
        df_filtered['vote_numeric'] = df_filtered['vote'].map(VOTE_MAP)
        logger.info(f"Unique votes before mapping: {df_filtered['vote'].unique()}")
        logger.info(f"Unique numeric votes after mapping: {df_filtered['vote_numeric'].unique()}")
        
        # Create pivot table
        df_analysis = df_filtered.drop_duplicates(subset=['rcid', 'country_identifier'])
        logger.info(f"After dropping duplicates: {df_analysis.shape} rows")
        
        vote_matrix = df_analysis.pivot_table(
            index='country_identifier',
            columns='rcid',
            values='vote_numeric'
        )
        
        logger.info(f"Vote matrix shape: {vote_matrix.shape}")
        
        # Fill NaN values
        vote_matrix_filled = vote_matrix.fillna(0)
        country_list = vote_matrix_filled.index.tolist()
        logger.info(f"Number of countries in matrix: {len(country_list)}")
        
        return vote_matrix_filled, country_list, df_filtered
        
    except Exception as e:
        logger.error(f"Error creating vote matrix: {str(e)}")
        return None, None, None

def calculate_country_entropy(df: pd.DataFrame) -> pd.Series:
    """Calculates vote entropy for each country."""
    try:
        entropy_scores = df.groupby('country_identifier')['vote'].apply(
            lambda x: entropy(x.value_counts(), base=2)
        ).sort_values()
        return entropy_scores
    except Exception as e:
        logger.error(f"Error calculating country entropy: {str(e)}")
        return pd.Series()

def get_issue_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculates various statistics for issues."""
    try:
        # Count votes per issue
        issue_votes = df.groupby('issue')['rcid'].nunique().sort_values(ascending=False)
        
        # Calculate vote distribution per issue
        issue_distributions = df.groupby(['issue', 'vote']).size().unstack(fill_value=0)
        issue_distributions = issue_distributions.div(issue_distributions.sum(axis=1), axis=0)
        
        # Calculate entropy per issue
        issue_entropy = df.groupby('issue')['vote'].apply(
            lambda x: entropy(x.value_counts(), base=2)
        ).sort_values()
        
        return {
            'vote_counts': issue_votes,
            'distributions': issue_distributions,
            'entropy': issue_entropy
        }
    except Exception as e:
        logger.error(f"Error calculating issue statistics: {str(e)}")
        return {} 