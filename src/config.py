"""
Configuration settings for the UN Voting Records Analysis
"""

import os
import logging
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
UN_VOTES_CSV_PATH = DATA_DIR / '2025_03_31_ga_voting_corr1.csv'

# Vote mapping
VOTE_MAP = {
    'Yes': 1,     # Yes
    'No': -1,     # No
    'Abstain': 0, # Abstain
    'Other': 0,   # Other/Not Voting
    'Absent': 0   # Absent
}

# Column mapping for data processing
COLUMN_RENAME_MAP = {
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

# Essential columns required for analysis
ESSENTIAL_COLUMNS = [
    'rcid',
    'country_code',
    'vote',
    'date',
    'year',
    'issue'
]

# Optional columns that provide additional context
OPTIONAL_COLUMNS = [
    'country_name',
    'descr',
    'session',
    'resolution',
    'agenda',
    'undl_link'
]

# Default settings for analysis
DEFAULT_N_CLUSTERS = 10
DEFAULT_TRAIN_TEST_SPLIT_YEAR = 2020
DEFAULT_N_TOP_ISSUES = 10
DEFAULT_FIG_SIZE = (10, 8)
DEFAULT_MARKER_SIZE = 50
DEFAULT_ALPHA = 0.7

# Plot colors
PLOT_COLORS = {
    'Y': 'green',  # Yes
    'N': 'red',    # No
    'A': 'gray',   # Abstain
    ' ': 'white'   # Absent/Not Voting
}

# Visualization Settings
DEFAULT_FIG_SIZE = (10, 8)
DEFAULT_MARKER_SIZE = 50
DEFAULT_ALPHA = 0.8
PLOT_COLORS = 'viridis' 