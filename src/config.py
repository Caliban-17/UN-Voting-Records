"""
Configuration settings for the UN Voting Records Analysis
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UN_VOTES_CSV_PATH = DATA_DIR / os.getenv(
    "UN_VOTING_DATA_PATH", "2025_03_31_ga_voting_corr1.csv"
)
CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", ".cache")
CACHE_DIR.mkdir(exist_ok=True)

# === STANDARDIZED VOTE ENCODING ===
# This is the single source of truth for vote mapping
VOTE_ENCODING = {
    "Y": 1,  # Yes
    "N": -1,  # No
    "A": 0,  # Abstain
    "X": None,  # Not Voting / Did not participate
    " ": None,  # Absent
}

# Legacy mapping for backward compatibility (deprecated, use VOTE_ENCODING)
VOTE_MAP = VOTE_ENCODING

# Column mapping for data processing
COLUMN_RENAME_MAP = {
    "undl_id": "rcid",
    "ms_code": "country_code",
    "ms_name": "country_name",
    "ms_vote": "vote",
    "date": "date",
    "session": "session",
    "title": "descr",
    "subjects": "issue",
    "resolution": "resolution",
    "agenda_title": "agenda",
    "undl_link": "undl_link",
}

# Essential columns required for analysis
ESSENTIAL_COLUMNS = ["rcid", "country_code", "vote", "date", "year", "issue"]

# Optional columns that provide additional context
OPTIONAL_COLUMNS = [
    "country_name",
    "descr",
    "session",
    "resolution",
    "agenda",
    "undl_link",
]

# === WEB SCRAPING CONFIGURATION ===
UN_DIGITAL_LIBRARY_URL = os.getenv(
    "UN_DIGITAL_LIBRARY_URL", "https://digitallibrary.un.org"
)
SCRAPING_DELAY_SECONDS = float(os.getenv("SCRAPING_DELAY_SECONDS", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
USER_AGENT = os.getenv("USER_AGENT", "UN-Voting-Analyzer/1.0")

# === CACHING CONFIGURATION ===
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
USE_PARQUET_CACHE = os.getenv("USE_PARQUET_CACHE", "true").lower() == "true"

# === AUTO UPDATE CONFIGURATION ===
ENABLE_AUTO_UPDATE = os.getenv("ENABLE_AUTO_UPDATE", "true").lower() == "true"
AUTO_UPDATE_CRON = os.getenv("AUTO_UPDATE_CRON", "0 2 * * *")  # Daily at 2 AM UTC

# === NETWORK ANALYSIS CONFIGURATION ===
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))
MIN_EDGE_WEIGHT = float(os.getenv("MIN_EDGE_WEIGHT", "0.5"))

# === SOFT POWER WEIGHTS ===
# These must sum to 1.0
PAGERANK_WEIGHT = float(os.getenv("PAGERANK_WEIGHT", "0.4"))
BETWEENNESS_WEIGHT = float(os.getenv("BETWEENNESS_WEIGHT", "0.3"))
EIGENVECTOR_WEIGHT = float(os.getenv("EIGENVECTOR_WEIGHT", "0.2"))
VOTE_SWAYING_WEIGHT = float(os.getenv("VOTE_SWAYING_WEIGHT", "0.1"))

# Validate weights sum to 1.0
_total_weight = (
    PAGERANK_WEIGHT + BETWEENNESS_WEIGHT + EIGENVECTOR_WEIGHT + VOTE_SWAYING_WEIGHT
)
if abs(_total_weight - 1.0) > 0.001:
    logging.warning(
        f"Soft power weights sum to {_total_weight}, not 1.0. Normalizing..."
    )
    PAGERANK_WEIGHT /= _total_weight
    BETWEENNESS_WEIGHT /= _total_weight
    EIGENVECTOR_WEIGHT /= _total_weight
    VOTE_SWAYING_WEIGHT /= _total_weight

# === TEMPORAL WEIGHTING ===
# Exponential decay factor for historical data (higher = more recent bias)
# 0.95 means data from 1 year ago has 95% of current weight
TEMPORAL_DECAY_FACTOR = float(os.getenv("TEMPORAL_DECAY_FACTOR", "0.95"))

# === PERFORMANCE SETTINGS ===
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "10000"))
MAX_ROWS_TO_LOAD = int(os.getenv("MAX_ROWS_TO_LOAD", "0"))

# === DEFAULT ANALYSIS SETTINGS ===
DEFAULT_N_CLUSTERS = 10
DEFAULT_TRAIN_TEST_SPLIT_YEAR = 2020
DEFAULT_N_TOP_ISSUES = 10
DEFAULT_FIG_SIZE = (10, 8)
DEFAULT_MARKER_SIZE = 50
DEFAULT_ALPHA = 0.8

# === VISUALIZATION SETTINGS ===
PLOT_COLORS = "viridis"

# Network graph layouts
NETWORK_LAYOUTS = ["force", "circular", "hierarchical", "kamada_kawai", "spring"]
DEFAULT_NETWORK_LAYOUT = "force"

# === LOGGING CONFIGURATION ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "output.log")
