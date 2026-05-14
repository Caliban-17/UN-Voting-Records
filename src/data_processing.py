import pandas as pd
from typing import Tuple, Optional, List, Dict
import logging
import os
from scipy.stats import entropy
from src.config import (
    VOTE_ENCODING,
    MAX_ROWS_TO_LOAD,
)

logger = logging.getLogger(__name__)


def _apply_post_load_transformations(df: pd.DataFrame) -> None:
    """Apply the standard schema normalization in-place.

    Handles both schemas seen in the wild: the legacy
    ``undl_id/ms_code/ms_vote/title/subjects`` from the archival CSV, and
    the fetcher's ``rcid/country_code/vote`` from UN Digital Library.
    """
    column_mapping = {
        "undl_id": "rcid",
        "ms_code": "country_code",
        "ms_name": "country_name",
        "ms_vote": "vote",
        "title": "issue",
    }
    # Only rename when target column doesn't already exist (post-merge CSVs
    # have BOTH undl_id and rcid columns).
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # country_identifier from country_code; tolerate missing column
    if "country_code" in df.columns:
        df["country_identifier"] = df["country_code"].astype(str).str.strip()
        df.dropna(subset=["country_identifier"], inplace=True)
    elif "country_identifier" not in df.columns:
        raise ValueError("CSV lacks both country_code and country_identifier columns.")

    # Country name — title case, fall back to identifier
    if "country_name" in df.columns:
        df["country_name"] = df["country_name"].astype(str).str.strip().str.title()
    else:
        df["country_name"] = df["country_identifier"]

    # Date + year
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["year"] = df["date"].dt.year

    # Vote encoding
    df["vote"] = df["vote"].astype(str).str.strip().str.upper()
    df["vote"] = df["vote"].map(VOTE_ENCODING)

    # Issue + topic
    if "issue" in df.columns:
        df["issue"] = df["issue"].fillna("Unknown").astype(str).str.strip()
    else:
        df["issue"] = "Unknown"
    if "subjects" in df.columns:
        df["topics"] = (
            df["subjects"].fillna("").astype(str)
            .apply(lambda s: [t.strip() for t in s.split("|") if t.strip()])
        )
        df["primary_topic"] = df["topics"].apply(lambda lst: lst[0] if lst else "Unknown")
    else:
        df["topics"] = [[] for _ in range(len(df))]
        df["primary_topic"] = "Unknown"


# Columns the rest of the app expects to find. If any are missing from a
# cached parquet, treat the cache as stale and rebuild from CSV.
REQUIRED_LOADED_COLUMNS: tuple[str, ...] = (
    "rcid",
    "country_identifier",
    "country_name",
    "vote",
    "date",
    "year",
    "issue",
)


def load_and_preprocess_data(file_path, use_cache=True):
    """
    Load and preprocess the UN voting data with optional parquet cache.
    """
    return _load_and_preprocess_data_impl(file_path, use_cache)


def _load_and_preprocess_data_impl(file_path, use_cache=True):
    """
    Args:
        file_path (str): Path to the CSV file containing UN voting data

    Returns:
        tuple: (processed DataFrame, list of unique issues)
    """
    try:
        parquet_path = file_path.rsplit(".", 1)[0] + ".parquet"

        if use_cache and os.path.exists(parquet_path):
            logger.info("Loading cached data from: %s", parquet_path)
            try:
                df = pd.read_parquet(parquet_path)
                missing = [c for c in REQUIRED_LOADED_COLUMNS if c not in df.columns]
                if missing:
                    logger.warning(
                        "Parquet cache missing columns %s — rebuilding from CSV.",
                        missing,
                    )
                else:
                    unique_issues = df["issue"].unique().tolist()
                    logger.info("Loaded %d rows from cache", len(df))
                    return df, unique_issues
            except Exception as e:
                logger.warning(
                    "Failed to load parquet cache: %s. Falling back to CSV.", e
                )

        logger.info("Attempting to load data from: %s", file_path)

        # First try a plain pandas read — modern CSVs (e.g. the post-merge
        # file produced by scripts/refresh_data.py) are well-quoted and don't
        # need the legacy line-stitcher. Fall back to stitching only if
        # the simple read yields suspiciously few rows.
        try:
            df_simple = pd.read_csv(file_path, low_memory=False)
            if len(df_simple) > 100:  # 100-row threshold = obviously real data
                logger.info("Loaded %s rows via plain pd.read_csv (no stitching needed).", len(df_simple))
                df = df_simple
                # Apply the same downstream transformations as the stitched path.
                df.columns = df.columns.str.strip()
                _apply_post_load_transformations(df)
                if use_cache:
                    parquet_path = file_path.rsplit(".", 1)[0] + ".parquet"
                    try:
                        df.to_parquet(parquet_path, index=False)
                    except Exception as e:
                        logger.warning("Failed to save parquet cache: %s", e)
                unique_issues = df["issue"].astype(str).unique().tolist() if "issue" in df.columns else []
                return df, unique_issues
            else:
                logger.info(
                    "Plain read returned %d rows — falling through to line-stitcher.",
                    len(df_simple),
                )
        except Exception as e:
            logger.info("Plain pd.read_csv failed (%s); using legacy stitcher.", e)

        # Legacy stitcher: the original archival CSV has embedded newlines
        # without proper quoting. Stitch lines together until one starts
        # with a digit (the record ID).
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().replace("\n", "")
            header = ",".join(col.strip() for col in header.split(","))

            current_line = []
            for line in f:
                if MAX_ROWS_TO_LOAD > 0 and len(lines) >= MAX_ROWS_TO_LOAD:
                    break

                line = line.strip()
                if not line:
                    continue

                if line[0].isdigit():
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [line]
                else:
                    current_line.append(line)

            if current_line and (
                MAX_ROWS_TO_LOAD == 0 or len(lines) < MAX_ROWS_TO_LOAD
            ):
                lines.append(" ".join(current_line))

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as temp_file:
            temp_file.write(header + "\n")
            temp_file.write("\n".join(lines))
            temp_path = temp_file.name

        df = pd.read_csv(temp_path, low_memory=False, skipinitialspace=True)
        os.unlink(temp_path)

        logger.info("Initial data shape: %s", df.shape)
        df.columns = df.columns.str.strip()

        # Schema Validation
        column_mapping = {
            "undl_id": "rcid",
            "ms_code": "country_code",
            "ms_name": "country_name",
            "ms_vote": "vote",
            "date": "date",
            "title": "issue",
            "subjects": "subjects",
        }
        required_columns = ["undl_id", "ms_code", "ms_vote", "date", "title"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required CSV columns: {missing_cols}")

        df = df.rename(columns=column_mapping)

        # Process Country ID
        df["country_identifier"] = df["country_code"].astype(str).str.strip()
        df = df.dropna(subset=["country_identifier"])

        # Process Country Name — title case for display, keep raw if absent.
        if "country_name" in df.columns:
            df["country_name"] = (
                df["country_name"].astype(str).str.strip().str.title()
            )
        else:
            df["country_name"] = df["country_identifier"]

        # Process Date and Year
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["date"].dt.year

        # Process Votes
        df["vote"] = df["vote"].astype(str).str.strip().str.upper()
        df["vote"] = df["vote"].map(VOTE_ENCODING)

        # Process Issues
        df["issue"] = df["issue"].fillna("Unknown").astype(str).str.strip()

        # Process Subjects into Topics
        if "subjects" in df.columns:
            # Split by | and strip whitespace
            df["topics"] = (
                df["subjects"]
                .fillna("")
                .astype(str)
                .apply(lambda s: [t.strip() for t in s.split("|") if t.strip()])
            )
            # Extract primary topic (first topic, or 'Unknown' if none)
            df["primary_topic"] = df["topics"].apply(
                lambda lst: lst[0] if lst else "Unknown"
            )
        else:
            df["topics"] = [[] for _ in range(len(df))]
            df["primary_topic"] = "Unknown"

        logger.info("Final data shape before cache: %s", df.shape)

        if use_cache:
            try:
                logger.info("Saving processed data to cache: %s", parquet_path)
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                logger.warning("Failed to save parquet cache: %s", e)

        unique_issues = df["issue"].unique().tolist()
        logger.info("Number of unique issues: %d", len(unique_issues))

        return df, unique_issues

    except Exception as e:
        logger.error("Error loading and preprocessing data: %s", str(e), exc_info=True)
        raise


def create_vote_matrix(
    df: pd.DataFrame, start_year: int, end_year: int
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[pd.DataFrame]]:
    """Creates a vote matrix for the specified year range."""
    try:
        logger.info("Creating vote matrix for years %s-%s", start_year, end_year)
        logger.info("Input DataFrame shape: %s", df.shape)

        required_cols = ["year", "country_identifier", "rcid", "vote"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logger.warning("Missing required columns in DataFrame: %s", missing_cols)
            return None, None, None

        # Filter by year range
        df_filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
        logger.info("After year filtering: %s rows", df_filtered.shape)

        if df_filtered.empty:
            logger.warning("No data found for years %s-%s", start_year, end_year)
            return None, None, None

        # Votes are already mapped to numeric (-1, 0, 1) in load_and_preprocess_data.
        # Ensure numeric type
        df_filtered["vote_numeric"] = pd.to_numeric(
            df_filtered["vote"], errors="coerce"
        )

        # Create pivot table
        df_analysis = df_filtered.drop_duplicates(subset=["rcid", "country_identifier"])
        logger.info("After dropping duplicates: %s rows", df_analysis.shape)

        vote_matrix = df_analysis.pivot_table(
            index="country_identifier", columns="rcid", values="vote_numeric"
        )
        logger.info("Vote matrix shape: %s", vote_matrix.shape)

        # Fill NaN values (missing votes are treated as neutrals/abstentions = 0)
        vote_matrix_filled = vote_matrix.fillna(0)
        country_list = vote_matrix_filled.index.tolist()
        logger.info("Number of countries in matrix: %s", len(country_list))

        return vote_matrix_filled, country_list, df_filtered

    except Exception as e:
        logger.error("Error creating vote matrix: %s", str(e), exc_info=True)
        return None, None, None


def calculate_country_entropy(df: pd.DataFrame) -> pd.Series:
    """Calculates vote entropy for each country."""
    try:
        entropy_scores = (
            df.groupby("country_identifier")["vote"]
            .apply(lambda x: entropy(x.value_counts(), base=2))
            .sort_values()
        )
        return entropy_scores
    except Exception as e:
        logger.error(f"Error calculating country entropy: {str(e)}")
        return pd.Series()


def get_issue_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculates various statistics for issues."""
    try:
        # Count votes per issue
        issue_votes = df.groupby("issue")["rcid"].nunique().sort_values(ascending=False)

        # Calculate vote distribution per issue
        issue_distributions = df.groupby(["issue", "vote"]).size().unstack(fill_value=0)
        issue_distributions = issue_distributions.div(
            issue_distributions.sum(axis=1), axis=0
        )

        # Calculate entropy per issue
        issue_entropy = (
            df.groupby("issue")["vote"]
            .apply(lambda x: entropy(x.value_counts(), base=2))
            .sort_values()
        )

        return {
            "vote_counts": issue_votes,
            "distributions": issue_distributions,
            "entropy": issue_entropy,
        }
    except Exception as e:
        logger.error(f"Error calculating issue statistics: {str(e)}")
        return {}
