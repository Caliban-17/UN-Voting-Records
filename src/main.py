# Core data processing and analysis utilities for UN voting analytics.
# RELIES ON LOCAL CSV FILE: data/2025_03_31_ga_voting_corr1.csv

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.similarity_utils import compute_cosine_similarity_matrix

logger = logging.getLogger(__name__)


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

        source_columns_needed = list(column_rename_map.keys())
        missing_source_cols = [
            col for col in source_columns_needed if col not in df.columns
        ]
        if missing_source_cols:
            raise ValueError(
                f"CSV missing source columns: {', '.join(missing_source_cols)}"
            )

        df.rename(columns=column_rename_map, inplace=True)

        essential_internal_cols = ["rcid", "country_code", "vote", "date"]
        missing_internal = [
            col for col in essential_internal_cols if col not in df.columns
        ]
        if missing_internal:
            raise ValueError(
                f"Essential columns missing after renaming: {', '.join(missing_internal)}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        initial_rows = len(df)
        df.dropna(subset=["date"], inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} invalid date rows.")
        if df.empty:
            raise ValueError("No valid date entries found.")

        df["year"] = df["date"].dt.year
        df["country_identifier"] = df["country_code"].astype(str)
        df.dropna(subset=["country_identifier"], inplace=True)

        df["vote"] = df["vote"].astype(str).str.strip()
        df.dropna(subset=["vote"], inplace=True)

        vote_map = {
            "Y": "Yes",
            "N": "No",
            "A": "Abstain",
            "Yes": "Yes",
            "No": "No",
            "Abstain": "Abstain",
            " ": "Absent",
        }
        df["vote"] = df["vote"].map(vote_map).fillna("Other")

        if "issue" not in df.columns:
            if "descr" in df.columns:
                df["issue"] = (
                    df["descr"]
                    .astype(str)
                    .str.split()
                    .str[:5]
                    .str.join(" ")
                    .fillna("Unknown/No Issue")
                )
            else:
                df["issue"] = "Unknown/No Issue"
        else:
            df["issue"] = df["issue"].fillna("Unknown/No Issue")

        if df["issue"].dtype == "object":
            df["issue"] = df["issue"].astype(str).str.split(";").str[0].str.strip()

        unique_issues = sorted(df["issue"].astype(str).unique().tolist())

        final_cols_needed = [
            "rcid",
            "country_identifier",
            "vote",
            "date",
            "year",
            "issue",
        ]
        optional_context_cols = [
            "country_name",
            "country_code",
            "descr",
            "session",
            "resolution",
            "agenda",
            "undl_link",
        ]
        if "importantvote" in df.columns:
            optional_context_cols.append("importantvote")

        final_cols = final_cols_needed + [
            col for col in optional_context_cols if col in df.columns
        ]
        missing_final = [col for col in final_cols_needed if col not in df.columns]
        if missing_final:
            raise ValueError(
                f"Essential columns missing before final selection: {', '.join(missing_final)}"
            )

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


def preprocess_for_similarity(
    df: pd.DataFrame, start_year: int, end_year: int
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[pd.DataFrame]]:
    """Filter `df` to a year range and return (vote_matrix, countries, filtered_df)."""
    try:
        from src.data_processing import create_vote_matrix

        vote_matrix, country_list, filtered_df = create_vote_matrix(
            df, start_year, end_year
        )

        if vote_matrix is None or country_list is None:
            logger.warning("Failed to create vote matrix")
            return None, None, None

        return vote_matrix, country_list, filtered_df

    except Exception as e:
        logger.error(f"Error in preprocess_for_similarity: {str(e)}")
        return None, None, None


def calculate_similarity(vote_matrix: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate cosine similarity between countries based on voting patterns."""
    try:
        similarity_matrix = compute_cosine_similarity_matrix(
            vote_matrix,
            min_norm=1e-8,
            drop_zero_rows=True,
        )

        if similarity_matrix.shape[0] < 2:
            return None

        return similarity_matrix

    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {e}")
        return None


def perform_clustering(
    similarity_matrix: pd.DataFrame, n_clusters: int, country_list: list
) -> tuple[dict | None, int | None, np.ndarray | None]:
    """Hierarchical clustering on the similarity matrix → (clusters, n, labels)."""
    try:
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        clusters: dict[int, list[str]] = {}
        for i, country in enumerate(country_list):
            cluster_id = int(cluster_labels[i])
            clusters.setdefault(cluster_id, []).append(country)

        return clusters, len(clusters), cluster_labels

    except Exception as e:
        logger.error(f"Error performing clustering: {e}")
        return None, None, None
