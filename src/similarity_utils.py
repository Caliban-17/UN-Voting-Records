"""
Utilities for numerically stable cosine similarity computations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cosine_similarity_matrix(
    matrix: pd.DataFrame,
    min_norm: float = 1e-8,
    drop_zero_rows: bool = True,
) -> pd.DataFrame:
    """
    Compute cosine similarity with explicit numeric sanitization.

    Args:
        matrix: Input matrix (rows are entities, columns are features).
        min_norm: Minimum L2 norm required for a row to be considered valid.
        drop_zero_rows: If True, remove rows with near-zero norm before computing
            similarity (matches the previous behavior in analysis code paths).

    Returns:
        Square similarity matrix as a DataFrame.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame()

    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = numeric.to_numpy(dtype=np.float64, copy=True)

    # Bound extreme values to keep downstream linear algebra stable.
    values = np.clip(values, -1e6, 1e6)

    row_norms = np.linalg.norm(values, axis=1)
    valid_rows = row_norms > min_norm

    if drop_zero_rows:
        numeric = numeric.iloc[valid_rows]
        values = values[valid_rows]
        row_norms = row_norms[valid_rows]

    n_rows = values.shape[0]
    if n_rows == 0:
        return pd.DataFrame()
    if n_rows == 1:
        idx = numeric.index
        return pd.DataFrame([[1.0]], index=idx, columns=idx)

    safe_norms = np.where(row_norms > min_norm, row_norms, 1.0)
    normalized = values / safe_norms[:, np.newaxis]

    similarity = np.einsum("ij,kj->ik", normalized, normalized, optimize=True)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = (similarity + similarity.T) / 2.0
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)

    idx = numeric.index
    return pd.DataFrame(similarity, index=idx, columns=idx)
