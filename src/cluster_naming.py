"""
Cluster auto-naming.

Replaces opaque cluster IDs ("Cluster 3") with human-readable labels:

    EU-aligned bloc — human rights
    Anti-sanctions axis — sovereignty
    Latin-American moderates — armed conflicts prevention

Each cluster's name is composed of:
- a **lead member** — the country most central to the cluster (highest mean
  similarity to other members), chosen as the anchor in the label
- a **signature topic** — the topic where this cluster's mean vote diverges
  most sharply from the rest of the world; this is what makes the cluster
  *distinctive* rather than just "a group of countries"
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _signature_topic(
    df_window: pd.DataFrame, members: Iterable[str], topic_col: str
) -> str | None:
    """Find the topic where this cluster votes most differently from the rest."""
    members_set = set(members)
    if topic_col not in df_window.columns:
        return None

    relevant = df_window.dropna(subset=[topic_col, "vote", "country_identifier"])
    if relevant.empty:
        return None

    relevant = relevant[["rcid", "country_identifier", "vote", topic_col]]
    relevant["in_cluster"] = relevant["country_identifier"].isin(members_set)

    # Per-topic mean vote inside vs outside the cluster.
    grouped = relevant.groupby([topic_col, "in_cluster"])["vote"].agg(["mean", "count"])
    grouped = grouped.reset_index()
    if grouped.empty:
        return None

    pivot = grouped.pivot_table(
        index=topic_col, columns="in_cluster", values=["mean", "count"]
    )
    if ("mean", True) not in pivot.columns or ("mean", False) not in pivot.columns:
        return None

    # Require a minimum sample size so single-vote topics don't dominate.
    min_inside = 3
    min_outside = 5
    valid = (
        (pivot[("count", True)] >= min_inside) & (pivot[("count", False)] >= min_outside)
    )
    pivot = pivot.loc[valid]
    if pivot.empty:
        return None

    distinctiveness = (pivot[("mean", True)] - pivot[("mean", False)]).abs()
    # Tie-break on inside-count so we prefer topics the cluster cares about often.
    ordered = distinctiveness.sort_values(ascending=False)
    if ordered.empty:
        return None
    return str(ordered.index[0])


def _lead_member(
    similarity: pd.DataFrame, members: list[str], name_lookup: dict[str, str] | None
) -> tuple[str, str]:
    """Pick the cluster's anchor — most central member by mean intra-cluster similarity."""
    in_matrix = [m for m in members if m in similarity.index]
    if not in_matrix:
        code = members[0]
    elif len(in_matrix) == 1:
        code = in_matrix[0]
    else:
        sub = similarity.loc[in_matrix, in_matrix]
        np.fill_diagonal(sub.values, 0.0)
        # Highest mean similarity to the rest of the cluster = most "central".
        code = str(sub.mean(axis=1).idxmax())
    name = (name_lookup or {}).get(code, code)
    return code, name


def _humanize_topic(raw: str | None) -> str | None:
    if not raw:
        return raw
    cleaned = str(raw).strip()
    if cleaned.isupper():
        return cleaned.capitalize()
    return cleaned


def label_clusters(
    clusters: dict[int, list[str]],
    similarity: pd.DataFrame,
    df_window: pd.DataFrame,
    name_lookup: dict[str, str] | None = None,
) -> dict[int, dict]:
    """
    Return ``{cluster_id: {label, lead, signature_topic, n_members}}``.

    The label is templated as ``"{lead}-aligned bloc — {signature_topic}"``
    when both pieces are available; falls back to ``"{lead}-led group"`` if
    no signature topic can be identified for the cluster size.
    """
    topic_col = (
        "primary_topic"
        if "primary_topic" in df_window.columns
        else "issue"
        if "issue" in df_window.columns
        else None
    )

    labels: dict[int, dict] = {}
    for cid, members in clusters.items():
        members = sorted(set(members))
        lead_code, lead_name = _lead_member(similarity, members, name_lookup)
        signature = _humanize_topic(
            _signature_topic(df_window, members, topic_col)
            if topic_col
            else None
        )
        if len(members) == 1:
            label = f"Solo: {lead_name}"
        elif signature:
            label = f"{lead_name}-aligned bloc — {signature.lower()}"
        else:
            label = f"{lead_name}-led group"
        labels[int(cid)] = {
            "label": label,
            "lead": lead_code,
            "lead_name": lead_name,
            "signature_topic": signature,
            "n_members": len(members),
        }
    return labels
