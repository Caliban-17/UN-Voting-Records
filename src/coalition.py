"""
Coalition Builder — the most diplomatically usable artifact in the platform.

Given a topic query ("nuclear weapons", "climate change", "Israel"), return:

- the **predicted vote tally** if a fresh resolution on that topic were held
  today (counts of Yes / No / Abstain across all member states)
- a **tiered country list**: champions, supporters, fence-sitters, hard-opposed
- the **fence-sitters** specifically — countries that abstain often or split
  votes 50/50 on the topic. These are the actual lobbying targets.

Output is meant to be read by a foreign-ministry analyst, dropped into a
briefing note, or quoted in a journalism piece. Each row carries enough
context to be trustworthy: n_votes, abstain rate, sample resolution titles.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tier thresholds applied to mean signed vote (-1 to +1) on matching resolutions.
# Names chosen for *action-readability* — these are the words a diplomat would
# use when handing this list to someone.
TIER_THRESHOLDS: list[tuple[str, float]] = [
    ("Champion supporter", 0.85),     # > 0.85  — co-sponsor candidate
    ("Reliable supporter", 0.55),     # 0.55 to 0.85
    ("Leans supporter", 0.20),        # 0.20 to 0.55
    ("Fence-sitter", -0.20),          # -0.20 to 0.20  — lobbying target
    ("Leans opposed", -0.55),         # -0.55 to -0.20
    ("Reliable opposed", -0.85),      # -0.85 to -0.55
    ("Champion opposed", -1.01),      # < -0.85
]


def _normalize(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _matching_resolutions(df: pd.DataFrame, topic_query: str) -> pd.DataFrame:
    """Return rows whose issue / topic / agenda contain the query (case-insensitive)."""
    query = _normalize(topic_query)
    if not query:
        return df.iloc[0:0]

    text_columns: list[str] = []
    for col in ("issue", "primary_topic", "agenda", "descr", "subjects"):
        if col in df.columns:
            text_columns.append(col)
    if not text_columns:
        return df.iloc[0:0]

    haystack = pd.Series([""] * len(df), index=df.index)
    for col in text_columns:
        haystack = haystack + " " + df[col].fillna("").astype(str).str.lower()
    mask = haystack.str.contains(re.escape(query), na=False)
    return df[mask]


def _tier_for(mean_vote: float) -> str:
    for label, threshold in TIER_THRESHOLDS:
        if mean_vote > threshold:
            return label
    return "Champion opposed"


def build_coalition(
    df: pd.DataFrame,
    topic_query: str,
    start_year: int,
    end_year: int,
    min_votes: int = 2,
    fence_sitter_abstain_rate: float = 0.40,
    name_lookup: Optional[dict[str, str]] = None,
) -> dict:
    """
    Produce a coalition report for a hypothetical resolution on ``topic_query``.

    Parameters
    ----------
    min_votes : countries with fewer matching votes than this are reported in a
        separate "no_history" list so callers don't quietly conflate "abstains"
        with "never voted".
    fence_sitter_abstain_rate : floor for promoting a country to "Fence-sitter"
        regardless of mean lean — captures strategic non-commitment.
    """
    if df is None or df.empty:
        raise ValueError("Data not loaded")
    if not topic_query or not topic_query.strip():
        raise ValueError("topic query is required")

    df_window = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df_window.empty:
        raise ValueError("No data for selected period")

    df_topic = _matching_resolutions(df_window, topic_query)
    if df_topic.empty:
        return {
            "topic": topic_query,
            "window": {"start_year": int(start_year), "end_year": int(end_year)},
            "matched_resolutions": 0,
            "predicted_tally": {"yes": 0, "no": 0, "abstain": 0, "no_history": 0},
            "tiers": {label: [] for label, _ in TIER_THRESHOLDS},
            "sample_resolutions": [],
        }

    matched_rcids = df_topic["rcid"].nunique()

    # Per-country aggregation. Mean vote ignores NaN; abstain rate is computed
    # on the votes a country actually cast on these resolutions.
    grouped = df_topic.groupby("country_identifier")["vote"].agg(
        n_votes="count",
        mean_vote="mean",
        n_yes=lambda v: int((v == 1).sum()),
        n_no=lambda v: int((v == -1).sum()),
        n_abstain=lambda v: int((v == 0).sum()),
    )

    tiers: dict[str, list[dict]] = {label: [] for label, _ in TIER_THRESHOLDS}
    no_history: list[dict] = []
    predicted = {"yes": 0, "no": 0, "abstain": 0, "no_history": 0}

    for country, row in grouped.iterrows():
        n = int(row["n_votes"])
        mean_vote = float(row["mean_vote"]) if pd.notna(row["mean_vote"]) else 0.0
        abstain_rate = float(row["n_abstain"]) / max(1, n)
        name = (name_lookup or {}).get(str(country), str(country))
        entry = {
            "country": str(country),
            "name": name,
            "n_votes": n,
            "mean_vote": round(mean_vote, 3),
            "abstain_rate": round(abstain_rate, 3),
            "n_yes": int(row["n_yes"]),
            "n_no": int(row["n_no"]),
            "n_abstain": int(row["n_abstain"]),
        }
        if n < min_votes:
            no_history.append(entry)
            predicted["no_history"] += 1
            continue

        # Promote chronic abstainers to fence-sitter even if their lean is
        # otherwise clear — this is the diplomatic reality of strategic abstention.
        if abstain_rate >= fence_sitter_abstain_rate:
            tier = "Fence-sitter"
        else:
            tier = _tier_for(mean_vote)
        entry["tier"] = tier
        tiers[tier].append(entry)

        # Predicted tally on a fresh resolution: country votes their mean lean.
        if mean_vote >= 0.33:
            predicted["yes"] += 1
        elif mean_vote <= -0.33:
            predicted["no"] += 1
        else:
            predicted["abstain"] += 1

    for tier_list in tiers.values():
        tier_list.sort(key=lambda r: r["mean_vote"], reverse=True)
    no_history.sort(key=lambda r: r["name"])

    # Sample resolution titles so the reader can sanity-check the topic match.
    sample_rows = (
        df_topic.drop_duplicates(subset=["rcid"])
        .head(6)
    )
    title_col = "descr" if "descr" in df_topic.columns else "issue"
    sample_resolutions = [
        {
            "rcid": int(row["rcid"]),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "title": str(row.get(title_col, "")).strip()[:160],
        }
        for _, row in sample_rows.iterrows()
    ]

    return {
        "topic": topic_query,
        "window": {"start_year": int(start_year), "end_year": int(end_year)},
        "matched_resolutions": int(matched_rcids),
        "predicted_tally": predicted,
        "tiers": tiers,
        "no_history": no_history,
        "sample_resolutions": sample_resolutions,
    }
