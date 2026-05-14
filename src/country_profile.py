"""
Country-first profile for UN-Scrupulous.

Pulls together the existing primitives (vote matrix, cosine similarity,
divergence detection) into a single payload that answers one question:

    "Who does country X vote with at the UN, and how has that shifted?"
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_processing import create_vote_matrix
from src.drift_analysis import find_alignment_drifts, pair_percentile
from src.similarity_utils import compute_cosine_similarity_matrix

logger = logging.getLogger(__name__)

# Permanent Security Council members — the natural reference points for
# narrative alignment analysis. Order matters for the legend.
P5_REFERENCE = ["USA", "GBR", "FRA", "RUS", "CHN"]

# Three small reference clusters used to produce a single "where on the map"
# headline number per country. These are intentionally simple and named for
# UN-voting blocs rather than political alignment — see README caveats.
BLOC_REFERENCE: dict[str, list[str]] = {
    "Western": ["USA", "GBR", "FRA", "DEU", "CAN", "AUS", "JPN", "ITA", "NLD"],
    "Eastern": ["RUS", "CHN", "BLR", "IRN", "SYR", "PRK", "VEN", "CUB", "NIC"],
    "Non-aligned": ["IND", "BRA", "ZAF", "IDN", "EGY", "MEX", "ARG", "NGA", "SAU"],
}

# Per-vote agreement scoring. Mirrors DivergenceDetector but vectorized
# so we can aggregate across thousands of rcids per year quickly.
_AGREEMENT_LOOKUP: dict[tuple[float, float], float] = {
    (1.0, 1.0): 1.0,
    (-1.0, -1.0): 1.0,
    (0.0, 0.0): 1.0,
    (1.0, -1.0): 0.0,
    (-1.0, 1.0): 0.0,
    (1.0, 0.0): 0.5,
    (0.0, 1.0): 0.5,
    (-1.0, 0.0): 0.5,
    (0.0, -1.0): 0.5,
}


def _vote_totals(df_country: pd.DataFrame) -> dict[str, int]:
    counts = df_country["vote"].value_counts(dropna=True).to_dict()
    return {
        "votes_cast": int(len(df_country)),
        "yes": int(counts.get(1, 0)),
        "no": int(counts.get(-1, 0)),
        "abstain": int(counts.get(0, 0)),
    }


def _pairwise_agreement(
    df: pd.DataFrame, country_a: str, country_b: str
) -> tuple[float, int]:
    """Return (mean agreement, n_overlap) for a pair of countries."""
    if country_a == country_b:
        return 1.0, int((df["country_identifier"] == country_a).sum())

    sub = df[df["country_identifier"].isin([country_a, country_b])]
    # Only keep rcids both countries voted on.
    pivot = sub.pivot_table(
        index="rcid",
        columns="country_identifier",
        values="vote",
        aggfunc="first",
    )
    if country_a not in pivot.columns or country_b not in pivot.columns:
        return float("nan"), 0
    pivot = pivot[[country_a, country_b]].dropna()
    if pivot.empty:
        return float("nan"), 0

    agreements = [
        _AGREEMENT_LOOKUP.get((float(va), float(vb)), 0.5)
        for va, vb in zip(pivot[country_a].to_numpy(), pivot[country_b].to_numpy())
    ]
    return float(np.mean(agreements)), int(len(agreements))


def _reference_alignment_series(
    df: pd.DataFrame,
    country: str,
    references: Iterable[str],
) -> dict[str, list[dict]]:
    """For each reference country, return a per-year agreement series."""
    series: dict[str, list[dict]] = {}
    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    for ref in references:
        ref_points: list[dict] = []
        if ref == country:
            for year in years:
                ref_points.append({"year": int(year), "agreement": 1.0, "n_votes": 0})
            series[ref] = ref_points
            continue
        for year in years:
            df_year = df[df["year"] == year]
            agreement, n = _pairwise_agreement(df_year, country, ref)
            if n == 0 or np.isnan(agreement):
                continue
            ref_points.append(
                {
                    "year": int(year),
                    "agreement": float(agreement),
                    "n_votes": int(n),
                }
            )
        series[ref] = ref_points
    return series


def _biggest_divergences(
    df: pd.DataFrame,
    country: str,
    peer: str,
    limit: int = 8,
) -> list[dict]:
    """Surface the most surprising recent disagreements with a chosen peer."""
    if peer == country:
        return []
    pair = df[df["country_identifier"].isin([country, peer])]
    pivot = pair.pivot_table(
        index="rcid",
        columns="country_identifier",
        values="vote",
        aggfunc="first",
    )
    if country not in pivot.columns or peer not in pivot.columns:
        return []
    pivot = pivot[[country, peer]].dropna()
    if pivot.empty:
        return []

    # An opposite-sign vote pair is the most informative — surface those first.
    opposite_mask = (
        ((pivot[country] == 1) & (pivot[peer] == -1))
        | ((pivot[country] == -1) & (pivot[peer] == 1))
    )
    rcids = pivot.index[opposite_mask].tolist()
    if not rcids:
        return []

    context_cols = [
        c for c in ("issue", "date", "year", "resolution") if c in df.columns
    ]
    meta = (
        df[df["rcid"].isin(rcids)]
        .drop_duplicates(subset=["rcid"])
        .set_index("rcid")[context_cols]
    )

    out: list[dict] = []
    vote_label = {1: "Yes", -1: "No", 0: "Abstain"}
    for rcid in rcids:
        row = meta.loc[rcid] if rcid in meta.index else None
        out.append(
            {
                "rcid": int(rcid) if pd.notna(rcid) else None,
                "issue": str(row["issue"]) if row is not None and "issue" in row else None,
                "date": str(row["date"]) if row is not None and "date" in row else None,
                "year": int(row["year"]) if row is not None and "year" in row else None,
                "resolution": (
                    str(row["resolution"])
                    if row is not None and "resolution" in row
                    else None
                ),
                "vote_self": vote_label.get(int(pivot.loc[rcid, country]), "?"),
                "vote_peer": vote_label.get(int(pivot.loc[rcid, peer]), "?"),
                "peer": peer,
            }
        )
    # Most recent first.
    out.sort(key=lambda r: (r.get("year") or 0, r.get("date") or ""), reverse=True)
    return out[:limit]


def _bloc_alignment(
    df_window: pd.DataFrame, country: str
) -> dict[str, dict[str, float | int]]:
    """Per-bloc mean agreement with the country, ignoring the country itself."""
    out: dict[str, dict[str, float | int]] = {}
    for bloc, members in BLOC_REFERENCE.items():
        peers = [c for c in members if c != country]
        agreements: list[float] = []
        n_total = 0
        coverage: list[str] = []
        for peer in peers:
            mean_agreement, n = _pairwise_agreement(df_window, country, peer)
            if n == 0 or np.isnan(mean_agreement):
                continue
            agreements.append(float(mean_agreement))
            n_total += int(n)
            coverage.append(peer)
        if not agreements:
            out[bloc] = {"alignment": None, "n_pairs": 0, "coverage": []}
        else:
            out[bloc] = {
                "alignment": float(np.mean(agreements)),
                "n_pairs": int(len(agreements)),
                "coverage": coverage,
                "n_votes_total": n_total,
            }
    return out


def build_country_profile(
    df: pd.DataFrame,
    country: str,
    start_year: int,
    end_year: int,
    n_neighbours: int = 5,
    name_lookup: dict[str, str] | None = None,
) -> dict:
    """Assemble the country profile payload."""
    if df is None or df.empty:
        raise ValueError("Data not loaded")

    df_window = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df_window.empty:
        raise ValueError("No data for selected period")
    df_country = df_window[df_window["country_identifier"] == country]
    if df_country.empty:
        raise ValueError(f"No votes for {country} between {start_year} and {end_year}")

    def _name_for(code: str) -> str | None:
        if name_lookup and code in name_lookup:
            return name_lookup[code]
        if "country_name" not in df_window.columns:
            return None
        names = (
            df_window.loc[df_window["country_identifier"] == code, "country_name"]
            .dropna()
            .astype(str)
        )
        return str(names.iloc[0]) if not names.empty else None

    # 1. Similarity row — top allies / opponents.
    vote_matrix, _, _ = create_vote_matrix(df_window, start_year, end_year)
    allies: list[dict] = []
    opponents: list[dict] = []
    if vote_matrix is not None and not vote_matrix.empty:
        sim = compute_cosine_similarity_matrix(vote_matrix)
        if not sim.empty and country in sim.index:
            row = sim.loc[country].drop(labels=[country], errors="ignore")
            row = row.dropna().sort_values(ascending=False)
            allies = [
                {
                    "country": str(c),
                    "name": _name_for(str(c)),
                    "similarity": float(s),
                }
                for c, s in row.head(n_neighbours).items()
            ]
            opponents = [
                {
                    "country": str(c),
                    "name": _name_for(str(c)),
                    "similarity": float(s),
                }
                for c, s in row.tail(n_neighbours).iloc[::-1].items()
            ]

    # 2. P5 alignment time series (always include the country itself for context).
    references = [country] + [c for c in P5_REFERENCE if c != country]
    p5_alignment = _reference_alignment_series(df_window, country, references)

    # 3. Biggest recent divergences from the top ally — the most narratively
    # interesting "where did they break with their closest partner?" cases.
    biggest = []
    if allies:
        biggest = _biggest_divergences(df_window, country, allies[0]["country"])

    # 4. Country name (best-effort).
    country_name = _name_for(country)

    # 5. Bloc-alignment headline strip — one comparable number per bloc.
    bloc_alignment = _bloc_alignment(df_window, country)

    # 6. Attach percentile context to top allies/opponents so each number
    # carries its own comparison: "92nd percentile of all pairs this window".
    for row in allies + opponents:
        info = pair_percentile(df_window, country, row["country"])
        if info is not None:
            row["percentile"] = info["percentile"]
            row["n_overlap"] = info["n_overlap"]

    # 7. Country-scoped drift alerts — biggest shifts in the most recent year
    # of the window compared to the years before it.
    drift_alerts: list[dict] = []
    if end_year > start_year:
        try:
            drift_alerts = find_alignment_drifts(
                df_window,
                recent_year=end_year,
                baseline_window=max(1, end_year - start_year),
                top_n=5,
                country_filter=country,
                min_baseline_overlap=10,
                min_recent_overlap=5,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Drift alert computation failed for %s", country)

    return {
        "country": country,
        "country_name": country_name,
        "window": {"start_year": int(start_year), "end_year": int(end_year)},
        "totals": _vote_totals(df_country),
        "top_allies": allies,
        "top_opponents": opponents,
        "p5_alignment": p5_alignment,
        "biggest_divergences": biggest,
        "bloc_alignment": bloc_alignment,
        "drift_alerts": drift_alerts,
        "references": P5_REFERENCE,
        "bloc_reference": BLOC_REFERENCE,
    }
