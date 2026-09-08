"""
The scorecard: where the theories of international relations disagree about
the General Assembly, and what the record says.

Descriptive fingerprints cannot adjudicate between theories, because every
theory predicts *some* structure in the votes. What discriminates is a
prediction one theory makes and its rivals do not. Each test below states the
theory, its distinctive prediction, what the rivals expect instead, the
measurement, and a verdict: supported, not supported, mixed, or insufficient
evidence. The numbers are pooled over the record and reported by era.

Tests
-----
1. Unique explained variance: when alliance camp, income tier, regime type and
   identity group are all held at once, how much of the vote each explains on
   its own (Wendt 1999; Katzenstein 1996 for identity; Krasner 1985 for the
   North–South structure; Voeten 2000, Bailey/Strezhnev/Voeten 2017).
2. Alliance loyalty when the patron stands alone: realism's alliance mechanism
   at the point where it should bind (Walt 1987; Voeten 2004).
3. Alignment before or after the treaty: for each NATO accession, how many
   years before the treaty the state already voted with the alliance
   (identity and regime change first: Risse-Kappen 1995; Schimmelfennig 2001).
4. Norm cascades cross blocs: switchers to 'yes' spread across camps and
   tiers, with a tipping point near a third of states (Finnemore & Sikkink 1998).
5. The lonely superpower: under unipolarity realism's bandwagoning predicts
   alignment with the hegemon; the record shows the opposite (Voeten 2004;
   Pape 2005 on soft balancing).
6. North–South persistence: the tier cleavage outlives the Cold War, and on
   economic items support orders core < semi-periphery < periphery
   (Wallerstein; Krasner 1985; Kim & Russett 1996).
7. Democracies among the non-aligned: regime type explains votes where
   alliance cannot (Doyle 1986; Russett & Oneal 2001; Voeten 2004 on the
   liberal gap).
8. The gender cleavage: on the few recorded gender votes, does a
   religious-conservative identity line explain more than alliance or
   wealth, and do feminist-policy states or women's representation predict
   support there (True 2003; Sanders 2018 on the anti-gender backlash)?
9. Consensus as internalisation: the share of resolutions adopted without a
   vote, overall and for gender items (Finnemore & Sikkink 1998 on
   internalisation; the feminist 'consensus trap').
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import story_analysis as story
from src.lenses import (
    NORM_CASCADES, THEMES_BY_LENS, GENDER_PATTERN, adjusted_explained_variance, _theme_columns,
)
from src.lenses_partitions import (
    NATO_AND_US_TREATY_ALLIES, NON_ALIGNED, SOVIET_LED, US_LED, CORE, PERIPHERY, SEMI, COLONIAL_POWERS,
    NORTH_GROUPS, alliance_camp, colonial_group, former_ruler, partition_labels, world_system_tier,
)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
CONSENSUS_CSV = BASE_DIR / "data" / "consensus_by_session.csv"

SUPPORTED, NOT_SUPPORTED, MIXED, INSUFFICIENT = "supported", "not supported", "mixed", "insufficient evidence"
PERMUTATIONS = 10
LEADERS = {"USA": US_LED, "SUN": SOVIET_LED, "RUS": SOVIET_LED}
NATO_ACCESSIONS = [(c, start) for c, start, end in NATO_AND_US_TREATY_ALLIES if start >= 1952 and c not in ("GER", "JPN", "KOR", "PHL", "AUS", "NZL", "THA", "ISR")]


def _decade(year: int) -> int:
    return (year // 10) * 10


def _mean(values) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.mean(vals)), 4) if vals else None


def _by_decade(pairs: list[tuple[int, Optional[float]]]) -> dict[str, Optional[float]]:
    buckets: dict[int, list] = defaultdict(list)
    for year, value in pairs:
        if value is not None:
            buckets[_decade(year)].append(value)
    return {f"{d}s": round(float(np.mean(v)), 4) for d, v in sorted(buckets.items())}


# ── 1. unique explained variance ─────────────────────────────────────────────

FACTORS = ("alliance", "tier", "democracy", "identity")
FACTOR_LABELS = {"alliance": "alliance camp", "tier": "income tier", "democracy": "regime type", "identity": "identity group"}


def _cross_labels(codes: list[str], year: int, factors: tuple[str, ...]) -> dict[str, str]:
    parts = {f: partition_labels(codes, year, f) for f in factors}
    out = {}
    for c in codes:
        if all(c in parts[f] for f in factors):
            out[c] = " × ".join(parts[f][c] for f in factors)
    return out


def unique_variance(df: pd.DataFrame, years: list[int], permutations: int = PERMUTATIONS) -> dict:
    """Per year: adjusted explained variance of the full cross-partition and
    of each leave-one-factor-out cross-partition; a factor's unique share is
    the difference. Returns the per-year series and decade means."""
    series = {f: [] for f in FACTORS}
    series["all_four"] = []
    for year in years:
        matrix = story._side_matrix(df, [year])
        codes = list(matrix.index)
        full = adjusted_explained_variance(matrix, _cross_labels(codes, year, FACTORS), permutations, seed=year)["adjusted"]
        series["all_four"].append(full)
        for f in FACTORS:
            rest = tuple(x for x in FACTORS if x != f)
            without = adjusted_explained_variance(matrix, _cross_labels(codes, year, rest), permutations, seed=year)["adjusted"]
            unique = None if full is None or without is None else round(max(0.0, full - without), 4)
            series[f].append(unique)
    decades = {f: _by_decade(list(zip(years, series[f]))) for f in series}
    return {"series": series, "decades": decades}


# ── 2. alliance loyalty when the patron stands alone ─────────────────────────


def alliance_loyalty(df: pd.DataFrame, years: list[int], max_share: float = 0.10, min_side_takers: int = 20) -> dict:
    """For votes where a camp leader's side held under ``max_share`` of the
    Yes+No, the share of its treaty allies (with a recorded vote) that voted
    with it, abstained, or voted against."""
    out = {}
    for leader in ("USA", "SUN", "RUS"):
        camp = LEADERS[leader]
        rows = []
        sub = df[df["year"].isin(years)]
        for year, ydf in sub.groupby("year"):
            if leader not in set(ydf["country_identifier"]):
                continue
            pivot = ydf.pivot_table(index="country_identifier", columns="rcid", values="vote", aggfunc="first")
            if leader not in pivot.index:
                continue
            allies = [c for c in pivot.index if c != leader and alliance_camp(c, int(year)) == camp]
            if not allies:
                continue
            for rcid in pivot.columns:
                col = pivot[rcid]
                lv = col.get(leader)
                if lv is None or pd.isna(lv) or lv == 0:
                    continue
                sided = col[col != 0].dropna()
                if len(sided) < min_side_takers:
                    continue
                share = float((sided == lv).mean())
                if share > max_share:
                    continue
                votes = col.loc[allies].dropna()
                if votes.empty:
                    continue
                rows.append({
                    "year": int(year), "rcid": int(rcid), "leader_side_share": share,
                    "with": float((votes == lv).mean()), "abstain": float((votes == 0).mean()),
                    "against": float((votes == -lv).mean()), "allies": int(len(votes)),
                })
        if not rows:
            out[leader] = {"votes": 0}
            continue
        frame = pd.DataFrame(rows)
        by_decade = {}
        for dec, g in frame.groupby(frame["year"].map(_decade)):
            by_decade[f"{dec}s"] = {
                "votes": int(len(g)), "with": round(float(g["with"].mean()), 4),
                "abstain": round(float(g["abstain"].mean()), 4), "against": round(float(g["against"].mean()), 4),
            }
        out[leader] = {
            "votes": int(len(frame)), "with": round(float(frame["with"].mean()), 4),
            "abstain": round(float(frame["abstain"].mean()), 4), "against": round(float(frame["against"].mean()), 4),
            "by_decade": by_decade,
        }
    return out


# ── 3. alignment before the treaty ───────────────────────────────────────────


def _pooled_agreement(matrix: pd.DataFrame, a: str, others: list[str]) -> Optional[float]:
    if a not in matrix.index:
        return None
    x = matrix.loc[a].to_numpy()
    shared = matched = 0
    for b in others:
        if b == a or b not in matrix.index:
            continue
        y = matrix.loc[b].to_numpy()
        both = (x != 0) & (y != 0)
        shared += int(both.sum())
        matched += int(((x == y) & both).sum())
    return matched / shared if shared >= 10 else None


def _insider_threshold(matrix: pd.DataFrame, camp: list[str]) -> Optional[float]:
    """The median camp member's agreement with the rest of its camp."""
    vals = [_pooled_agreement(matrix, m, camp) for m in camp]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if len(vals) >= 5 else None


def alignment_before_treaty(df: pd.DataFrame, years: list[int], window: int = 15, margin: float = 0.05) -> dict:
    """For each NATO accession since 1952: the number of years before the
    treaty from which the state already voted like an insider, its agreement
    with the US-led camp within ``margin`` of the median member's, in every
    remaining pre-accession year. Measured against the camp itself, so the
    Soviet Union's own swing toward the West in 1990 cannot move it."""
    cache: dict[int, tuple] = {}
    year_set = set(years)
    cases = []
    for code, accession in NATO_ACCESSIONS:
        span = [y for y in range(accession - window, accession) if y in year_set]
        if len(span) < 3:
            continue
        aligned = []
        for y in span:
            if y not in cache:
                matrix = story._side_matrix(df, [y])
                camp = [c for c in matrix.index if alliance_camp(c, y) == US_LED]
                cache[y] = (matrix, camp, _insider_threshold(matrix, camp))
            matrix, camp, threshold = cache[y]
            a_us = _pooled_agreement(matrix, code, [c for c in camp if c != code])
            if a_us is None or threshold is None:
                aligned.append(None)
            else:
                aligned.append(a_us >= threshold - margin)
        known = [(y, v) for y, v in zip(span, aligned) if v is not None]
        if len(known) < 3:
            continue
        # first year after which the state stays aligned through accession
        lead_year = None
        for i, (y, v) in enumerate(known):
            if all(w for _, w in known[i:]):
                lead_year = y
                break
        if lead_year is None:
            lead = 0
            note = "not aligned before the treaty"
        else:
            lead = accession - lead_year
            note = "aligned throughout the window" if lead_year == known[0][0] else f"aligned from {lead_year}"
        cases.append({"code": code, "accession": accession, "lead_years": lead, "window_start": known[0][0], "note": note})
    leads = [c["lead_years"] for c in cases]
    return {
        "cases": cases,
        "median_lead": float(np.median(leads)) if leads else None,
        "share_aligned_5_years_before": round(sum(1 for c in cases if c["lead_years"] >= 5) / len(cases), 4) if cases else None,
    }


# ── 4. norm cascades cross blocs ─────────────────────────────────────────────


def cascade_spread(df: pd.DataFrame) -> list[dict]:
    res = story.resolutions_table(df)
    out = []
    for key, label, pattern in NORM_CASCADES:
        hits = res[res["title"].fillna("").str.contains(pattern, case=False, regex=True)].sort_values("date")
        hits = hits.drop_duplicates("year", keep="last")
        if len(hits) < 3:
            continue
        first, last = hits.iloc[0], hits.iloc[-1]
        votes = df[df["rcid"].isin([first["rcid"], last["rcid"]])]
        pivot = votes.pivot_table(index="country_identifier", columns="rcid", values="vote", aggfunc="first")
        if first["rcid"] not in pivot.columns or last["rcid"] not in pivot.columns:
            continue
        both = pivot[[first["rcid"], last["rcid"]]].dropna()
        switchers = both[(both.iloc[:, 0] != 1) & (both.iloc[:, 1] == 1)].index.tolist()
        leavers = both[(both.iloc[:, 0] == 1) & (both.iloc[:, 1] != 1)].index.tolist()
        year_last = int(last["year"])
        camps = defaultdict(int)
        tiers = defaultdict(int)
        for c in switchers:
            camps[alliance_camp(c, year_last)] += 1
            tiers[world_system_tier(c, year_last) or "unknown"] += 1
        n = len(switchers)
        largest_camp = max(camps.values()) / n if n else None
        largest_tier = max(tiers.values()) / n if n else None
        # concentration relative to each tier's share of the membership: 1 = diffusion in proportion
        members = list(both.index)
        member_tiers = defaultdict(int)
        for c in members:
            member_tiers[world_system_tier(c, year_last) or "unknown"] += 1
        ratio = None
        if n:
            top_tier = max(tiers, key=tiers.get)
            expected = member_tiers.get(top_tier, 0) / len(members) if members else None
            ratio = round((tiers[top_tier] / n) / expected, 3) if expected else None
        tiers_reached = sum(1 for t in (CORE, SEMI, PERIPHERY) if tiers.get(t, 0) >= 2)
        # support path and the tipping point (a third of side-takers)
        path = []
        tipping = None
        for r in hits.itertuples(index=False):
            tot = (r.total_yes or 0) + (r.total_no or 0) + (r.total_abstentions or 0)
            share = (r.total_yes or 0) / tot if tot else None
            path.append({"year": int(r.year), "support": round(share, 4) if share is not None else None})
            if tipping is None and share is not None and share >= 1 / 3:
                tipping = int(r.year)
        out.append({
            "key": key, "label": label, "first_year": int(first["year"]), "last_year": year_last,
            "switchers": n, "leavers": len(leavers), "switchers_by_camp": dict(camps), "switchers_by_tier": dict(tiers),
            "largest_camp_share": round(largest_camp, 4) if largest_camp is not None else None,
            "largest_tier_share": round(largest_tier, 4) if largest_tier is not None else None,
            "largest_tier_concentration": ratio, "tiers_reached": tiers_reached,
            "saturated": bool(path[0]["support"] is not None and path[0]["support"] >= 0.9),
            "support_first": path[0]["support"], "support_last": path[-1]["support"], "tipping_year": tipping,
        })
    return out


# ── 5. the lonely superpower ─────────────────────────────────────────────────


def lonely_superpower(df: pd.DataFrame, name_lookup=None) -> dict:
    usa = story.world_alignment_with(df, "USA", name_lookup)
    rows = {r["year"]: r for r in usa["series"]}

    def period(a, b):
        agr = _mean([rows[y]["agreement"] for y in rows if a <= y <= b])
        iso = _mean([rows[y]["isolated_votes"] for y in rows if a <= y <= b])
        return {"agreement": agr, "isolated_votes_per_year": iso}
    return {"late_cold_war_1975_1990": period(1975, 1990), "unipolar_1992_2007": period(1992, 2007), "since_2008": period(2008, 2100)}


# ── 6. North–South persistence ───────────────────────────────────────────────


def north_south(df: pd.DataFrame, years: list[int], partitions: dict) -> dict:
    def mean_adj(scheme, a, b):
        return _mean([p["adjusted"] for y, p in zip(years, partitions[scheme]) if a <= y <= b])
    persistence = {}
    for scheme in ("tier", "alliance", "regime", "region"):
        before, after = mean_adj(scheme, 1980, 1990), mean_adj(scheme, 1992, 2002)
        persistence[scheme] = {"1980_1990": before, "1992_2002": after,
                               "retained": round(after / before, 4) if before and after else None}
    # ordering on economic items: core < semi < periphery in net support
    res = story.resolutions_table(df)
    ordered = 0
    checked = 0
    gaps = []
    for year in years:
        cols = _theme_columns(res, year, THEMES_BY_LENS["economic"])
        if len(cols) < 5:
            continue
        matrix = story._side_matrix(df, [year])
        cols = [c for c in cols if c in matrix.columns]
        if len(cols) < 5:
            continue
        tiers = partition_labels(list(matrix.index), year, "tier")
        support = matrix[cols].mean(axis=1)
        means = {}
        for tier in (CORE, SEMI, PERIPHERY):
            members = [c for c, t in tiers.items() if t == tier]
            if len(members) >= 5:
                means[tier] = float(support.loc[members].mean())
        if len(means) == 3:
            checked += 1
            if means[CORE] < means[SEMI] < means[PERIPHERY]:
                ordered += 1
            gaps.append(means[PERIPHERY] - means[CORE])
    return {
        "persistence": persistence,
        "ordering_years_checked": checked,
        "ordering_share": round(ordered / checked, 4) if checked else None,
        "mean_periphery_minus_core": _mean(gaps),
    }


# ── 6b. the semi-periphery: positions, lean and mobility ─────────────────────


def tier_positions(df: pd.DataFrame, years: list[int], min_votes: int = 5, min_members: int = 5) -> dict:
    """Per year, the mean net support (+1 yes, −1 no, 0 otherwise) of each
    income tier on the year's economic items, and the semi-periphery's lean:
    0 when it sits with the periphery, 1 when it sits with the core."""
    res = story.resolutions_table(df)
    out = {CORE: [], SEMI: [], PERIPHERY: [], "lean": [], "ordered": []}
    for year in years:
        cols = _theme_columns(res, year, THEMES_BY_LENS["economic"])
        matrix = story._side_matrix(df, [year]) if len(cols) >= min_votes else None
        cols = [c for c in cols if matrix is not None and c in matrix.columns]
        if matrix is None or len(cols) < min_votes:
            for k in out:
                out[k].append(None)
            continue
        tiers = partition_labels(list(matrix.index), year, "tier")
        support = matrix[cols].mean(axis=1)
        means = {}
        for tier in (CORE, SEMI, PERIPHERY):
            members = [c for c, t in tiers.items() if t == tier]
            means[tier] = round(float(support.loc[members].mean()), 4) if len(members) >= min_members else None
        for tier in (CORE, SEMI, PERIPHERY):
            out[tier].append(means[tier])
        c, m, p_ = means[CORE], means[SEMI], means[PERIPHERY]
        if None in (c, m, p_) or abs(p_ - c) < 0.05:
            out["lean"].append(None)
            out["ordered"].append(None)
        else:
            out["lean"].append(round(float(np.clip((m - p_) / (c - p_), -0.5, 1.5)), 4))
            out["ordered"].append(1.0 if (c < m < p_) else 0.0)
    return out


def semi_periphery_swing(positions: dict, years: list[int]) -> dict:
    """The semi-periphery's lean by era: world-systems expects it to move
    with the world-economy's phases, toward the periphery when blocked from
    the core (the 1970s), toward the core when incorporated (the 1990s)."""
    lean = list(zip(years, positions["lean"]))
    eras = {
        "1960s": _mean([v for y, v in lean if 1960 <= y <= 1969]),
        "1970s": _mean([v for y, v in lean if 1970 <= y <= 1979]),
        "1980s": _mean([v for y, v in lean if 1980 <= y <= 1989]),
        "1990s": _mean([v for y, v in lean if 1990 <= y <= 1999]),
        "2000s": _mean([v for y, v in lean if 2000 <= y <= 2009]),
        "2010s": _mean([v for y, v in lean if 2010 <= y <= 2019]),
        "2020s": _mean([v for y, v in lean if 2020 <= y <= 2029]),
    }
    ordered = [v for v in positions["ordered"] if v is not None]
    return {"lean_by_era": eras, "ordered_share": round(float(np.mean(ordered)), 4) if ordered else None,
            "ordered_years": len(ordered), "lean_overall": _mean(positions["lean"])}


def tier_mobility(df: pd.DataFrame, years: list[int], min_years: int = 4) -> dict:
    """States the World Bank promoted from upper-middle to high income: their
    lean on economic items in the years before and after promotion. World-
    systems expects position to follow structural location; a constructivist
    reading expects Southern identity (G77, NAM) to hold."""
    from src.lenses_partitions import _income_table
    table = _income_table()
    cases = []
    positions = tier_positions(df, years)
    year_index = {y: i for i, y in enumerate(years)}
    res = story.resolutions_table(df)
    cache: dict[int, tuple] = {}

    def lean_of(code: str, year: int) -> Optional[float]:
        i = year_index.get(year)
        if i is None or positions["lean"][i] is None:
            return None
        if year not in cache:
            cols = _theme_columns(res, year, THEMES_BY_LENS["economic"])
            matrix = story._side_matrix(df, [year])
            cols = [c for c in cols if c in matrix.columns]
            cache[year] = (matrix, cols)
        matrix, cols = cache[year]
        if code not in matrix.index or len(cols) < 5:
            return None
        own = float(matrix.loc[code, cols].mean())
        c, p_ = positions[CORE][i], positions[PERIPHERY][i]
        if c is None or p_ is None or abs(p_ - c) < 0.05:
            return None
        return float(np.clip((own - p_) / (c - p_), -1.0, 2.0))

    for code, groups in table.items():
        ys = sorted(groups)
        promo = None
        for a, b in zip(ys, ys[1:]):
            if groups[a] == "UM" and groups[b] == "H" and all(groups.get(y) == "H" for y in ys if b <= y <= b + 4):
                promo = b
                break
        if promo is None or promo < years[0] + min_years:
            continue
        before = [lean_of(code, y) for y in range(promo - 8, promo) if y in year_index]
        after = [lean_of(code, y) for y in range(promo, promo + 8) if y in year_index]
        before = [v for v in before if v is not None]
        after = [v for v in after if v is not None]
        if len(before) >= min_years and len(after) >= min_years:
            cases.append({"code": code, "promoted": promo, "lean_before": round(float(np.mean(before)), 3),
                          "lean_after": round(float(np.mean(after)), 3), "shift": round(float(np.mean(after) - np.mean(before)), 3)})
    shifts = [c["shift"] for c in cases]
    return {"cases": sorted(cases, key=lambda c: c["promoted"]), "mean_shift": _mean(shifts),
            "share_moved_toward_core": round(sum(1 for v in shifts if v > 0.1) / len(shifts), 4) if shifts else None}


# ── 7. democracies among the non-aligned ─────────────────────────────────────


def democracy_within_nonaligned(df: pd.DataFrame, years: list[int], permutations: int = PERMUTATIONS) -> dict:
    series = []
    for year in years:
        matrix = story._side_matrix(df, [year])
        codes = [c for c in matrix.index if alliance_camp(c, year) == NON_ALIGNED]
        labels = partition_labels(codes, year, "democracy")
        if len(codes) < 10 or len(set(labels.values())) < 2:
            series.append(None)
            continue
        series.append(adjusted_explained_variance(matrix.loc[codes], labels, permutations, seed=year)["adjusted"])
    return {"series": series, "decades": _by_decade(list(zip(years, series)))}


# ── 8. the gender cleavage ───────────────────────────────────────────────────


def gender_cleavage(df: pd.DataFrame, years: list[int], permutations: int = PERMUTATIONS) -> dict:
    res = story.resolutions_table(df)
    gender_mask = (res["issue"].fillna("") + " || " + res["subjects"].fillna("")).str.upper().str.contains(GENDER_PATTERN)
    gender = res[gender_mask]
    out = []
    for dec in sorted({_decade(y) for y in years}):
        span = [y for y in years if _decade(y) == dec]
        rcids = gender[gender["year"].isin(span)]["rcid"].tolist()
        if len(rcids) < 5:
            out.append({"decade": f"{dec}s", "votes": len(rcids)})
            continue
        matrix = story._side_matrix(df, span)
        cols = [c for c in rcids if c in matrix.columns]
        if len(cols) < 5:
            out.append({"decade": f"{dec}s", "votes": len(cols)})
            continue
        mid = span[len(span) // 2]
        codes = list(matrix.index)
        sided = matrix[cols].to_numpy()
        yes_share = float((sided == 1).sum() / max(1, (sided != 0).sum()))
        row = {"decade": f"{dec}s", "votes": len(cols), "yes_share": round(yes_share, 4)}
        for scheme in ("oic", "alliance", "tier", "democracy", "ffp", "representation"):
            labels = partition_labels(codes, mid, scheme)
            if len(set(labels.values())) < 2:
                row[scheme] = None
                continue
            row[scheme] = adjusted_explained_variance(matrix[cols], labels, permutations, seed=mid)["adjusted"]
        out.append(row)
    return out


# ── 10–12. the colonial line ─────────────────────────────────────────────────

COUNTRY_SPECIFIC = r"SITUATION OF HUMAN RIGHTS IN|HUMAN RIGHTS SITUATION IN|SITUATION IN (THE )?(ISLAMIC REPUBLIC|MYANMAR|SYRIAN|DEMOCRATIC PEOPLE|BELARUS|CRIMEA|ERITREA|IRAN|IRAQ|SUDAN|CUBA|CHILE|EL SALVADOR|GUATEMALA|AFGHANISTAN|KOSOVO|BOSNIA|RWANDA|BURUNDI|NIGERIA|ZAIRE|CAMBODIA|KAMPUCHEA|OCCUPIED|LEBANON|SOMALIA|HAITI|TURKMENISTAN|UZBEKISTAN)"
STRUCTURAL_RIGHTS = r"RIGHT TO DEVELOPMENT|INTERNATIONAL ORDER|COERCIVE MEASURES|RIGHT TO FOOD|SELF-DETERMINATION|RACISM|RACIAL DISCRIMINATION|MERCENAR|GLOBALIZATION|SOLIDARITY|CULTURAL DIVERSITY|EQUITABLE|DEBT|RIGHT OF PEOPLES TO PEACE|RIGHT TO PEACE|APARTHEID|NEW INTERNATIONAL ECONOMIC ORDER|PERMANENT SOVEREIGNTY"


def colonial_line(df: pd.DataFrame, years: list[int], permutations: int = PERMUTATIONS) -> dict:
    """Per year: what the colonial partition explains beyond income tier and
    alliance camp together (cross-partition difference), and the North–South
    gap in net support on decolonisation and self-determination items."""
    res = story.resolutions_table(df)
    unique, gap, north_support, south_support = [], [], [], []
    for year in years:
        matrix = story._side_matrix(df, [year])
        codes = list(matrix.index)
        base = _cross_labels(codes, year, ("tier", "alliance"))
        withc = _cross_labels(codes, year, ("tier", "alliance", "colonial"))
        a = adjusted_explained_variance(matrix, base, permutations, seed=year)["adjusted"]
        b = adjusted_explained_variance(matrix, withc, permutations, seed=year)["adjusted"]
        unique.append(None if a is None or b is None else round(max(0.0, b - a), 4))
        cols = [c for c in _theme_columns(res, year, THEMES_BY_LENS["decolonial"]) if c in matrix.columns]
        if len(cols) < 5:
            gap.append(None)
            north_support.append(None)
            south_support.append(None)
            continue
        groups = partition_labels(codes, year, "north_south")
        support = matrix[cols].mean(axis=1)
        north = [c for c, g in groups.items() if g == "North"]
        south = [c for c, g in groups.items() if g == "South"]
        if len(north) < 5 or len(south) < 10:
            gap.append(None)
            north_support.append(None)
            south_support.append(None)
            continue
        n_s, s_s = float(support.loc[north].mean()), float(support.loc[south].mean())
        north_support.append(round(n_s, 4))
        south_support.append(round(s_s, 4))
        gap.append(round(s_s - n_s, 4))
    return {"unique": unique, "gap": gap, "north_support": north_support, "south_support": south_support,
            "unique_by_decade": _by_decade(list(zip(years, unique))), "gap_by_decade": _by_decade(list(zip(years, gap)))}


def whose_rights(df: pd.DataFrame, years: list[int]) -> list[dict]:
    """Per decade, North and South net support on two kinds of recorded
    human-rights vote: resolutions on a named country's record, and
    structural or collective-rights resolutions."""
    res = story.resolutions_table(df)
    rights = res[res["theme"].isin(THEMES_BY_LENS["normative"]) | res["title"].fillna("").str.upper().str.contains(STRUCTURAL_RIGHTS, regex=True)]
    titles = rights["title"].fillna("").str.upper()
    specific = rights[titles.str.contains(COUNTRY_SPECIFIC, regex=True)]
    structural = rights[titles.str.contains(STRUCTURAL_RIGHTS, regex=True) & ~titles.str.contains(COUNTRY_SPECIFIC, regex=True)]
    out = []
    for dec in sorted({_decade(y) for y in years}):
        span = [y for y in years if _decade(y) == dec]
        row = {"decade": f"{dec}s"}
        matrix = None
        for kind, table in (("country_specific", specific), ("structural", structural)):
            rcids = table[table["year"].isin(span)]["rcid"].tolist()
            if len(rcids) < 5:
                row[kind] = None
                continue
            if matrix is None:
                matrix = story._side_matrix(df, span)
            cols = [c for c in rcids if c in matrix.columns]
            if len(cols) < 5:
                row[kind] = None
                continue
            mid = span[len(span) // 2]
            groups = partition_labels(list(matrix.index), mid, "north_south")
            support = matrix[cols].mean(axis=1)
            north = [c for c, g in groups.items() if g == "North"]
            south = [c for c, g in groups.items() if g == "South"]
            if len(north) < 5 or len(south) < 10:
                row[kind] = None
                continue
            row[kind] = {"votes": len(cols), "north": round(float(support.loc[north].mean()), 4), "south": round(float(support.loc[south].mean()), 4)}
        out.append(row)
    return out


def metropole_ties(df: pd.DataFrame, years: list[int]) -> dict:
    """Per decade, ex-colonies' pooled agreement with their former ruler
    against their agreement with the other former colonial powers."""
    rows = []
    for year in years:
        matrix = story._side_matrix(df, [year])
        codes = list(matrix.index)
        powers = [c for c in codes if colonial_group(c, year) == COLONIAL_POWERS]
        for code in codes:
            ruler = former_ruler(code)
            if not ruler or ruler not in codes or colonial_group(code, year) in NORTH_GROUPS:
                continue
            own = _pooled_agreement(matrix, code, [ruler])
            others = _pooled_agreement(matrix, code, [p for p in powers if p != ruler])
            if own is not None and others is not None:
                rows.append({"year": year, "code": code, "ruler": ruler, "own": own, "others": others})
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame["edge"] = frame["own"] - frame["others"]
    by_decade = {f"{d}s": round(float(g["edge"].mean()), 4) for d, g in frame.groupby(frame["year"].map(_decade))}
    by_ruler = {}
    for ruler, g in frame.groupby("ruler"):
        if len(g) >= 50:
            by_ruler[ruler] = {"edge": round(float(g["edge"].mean()), 4), "pairs": int(len(g)),
                               "by_decade": {f"{d}s": round(float(gg["edge"].mean()), 4) for d, gg in g.groupby(g["year"].map(_decade))}}
    return {"edge_by_decade": by_decade, "by_ruler": by_ruler, "edge_overall": round(float(frame["edge"].mean()), 4)}


# ── 9. consensus as internalisation ──────────────────────────────────────────


def consensus_shares() -> dict:
    if not CONSENSUS_CSV.exists():
        return {}
    rows = list(csv.DictReader(CONSENSUS_CSV.open(encoding="utf-8")))
    totals = {int(r["session"]): int(r["total"]) for r in rows if r["theme"] == "all"}
    if not totals:
        return {}
    # a session still being extracted is short of a full one; leave the latest
    # out while it holds under 85% of the others' median
    ordered = sorted(totals)
    if len(ordered) >= 3 and totals[ordered[-1]] < 0.85 * float(np.median([totals[s] for s in ordered[:-1]])):
        ordered = ordered[:-1]
    sessions = ordered
    lens_of = {theme: lens for lens, themes in THEMES_BY_LENS.items() for theme in themes}
    by_theme: dict[str, dict] = defaultdict(lambda: {"total": 0, "without_vote": 0})
    for r in rows:
        if int(r["session"]) not in sessions:
            continue
        key = "all" if r["theme"] == "all" else ("gender" if r["theme"] == "gender" else lens_of.get(r["theme"], "other"))
        by_theme[key]["total"] += int(r["total"])
        by_theme[key]["without_vote"] += int(r["without_vote"])
    shares = {t: round(v["without_vote"] / v["total"], 4) for t, v in by_theme.items() if v["total"]}
    return {"sessions": [sessions[0], sessions[-1]], "years": [1945 + sessions[0], 1945 + sessions[-1]],
            "totals": {t: v["total"] for t, v in by_theme.items()}, "without_vote_share": shares}


# ── verdicts ─────────────────────────────────────────────────────────────────


def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def scorecard(df: pd.DataFrame, years: list[int], partitions: dict, name_lookup=None, permutations: int = PERMUTATIONS) -> dict:
    """Run every test and attach a verdict. ``partitions`` is the timeline's
    per-year explained-variance table, reused for test 6."""
    tests: list[dict] = []

    uv = unique_variance(df, years, permutations)
    recent = {f: _mean(uv["series"][f][-10:]) for f in FACTORS}
    cold = {f: _mean([v for y, v in zip(years, uv["series"][f]) if 1975 <= y <= 1990]) for f in FACTORS}
    ranking_recent = sorted(((v or 0), f) for f, v in recent.items())[::-1]
    top_recent = FACTOR_LABELS[ranking_recent[0][1]]
    identity_recent, alliance_recent = recent["identity"], recent["alliance"]
    tests.append({
        "key": "unique_variance", "theory": "realism", "rivals": ["world_systems", "liberalism", "constructivism"], "mirrored": False,
        "tests": ["realism", "world_systems", "liberalism", "constructivism"],
        "prediction": "Realism: alliance camp carries the unique information; world-systems: income tier; liberalism: regime type; constructivism: identity group beyond all three.",
        "measure": "Explained variance of the four-way cross-partition minus the same without each factor, permutation-corrected, per year.",
        "result": {"last_decade": recent, "late_cold_war": cold, "decades": uv["decades"]},
        "verdict": MIXED,
        "reading": (
            f"Over the last ten full years the {top_recent} carries the most unique information "
            f"({_pct(ranking_recent[0][0])} of the vote), then {FACTOR_LABELS[ranking_recent[1][1]]} ({_pct(ranking_recent[1][0])}); "
            f"identity groups add {_pct(identity_recent)} beyond alliance, wealth and regime, against {_pct(alliance_recent)} for alliances. "
            f"In the late Cold War alliances added {_pct(cold['alliance'])} and identity {_pct(cold['identity'])}."
        ),
    })

    loyalty = alliance_loyalty(df, years)
    us, su, ru = loyalty.get("USA", {}), loyalty.get("SUN", {}), loyalty.get("RUS", {})
    us_with, su_with, ru_with = us.get("with"), su.get("with"), ru.get("with")
    verdict = MIXED
    if us_with is not None and su_with is not None:
        verdict = NOT_SUPPORTED if us_with < 0.34 and (ru_with is None or ru_with < 0.5) else (SUPPORTED if us_with > 0.5 else MIXED)
    tests.append({
        "key": "alliance_loyalty", "theory": "realism", "rivals": ["constructivism", "liberalism"], "mirrored": False, "tests": ["realism"],
        "prediction": "Treaty allies stand with their patron when it is outvoted; the alliance binds hardest when the patron is alone.",
        "rival": "Constructivism and liberalism: legitimacy and domestic values pull allies away from an isolated patron; only a coerced bloc stays loyal.",
        "measure": "On votes where the patron's side held under 10% of Yes+No, the share of its treaty allies voting with it.",
        "result": loyalty, "verdict": verdict,
        "reading": (
            f"When the United States was in a small minority, its treaty allies voted with it {_pct(us_with)} of the time "
            f"({us.get('votes', 0)} votes), abstained {_pct(us.get('abstain'))} and voted against it {_pct(us.get('against'))}. "
            f"The Soviet bloc stood with Moscow {_pct(su_with)} of the time ({su.get('votes', 0)} votes)"
            + (f"; Russia's treaty allies since 1992, {_pct(ru_with)} ({ru.get('votes', 0)} votes)." if ru_with is not None else ".")
        ),
    })

    lead = alignment_before_treaty(df, years)
    med = lead["median_lead"]
    verdict = INSUFFICIENT if med is None else (NOT_SUPPORTED if med >= 5 else (SUPPORTED if med <= 1 else MIXED))
    tests.append({
        "key": "alignment_before_treaty", "theory": "realism", "rivals": ["constructivism", "liberalism"], "mirrored": True, "tests": ["realism", "constructivism", "liberalism"],
        "prediction": "Realism: voting alignment follows the treaty, because the alliance creates the shared interest.",
        "rival": "Constructivism and liberalism: identity and regime change come first; the treaty ratifies an alignment already visible in the votes.",
        "measure": "For each NATO accession since 1952, the years before the treaty from which the state already voted closer to the US-led camp than to the Soviet/Russian-led camp.",
        "result": lead, "verdict": verdict,
        "reading": (
            f"Across {len(lead['cases'])} accessions the median state had voted with the alliance for {med:.0f} years before joining; "
            f"{_pct(lead['share_aligned_5_years_before'])} were aligned five or more years ahead of the treaty."
            if med is not None else "Too few accessions with a voting record before the treaty."
        ),
    })

    spread = cascade_spread(df)
    live = [c for c in spread if not c["saturated"] and c["switchers"] >= 10]
    crossing = [c for c in live if c["tiers_reached"] >= 2 and (c["largest_tier_concentration"] or 9) <= 1.5]
    verdict = INSUFFICIENT if not live else (SUPPORTED if len(crossing) >= max(1, (len(live) + 1) // 2) else (MIXED if crossing else NOT_SUPPORTED))
    tests.append({
        "key": "cascade_spread", "theory": "constructivism", "rivals": ["realism"], "mirrored": False, "tests": ["constructivism"],
        "prediction": "Support for a norm grows by diffusion: new adopters come from every camp and tier, and adoption accelerates after about a third of states have signed on.",
        "rival": "Realism: support moves only when blocs move, so switchers cluster in one camp.",
        "measure": "For each recurring normative resolution, the members that moved from no or abstain to yes between its first and latest vote, by alliance camp and income tier.",
        "result": spread, "verdict": verdict,
        "reading": (
            "; ".join(
                (f"{c['label']}: already at {_pct(c['support_first'])} when first recorded, nothing left to cascade" if c["saturated"] else
                 f"{c['label']}: {c['switchers']} switchers from {c['tiers_reached']} of 3 tiers, the largest tier {c['largest_tier_concentration']:.1f}× its share of members, support {_pct(c['support_first'])} → {_pct(c['support_last'])}")
                for c in spread
            ) + "." if spread else "No recurring normative resolution has three recorded votes."
        ),
    })

    lonely = lonely_superpower(df, name_lookup)
    a1, a2 = lonely["late_cold_war_1975_1990"]["agreement"], lonely["unipolar_1992_2007"]["agreement"]
    verdict = INSUFFICIENT if a1 is None or a2 is None else (NOT_SUPPORTED if a2 <= a1 + 0.02 else SUPPORTED)
    tests.append({
        "key": "lonely_superpower", "theory": "realism", "rivals": ["world_systems", "constructivism"], "mirrored": False, "tests": ["realism"],
        "prediction": "Under unipolarity states bandwagon: alignment with the hegemon rises after 1991.",
        "rival": "Soft balancing and legitimacy accounts: an unchecked hegemon is resisted, not followed. World-systems reads the same isolation as hegemonic decline.",
        "measure": "Share of side-takings that matched the United States, and the number of votes it cast with two or fewer companions, by period.",
        "result": lonely, "verdict": verdict,
        "reading": (
            f"Members sided with the United States {_pct(a1)} of the time in 1975–1990, {_pct(a2)} in 1992–2007 and "
            f"{_pct(lonely['since_2008']['agreement'])} since 2008; its isolated votes went from "
            f"{lonely['late_cold_war_1975_1990']['isolated_votes_per_year']:.0f} a year to {lonely['unipolar_1992_2007']['isolated_votes_per_year']:.0f} and "
            f"{lonely['since_2008']['isolated_votes_per_year']:.0f}."
            if a1 is not None and a2 is not None else "No United States series."
        ),
    })

    ns = north_south(df, years, partitions)
    tier_kept, all_kept = ns["persistence"]["tier"]["retained"], ns["persistence"]["alliance"]["retained"]
    ordering = ns["ordering_share"]
    verdict = INSUFFICIENT if tier_kept is None or ordering is None else (
        SUPPORTED if tier_kept > all_kept and ordering >= 0.6 else (MIXED if tier_kept > all_kept or ordering >= 0.6 else NOT_SUPPORTED)
    )
    tests.append({
        "key": "north_south", "theory": "world_systems", "rivals": ["realism"], "mirrored": False, "tests": ["world_systems"],
        "prediction": "The North–South cleavage is structural: it outlives the Cold War, and on economic items support runs core < semi-periphery < periphery.",
        "rival": "Realism: cleavages follow the distribution of power, so every partition should weaken together after 1991.",
        "measure": "Explained variance retained from 1980–1990 to 1992–2002, by partition; the share of years in which net support on economic items is ordered core < semi-periphery < periphery.",
        "result": ns, "verdict": verdict,
        "reading": (
            f"Income tiers kept {_pct(tier_kept)} of their explanatory power across 1991, alliances {_pct(all_kept)}, regime type "
            f"{_pct(ns['persistence']['regime']['retained'])}, regional groups {_pct(ns['persistence']['region']['retained'])}. "
            f"On economic items the tiers were ordered core < semi-periphery < periphery in {_pct(ordering)} of {ns['ordering_years_checked']} years, "
            f"with the periphery's net support {ns['mean_periphery_minus_core']:+.2f} above the core's on a −1 to +1 scale."
            if tier_kept is not None and ordering is not None else "Not enough economic votes."
        ),
    })

    positions = tier_positions(df, years)
    swing = semi_periphery_swing(positions, years)
    eras_lean = swing["lean_by_era"]
    l70, l90, l10 = eras_lean.get("1970s"), eras_lean.get("1990s"), eras_lean.get("2010s")
    distinct = swing["ordered_share"] is not None and swing["ordered_share"] >= 0.6
    leans = [v for v in eras_lean.values() if v is not None]
    mobile = len(leans) >= 3 and (max(leans) - min(leans)) >= 0.15
    verdict = INSUFFICIENT if swing["ordered_share"] is None else (
        SUPPORTED if distinct and mobile else (MIXED if distinct or mobile else NOT_SUPPORTED)
    )
    drift = "toward the periphery" if (l70 is not None and l10 is not None and l10 < l70) else "toward the core"
    tests.append({
        "key": "semi_periphery", "theory": "world_systems", "rivals": ["postcolonial", "constructivism"], "mirrored": False, "tests": ["world_systems"],
        "prediction": "Wallerstein's world-economy is trimodal: a semi-periphery exists as a stratum of its own, sitting between core and periphery on distributive questions, and it is politically mobile, its alignment tracking its prospects of promotion rather than a fixed identity.",
        "rival": "Post-colonial and identity accounts: the South is one bloc, the decolonised and the Non-Aligned; a middle stratum is not visible and does not move.",
        "measure": "Mean net support of each tier on the year's economic items; the semi-periphery's lean from 0 (with the periphery) to 1 (with the core), by era; the share of years ordered core < semi-periphery < periphery.",
        "result": swing, "verdict": verdict,
        "reading": (
            f"The semi-periphery leaned {l70:.2f} of the way from the periphery to the core in the 1970s"
            + (f", {eras_lean['1980s']:.2f} in the 1980s" if eras_lean.get("1980s") is not None else "")
            + (f", {l90:.2f} in the 1990s" if l90 is not None else "")
            + (f", {eras_lean['2000s']:.2f} in the 2000s" if eras_lean.get("2000s") is not None else "")
            + (f" and {l10:.2f} in the 2010s" if l10 is not None else "")
            + f"; the three tiers were ordered core < semi-periphery < periphery in {_pct(swing['ordered_share'])} of {swing['ordered_years']} years, so the middle stratum is real, and it has drifted {drift} as its promotion stalled."
            if l70 is not None else "Not enough economic votes with all three tiers present."
        ),
    })

    mobility = tier_mobility(df, years)
    ms, share_up = mobility["mean_shift"], mobility["share_moved_toward_core"]
    verdict = INSUFFICIENT if ms is None or len(mobility["cases"]) < 5 else (
        SUPPORTED if ms >= 0.15 and share_up >= 0.6 else (NOT_SUPPORTED if ms <= 0.05 else MIXED)
    )
    tests.append({
        "key": "tier_mobility", "theory": "world_systems", "rivals": ["constructivism"], "mirrored": True, "tests": ["world_systems", "constructivism"],
        "prediction": "World-systems: a state's position follows its structural location, so states promoted into the core move toward the core's positions on economic items.",
        "rival": "Constructivism: Southern identity (G77, non-alignment) outlives promotion, so the promoted keep voting with the periphery.",
        "measure": "For every member the World Bank moved from upper-middle to high income, its lean on economic items in the eight years before and after promotion (0 with the periphery, 1 with the core).",
        "result": mobility, "verdict": verdict,
        "reading": (
            f"{len(mobility['cases'])} members were promoted into the core; on average their lean moved {ms:+.2f} "
            f"(from {_mean([c['lean_before'] for c in mobility['cases']]):.2f} to {_mean([c['lean_after'] for c in mobility['cases']]):.2f}), "
            f"and {_pct(share_up)} moved toward the core by more than a tenth."
            if ms is not None and mobility["cases"] else "Too few promotions with votes on both sides."
        ),
    })

    dna = democracy_within_nonaligned(df, years, permutations)
    post = _mean([v for y, v in zip(years, dna["series"]) if y >= 1992])
    pre = _mean([v for y, v in zip(years, dna["series"]) if y < 1992])
    verdict = INSUFFICIENT if post is None else (SUPPORTED if post >= 0.05 else (MIXED if post > 0.02 else NOT_SUPPORTED))
    tests.append({
        "key": "democracy_within_nonaligned", "theory": "liberalism", "rivals": ["realism"], "mirrored": False, "tests": ["liberalism"],
        "prediction": "Regime type shapes preferences on its own: among states outside both alliance camps, democracies and autocracies vote apart, more so after 1991.",
        "rival": "Realism: outside the camps, votes follow interest and patronage, not domestic institutions.",
        "measure": "Permutation-corrected explained variance of democracy vs autocracy among non-aligned members, per year.",
        "result": dna, "verdict": verdict,
        "reading": (
            f"Among the non-aligned, regime type explained {_pct(pre)} of the vote beyond chance before 1992 and {_pct(post)} since."
            if post is not None else "Too few non-aligned members with regime data."
        ),
    })

    gender = gender_cleavage(df, years, permutations)
    usable = [g for g in gender if g.get("oic") is not None]
    contested = [g for g in usable if g.get("yes_share", 1.0) < 0.9 and max((g.get(k) or 0) for k in ("oic", "alliance", "tier", "democracy")) >= 0.03]
    if usable:
        latest = usable[-1]
        if not contested:
            verdict = INSUFFICIENT
        else:
            oic_wins = sum(1 for g in contested if g["oic"] >= max((g.get("alliance") or 0), (g.get("tier") or 0)))
            verdict = SUPPORTED if oic_wins >= max(1, (len(contested) + 1) // 2) else MIXED
    else:
        latest, verdict = None, INSUFFICIENT
    tests.append({
        "key": "gender_cleavage", "theory": "feminism", "rivals": ["realism", "world_systems"], "mirrored": False, "tests": ["feminism", "constructivism"],
        "prediction": "On gender votes the line is a values line: a religious-conservative identity (the OIC members) explains more than alliance or wealth, and feminist-policy states and women's representation predict support.",
        "rival": "Realism and world-systems: gender votes follow the same alliance and wealth lines as everything else.",
        "measure": "Per decade, on the recorded gender votes pooled, explained variance of OIC membership, alliance camp, income tier, regime type, the feminist-policy cohort and women's representation.",
        "result": gender, "verdict": verdict,
        "reading": (
            (f"The recorded gender votes are close to unanimous ({_pct(latest.get('yes_share'))} yes in the {latest['decade']}), so no partition has variance to explain: the contest over gender happens before the vote, in the drafting, and in the consensus texts this record does not hold. " if not contested else "")
            + f"In the {latest['decade']} ({latest['votes']} recorded gender votes) OIC membership explained {_pct(latest.get('oic'))}, "
            f"alliances {_pct(latest.get('alliance'))}, income tiers {_pct(latest.get('tier'))}, regime type {_pct(latest.get('democracy'))}, "
            f"the feminist-policy cohort {_pct(latest.get('ffp'))} and women's representation {_pct(latest.get('representation'))}."
            if latest else "Fewer than five recorded gender votes in any decade."
        ),
    })

    colonial = colonial_line(df, years, permutations)
    c_recent = _mean(colonial["unique"][-10:])
    c_cold = _mean([v for y, v in zip(years, colonial["unique"]) if 1960 <= y <= 1980])
    g_recent = _mean(colonial["gap"][-10:])
    g_60s = colonial["gap_by_decade"].get("1960s")
    verdict = INSUFFICIENT if c_recent is None or g_recent is None else (
        SUPPORTED if (c_recent >= 0.03 or c_cold >= 0.03) and g_recent >= 0.3 else (MIXED if g_recent >= 0.15 or (c_cold or 0) >= 0.03 else NOT_SUPPORTED)
    )
    tests.append({
        "key": "colonial_line", "theory": "postcolonial", "rivals": ["world_systems", "realism"], "mirrored": False, "tests": ["postcolonial"],
        "prediction": "The colonial encounter organises the Assembly: former colonial powers and settler states stand apart from the post-1945 decolonised states beyond what wealth or alliance explain, above all on decolonisation and self-determination, and the line persists after formal empire ends.",
        "rival": "World-systems: the line is income, and it dissolves once colonial history is netted out. Realism: it is alliance.",
        "measure": "Explained variance of tier × alliance × colonial line minus tier × alliance, per year; the South's net support minus the North's on decolonisation and self-determination items.",
        "result": {k: v for k, v in colonial.items() if k in ("unique_by_decade", "gap_by_decade")},
        "verdict": verdict,
        "reading": (
            f"Beyond wealth and alliance, colonial history explained {_pct(c_cold)} of the vote in 1960–1980 and {_pct(c_recent)} in the last ten years. "
            f"On decolonisation and self-determination items the decolonised states' net support ran {g_60s:+.2f} above the former colonial powers' and settler states' in the 1960s and {g_recent:+.2f} in the last ten years (on a −1 to +1 scale)."
            if c_recent is not None and g_recent is not None and g_60s is not None else "Not enough decolonisation votes with both groups present."
        ),
    })

    rights = whose_rights(df, years)
    usable = [r for r in rights if r.get("country_specific") and r.get("structural")]
    flips = [r for r in usable if r["country_specific"]["north"] > r["country_specific"]["south"] and r["structural"]["south"] > r["structural"]["north"]]
    verdict = INSUFFICIENT if not usable else (SUPPORTED if len(flips) >= max(1, (len(usable) + 1) // 2) else (MIXED if flips else NOT_SUPPORTED))
    latest_r = usable[-1] if usable else None
    tests.append({
        "key": "whose_rights", "theory": "postcolonial", "rivals": ["liberalism"], "mirrored": True, "tests": ["postcolonial", "liberalism"],
        "prediction": "Post-colonial theory: 'human rights' is contested ground. The North backs resolutions on named Southern states' records; the South backs structural rights (development, self-determination, racism, coercive measures, an equitable order) that the North resists.",
        "rival": "Liberalism: democracies support human-rights resolutions of every kind; the split is regime type, not colonial history.",
        "measure": "Per decade, North and South net support on recorded votes about a named country's human-rights record, and on structural or collective-rights resolutions.",
        "result": rights, "verdict": verdict,
        "reading": (
            f"In the {latest_r['decade']}, on {latest_r['country_specific']['votes']} country-specific rights votes the North's net support was {latest_r['country_specific']['north']:+.2f} and the South's {latest_r['country_specific']['south']:+.2f}; "
            f"on {latest_r['structural']['votes']} structural-rights votes the North's was {latest_r['structural']['north']:+.2f} and the South's {latest_r['structural']['south']:+.2f}. "
            f"The pattern held in {len(flips)} of {len(usable)} decades with both kinds of vote."
            if latest_r else "Too few recorded rights votes of both kinds."
        ),
    })

    ties = metropole_ties(df, years)
    e60, e_all = (ties.get("edge_by_decade") or {}).get("1960s"), ties.get("edge_overall")
    latest_edge = list((ties.get("edge_by_decade") or {}).values())[-1] if ties.get("edge_by_decade") else None
    verdict = INSUFFICIENT if e_all is None else (SUPPORTED if (e60 or 0) >= 0.03 else (MIXED if (e60 or 0) > 0 else NOT_SUPPORTED))
    tests.append({
        "key": "metropole_ties", "theory": "postcolonial", "rivals": ["realism", "world_systems"], "mirrored": False, "tests": ["postcolonial"],
        "prediction": "Neo-colonial ties outlast independence: an ex-colony votes closer to its own former ruler than to the other former colonial powers, most in the first decades after independence.",
        "rival": "Realism and world-systems: once independent, a state's alignment follows its patron or its tier, and the old metropole is just another Northern state.",
        "measure": "Per decade, an ex-colony's pooled agreement with its former ruler minus its agreement with the other former colonial powers, averaged over ex-colonies.",
        "result": ties, "verdict": verdict,
        "reading": (
            f"Ex-colonies agreed with their own former ruler {e60 * 100:+.1f} points more than with the other colonial powers in the 1960s and {latest_edge * 100:+.1f} points in the latest decade"
            + ("; " + ", ".join(f"former {r} colonies {v['edge'] * 100:+.1f}" for r, v in sorted(ties["by_ruler"].items())) if ties.get("by_ruler") else "") + "."
            if e60 is not None and latest_edge is not None else "Too few ex-colony and ruler pairs."
        ),
    })

    cons = consensus_shares()
    if cons:
        share_all, share_gender = cons["without_vote_share"].get("all"), cons["without_vote_share"].get("gender")
        share_sec = cons["without_vote_share"].get("security")
        verdict = INSUFFICIENT if share_gender is None else (SUPPORTED if share_gender > (share_all or 0) + 0.1 else MIXED)
    else:
        share_all = share_gender = share_sec = None
        verdict = INSUFFICIENT
    tests.append({
        "key": "consensus", "theory": "constructivism", "rivals": ["realism"], "mirrored": False, "tests": ["constructivism", "feminism"],
        "prediction": "Internalised norms leave the roll-call record: settled agendas are adopted without a vote. Feminist IR adds that gender equality is settled by consensus while 'high politics' is fought out in votes.",
        "rival": "Realism: consensus marks what nobody cares enough to contest, not agreement on norms.",
        "measure": "Share of General Assembly resolutions adopted without a vote, all items and gender items, sessions 74–80 (the years the DGACM extracts cover).",
        "result": cons, "verdict": verdict,
        "reading": (
            f"In sessions {cons['sessions'][0]}–{cons['sessions'][1]} ({cons['years'][0]}–{cons['years'][1]}) {_pct(share_all)} of resolutions were adopted without a vote; "
            f"{_pct(share_gender)} of gender resolutions and {_pct(share_sec)} of security resolutions."
            if cons else "No adoption-type data (data/consensus_by_session.csv missing)."
        ),
    })

    by_theory: dict[str, dict] = defaultdict(lambda: {"supported": 0, "not_supported": 0, "mixed": 0, "insufficient": 0})
    slot_of = {SUPPORTED: "supported", NOT_SUPPORTED: "not_supported", MIXED: "mixed", INSUFFICIENT: "insufficient"}
    mirror = {SUPPORTED: "not_supported", NOT_SUPPORTED: "supported", MIXED: "mixed", INSUFFICIENT: "insufficient"}
    for t in tests:
        owners = [t["theory"]] + ([x for x in t["tests"] if x not in (t["theory"], *t.get("rivals", []))])
        for theory in owners:
            by_theory[theory][slot_of[t["verdict"]]] += 1
        if t.get("mirrored"):
            for rival in t.get("rivals", []):
                by_theory[rival][mirror[t["verdict"]]] += 1
    return {"tests": tests, "by_theory": dict(by_theory), "unique_variance_series": uv["series"],
            "democracy_within_nonaligned_series": dna["series"], "tier_positions": positions,
            "colonial_line_series": {k: colonial[k] for k in ("unique", "gap", "north_support", "south_support")}}
