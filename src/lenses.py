"""
Through which lens? — the record read through competing theories of
international relations, year by year.

Each theory is treated as a set of claims about what organises voting in
the General Assembly, and each claim as a measurement the roll-call record
can support:

* **Realism**: states follow power and security. Signature: members bound
  to Washington or Moscow by treaty vote as blocs, and that partition
  explains a large share of the year's votes, above all on security items.
* **Liberalism**: institutions socialise states into cooperation. Signature:
  high agreement across the membership, few divided votes, and a busy
  institution-building agenda.
* **World-systems (Marxist)**: economic position — core, semi-periphery,
  periphery — organises conflict. Signature: the income tiers explain votes,
  above all on economic items, with a cohesive core facing a cohesive
  periphery.
* **Constructivism**: identities and norms matter beyond power. Signature:
  the UN's own regional groups explain votes, the normative (human-rights)
  agenda grows, and support for norm resolutions climbs in cascades.
* **Feminist IR**: gender is an organising principle of international
  politics. Signature: gender items reach recorded votes, and the states
  with a declared feminist foreign policy vote as a distinct bloc.

The core measurement is *explained variance*: for a partition of the
membership, the share of the year's voting variance (Yes = +1, No = −1,
side-takers only) that lies between groups rather than within them, pooled
over the year's resolutions. Because a partition with more groups explains
more by chance, each value is reported against a permutation baseline and
the adjusted excess is what the indices use.

Indices are the average of a theory's components, each scaled to its own
1946–today range, so they answer "how strongly is this lens's fingerprint
present this year compared with other years", not "is this theory true".
That distinction is stated in the UI.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.lenses_partitions import CORE, PERIPHERY, SOVIET_LED, US_LED, partition_labels
from src.story_analysis import (
    _complete_years, _side_matrix, division_by_year, last_full_year, resolutions_table,
)

logger = logging.getLogger(__name__)

THEMES_BY_LENS = {
    "security": {"Nuclear weapons & disarmament", "Peace, security & conflicts"},
    "economic": {"Development, economy & environment"},
    "normative": {"Human rights"},
    "institutional": {"UN institutions, budget & law"},
    "decolonial": {"Decolonization & self-determination"},
}

GENDER_PATTERN = re.compile(
    r"\bWOMEN\b|\bWOMAN\b|GENDER|\bGIRLS?\b|FEMALE|FEMINI|CEDAW|SEXUAL VIOLENCE"
    r"|VIOLENCE AGAINST WOMEN|WOMEN, PEACE|MATERNAL|OBSTETRIC|TRAFFICKING IN WOMEN"
)

# Recurring normative resolutions whose support has climbed: the
# constructivist "norm cascade" signature. Matched on the title.
NORM_CASCADES = [
    ("death_penalty", "Moratorium on the use of the death penalty", r"moratorium on the use of the death penalty"),
    ("religious_intolerance", "Elimination of religious intolerance", r"^Elimination of all forms of religious intolerance"),
    ("right_to_food", "The right to food", r"^The right to food"),
    ("extrajudicial", "Extrajudicial, summary or arbitrary executions", r"^Extrajudicial, summary or arbitrary executions"),
]

# Recurring economic resolutions that split North from South: the
# world-systems signature on its home ground.
NORTH_SOUTH_MARKERS = [
    ("equitable_order", "A democratic and equitable international order", r"^Promotion of a democratic and equitable international order"),
    ("coercive_measures", "Human rights and unilateral coercive measures", r"^Human rights and unilateral coercive measures"),
    ("right_to_development", "The right to development", r"^The right to development"),
]

PERMUTATIONS = 30
CLOSE_CALL = 0.05  # index points (0–1) separating a clear top lens from a close call
LENSES = ["realism", "liberalism", "world_systems", "constructivism", "feminism"]
LENS_LABELS = {
    "realism": "Realism",
    "liberalism": "Liberalism",
    "world_systems": "World-systems",
    "constructivism": "Constructivism",
    "feminism": "Feminist IR",
}


# ── explained variance ───────────────────────────────────────────────────────


def explained_variance(matrix: pd.DataFrame, labels: dict[str, str]) -> Optional[float]:
    """Share of voting variance between groups, pooled over resolutions.

    ``matrix`` is countries × resolutions with +1/−1 for Yes/No and 0 for no
    side; ``labels`` maps a subset of the countries to groups. Countries the
    partition does not cover are ignored. Vectorised: one-hot group matrix
    against the vote matrix, so a year costs a few matrix products."""
    codes = [c for c in matrix.index if c in labels]
    if len(codes) < 3:
        return None
    # Integer matrices throughout the products: exact, and they sidestep the
    # spurious matmul warnings some BLAS builds emit for floats.
    m = matrix.loc[codes].to_numpy(dtype=np.int64)
    names = sorted({labels[c] for c in codes})
    if len(names) < 2:
        return None
    index = {g: i for i, g in enumerate(names)}
    onehot = np.zeros((len(codes), len(names)), dtype=np.int64)
    for i, c in enumerate(codes):
        onehot[i, index[labels[c]]] = 1
    sided = (m != 0).astype(np.int64)
    n_j = sided.sum(axis=0)                       # side-takers per resolution
    keep = n_j >= 3
    if not keep.any():
        return None
    m, sided, n_j = m[:, keep], sided[:, keep], n_j[keep]
    mean_j = m.sum(axis=0) / n_j                  # resolution means (0s are non-voters)
    total_ss = float((((m - mean_j) ** 2) * sided).sum())
    group_n = onehot.T @ sided                    # k × r, counts
    group_sum = onehot.T @ m                      # k × r, sums of ±1
    safe_n = np.where(group_n > 0, group_n, 1)
    group_mean = np.where(group_n > 0, group_sum / safe_n, 0.0)
    between_ss = float((group_n * (group_mean - mean_j) ** 2).sum())
    return between_ss / total_ss if total_ss > 0 else None


def adjusted_explained_variance(
    matrix: pd.DataFrame, labels: dict[str, str], permutations: int = PERMUTATIONS, seed: int = 0
) -> dict:
    """The raw value, its permutation baseline (labels shuffled among the
    same countries), and the excess over chance rescaled to [0, 1]."""
    raw = explained_variance(matrix, labels)
    if raw is None:
        return {"raw": None, "baseline": None, "adjusted": None}
    rng = np.random.default_rng(seed)
    codes = [c for c in matrix.index if c in labels]
    values = [labels[c] for c in codes]
    baseline_samples = []
    for _ in range(permutations):
        shuffled = list(values)
        rng.shuffle(shuffled)
        b = explained_variance(matrix, dict(zip(codes, shuffled)))
        if b is not None:
            baseline_samples.append(b)
    baseline = float(np.mean(baseline_samples)) if baseline_samples else 0.0
    adjusted = max(0.0, (raw - baseline) / (1 - baseline)) if baseline < 1 else 0.0
    return {"raw": round(raw, 4), "baseline": round(baseline, 4), "adjusted": round(adjusted, 4)}


def group_cohesion(matrix: pd.DataFrame, labels: dict[str, str]) -> dict[str, Optional[float]]:
    """Pooled pairwise agreement within each group (share of shared
    side-takings that matched)."""
    out: dict[str, Optional[float]] = {}
    for grp in sorted(set(labels.values())):
        codes = [c for c in matrix.index if labels.get(c) == grp]
        if len(codes) < 2:
            out[grp] = None
            continue
        m = matrix.loc[codes].to_numpy(dtype=float)
        yes = (m == 1).astype(np.int64)
        no = (m == -1).astype(np.int64)
        both = yes + no
        same = yes @ yes.T + no @ no.T
        comps = both @ both.T
        iu = np.triu_indices(len(codes), k=1)
        total = comps[iu].sum()
        out[grp] = round(float(same[iu].sum() / total), 4) if total > 0 else None
    return out


# ── the per-year measurements ────────────────────────────────────────────────


def _theme_columns(res: pd.DataFrame, year: int, themes: set[str]) -> list[int]:
    sub = res[(res["year"] == year) & (res["theme"].isin(themes))]
    return sub["rcid"].tolist()


def _support_series(df: pd.DataFrame, pattern: str) -> dict[int, float]:
    res = resolutions_table(df)
    hits = res[res["title"].fillna("").str.contains(pattern, case=False, regex=True)]
    hits = hits.sort_values("date").drop_duplicates("year", keep="last")
    out = {}
    for r in hits.itertuples(index=False):
        total = (r.total_yes or 0) + (r.total_no or 0) + (r.total_abstentions or 0)
        if total:
            out[int(r.year)] = round(float(r.total_yes) / total, 4)
    return out


def _scale(values: list[Optional[float]]) -> list[Optional[float]]:
    """Min–max scale a series to [0, 1] over its non-missing values."""
    present = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not present:
        return [None for _ in values]
    lo, hi = min(present), max(present)
    if hi - lo < 1e-12:
        return [0.5 if v is not None else None for v in values]
    return [None if v is None else round((v - lo) / (hi - lo), 4) for v in values]


def _last_complete_year(res: pd.DataFrame, last_full: Optional[int]) -> Optional[int]:
    """The last year whose main session part is in the record: the record's
    final year counts only once it holds a December vote. Earlier years are
    taken as the story layer's last full year."""
    if last_full is None or res.empty:
        return last_full
    latest = pd.to_datetime(res["date"], errors="coerce").max()
    if pd.isna(latest):
        return last_full
    if last_full >= latest.year and latest.month < 12:
        return latest.year - 1
    return last_full


def _smooth(series: list[Optional[float]], window: int = 5, min_points: int = 3) -> list[Optional[float]]:
    """Centred moving average that skips gaps; None where fewer than
    ``min_points`` of the window are present."""
    half = window // 2
    out: list[Optional[float]] = []
    for i in range(len(series)):
        vals = [v for v in series[max(0, i - half):i + half + 1] if v is not None]
        out.append(round(float(np.mean(vals)), 4) if len(vals) >= min_points else None)
    return out


def _mean_of(components: list[list[Optional[float]]]) -> list[Optional[float]]:
    out = []
    for i in range(len(components[0])):
        vals = [c[i] for c in components if c[i] is not None]
        out.append(round(float(np.mean(vals)), 4) if vals else None)
    return out


def lens_timeline(df: pd.DataFrame, permutations: int = PERMUTATIONS) -> dict:
    """Everything the 'Through which lens?' view needs, for every complete
    year in the record."""
    res = resolutions_table(df)
    last_full = _last_complete_year(res, last_full_year(df))
    years = [y for y in _complete_years(res) if last_full is None or y <= last_full]
    division = {r["year"]: r for r in division_by_year(df)["series"]}
    gender_mask = (res["issue"].fillna("") + " || " + res["subjects"].fillna("")).str.upper().str.contains(GENDER_PATTERN)
    res = res.assign(gender=gender_mask)

    partitions = {"alliance": [], "tier": [], "region": [], "ffp": []}
    home_turf = {"alliance": [], "tier": [], "region": []}
    cohesion = {"us_led": [], "soviet_led": [], "core": [], "periphery": [], "regions": []}
    agenda = {k: [] for k in ("security", "economic", "normative", "institutional", "decolonial", "gender")}
    gender_support = []
    ffp_cohesion = []

    for year in years:
        matrix = _side_matrix(df, [year])
        codes = list(matrix.index)
        year_res = res[res["year"] == year]
        for key, themes in THEMES_BY_LENS.items():
            agenda[key].append(round(float(year_res["theme"].isin(themes).mean()), 4) if len(year_res) else None)
        agenda["gender"].append(round(float(year_res["gender"].mean()), 4) if len(year_res) else None)
        g = year_res[year_res["gender"]]
        if len(g):
            tot = (g["total_yes"].fillna(0) + g["total_no"].fillna(0) + g["total_abstentions"].fillna(0))
            gender_support.append(round(float((g["total_yes"].fillna(0) / tot.replace(0, np.nan)).mean()), 4))
        else:
            gender_support.append(None)

        labels = {scheme: partition_labels(codes, year, scheme) for scheme in partitions}
        for scheme in partitions:
            partitions[scheme].append(adjusted_explained_variance(matrix, labels[scheme], permutations, seed=year))
        # each theory on its home ground
        for scheme, themes in (("alliance", THEMES_BY_LENS["security"]), ("tier", THEMES_BY_LENS["economic"]), ("region", THEMES_BY_LENS["normative"])):
            cols = [c for c in _theme_columns(res, year, themes) if c in matrix.columns]
            if len(cols) >= 5:
                home_turf[scheme].append(adjusted_explained_variance(matrix[cols], labels[scheme], permutations, seed=year))
            else:
                home_turf[scheme].append({"raw": None, "baseline": None, "adjusted": None})
        alliance_coh = group_cohesion(matrix, labels["alliance"])
        tier_coh = group_cohesion(matrix, labels["tier"])
        region_coh = group_cohesion(matrix, labels["region"])
        cohesion["us_led"].append(alliance_coh.get(US_LED))
        cohesion["soviet_led"].append(alliance_coh.get(SOVIET_LED))
        cohesion["core"].append(tier_coh.get(CORE))
        cohesion["periphery"].append(tier_coh.get(PERIPHERY))
        region_vals = [v for v in region_coh.values() if v is not None]
        cohesion["regions"].append(round(float(np.mean(region_vals)), 4) if region_vals else None)
        ffp_coh = group_cohesion(matrix, labels["ffp"])
        ffp_cohesion.append(ffp_coh.get("Feminist foreign policy"))

    agreement = [division.get(y, {}).get("agreement") for y in years]
    divided = [division.get(y, {}).get("divided_share") for y in years]
    lopsided = [None if d is None else round(1 - d, 4) for d in divided]

    cascades = []
    for key, label, pattern in NORM_CASCADES:
        series = _support_series(df, pattern)
        if len(series) >= 3:
            cascades.append({"key": key, "label": label, "series": [{"year": y, "support": s} for y, s in sorted(series.items())]})
    cascade_level = []
    for y in years:
        vals = [c["series"] for c in cascades]
        active = [next((p["support"] for p in s if p["year"] == y), None) for s in vals]
        active = [a for a in active if a is not None]
        cascade_level.append(round(float(np.mean(active)), 4) if active else None)

    markers = []
    for key, label, pattern in NORTH_SOUTH_MARKERS:
        series = recurring_series_support(df, pattern)
        if series:
            markers.append({"key": key, "label": label, "series": series})

    adj = lambda scheme: [p["adjusted"] for p in partitions[scheme]]  # noqa: E731
    adj_home = lambda scheme: [p["adjusted"] for p in home_turf[scheme]]  # noqa: E731
    two_camps = [None if a is None or b is None else round((a + b) / 2, 4) for a, b in zip(cohesion["us_led"], cohesion["soviet_led"])]
    components = {
        "realism": {
            "alliance_blocs_explain_votes": adj("alliance"),
            "alliance_blocs_on_security_items": adj_home("alliance"),
            "camp_cohesion": two_camps,
            "security_agenda_share": agenda["security"],
        },
        "liberalism": {
            "agreement_across_members": agreement,
            "lopsided_votes_share": lopsided,
            "institution_building_agenda_share": agenda["institutional"],
        },
        "world_systems": {
            "income_tiers_explain_votes": adj("tier"),
            "income_tiers_on_economic_items": adj_home("tier"),
            "core_cohesion": cohesion["core"],
            "economic_agenda_share": agenda["economic"],
        },
        "constructivism": {
            "regional_groups_explain_votes": adj("region"),
            "regional_groups_on_rights_items": adj_home("region"),
            "normative_agenda_share": agenda["normative"],
            "norm_cascade_support": cascade_level,
        },
        "feminism": {
            "gender_agenda_share": agenda["gender"],
            "gender_vote_support": gender_support,
            "feminist_policy_bloc_cohesion": ffp_cohesion,
            "feminist_policy_bloc_explains_votes": adj("ffp"),
        },
    }
    indices = {}
    for lens, comps in components.items():
        scaled = [_scale(series) for series in comps.values()]
        indices[lens] = _mean_of(scaled)
    # No gender vote and no feminist-policy cohort in a year: nothing to read.
    ffp = components["feminism"]
    for i in range(len(years)):
        if not agenda["gender"][i] and ffp["feminist_policy_bloc_cohesion"][i] is None:
            indices["feminism"][i] = None
    indices_smoothed = {lens: _smooth(series) for lens, series in indices.items()}
    home_turf_smoothed = {k: _smooth([p["adjusted"] for p in v]) for k, v in home_turf.items()}

    eras = _eras(years, indices, components, last_full)
    return {
        "years": years,
        "last_full_year": last_full,
        "indices_smoothed": indices_smoothed,
        "lenses": [{"key": k, "label": LENS_LABELS[k]} for k in LENSES],
        "indices": indices,
        "components": components,
        "partitions": {k: v for k, v in partitions.items()},
        "home_turf": home_turf,
        "home_turf_smoothed": home_turf_smoothed,
        "cohesion": cohesion,
        "agenda": agenda,
        "cascades": cascades,
        "north_south_markers": markers,
        "eras": eras,
        "definitions": DEFINITIONS,
        "caveats": CAVEATS,
    }


def recurring_series_support(df: pd.DataFrame, pattern: str) -> list[dict]:
    res = resolutions_table(df)
    hits = res[res["title"].fillna("").str.contains(pattern, case=False, regex=True)]
    hits = hits.sort_values("date").drop_duplicates("year", keep="last")
    return [
        {"year": int(r.year), "yes": int(r.total_yes), "no": int(r.total_no), "abstain": int(r.total_abstentions)}
        for r in hits.itertuples(index=False)
        if pd.notna(r.total_yes) and pd.notna(r.total_no) and pd.notna(r.total_abstentions)
    ]


def _eras(years: list[int], indices: dict, components: dict, last_full: Optional[int] = None) -> list[dict]:
    """Per decade: which lens's fingerprint was strongest, with the numbers
    that make the case. A partial current year is left out of the averages."""
    out = []
    usable = [y for y in years if last_full is None or y <= last_full]
    decades = sorted({(y // 10) * 10 for y in usable})
    for decade in decades:
        idx = [i for i, y in enumerate(years) if decade <= y < decade + 10 and y in usable]
        if not idx:
            continue
        means = {}
        for lens in LENSES:
            vals = [indices[lens][i] for i in idx if indices[lens][i] is not None]
            # a lens ranks in a decade only when it can be read in at least half its years
            enough = len(vals) >= max(2, len(idx) // 2)
            means[lens] = round(float(np.mean(vals)), 3) if enough else None
        ranking = sorted([(v, k) for k, v in means.items() if v is not None], reverse=True)
        top = ranking[0][1] if ranking else None
        # a runner-up within five points is a close call, and is said so
        close = ranking[1][1] if len(ranking) > 1 and ranking[0][0] - ranking[1][0] < CLOSE_CALL else None

        def avg(lens, comp):
            vals = [components[lens][comp][i] for i in idx if components[lens][comp][i] is not None]
            return float(np.mean(vals)) if vals else None

        facts = []
        a = avg("realism", "alliance_blocs_explain_votes")
        t = avg("world_systems", "income_tiers_explain_votes")
        r = avg("constructivism", "regional_groups_explain_votes")
        if a is not None and t is not None and r is not None:
            best = max((a, "treaty alliances"), (t, "income tiers"), (r, "regional groups"))
            facts.append(f"{best[1]} explained the most voting variance ({best[0] * 100:.0f}% above chance)")
        g = avg("liberalism", "agreement_across_members")
        if g is not None:
            facts.append(f"members agreed {g * 100:.0f}% of the time")
        n = avg("constructivism", "normative_agenda_share")
        e = avg("world_systems", "economic_agenda_share")
        s = avg("realism", "security_agenda_share")
        if n is not None and e is not None and s is not None:
            facts.append(f"agenda: security {s * 100:.0f}%, economic {e * 100:.0f}%, human rights {n * 100:.0f}%")
        out.append({
            "decade": decade,
            "label": f"{decade}s",
            "top": top,
            "top_label": LENS_LABELS.get(top, ""),
            "close": close,
            "close_label": LENS_LABELS.get(close, "") if close else "",
            "ranking": [{"lens": k, "label": LENS_LABELS[k], "score": v} for v, k in ranking],
            "facts": facts,
        })
    return out


DEFINITIONS = {
    "explained_variance": (
        "For a partition of the membership (treaty alliances, income tiers, regional groups), the "
        "share of the year's voting variance that lies between groups rather than within them, "
        "pooled over the year's resolutions, minus what the same partition explains when its "
        "labels are shuffled, rescaled to 0–1."
    ),
    "cohesion": "Within a group, the share of shared side-takings that matched (pooled pairwise agreement).",
    "index": (
        "The average of a lens's components, each scaled to its own 1946–today range. It says how "
        "strongly that lens's fingerprint is present in a year compared with other years, not whether "
        "the theory is true."
    ),
    "alliance_camps": (
        "US-led: NATO by accession year plus the United States' bilateral defence treaties (Japan, "
        "Korea, the Philippines, Australia, New Zealand to 1986, Thailand) and Israel from 1967. "
        "Soviet/Russian-led: the USSR, the Warsaw Pact, Mongolia, Cuba and Viet Nam to 1991, then the "
        "Collective Security Treaty and CSTO. Everyone else non-aligned that year."
    ),
    "tiers": (
        "World Bank income groups from 1987 (high = core, upper-middle = semi-periphery, lower-middle "
        "and low = periphery), carried back to earlier years with the socialist bloc in the "
        "semi-periphery."
    ),
    "feminist_cohort": (
        "States with a declared feminist foreign policy from the year of adoption (Sweden 2014–2022, "
        "Canada 2017, France and Luxembourg 2019, Mexico 2020, Spain 2021, Germany, Chile, the "
        "Netherlands, Colombia and Liberia 2022, Slovenia and Mongolia 2023)."
    ),
}

CAVEATS = [
    "Recorded votes only: resolutions adopted by consensus, including most gender-equality texts, never appear, so the feminist and liberal readings are measured on the contested end of the record.",
    "Every partition is a proxy chosen in advance and listed above; a different proxy would move the numbers. The permutation baseline removes the mechanical advantage of partitions with more groups.",
    "Lenses are not rivals in the data: two can score high in the same year. The timeline shows whose fingerprint is strongest when, not which theory is right.",
    "The feminist index has thin evidence before 2014, when the first feminist foreign policy was declared, and rests on a handful of recorded gender votes per decade; years with neither are left blank.",
    "The current year is left out until its main session part, September to December, is in the record.",
]
