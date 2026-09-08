"""
The big picture — what the General Assembly votes on, how divided it is, and
who stands where, across the whole record (1946 → today).

Everything here is written for a reader, not a modeller: each function
returns plain aggregates a chart can draw directly, plus a one-sentence
``takeaway`` computed from the numbers so the finding travels with the data
(the newsletter and the web app both use it).

Definitions (kept deliberately simple and stated in the UI):

* **Recorded votes only.** The dataset holds roll-call votes. Resolutions
  adopted by consensus never appear, so "how divided is the Assembly" means
  "how divided were the votes it chose to record" — the contested end of
  its work by construction.
* **Taking a side** means voting Yes or No. Abstentions and absences are
  ignored when measuring agreement, exactly as they are ignored when the
  Assembly counts a majority (Charter, Article 18).
* **Divided** means at least one in ten of the members that took a side
  were on the losing side (winning side under 90% of Yes+No). **Contested**
  is the stricter Charter test — winning side under two-thirds — which the
  Assembly's lopsided majorities almost never fail, so it is reported but
  not charted.
* **Agreement** between two countries is the share of resolutions on which
  both took a side and it was the same side.
* **Isolated votes** are resolutions on which a country took a side with at
  most two other members.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.regional_groups import GROUP_ORDER, lineage_codes, regional_group

logger = logging.getLogger(__name__)

# ── Agenda themes ────────────────────────────────────────────────────────────
# First match wins, so order matters: the occupied-territories items are filed
# under Israel & Palestine even though they mention human rights or nuclear
# armament, and the colonial questions before the generic human-rights rules.

THEMES: list[tuple[str, str]] = [
    (
        "Israel & Palestine",
        r"PALESTIN|ISRAEL|UNRWA|GAZA|JERUSALEM|GOLAN|MIDDLE EAST SITUATION|MIDDLE EAST--"
        r"|WEST BANK|OCCUPIED (ARAB|SYRIAN)|LEBAN",
    ),
    (
        "Decolonization & self-determination",
        r"DECOLONI|SELF-DETERMINATION|NON-SELF-GOVERNING|COLONIAL|NAMIBIA|SOUTH WEST AFRICA"
        r"|APARTHEID|SOUTH AFRICA|RHODESIA|PORTUGUESE|SAHARA|IFNI|CHAGOS|GIBRALTAR|FALKLAND"
        r"|MALVINAS|PUERTO RICO|NEW CALEDONIA|TRUST TERRITOR|TRUSTEESHIP|GUAM|TOKELAU"
        r"|PITCAIRN|SAINT HELENA|BERMUDA|CAYMAN|MONTSERRAT|TURKS AND CAICOS|VIRGIN ISLANDS"
        r"|ANGUILLA|AMERICAN SAMOA|FRENCH POLYNESIA|EAST TIMOR|TIMOR|MAYOTTE|COMOR"
        r"|RUANDA-URUNDI|QUESTION OF OMAN|TERRITORY OF",
    ),
    (
        "Nuclear weapons & disarmament",
        r"NUCLEAR|DISARMAMENT|ARMS|WEAPON|MISSILE|LANDMINE|MINES|OUTER SPACE|MILITARY|IAEA"
        r"|ATOMIC|CHEMICAL|BIOLOGICAL|BACTERIOLOGICAL|ZONES? OF PEACE|CONFIDENCE-BUILDING"
        r"|NON-PROLIFERATION|TEST BAN|FISSILE|CLUSTER MUNITION|ARMAMENT|SMALL ARMS"
        r"|GENEVA PROTOCOL",
    ),
    (
        "Human rights",
        r"HUMAN RIGHTS|RACIAL|RACISM|GENOCIDE|TORTURE|DEATH PENALTY|EXTRAJUDICIAL|RELIGIO"
        r"|XENOPHOB|NAZI|RIGHT TO DEVELOPMENT|DISCRIMINATION|MIGRANT|REFUGEE|INDIGENOUS"
        r"|WOMEN|CHILD|DEMOCRA|FORCED LABOUR|AGING|AGEING|DISABILIT|CULTURAL PROPERTY",
    ),
    (
        "Development, economy & environment",
        r"DEVELOPMENT|ECONOMIC|TRADE|DEBT|FINANC(E|ING) FOR|POVERTY|FOOD|AGRICULTUR|AGRARIAN"
        r"|ENVIRONMENT|CLIMATE|ENERGY|OIL|NATURAL RESOURCES|SOVEREIGNTY OVER|EMBARGO|CUBA"
        r"|UNILATERAL|COERCIVE|SANCTION|GLOBALIZATION|INDUSTRIAL|TECHNOLOGY|SCIENCE"
        r"|INFORMATION|COMMUNICATION|HEALTH|HOUSING|WATER|OCEAN|SEA|ANTARCTIC|CORPORATION"
        r"|TRANSNATIONAL|INVESTMENT|COMMODIT|LAND-LOCKED|TRANSMISSION OF NEWS",
    ),
    (
        "UN institutions, budget & law",
        r"BUDGET|SCALE OF ASSESSMENT|CONTRIBUTIONS|PROGRAMME PLANNING|SECRETARIAT|STAFF"
        r"|PENSION|SECURITY COUNCIL--|ADMISSION|MEMBERSHIP|CREDENTIALS|RULES OF PROCEDURE"
        r"|GENERAL ASSEMBLY--|ECONOMIC AND SOCIAL COUNCIL|INTERNATIONAL COURT|CHARTER"
        r"|INTERNATIONAL LAW|LAW COMMISSION|CRIMINAL COURT|COOPERATION BETWEEN THE UNITED"
        r" NATIONS AND|CO-OPERATION BETWEEN THE UNITED NATIONS AND|OBSERVER STATUS|UN\. |UN--"
        r"|UNITED NATIONS (SYSTEM|REFORM|ORGANIZATION)|ACCOMMODATION|CONFERENCE|COMMITTEE"
        r"|COMMISSION|ELECTION|APPOINTMENT|JOINT INSPECTION|INCOME ESTIMATES|APPROPRIATIONS"
        r"|WORKING CAPITAL|FINANCIAL SITUATION|PALAIS|PRIVILEGES AND IMMUNITIES|DIPLOMAT"
        r"|REPRESENTATION OF|TREATY-MAKING|WORK PROGRAMME",
    ),
    (
        "Peace, security & conflicts",
        r"AGGRESSION|UKRAINE|CRIMEA|GEORGIA|KOREA|CYPRUS|AFGHANISTAN|KAMPUCHEA|CAMBODIA|VIET"
        r"|KOSOVO|BOSNIA|YUGOSLAV|RWANDA|SOMALIA|CONGO|GRENADA|PANAMA|NICARAGUA"
        r"|CENTRAL AMERICA|HUNGARY|TIBET|IRAN-IRAQ|KUWAIT|LIBYA|CHAD|ANGOLA|MOZAMBIQUE"
        r"|NAGORNO|AZERBAIJAN|ARMENIA|MOLDOVA|OCCUPIED TERRITORIES OF|PEACEKEEPING"
        r"|PEACE-KEEPING|INTERIM FORCE|OBSERVER FORCE|EMERGENCY FORCE|TERRORIS|MERCENAR"
        r"|GOOD-NEIGHBOUR|FRIENDLY RELATIONS|INTERVENTION|NON-INTERFERENCE|SITUATION IN"
        r"|TRUCE|CEASEFIRE|CEASE-FIRE|HOSTILITIES|WAR|INTERNATIONAL SECURITY"
        r"|SETTLEMENT OF DISPUTES|PEACE|MEDITERRANEAN|GOOD OFFICES|CODE OF OFFENCES",
    ),
]
OTHER_THEME = "Other"
THEME_ORDER = [name for name, _ in THEMES] + [OTHER_THEME]
_THEME_PATTERNS = [(name, re.compile(pat)) for name, pat in THEMES]

# ── Landmark votes ───────────────────────────────────────────────────────────
# Curated: resolution symbol as it appears in the dataset, a short label, and
# a one-sentence "why it matters". Tallies come from the data at run time (and
# are checked against the record in tests), never from this table.

LANDMARK_VOTES: list[dict] = [
    {
        "symbol": "A/RES/181(II)[A]",
        "label": "Partition of Palestine",
        "why": "The plan that preceded Israel's founding and the first Arab–Israeli war: the Assembly's most consequential early decision, carried 33 to 13.",
    },
    {
        "symbol": "A/RES/217(III)[A]",
        "label": "Universal Declaration of Human Rights",
        "why": "Adopted with no votes against; the Soviet bloc, Saudi Arabia and South Africa abstained.",
    },
    {
        "symbol": "A/RES/377(V)[A-C]",
        "label": "Uniting for Peace",
        "why": "Let the Assembly act when a veto deadlocks the Security Council — the basis for today's emergency sessions on Ukraine and Gaza.",
    },
    {
        "symbol": "A/RES/1514(XV)",
        "label": "Decolonization declaration",
        "why": "89 in favour, none against, the colonial powers abstaining. UN membership doubled over the following decade.",
    },
    {
        "symbol": "A/RES/2758(XXVI)",
        "label": "China's seat passes to Beijing",
        "why": "Over US objections, the Assembly recognised the People's Republic — a marker of the new post-colonial majority.",
    },
    {
        "symbol": "A/RES/3379(XXX)",
        "label": "\"Zionism is a form of racism\"",
        "why": "High-water mark of the Arab–Soviet–non-aligned majority; revoked in 1991 as the Cold War ended.",
    },
    {
        "symbol": "A/RES/47/19",
        "label": "First vote against the US embargo on Cuba",
        "why": "59 in favour and 71 abstentions in 1992; three decades later the same annual resolution passed 187 to 2.",
    },
    {
        "symbol": "A/RES/67/19",
        "label": "Palestine becomes a non-member observer state",
        "why": "138 in favour; the US, Israel, Canada and six others against.",
    },
    {
        "symbol": "A/RES/68/262",
        "label": "Territorial integrity of Ukraine (Crimea)",
        "why": "100 states rejected the Crimea referendum, but 58 abstentions and 24 absences showed the limits of Western reach.",
    },
    {
        "symbol": "A/RES/ES-10/19",
        "label": "Status of Jerusalem",
        "why": "128 states rejected the US recognition of Jerusalem as Israel's capital despite explicit US warnings about aid.",
    },
    {
        "symbol": "A/RES/ES-11/1",
        "label": "Aggression against Ukraine",
        "why": "141 demanded Russia's withdrawal. Only Belarus, Eritrea, North Korea and Syria sided with Moscow; China and India were among 35 abstentions.",
    },
    {
        "symbol": "A/RES/ES-10/22",
        "label": "Humanitarian ceasefire in Gaza",
        "why": "153 to 10 for an immediate ceasefire; the United States and Israel among the ten.",
    },
    {
        "symbol": "A/RES/ES-10/23",
        "label": "Palestinian UN membership",
        "why": "143 states backed Palestine's bid for full membership after a US veto in the Security Council.",
    },
    {
        "symbol": "A/DEC/80/506",
        "label": "New York Declaration on a two-state solution",
        "why": "142 to 10 after two years of war in Gaza; the United States and Israel opposed.",
    },
    {
        "symbol": "A/RES/80/4",
        "label": "Cuba embargo vote, 2025",
        "why": "The first erosion in decades: 165 in favour after years at 187, with 7 against and 12 abstentions.",
    },
]

RECURRING_VOTES: dict[str, dict] = {
    "cuba": {
        "label": "US embargo on Cuba",
        "pattern": r"embargo imposed by the United States of America against Cuba",
        "why": "One resolution, voted on every autumn since 1992: the clearest single measure of how the Assembly relates to Washington.",
    },
    "nazism": {
        "label": "Glorification of Nazism resolution",
        "pattern": r"glorification of Nazism",
        "why": "Tabled by Russia every year since 2012. Western states abstained for a decade, then switched to voting no after the invasion of Ukraine — the Ukraine effect in a single series.",
    },
    "unrwa": {
        "label": "UNRWA assistance to Palestine refugees",
        "pattern": r"^Assistance to Palestine refugees",
        "why": "The Assembly's annual backing for UNRWA, near-unanimous for decades; the count of no votes tracks the US–Israel position.",
    },
    "palestine_self_determination": {
        "label": "Palestinian self-determination",
        "pattern": r"^The right of the Palestinian people to self-determination",
        "why": "Voted on every year since 1994 with more than 150 in favour; the handful against is the story.",
    },
    "golan": {
        "label": "Occupied Syrian Golan",
        "pattern": r"^The occupied Syrian Golan",
        "why": "Annual since 1996; abstentions rose after the US recognised Israeli sovereignty over the Golan in 2019.",
    },
    "iran_rights": {
        "label": "Human rights in Iran",
        "pattern": r"human rights in the Islamic Republic of Iran",
        "why": "The most evenly split of the annual country resolutions: roughly a third of the room votes no or abstains every year.",
    },
    "coercive_measures": {
        "label": "Unilateral coercive measures (sanctions)",
        "pattern": r"^Human rights and unilateral coercive measures",
        "why": "The Global South's annual rebuke of unilateral sanctions; the West votes no as a bloc, with almost no abstentions.",
    },
    "nuclear_ban": {
        "label": "Nuclear ban treaty",
        "pattern": r"Treaty on the Prohibition of Nuclear Weapons",
        "why": "The ban treaty, opposed every year by the nuclear-armed states and most of their allies.",
    },
    "outer_space": {
        "label": "Arms race in outer space",
        "pattern": r"^Prevention of an arms race in outer space",
        "why": "Adopted almost unanimously every year since 1982; the one or two votes against are usually the United States and Israel.",
    },
}

# ── Resolution table (one row per recorded vote) ─────────────────────────────

_res_cache: dict[tuple[int, int], pd.DataFrame] = {}


def classify_theme(text: str) -> str:
    """Theme for a resolution given its title + subject strings (any case)."""
    up = str(text or "").upper()
    for name, pattern in _THEME_PATTERNS:
        if pattern.search(up):
            return name
    return OTHER_THEME


def resolutions_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded vote with date, title, official tallies and theme.

    Cached per DataFrame object: the Flask app holds one frame for the life of
    the process, so this is computed once.
    """
    key = (id(df), len(df))
    cached = _res_cache.get(key)
    if cached is not None:
        return cached
    cols = [c for c in (
        "rcid", "year", "date", "issue", "subjects", "primary_topic", "resolution",
        "total_yes", "total_no", "total_abstentions",
    ) if c in df.columns]
    res = df.drop_duplicates("rcid")[cols].copy()
    for c in ("subjects", "primary_topic", "issue"):
        if c not in res.columns:
            res[c] = ""
    text = (
        res["issue"].fillna("").astype(str)
        + " || " + res["subjects"].fillna("").astype(str)
        + " || " + res["primary_topic"].fillna("").astype(str)
    )
    res["theme"] = text.map(classify_theme)
    res["title"] = res["issue"].fillna("").astype(str).str.split(" : ").str[0]
    for c in ("total_yes", "total_no", "total_abstentions"):
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors="coerce")
    if "total_yes" not in res.columns:
        # Fall back to counting rows when the official tallies are absent.
        counts = df.groupby("rcid")["vote"].agg(
            total_yes=lambda s: int((s == 1).sum()),
            total_no=lambda s: int((s == -1).sum()),
            total_abstentions=lambda s: int((s == 0).sum()),
        )
        res = res.merge(counts, left_on="rcid", right_index=True, how="left")
    yn = res["total_yes"].fillna(0) + res["total_no"].fillna(0)
    winning = np.maximum(res["total_yes"].fillna(0), res["total_no"].fillna(0))
    res["winning_share"] = np.where(yn > 0, winning / yn.replace(0, np.nan), np.nan)
    res["contested"] = res["winning_share"] < (2 / 3)
    res["divided"] = res["winning_share"] < 0.9
    res["year"] = res["year"].astype(int)
    _res_cache.clear()
    _res_cache[key] = res
    return res


# A trailing year with fewer recorded votes than this is treated as still in
# progress and dropped from the time series (tests set it to 0).
INCOMPLETE_YEAR_MIN_VOTES = 10
# A year needs at least this many recorded votes to anchor a takeaway.
FULL_YEAR_MIN_VOTES = 20


def _complete_years(res: pd.DataFrame) -> list[int]:
    """Calendar years with a meaningful number of recorded votes (the current,
    in-progress year is dropped when it has fewer than
    ``INCOMPLETE_YEAR_MIN_VOTES``)."""
    counts = res.groupby("year")["rcid"].nunique()
    last = counts.index.max()
    years = [int(y) for y, n in counts.items() if y < last or n >= INCOMPLETE_YEAR_MIN_VOTES]
    return sorted(years)


def last_full_year(df: pd.DataFrame) -> int:
    """Most recent calendar year with at least 20 recorded votes."""
    return _last_full_year(resolutions_table(df))


def _last_full_year(res: pd.DataFrame) -> int:
    counts = res.groupby("year")["rcid"].nunique()
    eligible = counts[counts >= FULL_YEAR_MIN_VOTES]
    return int(eligible.index.max() if not eligible.empty else counts.index.max())


# ── 1. The agenda: what the Assembly votes on ────────────────────────────────


def agenda_by_year(df: pd.DataFrame) -> dict:
    res = resolutions_table(df)
    years = _complete_years(res)
    tab = (
        res[res["year"].isin(years)]
        .groupby(["year", "theme"])["rcid"].nunique()
        .unstack(fill_value=0)
        .reindex(index=years, fill_value=0)
        .reindex(columns=THEME_ORDER, fill_value=0)
    )
    res["decade"] = (res["year"] // 10) * 10
    dec = pd.crosstab(res["decade"], res["theme"], normalize="index").reindex(
        columns=THEME_ORDER, fill_value=0
    )
    last_dec = int(dec.index.max())
    shares_now = dec.loc[last_dec]
    top_now = shares_now.drop(OTHER_THEME).sort_values(ascending=False)
    decol_1960 = float(dec.loc[1960, "Decolonization & self-determination"]) if 1960 in dec.index else None
    decol_now = float(shares_now["Decolonization & self-determination"])
    takeaway = (
        f"In the {last_dec}s, {top_now.iloc[0] * 100:.0f}% of recorded votes concern "
        f"{top_now.index[0].lower()} and {top_now.iloc[1] * 100:.0f}% {top_now.index[1].lower()}."
    )
    if decol_1960 is not None:
        takeaway += (
            f" Decolonization, {decol_1960 * 100:.0f}% of votes in the 1960s, is now "
            f"{decol_now * 100:.0f}%."
        )
    return {
        "years": years,
        "themes": THEME_ORDER,
        "counts": {theme: [int(v) for v in tab[theme].tolist()] for theme in THEME_ORDER},
        "totals": [int(v) for v in tab.sum(axis=1).tolist()],
        "decade_shares": {
            str(int(d)): {t: round(float(v), 3) for t, v in row.items()} for d, row in dec.iterrows()
        },
        "takeaway": takeaway,
        "caveat": (
            "Recorded (roll-call) votes only. Resolutions adopted by consensus — the "
            "majority of the Assembly's output — never come to a recorded vote and are "
            "not in this data."
        ),
    }


# ── 2. Division: how split the votes are ────────────────────────────────────


def _side_matrix(df: pd.DataFrame, years: Optional[list[int]] = None) -> pd.DataFrame:
    """Countries × resolutions matrix of +1 (Yes), -1 (No), 0 (no side taken)."""
    sub = df if years is None else df[df["year"].isin(years)]
    sub = sub[sub["vote"].isin([1, -1])]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="country_identifier", columns="rcid", values="vote", aggfunc="first"
    ).fillna(0)


def pooled_agreement(matrix: pd.DataFrame) -> float | None:
    """Share of same-side outcomes over all country pairs and resolutions where
    both took a side. Pooled (vote-weighted), so busy years are not swamped by
    thin ones."""
    if matrix.empty or matrix.shape[0] < 2:
        return None
    m = matrix.to_numpy(dtype=float)
    yes = (m == 1).astype(np.int64)
    no = (m == -1).astype(np.int64)
    both = yes + no
    same = yes @ yes.T + no @ no.T
    comps = both @ both.T
    n = m.shape[0]
    iu = np.triu_indices(n, k=1)
    total = comps[iu].sum()
    return float(same[iu].sum() / total) if total > 0 else None


def division_by_year(df: pd.DataFrame) -> dict:
    res = resolutions_table(df)
    years = _complete_years(res)
    rows = []
    for y in years:
        r = res[res["year"] == y]
        n = int(len(r))
        contested = float(r["contested"].mean()) if n else None
        divided = float(r["divided"].mean()) if n else None
        rows.append({
            "year": y,
            "votes": n,
            "divided_share": None if divided is None else round(divided, 3),
            "contested_share": None if contested is None else round(contested, 3),
            "mean_winning_share": None if n == 0 else round(float(r["winning_share"].mean()), 3),
            "agreement": None,
        })
    agreement = {}
    for y in years:
        agreement[y] = pooled_agreement(_side_matrix(df, [y]))
    for row in rows:
        a = agreement.get(row["year"])
        row["agreement"] = None if a is None else round(a, 3)

    last = _last_full_year(res)
    series = {r["year"]: r for r in rows}
    now = series.get(last, {})
    ref_year = 1990 if 1990 in series else years[0]
    ref = series.get(ref_year, {})
    parts = []
    if now.get("divided_share") is not None and ref.get("divided_share") is not None:
        parts.append(
            f"In {last}, {now['divided_share'] * 100:.0f}% of recorded votes were divided "
            f"(at least one member in ten on the losing side), against "
            f"{ref['divided_share'] * 100:.0f}% in {ref_year}."
        )
    valid = [(r["year"], r["agreement"]) for r in rows if r["agreement"] is not None]
    if now.get("agreement") is not None and valid:
        lowest_year = min(valid, key=lambda t: t[1])[0]
        parts.append(
            f"Two members that both took a side agreed {now['agreement'] * 100:.0f}% of the time in {last}"
            + (f"; the low point of the record was {lowest_year}." if lowest_year != last else ", the lowest on record.")
        )
    return {
        "years": years,
        "series": rows,
        "last_full_year": last,
        "takeaway": " ".join(parts),
        "caveat": (
            "Recorded votes only, so this is the contested end of the Assembly's work. "
            "Abstentions do not count toward a majority and are left out of both measures. "
            "The Charter's two-thirds test is failed by only a few votes a year and is "
            "included as contested_share."
        ),
    }


# ── 3. A power and the world ─────────────────────────────────────────────────


def _anchor_rows(df: pd.DataFrame, code: str) -> pd.DataFrame:
    codes = lineage_codes(code)
    return df[df["country_identifier"].isin(codes)]


def world_alignment_with(df: pd.DataFrame, code: str, name_lookup: Optional[dict] = None) -> dict:
    """Per year: how often other members sided with ``code`` when both took a
    side, and how many votes ``code`` cast with at most two companions."""
    code = str(code).strip().upper()
    res = resolutions_table(df)
    years = _complete_years(res)
    sided = df[df["vote"].isin([1, -1])][["rcid", "year", "country_identifier", "vote"]]
    anchor = _anchor_rows(sided, code)
    if anchor.empty:
        raise ValueError(f"No recorded votes for {code}")
    # When a predecessor and successor both appear in one year (1991/92), keep
    # the successor's vote.
    anchor = anchor.sort_values("country_identifier", key=lambda s: s != code).drop_duplicates("rcid")
    anchor_vote = anchor.set_index("rcid")["vote"]
    others = sided[~sided["country_identifier"].isin(lineage_codes(code))].copy()
    others = others[others["rcid"].isin(anchor_vote.index)]
    others["same"] = others["vote"].to_numpy() == anchor_vote.reindex(others["rcid"]).to_numpy()
    per_year = others.groupby("year")["same"].agg(["mean", "size"])

    # Isolation: anchor took a side with ≤ 2 companions.
    side_counts = others.groupby(["rcid", "vote"]).size()
    iso_rows = []
    for rcid, v in anchor_vote.items():
        companions = int(side_counts.get((rcid, v), 0))
        if companions <= 2:
            iso_rows.append((rcid, companions))
    iso = pd.DataFrame(iso_rows, columns=["rcid", "companions"]).merge(
        res[["rcid", "year", "title", "date"]], on="rcid", how="left"
    ) if iso_rows else pd.DataFrame(columns=["rcid", "companions", "year", "title", "date"])
    iso_by_year = iso.groupby("year").size() if not iso.empty else pd.Series(dtype=int)
    cast_by_year = anchor.groupby("year").size()

    series = []
    for y in years:
        m = per_year["mean"].get(y)
        series.append({
            "year": y,
            "agreement": None if m is None or pd.isna(m) else round(float(m), 3),
            "comparisons": int(per_year["size"].get(y, 0)),
            "isolated_votes": int(iso_by_year.get(y, 0)),
            "votes_cast": int(cast_by_year.get(y, 0)),
        })
    last = _last_full_year(res)
    by_year = {s["year"]: s for s in series}
    now = by_year.get(last, {})
    ref_year = 1990 if 1990 in by_year else years[0]
    ref = by_year.get(ref_year, {})
    names = name_lookup or {}
    display = names.get(code, code)
    parts = []
    if now.get("agreement") is not None and ref.get("agreement") is not None:
        parts.append(
            f"Other members sided with {display} in {now['agreement'] * 100:.0f}% of their votes in {last}, "
            f"compared with {ref['agreement'] * 100:.0f}% in {ref_year}."
        )
    if now.get("votes_cast"):
        parts.append(
            f"{display} voted with two or fewer other members {now['isolated_votes']} times in {last}, "
            f"out of {now['votes_cast']} votes cast."
        )
    recent_iso = iso.sort_values("date", ascending=False).head(8) if not iso.empty else iso
    return {
        "anchor": code,
        "anchor_name": display,
        "years": years,
        "series": series,
        "last_full_year": last,
        "recent_isolated": [
            {"rcid": int(r.rcid), "year": int(r.year), "title": r.title, "companions": int(r.companions)}
            for r in recent_iso.itertuples(index=False)
        ],
        "takeaway": " ".join(parts),
    }


# ── 4. Between Washington and Beijing ────────────────────────────────────────


def _agreement_with_anchors(
    df: pd.DataFrame, years: list[int], anchors: list[str], min_comparisons: int
) -> pd.DataFrame:
    matrix = _side_matrix(df, years)
    if matrix.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=matrix.index)
    for anchor in anchors:
        rows = [c for c in lineage_codes(anchor) if c in matrix.index]
        if not rows:
            out[anchor] = np.nan
            continue
        # successor first, predecessor fills gaps
        a = matrix.loc[rows[0]].to_numpy(dtype=float)
        for extra in rows[1:]:
            b = matrix.loc[extra].to_numpy(dtype=float)
            a = np.where(a == 0, b, a)
        m = matrix.to_numpy(dtype=float)
        both = (m != 0) & (a != 0)
        same = (m == a) & both
        comps = both.sum(axis=1)
        share = np.where(comps >= min_comparisons, same.sum(axis=1) / np.maximum(comps, 1), np.nan)
        out[anchor] = share
    return out


def alignment_scatter(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    base_start: Optional[int] = None,
    base_end: Optional[int] = None,
    anchors: tuple[str, str] = ("USA", "CHN"),
    min_comparisons: int = 15,
    name_lookup: Optional[dict] = None,
) -> dict:
    """Each member's agreement with two anchors over a window, and over an
    earlier baseline window so the chart can draw the move."""
    a1, a2 = anchors
    years = list(range(int(start_year), int(end_year) + 1))
    now = _agreement_with_anchors(df, years, [a1, a2], min_comparisons)
    base = None
    if base_start is not None and base_end is not None:
        base = _agreement_with_anchors(
            df, list(range(int(base_start), int(base_end) + 1)), [a1, a2], min_comparisons
        )
    names = name_lookup or {}
    points = []
    skip = set(lineage_codes(a1)) | set(lineage_codes(a2))
    for code, row in now.iterrows():
        if code in skip or pd.isna(row[a1]) or pd.isna(row[a2]):
            continue
        p = {
            "code": code,
            "name": names.get(code, code),
            "region": regional_group(code),
            a1.lower(): round(float(row[a1]), 3),
            a2.lower(): round(float(row[a2]), 3),
        }
        if base is not None and code in base.index and not (pd.isna(base.loc[code, a1]) or pd.isna(base.loc[code, a2])):
            p[f"{a1.lower()}_base"] = round(float(base.loc[code, a1]), 3)
            p[f"{a2.lower()}_base"] = round(float(base.loc[code, a2]), 3)
        points.append(p)
    moved_to_a2 = sum(
        1 for p in points
        if f"{a1.lower()}_base" in p
        and (p[a2.lower()] - p[a1.lower()]) > (p[f"{a2.lower()}_base"] - p[f"{a1.lower()}_base"]) + 0.02
    )
    moved_to_a1 = sum(
        1 for p in points
        if f"{a1.lower()}_base" in p
        and (p[a1.lower()] - p[a2.lower()]) > (p[f"{a1.lower()}_base"] - p[f"{a2.lower()}_base"]) + 0.02
    )
    closer_a2 = sum(1 for p in points if p[a2.lower()] > p[a1.lower()])
    n1, n2 = names.get(a1, a1), names.get(a2, a2)
    takeaway = (
        f"Over {start_year}–{end_year}, {closer_a2} of {len(points)} members voted more often with "
        f"{n2} than with {n1}."
    )
    if base is not None:
        takeaway += (
            f" Compared with {base_start}–{base_end}, {moved_to_a2} moved toward {n2} and "
            f"{moved_to_a1} toward {n1}."
        )
    return {
        "anchors": [a1, a2],
        "anchor_names": [n1, n2],
        "window": [int(start_year), int(end_year)],
        "baseline": None if base is None else [int(base_start), int(base_end)],
        "points": points,
        "takeaway": takeaway,
        "caveat": (
            "Agreement counts only votes where both countries took a side; abstentions are "
            f"ignored. Countries with fewer than {min_comparisons} shared votes are left out."
        ),
    }


# ── 5. Landmark and recurring votes ──────────────────────────────────────────


def landmark_votes(df: pd.DataFrame) -> list[dict]:
    res = resolutions_table(df)
    if "resolution" not in res.columns:
        return []
    by_symbol = res.set_index(res["resolution"].astype(str).str.strip())
    out = []
    for item in LANDMARK_VOTES:
        if item["symbol"] not in by_symbol.index:
            continue
        r = by_symbol.loc[item["symbol"]]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        out.append({
            "rcid": int(r["rcid"]),
            "symbol": item["symbol"],
            "label": item["label"],
            "why": item["why"],
            "title": r["title"],
            "date": str(r["date"])[:10],
            "year": int(r["year"]),
            "yes": int(r["total_yes"]) if pd.notna(r["total_yes"]) else None,
            "no": int(r["total_no"]) if pd.notna(r["total_no"]) else None,
            "abstain": int(r["total_abstentions"]) if pd.notna(r["total_abstentions"]) else None,
            "theme": r["theme"],
        })
    return out


VOTE_LABELS = {1: "yes", -1: "no", 0: "abstain"}


def _region_note(regions: list[dict]) -> str:
    """One sentence on where a vote's support came from and where it split."""
    scored = []
    for r in regions:
        if r["region"] == "Other":
            continue
        sided = r["yes"] + r["no"]
        if sided == 0:
            continue
        scored.append((r["yes"] / sided, r))
    if not scored:
        return ""
    scored.sort(key=lambda t: t[0], reverse=True)
    top_share, top = scored[0]
    low_share, low = scored[-1]
    if low_share >= 0.9:
        return "Every regional group backed it by nine to one or better."
    if top_share <= 0.5:
        return f"No regional group gave it a majority; {top['region']} came closest."
    against = low["no"] + low["abstain"]
    return (
        f"{top['region']} backed it {top['yes']} to {top['no']}; the split ran through "
        f"{low['region']}, where {against} of {low['members']} members voted against or abstained."
    )


def resolution_vote_map(df: pd.DataFrame, rcid: int, name_lookup: Optional[dict] = None) -> dict:
    rows = df[df["rcid"] == rcid]
    if rows.empty:
        raise ValueError(f"Unknown resolution id {rcid}")
    res = resolutions_table(df)
    meta = res[res["rcid"] == rcid].iloc[0]
    names = name_lookup or {}
    votes = []
    for r in rows.drop_duplicates("country_identifier").itertuples(index=False):
        v = r.vote
        label = "absent" if pd.isna(v) else VOTE_LABELS.get(int(v), "absent")
        votes.append({"code": r.country_identifier, "name": names.get(r.country_identifier, r.country_identifier), "vote": label})
    tally = {k: sum(1 for v in votes if v["vote"] == k) for k in ("yes", "no", "abstain", "absent")}
    by_region: dict[str, dict] = {}
    for v in votes:
        group = regional_group(v["code"])
        cell = by_region.setdefault(group, {"yes": 0, "no": 0, "abstain": 0, "absent": 0, "members": 0})
        cell[v["vote"]] += 1
        cell["members"] += 1
    regions = [
        {"region": g, **by_region[g]} for g in GROUP_ORDER + ["Other"] if g in by_region
    ]
    return {
        "regions": regions,
        "region_note": _region_note(regions),
        "rcid": int(rcid),
        "symbol": str(meta.get("resolution", "")),
        "title": meta["title"],
        "date": str(meta["date"])[:10],
        "year": int(meta["year"]),
        "theme": meta["theme"],
        "official": {
            "yes": None if pd.isna(meta["total_yes"]) else int(meta["total_yes"]),
            "no": None if pd.isna(meta["total_no"]) else int(meta["total_no"]),
            "abstain": None if pd.isna(meta["total_abstentions"]) else int(meta["total_abstentions"]),
        },
        "tally": tally,
        "votes": votes,
    }


def recurring_vote_series(df: pd.DataFrame, key: str) -> dict:
    spec = RECURRING_VOTES.get(key)
    if spec is None:
        raise ValueError(f"Unknown recurring vote {key!r}")
    res = resolutions_table(df)
    hits = res[res["title"].fillna("").str.contains(spec["pattern"], case=False, regex=True)]
    hits = hits.sort_values("date").drop_duplicates("year", keep="last")
    series = [
        {
            "year": int(r.year),
            "rcid": int(r.rcid),
            "symbol": str(r.resolution),
            "yes": int(r.total_yes),
            "no": int(r.total_no),
            "abstain": int(r.total_abstentions),
        }
        for r in hits.itertuples(index=False)
    ]
    takeaway = ""
    if len(series) >= 2:
        first, last = series[0], series[-1]
        peak = max(series, key=lambda s: s["yes"])
        takeaway = (
            f"Support rose from {first['yes']} in {first['year']} to a peak of {peak['yes']} in {peak['year']}; "
            f"in {last['year']} it stood at {last['yes']} for, {last['no']} against, {last['abstain']} abstaining."
        )
    return {
        "key": key,
        "label": spec["label"],
        "why": spec["why"],
        "series": series,
        "takeaway": takeaway,
        "available": [{"key": k, "label": v["label"]} for k, v in RECURRING_VOTES.items()],
    }


# ── 6. A country's long view ─────────────────────────────────────────────────


def country_story(
    df: pd.DataFrame,
    code: str,
    name_lookup: Optional[dict] = None,
    anchors: tuple[str, ...] = ("USA", "RUS", "CHN"),
    min_votes: int = 5,
) -> dict:
    """One member across the whole record: per year, how often it sided with
    each anchor when both took a side, how often it was on the winning side,
    and how many votes it cast. Predecessor states are folded in (Russia's
    line continues the USSR's)."""
    code = str(code).strip().upper()
    res = resolutions_table(df)
    years = _complete_years(res)
    sided = df[df["vote"].isin([1, -1])][["rcid", "year", "country_identifier", "vote"]]

    def _rows(anchor: str) -> pd.Series:
        rows = sided[sided["country_identifier"].isin(lineage_codes(anchor))]
        rows = rows.sort_values("country_identifier", key=lambda s: s != anchor)
        return rows.drop_duplicates("rcid").set_index("rcid")

    mine = _rows(code)
    if mine.empty:
        raise ValueError(f"No recorded votes for {code}")
    tallies = res.set_index("rcid")
    majority = pd.Series(
        np.where(tallies["total_yes"] > tallies["total_no"], 1,
                 np.where(tallies["total_no"] > tallies["total_yes"], -1, 0)),
        index=tallies.index,
    )
    with_majority = (mine["vote"] == majority.reindex(mine.index)).groupby(mine["year"]).mean()
    votes_cast = mine.groupby("year").size()
    per_anchor: dict[str, pd.Series] = {}
    per_anchor_n: dict[str, pd.Series] = {}
    for anchor in anchors:
        if anchor in lineage_codes(code) or code in lineage_codes(anchor):
            continue
        arow = _rows(anchor)["vote"].rename("anchor")
        joined = mine.join(arow, how="inner")
        if joined.empty:
            continue
        per_anchor[anchor] = (joined["vote"] == joined["anchor"]).groupby(joined["year"]).mean()
        per_anchor_n[anchor] = joined.groupby("year").size()

    series = []
    for y in years:
        cast = int(votes_cast.get(y, 0))
        row = {
            "year": y,
            "votes_cast": cast,
            "with_majority": None if cast == 0 else round(float(with_majority.get(y, np.nan)), 3),
        }
        for anchor in anchors:
            s = per_anchor.get(anchor)
            n = int(per_anchor_n[anchor].get(y, 0)) if anchor in per_anchor_n else 0
            val = s.get(y) if s is not None else None
            row[anchor.lower()] = None if val is None or pd.isna(val) or n == 0 else round(float(val), 3)
        series.append(row)
    for row in series:
        if row["with_majority"] is not None and pd.isna(row["with_majority"]):
            row["with_majority"] = None

    names = name_lookup or {}
    name = names.get(code, code)
    active = [r for r in series if r["votes_cast"] >= min_votes]
    first_year = int(active[0]["year"]) if active else int(mine["year"].min())
    last = active[-1] if active else None
    parts = []
    if last:
        anchor_bits = []
        for anchor in anchors:
            v = last.get(anchor.lower())
            if v is not None:
                anchor_bits.append(f"{names.get(anchor, anchor)} {v * 100:.0f}%")
        if anchor_bits:
            parts.append(
                f"In {last['year']} {name} sided with " + ", ".join(anchor_bits)
                + " of the time when both took a side."
            )
        recent = [r for r in active if r["year"] > last["year"] - 5]
        means = {}
        for anchor in anchors:
            vals = [r[anchor.lower()] for r in recent if r.get(anchor.lower()) is not None]
            if vals:
                means[anchor] = sum(vals) / len(vals)
        if means:
            partner = max(means, key=means.get)
            parts.append(
                f"Its closest permanent member over {last['year'] - 4}–{last['year']} has been "
                f"{names.get(partner, partner)} ({means[partner] * 100:.0f}%)."
            )
        if last.get("with_majority") is not None:
            parts.append(
                f"It was on the winning side of {last['with_majority'] * 100:.0f}% of the votes it "
                f"took a side on in {last['year']}."
            )
    return {
        "code": code,
        "name": name,
        "region": regional_group(code),
        "first_year": first_year,
        "anchors": list(anchors),
        "anchor_names": {a: names.get(a, a) for a in anchors},
        "series": series,
        "latest": last,
        "takeaway": " ".join(parts),
        "caveat": (
            "Agreement counts only votes where both countries took a side; abstentions are "
            "ignored. Years with fewer than five votes cast are shown but not used for the "
            "summary. Predecessor states are folded into their successors."
        ),
    }


# ── 7. This week in the Assembly ─────────────────────────────────────────────


def recent_votes(
    df: pd.DataFrame,
    days: int = 14,
    limit: int = 10,
    name_lookup: Optional[dict] = None,
    as_of: Optional[str] = None,
) -> dict:
    """The recorded votes in the last ``days`` of the record, most recent
    first, with the dissenters named when there are few of them.

    ``sitting`` is true only when the latest recorded vote falls within
    ``days`` of ``as_of`` (the edition date, default today): the Assembly is
    voting *now*. The listing itself is always relative to the latest vote in
    the data, because the export lags the calendar by about a week.
    """
    res = resolutions_table(df)
    dates = pd.to_datetime(res["date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return {"sitting": False, "count": 0, "votes": [], "takeaway": ""}
    today = pd.Timestamp(as_of) if as_of else pd.Timestamp.utcnow().tz_localize(None).normalize()
    sitting = (today - latest).days <= days
    start = latest - pd.Timedelta(days=days)
    window = res[(dates >= start) & (dates <= latest)].assign(_d=dates).sort_values("_d", ascending=False)
    sided = df[df["vote"].isin([1, -1])][["rcid", "country_identifier", "vote"]]
    names = name_lookup or {}
    votes = []
    for r in window.head(limit).itertuples(index=False):
        yes_n = int(r.total_yes) if pd.notna(r.total_yes) else 0
        no_n = int(r.total_no) if pd.notna(r.total_no) else 0
        abst = int(r.total_abstentions) if pd.notna(r.total_abstentions) else 0
        losing = -1 if yes_n >= no_n else 1
        rows = sided[sided["rcid"] == r.rcid]
        dissenters = sorted(names.get(c, c) for c in rows[rows["vote"] == losing]["country_identifier"].unique())
        votes.append({
            "rcid": int(r.rcid),
            "date": str(r.date)[:10],
            "symbol": str(r.resolution) if "resolution" in window.columns else "",
            "title": r.title,
            "theme": r.theme,
            "yes": yes_n,
            "no": no_n,
            "abstain": abst,
            "winning_share": None if pd.isna(r.winning_share) else round(float(r.winning_share), 3),
            "dissenters": dissenters if len(dissenters) <= 8 else [],
            "dissenter_count": len(dissenters),
        })
    takeaway = ""
    if votes:
        closest = min(votes, key=lambda v: v["winning_share"] if v["winning_share"] is not None else 1.0)
        takeaway = (
            f"{len(window)} recorded vote{'s' if len(window) != 1 else ''} in the fortnight to "
            f"{latest.strftime('%-d %B %Y')}. The closest was \"{closest['title']}\", "
            f"{closest['yes']} to {closest['no']} with {closest['abstain']} abstaining."
        )
    return {
        "sitting": bool(votes) and bool(sitting),
        "as_of": today.strftime("%Y-%m-%d"),
        "days_since_latest": int((today - latest).days),
        "window_start": start.strftime("%Y-%m-%d"),
        "window_end": latest.strftime("%Y-%m-%d"),
        "count": int(len(window)),
        "votes": votes,
        "takeaway": takeaway,
    }
