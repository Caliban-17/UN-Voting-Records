"""The world, coloured by who votes together.

Members are grouped by how often they took the same side in a year's recorded
votes; each group gets one hue, and the shade says how tightly a country votes
with its group. The map is an inline SVG drawn from Natural Earth outlines
(vendored), so it works on the page and, converted to an image, in email.
Flags are Unicode regional-indicator pairs, so they need no image files.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from html import escape
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.regional_groups import LINEAGE, WESTERN, regional_group

ROOT = Path(__file__).resolve().parents[1]
OUTLINES = ROOT / "static" / "vendor" / "countries_110m.json"
ISO_TABLE = ROOT / "data" / "iso_codes.csv"
INK, MUTED, PAPER, NO_DATA = "#102c35", "#5d6f73", "#fffdf8", "#e4e1d8"
MIN_VOTES = 20

# Bloc key -> (label, shades from loosest to tightest).
BLOCS = {
    "washington": ("Votes with Washington", ("#9db8d9", "#4f7fb8", "#1f4e8c")),
    "west": ("The Western group", ("#b4dcd6", "#5fb3a8", "#126f66")),
    "majority": ("The majority", ("#f1dcae", "#dfb45a", "#b07d12")),
    "moscow": ("Votes with Moscow", ("#eab6a8", "#d4765f", "#a8351f")),
}
ORDER = ("washington", "west", "majority", "moscow")
# Outlines drawn separately that vote as part of a member state.
TERRITORIES = {"GRL": "DNK", "PRI": "USA", "NCL": "FRA", "FLK": "GBR", "ATF": "FRA"}
# Members a reader will look for first, in the order to name them.
NOTABLE = ("USA", "CHN", "RUS", "GBR", "FRA", "DEU", "IND", "JPN", "BRA", "ZAF", "TUR", "SAU", "IDN", "MEX",
           "NGA", "EGY", "PAK", "KOR", "AUS", "CAN", "ITA", "ESP", "POL", "UKR", "ISR", "IRN", "ARG", "HUN")


# ── codes and flags ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def iso_table() -> dict[str, dict]:
    with ISO_TABLE.open(newline="") as handle:
        return {row["alpha3"]: row for row in csv.DictReader(handle)}


def flag(code: str) -> str:
    """The flag emoji for an ISO-3 code, or '' when there is none (a state
    that no longer exists has no flag to show)."""
    row = iso_table().get(str(code).upper())
    if not row:
        return ""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in row["alpha2"])


# ── blocs from the votes ────────────────────────────────────────────────────


def agreement_matrix(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Share of shared Yes/No votes on which each pair took the same side."""
    from src.story_analysis import _side_matrix

    sides = _side_matrix(df, [year])
    sides = sides[(sides != 0).sum(axis=1) >= MIN_VOTES]
    values = sides.to_numpy(dtype=float)
    took = (values != 0).astype(float)
    same = (values @ values.T + took @ took.T) / 2     # both sided and matched
    shared = took @ took.T
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(shared > 0, same / shared, np.nan)
    return pd.DataFrame(share, index=sides.index, columns=sides.index)


def voting_blocs(df: pd.DataFrame, year: int) -> dict:
    """Group the members by agreement in ``year``.

    Average-linkage clustering finds the groups; they are then named for what
    they contain: the largest is the majority, the one holding most of the
    Western European and Others group is the Western group, and the ones
    around the United States and Russia are named for those capitals. A group
    too small to stand alone joins whichever named group it agrees with most.
    ``tightness`` is a member's mean agreement with the rest of its group.
    """
    from sklearn.cluster import AgglomerativeClustering

    agree = agreement_matrix(df, year)
    if len(agree) < 30:
        return {}
    filled = agree.fillna(0.5).to_numpy()
    labels = AgglomerativeClustering(n_clusters=6, metric="precomputed", linkage="average").fit_predict(1 - filled)
    codes = list(agree.index)
    clusters = [[c for c, lab in zip(codes, labels) if lab == i] for i in range(6)]
    clusters.sort(key=len, reverse=True)
    successors = {old: new for new, olds in LINEAGE.items() for old in olds}

    def holding(code):
        return next((c for c in clusters if code in c or successors.get(code) in c), None)

    named: dict[str, list[str]] = {"majority": clusters[0]}
    western = max(clusters[1:], key=lambda c: sum(regional_group(x) == WESTERN for x in c), default=None)
    if western and sum(regional_group(x) == WESTERN for x in western) >= 8:
        named["west"] = western
    for key, capital in (("washington", "USA"), ("moscow", "RUS")):
        cluster = holding(capital)
        if cluster is not None and all(cluster is not c for c in named.values()):
            named[key] = cluster
    taken = {id(c) for c in named.values()}
    for cluster in clusters:
        if id(cluster) in taken:
            continue
        # A stray handful: put each with the named group it sides with most.
        for code in cluster:
            best = max(named, key=lambda k: float(np.nanmean(agree.loc[code, named[k]])))
            named[best] = named[best] + [code]
    members = {}
    for key, group in named.items():
        for code in group:
            others = [c for c in group if c != code]
            tight = float(np.nanmean(agree.loc[code, others])) if others else 1.0
            members[code] = {"bloc": key, "tightness": round(tight, 3)}
    between = {}
    keys = [k for k in ORDER if k in named]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            between[f"{a}|{b}"] = round(float(np.nanmean(agree.loc[named[a], named[b]].to_numpy())), 3)
    return {
        "year": int(year),
        "blocs": [{"key": k, "label": BLOCS[k][0], "size": len(named[k]),
                   "cohesion": round(float(np.mean([members[c]["tightness"] for c in named[k]])), 3),
                   "members": sorted(named[k])} for k in keys],
        "members": members, "between": between,
    }


# ── drawing ─────────────────────────────────────────────────────────────────

_A = (1.340264, -0.081106, 0.000893, 0.003796)


def _project(lon: float, lat: float) -> tuple[float, float]:
    """Equal Earth: areas are true, so a bloc's share of the map is its share of land."""
    theta = math.asin(math.sqrt(3) / 2 * math.sin(math.radians(lat)))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = 2 * math.sqrt(3) * math.radians(lon) * math.cos(theta) / (3 * (9 * _A[3] * t6 * t2 + 7 * _A[2] * t6 + 3 * _A[1] * t2 + _A[0]))
    y = theta * (_A[3] * t6 * t2 + _A[2] * t6 + _A[1] * t2 + _A[0])
    return x, y


@lru_cache(maxsize=1)
def _outlines() -> list[tuple[str, str]]:
    """(ISO-3 code or '', SVG path) for every country, projected once."""
    topo = json.loads(OUTLINES.read_text())
    (sx, sy), (tx, ty) = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        points = []
        for dx, dy in arc:
            x, y = x + dx, y + dy
            points.append((x * sx + tx, y * sy + ty))
        arcs.append(points)
    numeric = {row["numeric"]: code for code, row in iso_table().items()}
    width, height, pad = 640.0, 310.0, 4.0
    x_max, y_max = _project(180, 0)[0], _project(0, 84)[1]
    y_min = _project(0, -58)[1]
    scale = min((width - 2 * pad) / (2 * x_max), (height - 2 * pad) / (y_max - y_min))

    def place(point):
        return (width / 2 + point[0] * scale, pad + (y_max - point[1]) * scale)

    def ring(indices):
        points = []
        for index in indices:
            segment = arcs[index] if index >= 0 else arcs[~index][::-1]
            points.extend(segment if not points else segment[1:])
        # A ring that crosses the date line (Russia's far east, Fiji) would be
        # drawn as a streak across the whole map: keep it on one side instead.
        if max(p[0] for p in points) - min(p[0] for p in points) > 180:
            points = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in points]
        return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in map(place, (_project(*p) for p in points))) + "Z"

    out = []
    for geometry in topo["objects"]["countries"]["geometries"]:
        code = numeric.get(str(geometry.get("id", "")).zfill(3), "")
        if code == "ATA":
            continue
        code = TERRITORIES.get(code, code)
        polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
        out.append((code, "".join(ring(r) for polygon in polygons for r in polygon)))
    return out


def _shade(member: dict, blocs: dict) -> str:
    """Three shades per bloc, by where the member sits among its own group."""
    group = [m["tightness"] for m in blocs["members"].values() if m["bloc"] == member["bloc"]]
    low, high = np.quantile(group, [0.33, 0.67]) if len(group) >= 3 else (0, 0)
    step = 2 if member["tightness"] >= high else (1 if member["tightness"] >= low else 0)
    return BLOCS[member["bloc"]][1][step]


def bloc_map_svg(blocs: dict, title: str) -> str:
    """The map: one hue per bloc, darker where a country votes more tightly
    with its bloc, grey where there is no vote to go on."""
    width, top = 640, 58
    rows = (len(blocs["blocs"]) + 1) // 2
    height = top + 310 + 26 + 20 * rows + 16
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-label="{escape(title, quote=True)}">',
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/><rect width="44" height="4" fill="{INK}"/>',
            f'<text x="0" y="28" font-family="Georgia,serif" font-size="19" font-weight="700" fill="{INK}">{escape(title)}</text>',
            f'<text x="0" y="48" font-family="Arial,sans-serif" font-size="12.5" fill="{MUTED}">Groups found from how members '
            f'voted in {blocs["year"]}; darker means voting more tightly with the group</text>',
            f'<g transform="translate(0,{top})" stroke="{PAPER}" stroke-width="0.5" stroke-linejoin="round">']
    for code, path in _outlines():
        member = blocs["members"].get(code)
        body.append(f'<path d="{path}" fill="{_shade(member, blocs) if member else NO_DATA}"/>')
    body.append("</g>")
    y = top + 310 + 22
    for index, bloc in enumerate(blocs["blocs"]):
        label, shades = BLOCS[bloc["key"]]
        x, y = (index % 2) * 320, top + 310 + 22 + 20 * (index // 2)
        for i, colour in enumerate(shades):
            body.append(f'<rect x="{x + i * 13}" y="{y - 10}" width="12" height="12" fill="{colour}"/>')
        text = f'{label} ({bloc["size"]})'
        body.append(f'<text x="{x + 44}" y="{y}" font-family="Arial,sans-serif" font-size="12" fill="{INK}">{escape(text)}</text>')
    body.append(f'<text x="0" y="{height - 6}" font-family="Arial,sans-serif" font-size="10.5" fill="{MUTED}">Grey: not a member, or '
                f'too few votes. Source: UN Digital Library voting record; outlines Natural Earth.</text></svg>')
    return "".join(body)


def bloc_lines(blocs: dict, names: dict | None = None, limit: int = 12) -> list[str]:
    """One line per bloc, with flags: every member of a small bloc; for a
    large one, the members a reader will look for first."""
    names = names or {}
    lines = []
    for bloc in blocs["blocs"]:
        members = bloc["members"]
        if len(members) <= limit:
            shown = sorted(members, key=lambda c: -blocs["members"][c]["tightness"])
        else:
            shown = [c for c in NOTABLE if c in members][:limit]
        listed = ", ".join(f"{flag(c)} {names.get(c, c)}".strip() for c in shown)
        more = f" and {len(members) - len(shown)} more" if len(members) > len(shown) else ""
        lead = "including " if more else ""
        lines.append(f"{bloc['label']} ({bloc['size']}; agreeing with each other {bloc['cohesion'] * 100:.0f}% "
                     f"of the time): {lead}{listed}{more}.")
    return lines


def bloc_title(blocs: dict, names: dict | None = None) -> str:
    """The map's finding, from the blocs themselves."""
    names = names or {}
    by_key = {b["key"]: b for b in blocs["blocs"]}
    washington = by_key.get("washington")
    if washington and washington["size"] <= 5:
        others = [names.get(c, c) for c in washington["members"] if c != "USA"]
        company = " and ".join(others) if len(others) <= 2 else f"{len(others)} others"
        return f"Only {company} vote{'s' if len(others) == 1 else ''} with Washington" if others else "Washington votes alone"
    return f"{len(blocs['blocs'])} voting blocs in {blocs['year']}"
