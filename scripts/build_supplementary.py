"""
Build the supplementary datasets the "Through which lens?" layer reads:

* ``data/vdem_regimes.csv`` — regime type (Regimes of the World, 0–3) and the
  V-Dem liberal democracy index per member and year, from Our World in
  Data's processed V-Dem series.
* ``data/women_in_parliament.csv`` — share of seats held by women, IPU via
  the World Bank where available (1997 on), V-Dem's series otherwise.
* ``data/cow_defense_communities.csv`` — for each year to 2012, the
  modularity community of the Correlates of War defence-pact graph each
  state belongs to (see ``build_cow``), keyed by ISO-3.
* ``data/colonial_history.csv`` — each state's former colonial ruler,
  independence year and independence type, from the ICOW Colonial History
  data (Hensel), keyed by ISO-3.

Every output is ``code,year,...`` keyed by the ISO-3 codes the voting record
uses, with UN lineages applied (the USSR takes Russia's V-Dem series, and so
on). Sources and licences are listed in ``data/supplementary_sources.md``.

Usage: ``python scripts/build_supplementary.py --raw <dir with the
downloads>`` (downloads them when ``--fetch`` is given).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
DATA = ROOT / "data"

OWID = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
SOURCES = {
    "political-regime": OWID.format(slug="political-regime"),
    "liberal-democracy-index": OWID.format(slug="liberal-democracy-index"),
    "share-of-women-in-parliament": OWID.format(slug="share-of-women-in-parliament"),
    "wb_women_parl": "https://api.worldbank.org/v2/country/all/indicator/SG.GEN.PARL.ZS?format=json&per_page=20000",
}

# OWID's codes for states the voting record keys differently.
OWID_TO_ISO = {"OWID_GDR": "DDR", "OWID_GFR": "GER", "OWID_YAR": "YEM", "OWID_YPR": "YMD", "OWID_ZAN": "EAZ"}
# Predecessor states take the successor's series for the years they existed.
LINEAGE_BACKFILL = [("SUN", "RUS", 1946, 1991), ("CSK", "CZE", 1946, 1992), ("YUG", "SRB", 1946, 1991),
                    ("SCG", "SRB", 1992, 2005)]
FIRST_YEAR = 1946


def _read_owid(path: Path) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        value_col = [c for c in reader.fieldnames if c not in ("entity", "code", "year", "owid_region")][0]
        for row in reader:
            code = OWID_TO_ISO.get(row["code"], row["code"])
            if not code or code.startswith(("OWID_", "WB_")) or row[value_col] in ("", None):
                continue
            year = int(row["year"])
            if year >= FIRST_YEAR:
                out[(code, year)] = row[value_col]
    return out


def _backfill(table: dict[tuple[str, int], str]) -> None:
    for old, new, start, end in LINEAGE_BACKFILL:
        for year in range(start, end + 1):
            if (old, year) not in table and (new, year) in table:
                table[(old, year)] = table[(new, year)]


def build_vdem(raw: Path) -> Path:
    regimes = _read_owid(raw / "political-regime.csv")
    libdem = _read_owid(raw / "liberal-democracy-index.csv")
    _backfill(regimes)
    _backfill(libdem)
    keys = sorted(set(regimes) | set(libdem))
    out = DATA / "vdem_regimes.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "regime", "libdem"])
        for code, year in keys:
            ld = libdem.get((code, year))
            w.writerow([code, year, regimes.get((code, year), ""), f"{float(ld):.3f}" if ld else ""])
    logger.info("wrote %s (%d rows)", out, len(keys))
    return out


def build_women(raw: Path) -> Path:
    vdem = _read_owid(raw / "share-of-women-in-parliament.csv")
    _backfill(vdem)
    ipu: dict[tuple[str, int], float] = {}
    with (raw / "wb_women_parl.json").open(encoding="utf-8") as f:
        _, rows = json.load(f)
    for r in rows:
        code, value = r.get("countryiso3code"), r.get("value")
        if code and value is not None and len(code) == 3:
            ipu[(code, int(r["date"]))] = float(value)
    keys = sorted(set(vdem) | set(ipu))
    out = DATA / "women_in_parliament.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "share", "source"])
        n = 0
        for key in keys:
            if key in ipu:
                w.writerow([key[0], key[1], f"{ipu[key]:.1f}", "ipu"])
            else:
                w.writerow([key[0], key[1], f"{float(vdem[key]):.1f}", "vdem"])
            n += 1
    logger.info("wrote %s (%d rows)", out, n)
    return out


# ── Correlates of War: defence-pact communities ──────────────────────────────

# COW state abbreviations that differ from ISO-3 (states in the UN record).
COW_TO_ISO = {
    "UKG": "GBR", "FRN": "FRA", "GMY": "DEU", "GFR": "GER", "GDR": "DDR", "SPN": "ESP", "POR": "PRT",
    "NTH": "NLD", "SWZ": "CHE", "AUS": "AUT", "CZE": "CSK", "CZR": "CZE", "SLO": "SVK", "ROM": "ROU",
    "BUL": "BGR", "YUG": "YUG", "GRC": "GRC", "CYP": "CYP", "MLT": "MLT", "ICE": "ISL", "IRE": "IRL",
    "DEN": "DNK", "SWD": "SWE", "LAT": "LVA", "LIT": "LTU", "UKR": "UKR", "BLR": "BLR", "MLD": "MDA",
    "CRO": "HRV", "BOS": "BIH", "SLV": "SVN", "MAC": "MKD", "MNG": "MNE", "KOS": "XKX", "GRG": "GEO",
    "AZE": "AZE", "KZK": "KAZ", "KYR": "KGZ", "TAJ": "TJK", "TKM": "TKM", "UZB": "UZB", "MON": "MNG",
    "TAW": "TWN", "PRK": "PRK", "ROK": "KOR", "JPN": "JPN", "CHN": "CHN", "IND": "IND", "BHU": "BTN",
    "PAK": "PAK", "BNG": "BGD", "MYA": "MMR", "SRI": "LKA", "MAD": "MDV", "NEP": "NPL", "THI": "THA",
    "CAM": "KHM", "LAO": "LAO", "DRV": "VNM", "RVN": "VDR", "MAL": "MYS", "SIN": "SGP", "BRU": "BRN",
    "PHI": "PHL", "INS": "IDN", "ETM": "TLS", "AUL": "AUS", "PNG": "PNG", "NEW": "NZL", "VAN": "VUT",
    "SOL": "SLB", "FIJ": "FJI", "KIR": "KIR", "NAU": "NRU", "TON": "TON", "TUV": "TUV", "MSI": "MHL",
    "PAL": "PLW", "FSM": "FSM", "WSM": "WSM", "IRN": "IRN", "TUR": "TUR", "IRQ": "IRQ", "EGY": "EGY",
    "SYR": "SYR", "LEB": "LBN", "JOR": "JOR", "ISR": "ISR", "SAU": "SAU", "YAR": "YEM", "YPR": "YMD",
    "YEM": "YEM", "KUW": "KWT", "BAH": "BHR", "QAT": "QAT", "UAE": "ARE", "OMA": "OMN", "AFG": "AFG",
    "MOR": "MAR", "ALG": "DZA", "TUN": "TUN", "LIB": "LBY", "SUD": "SDN", "SSD": "SSD", "MAA": "MRT",
    "MLI": "MLI", "SEN": "SEN", "BEN": "BEN", "NIR": "NER", "CDI": "CIV", "GUI": "GIN", "BFO": "BFA",
    "LBR": "LBR", "SIE": "SLE", "GHA": "GHA", "TOG": "TGO", "CAO": "CMR", "NIG": "NGA", "GAB": "GAB",
    "CEN": "CAF", "CHA": "TCD", "CON": "COG", "DRC": "COD", "UGA": "UGA", "KEN": "KEN", "TAZ": "TZA",
    "ZAN": "EAZ", "BUI": "BDI", "RWA": "RWA", "SOM": "SOM", "DJI": "DJI", "ETH": "ETH", "ERI": "ERI",
    "ANG": "AGO", "MZM": "MOZ", "ZAM": "ZMB", "ZIM": "ZWE", "MAW": "MWI", "SAF": "ZAF", "NAM": "NAM",
    "LES": "LSO", "BOT": "BWA", "SWA": "SWZ", "MAG": "MDG", "COM": "COM", "MAS": "MUS", "SEY": "SYC",
    "CAP": "CPV", "STP": "STP", "GNB": "GNB", "EQG": "GNQ", "GAM": "GMB", "USA": "USA", "CAN": "CAN",
    "BHM": "BHS", "CUB": "CUB", "HAI": "HTI", "DOM": "DOM", "JAM": "JAM", "TRI": "TTO", "BAR": "BRB",
    "DMA": "DMA", "GRN": "GRD", "SLU": "LCA", "SVG": "VCT", "AAB": "ATG", "SKN": "KNA", "MEX": "MEX",
    "BLZ": "BLZ", "GUA": "GTM", "HON": "HND", "SAL": "SLV", "NIC": "NIC", "COS": "CRI", "PAN": "PAN",
    "COL": "COL", "VEN": "VEN", "GUY": "GUY", "SUR": "SUR", "ECU": "ECU", "PER": "PER", "BRA": "BRA",
    "BOL": "BOL", "PAR": "PRY", "CHL": "CHL", "ARG": "ARG", "URU": "URY", "RUS": "RUS", "ARM": "ARM",
    "EST": "EST", "FIN": "FIN", "NOR": "NOR", "POL": "POL", "HUN": "HUN", "ALB": "ALB", "ITA": "ITA",
    "BEL": "BEL", "LUX": "LUX", "LIE": "LIE", "MNC": "MCO", "SNM": "SMR", "AND": "AND", "SUN": "SUN", "DEN": "DNK",
}


def _cow_ccode_to_iso(raw: Path) -> dict[str, str]:
    """COW numeric code -> ISO-3, through the COW state list's abbreviations."""
    out: dict[str, str] = {}
    src = raw / "states2016.csv"
    if not src.exists():
        return out
    with src.open(encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            iso = COW_TO_ISO.get(row["stateabb"])
            if iso:
                out[row["ccode"]] = iso
    return out


def build_cow(raw: Path) -> Path | None:
    """Louvain communities of the defence-pact graph per year (COW Formal
    Alliances v4.1, dyad-year file; ``defense == 1``), to 2012 where the
    data end. A community is labelled by its alphabetically first member."""
    src = next(iter(sorted(raw.glob("alliance_v4.1_by_dyad_yearly*"))), None)
    if src is None:
        logger.warning("no COW dyad-year file in %s; skipping", raw)
        return None
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            text = z.read(name).decode("utf-8", errors="replace")
    else:
        text = src.read_text(encoding="utf-8", errors="replace")
    iso_of = _cow_ccode_to_iso(raw)
    edges: dict[int, list[tuple[str, str]]] = defaultdict(list)
    unmapped: set[str] = set()
    for row in csv.DictReader(io.StringIO(text)):
        if row["defense"] != "1":
            continue
        year = int(row["year"])
        if year < FIRST_YEAR:
            continue
        a, b = iso_of.get(row["ccode1"]), iso_of.get(row["ccode2"])
        for code, name in ((a, row["state_name1"]), (b, row["state_name2"])):
            if code is None:
                unmapped.add(name)
        if a and b:
            edges[year].append((a, b))
    import networkx as nx  # noqa: E402  (project dependency; imported late to keep --help fast)

    out = DATA / "cow_defense_communities.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "community"])
        n = 0
        for year in sorted(edges):
            graph = nx.Graph()
            graph.add_edges_from(edges[year])
            # Modularity communities: multilateral pacts are cliques, and the
            # bilateral treaties that bridge them (UK-Jordan, Turkey-Azerbaijan)
            # must not fuse NATO, the Arab League and the CSTO into one block.
            communities = nx.community.louvain_communities(graph, seed=0)
            for members in communities:
                if len(members) < 2:
                    continue
                label = "pact:" + min(members)
                for code in sorted(members):
                    w.writerow([code, year, label])
                    n += 1
    if unmapped:
        logger.warning("COW states without an ISO code (left out): %s", sorted(unmapped))
    logger.info("wrote %s (%d rows)", out, n)
    return out


# ── ICOW colonial history ────────────────────────────────────────────────────

# COW codes of the overseas colonial empires whose former possessions make up
# the post-1945 decolonisation wave (Britain, France, Spain, Portugal, the
# Netherlands, Belgium, Italy, Germany, the United States, Japan, Denmark).
OVERSEAS_EMPIRES = {"200", "220", "230", "235", "210", "211", "325", "255", "2", "740", "390"}


def build_colonial(raw: Path) -> Path | None:
    """One row per state: former ruler (ISO-3, blank when none), independence
    year and the ICOW independence type (1 formation, 2 decolonisation,
    3 secession, 4 partition), from ICOW Colonial History v1.1."""
    src = next(iter(raw.rglob("coldata*.csv")), None)
    if src is None:
        logger.warning("no ICOW coldata csv under %s; skipping", raw)
        return None
    iso_of = _cow_ccode_to_iso(raw)
    out = DATA / "colonial_history.csv"
    n = 0
    with src.open(encoding="utf-8-sig", errors="replace") as f, out.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["code", "ruler", "independence_year", "independence_type", "overseas_empire"])
        for row in csv.DictReader(f):
            code = iso_of.get(row["State"])
            if not code:
                continue
            ruler_cc = row.get("ColRuler", "-9")
            ruler = iso_of.get(ruler_cc, "") if ruler_cc not in ("-9", "") else ""
            ind = row.get("IndDate", "-9")
            year = int(ind[:4]) if ind not in ("-9", "") and len(ind) >= 4 else ""
            w.writerow([code, ruler, year, row.get("IndType", ""), int(ruler_cc in OVERSEAS_EMPIRES)])
            n += 1
    logger.info("wrote %s (%d rows)", out, n)
    return out


def fetch(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = raw / (f"{name}.json" if name.startswith("wb_") else f"{name}.csv")
        if target.exists():
            continue
        logger.info("fetching %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "UN-Alignment-Atlas/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="directory holding (or receiving) the raw downloads")
    parser.add_argument("--fetch", action="store_true", help="download the OWID / World Bank sources first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.fetch:
        fetch(args.raw)
    build_vdem(args.raw)
    build_women(args.raw)
    build_cow(args.raw)
    build_colonial(args.raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
