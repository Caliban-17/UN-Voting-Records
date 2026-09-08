"""
UN regional groups by ISO-3 code.

The five regional groups are the Assembly's own political geography: seats on
the Security Council, ECOSOC and the Human Rights Council are allocated by
them, and members overwhelmingly caucus through them. Colouring a chart by
regional group is therefore the most honest "bloc" colouring available —
it is the UN's, not ours.

Historical members (USSR, Czechoslovakia, the two Germanies, the Yemens,
Yugoslavia and its successor state union, Tanganyika, Zanzibar) are assigned
to the group their territory sits in today so long time-series stay
continuous.

Source: UN Department for General Assembly and Conference Management,
"Regional groups of Member States". The United States is not formally a
member of any group but attends WEOG meetings as an observer and is
treated as WEOG here, as it is for electoral purposes.
"""

from __future__ import annotations

AFRICA = "Africa"
ASIA_PACIFIC = "Asia-Pacific"
EASTERN_EUROPE = "Eastern Europe"
LATIN_AMERICA = "Latin America & Caribbean"
WESTERN = "Western Europe & Others"

GROUP_ORDER = [WESTERN, EASTERN_EUROPE, LATIN_AMERICA, AFRICA, ASIA_PACIFIC]

_MEMBERS: dict[str, str] = {
    AFRICA: (
        "DZA AGO BEN BWA BFA BDI CMR CPV CAF TCD COM COG COD CIV DJI EGY GNQ ERI "
        "SWZ ETH GAB GMB GHA GIN GNB KEN LSO LBR LBY MDG MWI MLI MRT MUS MAR MOZ "
        "NAM NER NGA RWA STP SEN SYC SLE SOM ZAF SSD SDN TZA TGO TUN UGA ZMB ZWE "
        # historical: Tanganyika, Zanzibar
        "EAT EAZ"
    ),
    ASIA_PACIFIC: (
        "AFG BHR BGD BTN BRN KHM CHN CYP PRK FJI IND IDN IRN IRQ JPN JOR KAZ KIR "
        "KWT KGZ LAO LBN MYS MDV MHL FSM MNG MMR NRU NPL OMN PAK PLW PNG PHL QAT "
        "KOR WSM SAU SGP SLB LKA SYR TJK THA TLS TON TKM TUV ARE UZB VUT VNM YEM "
        # historical: South Yemen
        "YMD"
    ),
    EASTERN_EUROPE: (
        "ALB ARM AZE BLR BIH BGR HRV CZE EST GEO HUN LVA LTU MDA MNE MKD POL ROU "
        "RUS SRB SVK SVN UKR "
        # historical: USSR, Czechoslovakia, East Germany, Yugoslavia, Serbia and Montenegro
        "SUN CSK DDR YUG SCG"
    ),
    LATIN_AMERICA: (
        "ATG ARG BHS BRB BLZ BOL BRA CHL COL CRI CUB DMA DOM ECU SLV GRD GTM GUY "
        "HTI HND JAM MEX NIC PAN PRY PER KNA LCA VCT SUR TTO URY VEN"
    ),
    WESTERN: (
        "AND AUS AUT BEL CAN DNK FIN FRA DEU GRC ISL IRL ISR ITA LIE LUX MLT MCO "
        "NLD NZL NOR PRT SMR ESP SWE CHE TUR GBR USA "
        # historical: West Germany
        "GER"
    ),
}

REGIONAL_GROUPS: dict[str, str] = {
    code: group for group, codes in _MEMBERS.items() for code in codes.split()
}

# Predecessor codes folded into a present-day anchor so a country's line does
# not break in 1990–1993. Used when an *anchor* is chosen (e.g. "Russia since
# 1946" means the USSR before 1992); the successor's own rows always win.
LINEAGE: dict[str, list[str]] = {
    "RUS": ["SUN"],
    "DEU": ["GER"],
    "CZE": ["CSK"],
    "SRB": ["YUG", "SCG"],
}


def regional_group(code: str | None) -> str:
    """Regional group for an ISO-3 code, or "Other" when unknown."""
    if not code:
        return "Other"
    return REGIONAL_GROUPS.get(str(code).strip().upper(), "Other")


def lineage_codes(code: str) -> list[str]:
    """The code itself plus any predecessor states it succeeded."""
    code = str(code or "").strip().upper()
    return [code] + LINEAGE.get(code, [])
