"""The coalition builder's predicted tally counts the members of the window's last year."""

import pandas as pd

from src.coalition import build_coalition


def _frame():
    rows = []
    def vote(rcid, year, votes):
        for code, v in votes.items():
            rows.append({"rcid": rcid, "year": year, "date": f"{year}-12-01", "issue": "Nuclear disarmament",
                         "subjects": "NUCLEAR DISARMAMENT", "primary_topic": "NUCLEAR DISARMAMENT",
                         "country_identifier": code, "country_name": code, "vote": v})
    # 1988-89: the USSR, both Germanys and Yugoslavia are members.
    for rcid, year in ((1, 1988), (2, 1989)):
        vote(rcid, year, {"USA": -1, "SUN": 1, "DDR": 1, "YUG": 1, "FRA": 0})
    # 1993-94: Russia and Serbia continue two of them; the GDR is gone.
    for rcid, year in ((3, 1993), (4, 1994)):
        vote(rcid, year, {"USA": -1, "RUS": 1, "SRB": 1, "FRA": 0})
    return pd.DataFrame(rows)


def test_predecessors_fold_into_successors_and_defunct_states_are_left_out():
    out = build_coalition(_frame(), "nuclear", 1988, 1994)
    tally = out["predicted_tally"]
    # Four members in 1994, not seven "states".
    assert tally["yes"] + tally["no"] + tally["abstain"] + tally["no_history"] == 4
    assert tally == {"yes": 2, "no": 1, "abstain": 1, "no_history": 0}
    listed = {e["country"]: e for tier in out["tiers"].values() for e in tier}
    assert set(listed) == {"USA", "RUS", "SRB", "FRA"}
    assert listed["RUS"]["n_votes"] == 4 and listed["SRB"]["n_votes"] == 4   # USSR and Yugoslav votes folded in
    assert [m["country"] for m in out["former_members"]] == ["DDR"]


def test_a_window_that_ends_before_the_succession_keeps_the_predecessor():
    out = build_coalition(_frame(), "nuclear", 1988, 1989)
    listed = {e["country"] for tier in out["tiers"].values() for e in tier}
    assert listed == {"USA", "SUN", "DDR", "YUG", "FRA"}
    assert out["former_members"] == []
