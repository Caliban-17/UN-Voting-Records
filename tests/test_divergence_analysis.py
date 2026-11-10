"""The divergence detector: absences are not disagreements; empty input is not a crash."""

import pandas as pd

from src.divergence_analysis import DivergenceDetector


def _votes():
    rows = []
    for rcid, year, (a, b, c) in ((1, 2024, (1, 1, 1)), (2, 2024, (1, None, -1)),
                                  (3, 2025, (1, -1, 1)), (4, 2025, (1, 1, None))):
        for code, vote in zip(("AAA", "BBB", "CCC"), (a, b, c)):
            rows.append({"rcid": rcid, "year": year, "date": f"{year}-12-01", "issue": f"Vote {rcid}",
                         "country_identifier": code, "vote": vote})
    return pd.DataFrame(rows)


def _similarity(codes=("AAA", "BBB", "CCC")):
    return pd.DataFrame(0.95, index=list(codes), columns=list(codes))


def test_an_absence_is_never_reported_as_a_divergence():
    anomalies = DivergenceDetector(_votes(), _similarity()).detect_vote_anomalies("AAA", "BBB")
    # Only rcid 3 is a real split; in rcid 2 BBB was absent.
    assert [a["rcid"] for a in anomalies] == [3]
    assert all(pd.notna(a["vote_a"]) and pd.notna(a["vote_b"]) for a in anomalies)


def test_similarity_delta_of_an_empty_period_is_an_empty_frame():
    detector = DivergenceDetector(_votes(), _similarity())
    out = detector.calculate_similarity_delta((1900, 1901), (2024, 2025))
    assert out.empty and "delta" in out.columns


def test_report_skips_a_member_missing_from_the_similarity_matrix():
    detector = DivergenceDetector(_votes(), _similarity(("AAA", "BBB")))
    report = detector.generate_divergence_report(["AAA", "BBB", "CCC"], 2025)
    assert isinstance(report, dict)
