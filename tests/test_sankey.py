import pytest
import pandas as pd
from src.sankey_analysis import build_sankey_timeline


def test_build_sankey_timeline():
    # Construct a sample dataset spanning 3 windows
    # Window 1: 2000-2004
    # Window 2: 2005-2009
    # Window 3: 2010-2014

    rows = []
    countries = ["US", "UK", "FR", "CN", "RU"]

    # We will simulate all countries voting the same way to ensure clustering logic triggers
    # and they end up in 1 or more clusters
    for year in [2000, 2005, 2010]:
        for c in countries:
            rows.append(
                {
                    "country_identifier": c,
                    "year": year,
                    "rcid": f"res_{year}",
                    "vote": 1 if c in ["US", "UK"] else -1,
                }
            )

    df = pd.DataFrame(rows)

    result = build_sankey_timeline(df, 2000, 2014, window=5, num_clusters=2)

    assert "error" not in result
    assert "nodes" in result
    assert "links" in result
    assert len(result["windows"]) == 3
    assert result["windows"] == ["2000-2004", "2005-2009", "2010-2014"]

    # Check that we have nodes
    assert len(result["nodes"]) > 0

    # Check that we have links indicating flow
    assert len(result["links"]["source"]) > 0
    assert len(result["links"]["target"]) > 0
    assert len(result["links"]["value"]) > 0
    assert (
        len(result["links"]["source"])
        == len(result["links"]["target"])
        == len(result["links"]["value"])
    )


def test_build_sankey_timeline_insufficient_data():
    df = pd.DataFrame(columns=["country_identifier", "year", "rcid", "vote"])
    result = build_sankey_timeline(df, 2000, 2010)
    assert result.get("error") == "Not enough valid windows to trace bloc movement"
