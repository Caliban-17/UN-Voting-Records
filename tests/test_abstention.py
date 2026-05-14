import pytest
import pandas as pd
from src.abstention_analysis import calculate_abstention_rates


def test_calculate_abstention_rates():
    # Sample data
    df = pd.DataFrame(
        {
            "country_identifier": ["US", "US", "US", "UK", "UK"],
            "year": [2020, 2020, 2021, 2020, 2020],
            "vote": [1, 0, 0, 1, 1],
            "issue": [
                "Human Rights",
                "Disarmament",
                "Human Rights",
                "Human Rights",
                "Disarmament",
            ],
            "primary_topic": [
                "HUMAN RIGHTS",
                "DISARMAMENT",
                "HUMAN RIGHTS",
                "HUMAN RIGHTS",
                "DISARMAMENT",
            ],
        }
    )

    # Run with min_votes = 1 to capture this small mock dataset
    result = calculate_abstention_rates(df, 2020, 2021, min_votes=1)

    assert "error" not in result
    assert "global_rates" in result
    assert "top_strategic_topics" in result
    assert "country_topic_matrix" in result

    global_rates = result["global_rates"]
    # US voted 3 times, abstained 2 times -> rate 0.66
    assert "US" in global_rates
    assert round(global_rates["US"]["abstention_rate"], 2) == 0.67

    # UK voted 2 times, abstained 0 times -> rate 0.0
    assert "UK" in global_rates
    assert global_rates["UK"]["abstention_rate"] == 0.0

    # Check top strategic topics
    topics = result["top_strategic_topics"]
    assert "HUMAN RIGHTS" in topics
    assert "DISARMAMENT" in topics

    # Check country topic matrix
    matrix = result["country_topic_matrix"]
    assert "US" in matrix
    assert matrix["US"].get("HUMAN RIGHTS", 0) > 0  # 1 abstain out of 2 = 0.5
    assert matrix["US"].get("DISARMAMENT", 0) == 1.0  # 1 abstain out of 1 = 1.0


def test_calculate_abstention_rates_empty():
    df = pd.DataFrame(columns=["country_identifier", "year", "vote", "primary_topic"])

    result = calculate_abstention_rates(df, 2020, 2021)
    assert "error" in result
    assert result["error"] == "No data for specified period"
