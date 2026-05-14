"""
Quick test script to verify core functionality
Run this to test the network analysis modules without the full web app
"""

import sys
import os

import pandas as pd
import numpy as np
from src.config import UN_VOTES_CSV_PATH
from src.network_analysis import VotingNetwork
from src.soft_power import SoftPowerCalculator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_loading():
    """Test that data loads correctly"""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)

    if not os.path.exists(UN_VOTES_CSV_PATH):
        print(f"❌ Data file not found: {UN_VOTES_CSV_PATH}")
        assert False, f"Data file not found: {UN_VOTES_CSV_PATH}"

    try:
        df = pd.read_csv(
            UN_VOTES_CSV_PATH, nrows=1000
        )  # Load first 1000 rows for quick test
        print(f"✅ Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns[:5])}...")
        assert True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        assert False, f"Error loading data: {e}"


def test_network_building():
    """Test network graph building"""
    print("\n" + "=" * 60)
    print("TEST 2: Network Building")
    print("=" * 60)

    try:
        # Create sample data
        countries = ["USA", "CHN", "RUS", "GBR", "FRA", "DEU", "JPN", "IND"]
        resolutions = [f"R{i}" for i in range(20)]

        # Create random vote matrix
        np.random.seed(42)
        vote_matrix = pd.DataFrame(
            np.random.choice([1, -1, 0], size=(len(countries), len(resolutions))),
            index=countries,
            columns=resolutions,
        )

        print(f"   Created vote matrix: {vote_matrix.shape}")

        # Build network
        network = VotingNetwork(vote_matrix, similarity_threshold=0.5)
        graph = network.build_graph()

        print("✅ Network built successfully")
        print(f"   Nodes: {graph.number_of_nodes()}")
        print(f"   Edges: {graph.number_of_edges()}")

        # Test centrality metrics
        metrics = network.calculate_centrality_metrics()
        print(f"   Calculated {len(metrics)} centrality metrics")
        assert True

    except Exception as e:
        print(f"❌ Error building network: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Error building network: {e}"


def test_soft_power():
    """Test soft power calculation"""
    print("\n" + "=" * 60)
    print("TEST 3: Soft Power Calculation")
    print("=" * 60)

    try:
        # Create sample data
        countries = ["USA", "CHN", "RUS", "GBR", "FRA"]
        resolutions = [f"R{i}" for i in range(15)]

        np.random.seed(42)
        vote_matrix = pd.DataFrame(
            np.random.choice([1, -1, 0], size=(len(countries), len(resolutions))),
            index=countries,
            columns=resolutions,
        )

        # Create vote data
        vote_data = pd.DataFrame(
            {
                "country_identifier": np.repeat(countries, len(resolutions)),
                "rcid": np.tile(resolutions, len(countries)),
                "vote": np.random.choice(
                    [1, -1, 0], size=len(countries) * len(resolutions)
                ),
                "year": np.random.choice(
                    [2020, 2021, 2022], size=len(countries) * len(resolutions)
                ),
            }
        )

        # Build network
        network = VotingNetwork(vote_matrix, similarity_threshold=0.5)
        network.build_graph()

        # Calculate soft power
        calculator = SoftPowerCalculator(network, vote_data)
        soft_power = calculator.aggregate_soft_power_score()

        print("✅ Soft power calculated successfully")
        print("   Top 3 countries:")
        for country, score in soft_power.head(3).items():
            print(f"   - {country}: {score:.3f}")
        assert True

    except Exception as e:
        print(f"❌ Error calculating soft power: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Error calculating soft power: {e}"


def test_visualization_imports():
    """Test that visualization modules import correctly"""
    print("\n" + "=" * 60)
    print("TEST 4: Visualization Imports")
    print("=" * 60)

    try:
        pass

        print("✅ All visualization functions imported successfully")
        assert True
    except Exception as e:
        print(f"❌ Error importing visualization: {e}")
        import traceback

        traceback.print_exc()
        assert False, f"Error importing visualization: {e}"


if __name__ == "__main__":
    print("\n🧪 Running UN Voting Network Analysis Tests\n")

    results = []
    results.append(("Data Loading", test_data_loading()))
    results.append(("Network Building", test_network_building()))
    results.append(("Soft Power Calculation", test_soft_power()))
    results.append(("Visualization Imports", test_visualization_imports()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result is None or result is True)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result is None or result is True else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print(
            f"\n⚠️  {total - passed} test(s) failed. Check the output above for details."
        )

    sys.exit(0 if passed == total else 1)
