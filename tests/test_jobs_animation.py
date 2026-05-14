import pytest
from unittest.mock import patch, MagicMock
from app.services import compute_network_animation_payload
import numpy as np


def test_compute_network_animation_payload_success():
    # We mock plot_network_animation to avoid needing a real Plotly Figure and slow layout
    with patch("src.network_viz.plot_network_animation") as mock_plot:
        # Create a dummy figure that returns fake JSON
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_plot.return_value = mock_fig

        # Create vote items for years 2010 to 2015
        import pandas as pd

        rows = []
        for year in [2010, 2011, 2012, 2013, 2014, 2015]:
            rows.extend(
                [
                    {
                        "country_identifier": "US",
                        "year": year,
                        "rcid": f"r{year}",
                        "vote": 1,
                        "issue": "A",
                    },
                    {
                        "country_identifier": "UK",
                        "year": year,
                        "rcid": f"r{year}",
                        "vote": 1,
                        "issue": "A",
                    },
                ]
            )

        mock_df = pd.DataFrame(rows)
        with patch("app.services.df_global", mock_df):
            payload = compute_network_animation_payload(
                start_year=2010, end_year=2015, window="2Y"
            )

            assert "meta" in payload
            assert payload["meta"]["selected_window"]["start_year"] == 2010
            assert payload["meta"]["selected_window"]["end_year"] == 2015
            assert payload["meta"]["context"]["window"] == "2Y"

            # Since step=2 and years 2010..2015, loop is 2010, 2012, 2014
            # Networks should be built for 2010-2011, 2012-2013, 2014-2015 -> 3 frames
            assert payload["meta"]["context"]["frames"] == 3
            assert mock_plot.called
