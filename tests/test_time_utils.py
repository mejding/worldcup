from datetime import datetime

import pandas as pd

from time_utils import add_danish_kickoff_column, convert_to_danish_time, format_danish_kickoff


def test_convert_to_danish_time_uses_copenhagen_timezone():
    converted = convert_to_danish_time("2026-06-14T19:00:00Z")

    assert converted is not None
    assert converted.hour == 21
    assert converted.tzinfo is not None
    assert converted.tzname() == "CEST"


def test_format_danish_kickoff_uses_danish_month_and_time_label():
    assert format_danish_kickoff("2026-12-01T18:00:00Z") == "1. december 2026, 19:00 dansk tid"


def test_add_danish_kickoff_column_preserves_raw_kickoff_time():
    df = pd.DataFrame({"kickoff_time": ["2026-06-14T19:00:00Z"]})

    result = add_danish_kickoff_column(df)

    assert result.loc[0, "kickoff_time"] == "2026-06-14T19:00:00Z"
    assert result.loc[0, "kickoff_time_dk"] == "14. juni 2026, 21:00 dansk tid"
