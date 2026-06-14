"""Import and clean historical international football results."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from config import HISTORICAL_RESULTS_PATH
from historical_data import clean_historical_results_for_training, validate_historical_results


SOURCE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def import_historical_results(source_url: str = SOURCE_URL, output_path: Path = HISTORICAL_RESULTS_PATH) -> dict:
    raw = pd.read_csv(source_url)
    cleaned = clean_historical_results_for_training(raw)
    warnings, errors = validate_historical_results(cleaned)
    if errors:
        raise ValueError("; ".join(errors))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return {
        "source_url": source_url,
        "output_path": str(output_path),
        "rows": int(len(cleaned)),
        "date_min": str(cleaned["date"].min()),
        "date_max": str(cleaned["date"].max()),
        "world_cup_rows": int(cleaned["tournament"].str.contains("World Cup", case=False, na=False).sum()),
        "qualifier_rows": int(cleaned["tournament"].str.contains("qual", case=False, na=False).sum()),
        "warnings": warnings,
    }


if __name__ == "__main__":
    # Touch the URL early so network failures are reported clearly before pandas parsing.
    with urlopen(SOURCE_URL, timeout=30):
        pass
    summary = import_historical_results()
    for key, value in summary.items():
        print(f"{key}: {value}")
