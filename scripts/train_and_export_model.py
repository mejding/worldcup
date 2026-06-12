#!/usr/bin/env python3
"""Developer-only model export pipeline.

This script trains the bundled pre-trained model artifacts from a historical CSV.
It is not required at runtime by the Streamlit app.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest import run_walk_forward_backtest
from config import HISTORICAL_RESULTS_PATH
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results
from train_model import train_from_historical_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export pre-trained World Cup prediction model artifacts.")
    parser.add_argument("--input", default=str(HISTORICAL_RESULTS_PATH), help="Historical international results CSV.")
    parser.add_argument("--test-start-date", default=None, help="Optional train/test cutoff date, e.g. 2022-01-01.")
    parser.add_argument("--include-draw-context-features", action="store_true")
    parser.add_argument("--run-backtest", action="store_true", help="Also run the walk-forward backtest outputs.")
    parser.add_argument("--initial-train-end-date", default="2014-01-01")
    parser.add_argument("--test-window", default="365D")
    parser.add_argument("--step-size", default="365D")
    parser.add_argument("--min-train-matches", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Historical training data is missing: {input_path}. "
            "Add the CSV for development training, then rerun this export script."
        )

    metadata = train_from_historical_csv(
        input_path,
        test_start_date=args.test_start_date,
        include_draw_context_features=args.include_draw_context_features,
    )

    result = {"model_metadata": metadata}
    if args.run_backtest:
        raw = load_historical_results(input_path)
        warnings, errors = validate_historical_results(raw)
        if errors:
            raise ValueError("; ".join(errors))
        standardized = standardize_historical_results(raw)
        backtest = run_walk_forward_backtest(
            standardized,
            initial_train_end_date=args.initial_train_end_date,
            test_window=args.test_window,
            step_size=args.step_size,
            min_train_matches=args.min_train_matches,
            include_draw_context_features=args.include_draw_context_features,
        )
        result["backtest_predictions"] = len(backtest["predictions"])
        result["backtest_warnings"] = warnings

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
