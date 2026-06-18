from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HISTORICAL_RESULTS_PATH, PROCESSED_DATA_DIR
from historical_data import load_historical_results, standardize_historical_results, validate_historical_results
from walk_forward_backtest import run_full_walk_forward_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full walk-forward model validation.")
    parser.add_argument("--historical-path", type=Path, default=HISTORICAL_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--min-train-matches", type=int, default=1000)
    parser.add_argument("--test-window-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=12)
    args = parser.parse_args()

    raw = load_historical_results(args.historical_path)
    warnings, errors = validate_historical_results(raw)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    historical = standardize_historical_results(raw)
    result = run_full_walk_forward_backtest(
        historical,
        initial_train_end_date=None,
        test_window_months=args.test_window_months,
        step_months=args.step_months,
        min_train_matches=args.min_train_matches,
        output_dir=args.output_dir,
    )
    if result.get("status") == "validation_error":
        print(f"ERROR: {result.get('error')}")
        return 1
    summary = result.get("summary")
    if summary is not None and not summary.empty:
        row = summary.iloc[0]
        print(
            "Full validation complete: "
            f"matches={int(row.get('match_count', 0))}, "
            f"accuracy={row.get('accuracy')}, "
            f"log_loss={row.get('log_loss')}, "
            f"brier={row.get('brier_score')}, "
            f"ece={row.get('ece')}"
        )
    comparison = result.get("market_comparison")
    if comparison is None or comparison.empty:
        print("Market comparison cannot be calculated because historical market odds are not available.")
    else:
        print(f"Market comparison rows: {len(comparison)}")
    print(f"Report: {result['paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
