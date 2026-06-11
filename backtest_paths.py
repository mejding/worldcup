from pathlib import Path

try:
    from config import DATA_DIR, PROCESSED_DATA_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = DATA_DIR / "reports"

BACKTEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "backtest_predictions.csv"
BACKTEST_SUMMARY_PATH = PROCESSED_DATA_DIR / "backtest_summary.csv"
BACKTEST_BY_SEGMENT_PATH = PROCESSED_DATA_DIR / "backtest_by_segment.csv"
BACKTEST_DRAW_CALIBRATION_PATH = PROCESSED_DATA_DIR / "backtest_draw_calibration.csv"
BACKTEST_CALIBRATION_BINS_PATH = PROCESSED_DATA_DIR / "backtest_calibration_bins.csv"
WORLD_CUP_BACKTEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "world_cup_backtest_predictions.csv"
WORLD_CUP_BACKTEST_SUMMARY_PATH = PROCESSED_DATA_DIR / "world_cup_backtest_summary.csv"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_report.md"
DRAW_HYPOTHESIS_SUMMARY_PATH = PROCESSED_DATA_DIR / "draw_hypothesis_summary.csv"
DRAW_HYPOTHESIS_BY_SEGMENT_PATH = PROCESSED_DATA_DIR / "draw_hypothesis_by_segment.csv"
DRAW_FEATURE_COMPARISON_PATH = PROCESSED_DATA_DIR / "draw_feature_comparison.csv"
BACKTEST_PREDICTIONS_WITH_DRAW_FEATURES_PATH = PROCESSED_DATA_DIR / "backtest_predictions_with_draw_features.csv"
BACKTEST_SUMMARY_WITH_DRAW_FEATURES_PATH = PROCESSED_DATA_DIR / "backtest_summary_with_draw_features.csv"
GROUP_STATE_FEATURES_PATH = PROCESSED_DATA_DIR / "group_state_features.csv"
DRAW_HYPOTHESIS_REPORT_PATH = REPORTS_DIR / "draw_hypothesis_report.md"
