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
