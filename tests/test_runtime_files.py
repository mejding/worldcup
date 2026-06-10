import json

import pandas as pd

from bankroll import BANKROLL_HISTORY_COLUMNS, load_bankroll_history
from bet_log import BET_LOG_COLUMNS, load_bet_log
from data_loader import ensure_runtime_data_files


def test_ensure_runtime_data_files_creates_missing_files_from_examples(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    state_example = data_dir / "bankroll_state.example.json"
    history_example = data_dir / "bankroll_history.example.csv"
    bet_log_example = data_dir / "bet_log.example.csv"
    state_example.write_text('{"starting_bankroll": 1234.0, "current_bankroll": 1234.0}')
    history_example.write_text("timestamp,transaction_type,amount,bankroll_before,bankroll_after,note\n")
    bet_log_example.write_text(",".join(BET_LOG_COLUMNS) + "\n")
    runtime_pairs = {
        data_dir / "bankroll_state.json": state_example,
        data_dir / "bankroll_history.csv": history_example,
        data_dir / "bet_log.csv": bet_log_example,
    }

    created = ensure_runtime_data_files(data_dir=data_dir, runtime_file_pairs=runtime_pairs)

    assert len(created) == 3
    assert json.loads((data_dir / "bankroll_state.json").read_text())["starting_bankroll"] == 1234.0
    assert list(pd.read_csv(data_dir / "bankroll_history.csv").columns) == BANKROLL_HISTORY_COLUMNS
    assert list(pd.read_csv(data_dir / "bet_log.csv").columns) == BET_LOG_COLUMNS


def test_ensure_runtime_data_files_does_not_overwrite_existing_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runtime_state = data_dir / "bankroll_state.json"
    missing_example = data_dir / "missing.example.json"
    runtime_state.write_text('{"starting_bankroll": 2000.0, "current_bankroll": 1800.0}')

    created = ensure_runtime_data_files(
        data_dir=data_dir,
        runtime_file_pairs={runtime_state: missing_example},
    )

    assert created == []
    assert json.loads(runtime_state.read_text())["current_bankroll"] == 1800.0


def test_ensure_runtime_data_files_creates_safe_defaults_without_examples(tmp_path):
    data_dir = tmp_path / "data"
    runtime_pairs = {
        data_dir / "bankroll_state.json": data_dir / "bankroll_state.example.json",
        data_dir / "bankroll_history.csv": data_dir / "bankroll_history.example.csv",
        data_dir / "bet_log.csv": data_dir / "bet_log.example.csv",
    }

    ensure_runtime_data_files(data_dir=data_dir, runtime_file_pairs=runtime_pairs)

    assert json.loads((data_dir / "bankroll_state.json").read_text()) == {
        "starting_bankroll": 1000.0,
        "current_bankroll": 1000.0,
    }
    assert list(pd.read_csv(data_dir / "bankroll_history.csv").columns) == BANKROLL_HISTORY_COLUMNS
    assert list(pd.read_csv(data_dir / "bet_log.csv").columns) == BET_LOG_COLUMNS


def test_empty_csv_loads_with_correct_headers(tmp_path):
    history_path = tmp_path / "bankroll_history.csv"
    bet_log_path = tmp_path / "bet_log.csv"
    history_path.write_text("")
    bet_log_path.write_text("")

    history = load_bankroll_history(history_path)
    bet_log = load_bet_log(bet_log_path)

    assert history.empty
    assert list(history.columns) == BANKROLL_HISTORY_COLUMNS
    assert bet_log.empty
    assert list(bet_log.columns) == BET_LOG_COLUMNS

