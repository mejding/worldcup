import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd
from pandas.errors import EmptyDataError

from config import BANKROLL_HISTORY_PATH, BANKROLL_STATE_PATH


DEFAULT_BANKROLL_STATE = {
    "starting_bankroll": 1000.0,
    "current_bankroll": 1000.0,
}

BANKROLL_HISTORY_COLUMNS = [
    "timestamp",
    "transaction_type",
    "amount",
    "bankroll_before",
    "bankroll_after",
    "note",
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_bankroll_state(path: Union[str, Path] = BANKROLL_STATE_PATH) -> dict:
    state_path = Path(path)
    if not state_path.exists():
        save_bankroll_state(DEFAULT_BANKROLL_STATE.copy(), path)

    try:
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (json.JSONDecodeError, KeyError):
        state = DEFAULT_BANKROLL_STATE.copy()
        save_bankroll_state(state, path)

    starting_bankroll = float(state.get("starting_bankroll", DEFAULT_BANKROLL_STATE["starting_bankroll"]))
    current_bankroll = float(state.get("current_bankroll", DEFAULT_BANKROLL_STATE["current_bankroll"]))
    if starting_bankroll <= 0:
        raise ValueError("Starting bankroll must be greater than 0.")
    if current_bankroll < 0:
        raise ValueError("Current bankroll cannot be negative.")
    return {"starting_bankroll": starting_bankroll, "current_bankroll": current_bankroll}


def save_bankroll_state(state: dict, path: Union[str, Path] = BANKROLL_STATE_PATH) -> None:
    _ensure_parent_dir(path)
    starting_bankroll = float(state["starting_bankroll"])
    current_bankroll = float(state["current_bankroll"])
    if starting_bankroll <= 0:
        raise ValueError("Starting bankroll must be greater than 0.")
    if current_bankroll < 0:
        raise ValueError("Current bankroll cannot be negative.")
    state_to_save = {
        "starting_bankroll": starting_bankroll,
        "current_bankroll": current_bankroll,
    }
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(state_to_save, file, indent=2)


def get_current_bankroll(path: Union[str, Path] = BANKROLL_STATE_PATH) -> float:
    return load_bankroll_state(path)["current_bankroll"]


def set_current_bankroll(
    current_bankroll: float,
    path: Union[str, Path] = BANKROLL_STATE_PATH,
) -> dict:
    state = load_bankroll_state(path)
    state["current_bankroll"] = float(current_bankroll)
    save_bankroll_state(state, path)
    return state


def ensure_bankroll_history_exists(path: Union[str, Path] = BANKROLL_HISTORY_PATH) -> None:
    history_path = Path(path)
    if history_path.exists():
        return

    _ensure_parent_dir(path)
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BANKROLL_HISTORY_COLUMNS)
        writer.writeheader()


def add_bankroll_history_entry(
    transaction_type: str,
    amount: float,
    bankroll_before: float,
    bankroll_after: float,
    note: str = "",
    path: Union[str, Path] = BANKROLL_HISTORY_PATH,
) -> None:
    ensure_bankroll_history_exists(path)
    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BANKROLL_HISTORY_COLUMNS)
        writer.writerow(
            {
                "timestamp": _timestamp(),
                "transaction_type": transaction_type,
                "amount": float(amount),
                "bankroll_before": float(bankroll_before),
                "bankroll_after": float(bankroll_after),
                "note": note,
            }
        )


def load_bankroll_history(path: Union[str, Path] = BANKROLL_HISTORY_PATH) -> pd.DataFrame:
    ensure_bankroll_history_exists(path)
    path = Path(path)
    if path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=BANKROLL_HISTORY_COLUMNS)
            writer.writeheader()
        return pd.DataFrame(columns=BANKROLL_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        ensure_bankroll_history_exists(path)
        return pd.DataFrame(columns=BANKROLL_HISTORY_COLUMNS)
    missing_columns = [column for column in BANKROLL_HISTORY_COLUMNS if column not in df.columns]
    if missing_columns:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=BANKROLL_HISTORY_COLUMNS)
            writer.writeheader()
        return pd.DataFrame(columns=BANKROLL_HISTORY_COLUMNS)
    return df


def update_bankroll(
    amount: float,
    transaction_type: str,
    note: str = "",
    state_path: Union[str, Path] = BANKROLL_STATE_PATH,
    history_path: Union[str, Path] = BANKROLL_HISTORY_PATH,
    allow_negative_bankroll: bool = False,
) -> dict:
    state = load_bankroll_state(state_path)
    bankroll_before = float(state["current_bankroll"])
    bankroll_after = bankroll_before + float(amount)
    if bankroll_after < 0 and not allow_negative_bankroll:
        raise ValueError("Bankroll update would make current bankroll negative.")

    state["current_bankroll"] = bankroll_after
    save_bankroll_state(state, state_path)
    add_bankroll_history_entry(
        transaction_type=transaction_type,
        amount=float(amount),
        bankroll_before=bankroll_before,
        bankroll_after=bankroll_after,
        note=note,
        path=history_path,
    )
    return state


def reset_bankroll(
    starting_bankroll: float,
    state_path: Union[str, Path] = BANKROLL_STATE_PATH,
    history_path: Union[str, Path] = BANKROLL_HISTORY_PATH,
) -> dict:
    if float(starting_bankroll) <= 0:
        raise ValueError("Starting bankroll must be greater than 0.")
    state = {
        "starting_bankroll": float(starting_bankroll),
        "current_bankroll": float(starting_bankroll),
    }
    save_bankroll_state(state, state_path)
    add_bankroll_history_entry(
        transaction_type="reset",
        amount=0.0,
        bankroll_before=float(starting_bankroll),
        bankroll_after=float(starting_bankroll),
        note="Bankroll reset",
        path=history_path,
    )
    return state
