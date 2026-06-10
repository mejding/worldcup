import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_bankroll_state(path: str = "data/bankroll_state.json") -> dict:
    state_path = Path(path)
    if not state_path.exists():
        save_bankroll_state(DEFAULT_BANKROLL_STATE.copy(), path)

    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)

    return {
        "starting_bankroll": float(state["starting_bankroll"]),
        "current_bankroll": float(state["current_bankroll"]),
    }


def save_bankroll_state(state: dict, path: str = "data/bankroll_state.json") -> None:
    _ensure_parent_dir(path)
    state_to_save = {
        "starting_bankroll": float(state["starting_bankroll"]),
        "current_bankroll": float(state["current_bankroll"]),
    }
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(state_to_save, file, indent=2)


def get_current_bankroll(path: str = "data/bankroll_state.json") -> float:
    return load_bankroll_state(path)["current_bankroll"]


def set_current_bankroll(
    current_bankroll: float,
    path: str = "data/bankroll_state.json",
) -> dict:
    state = load_bankroll_state(path)
    state["current_bankroll"] = float(current_bankroll)
    save_bankroll_state(state, path)
    return state


def ensure_bankroll_history_exists(path: str) -> None:
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
    path: str = "data/bankroll_history.csv",
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


def load_bankroll_history(path: str = "data/bankroll_history.csv") -> pd.DataFrame:
    ensure_bankroll_history_exists(path)
    return pd.read_csv(path)


def update_bankroll(
    amount: float,
    transaction_type: str,
    note: str = "",
    state_path: str = "data/bankroll_state.json",
    history_path: str = "data/bankroll_history.csv",
) -> dict:
    state = load_bankroll_state(state_path)
    bankroll_before = float(state["current_bankroll"])
    bankroll_after = bankroll_before + float(amount)

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
    state_path: str = "data/bankroll_state.json",
    history_path: str = "data/bankroll_history.csv",
) -> dict:
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

