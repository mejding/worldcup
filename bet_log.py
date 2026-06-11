import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd
from pandas.errors import EmptyDataError

from bankroll import update_bankroll
from config import BANKROLL_HISTORY_PATH, BANKROLL_STATE_PATH, BET_LOG_PATH


BET_LOG_COLUMNS = [
    "bet_id",
    "timestamp",
    "match_id",
    "match",
    "kickoff_time_dk",
    "bookmaker",
    "outcome",
    "odds",
    "model_probability",
    "edge",
    "full_kelly",
    "fractional_kelly",
    "stake_dkk",
    "result",
    "profit_loss_dkk",
    "settled",
]

ALLOWED_RESULTS = {"won", "lost", "void"}
ALLOWED_BET_RESULTS = {"pending", "won", "lost", "void"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _normalize_settled(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def ensure_bet_log_exists(path: Union[str, Path] = BET_LOG_PATH) -> None:
    bet_log_path = Path(path)
    if bet_log_path.exists():
        return

    _ensure_parent_dir(path)
    with bet_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BET_LOG_COLUMNS)
        writer.writeheader()


def _empty_bet_log(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(columns=BET_LOG_COLUMNS)
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return df


def load_bet_log(path: Union[str, Path] = BET_LOG_PATH) -> pd.DataFrame:
    ensure_bet_log_exists(path)
    path = Path(path)
    if path.stat().st_size == 0:
        return _empty_bet_log(path)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return _empty_bet_log(path)
    for column in BET_LOG_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column == "kickoff_time_dk" else None
    df = df[BET_LOG_COLUMNS]
    if not df.empty:
        df["settled"] = df["settled"].apply(_normalize_settled)
    return df


def save_bet_log(df: pd.DataFrame, path: Union[str, Path] = BET_LOG_PATH) -> None:
    _ensure_parent_dir(path)
    df.to_csv(path, index=False, columns=BET_LOG_COLUMNS)


def _validate_bet_inputs(odds: float, model_probability: float, edge: float, stake_dkk: float) -> None:
    odds = float(odds)
    model_probability = float(model_probability)
    float(edge)
    stake_dkk = float(stake_dkk)
    if odds <= 1.0:
        raise ValueError("Odds must be greater than 1.0.")
    if not 0 <= model_probability <= 1:
        raise ValueError("Model probability must be between 0 and 1.")
    if stake_dkk <= 0:
        raise ValueError("Stake must be greater than 0.")


def add_bet(
    match_id: str,
    match: str,
    bookmaker: str,
    outcome: str,
    odds: float,
    model_probability: float,
    edge: float,
    full_kelly: float,
    fractional_kelly: float,
    stake_dkk: float,
    kickoff_time_dk: str = "",
    path: Union[str, Path] = BET_LOG_PATH,
) -> dict:
    _validate_bet_inputs(odds, model_probability, edge, stake_dkk)
    ensure_bet_log_exists(path)
    bet = {
        "bet_id": str(uuid.uuid4()),
        "timestamp": _timestamp(),
        "match_id": match_id,
        "match": match,
        "kickoff_time_dk": kickoff_time_dk,
        "bookmaker": bookmaker,
        "outcome": outcome,
        "odds": float(odds),
        "model_probability": float(model_probability),
        "edge": float(edge),
        "full_kelly": float(full_kelly),
        "fractional_kelly": float(fractional_kelly),
        "stake_dkk": float(stake_dkk),
        "result": "pending",
        "profit_loss_dkk": 0.0,
        "settled": False,
    }

    with Path(path).open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BET_LOG_COLUMNS)
        writer.writerow(bet)

    return bet


def _profit_loss_for_result(row, result: str) -> float:
    stake = float(row["stake_dkk"])
    odds = float(row["odds"])
    if result == "won":
        return stake * (odds - 1)
    if result == "lost":
        return -stake
    return 0.0


def settle_bet(
    bet_id: str,
    result: str,
    bet_log_path: Union[str, Path] = BET_LOG_PATH,
    bankroll_state_path: Union[str, Path] = BANKROLL_STATE_PATH,
    bankroll_history_path: Union[str, Path] = BANKROLL_HISTORY_PATH,
) -> dict:
    if result not in ALLOWED_RESULTS:
        raise ValueError("Result must be one of: won, lost, void.")

    df = load_bet_log(bet_log_path)
    matches = df.index[df["bet_id"] == bet_id].tolist()
    if not matches:
        raise ValueError(f"Bet {bet_id} was not found.")

    index = matches[0]
    row = df.loc[index]
    if _normalize_settled(row["settled"]):
        raise ValueError(f"Bet {bet_id} has already been settled.")

    profit_loss_dkk = _profit_loss_for_result(row, result)
    df.loc[index, "result"] = result
    df.loc[index, "profit_loss_dkk"] = profit_loss_dkk
    df.loc[index, "settled"] = True
    save_bet_log(df, bet_log_path)

    note = (
        f"Settled bet {bet_id}: {row['match']}, {row['outcome']}, "
        f"{row['bookmaker']}, result={result}"
    )
    update_bankroll(
        amount=profit_loss_dkk,
        transaction_type=f"bet {result}",
        note=note,
        state_path=bankroll_state_path,
        history_path=bankroll_history_path,
    )

    return df.loc[index].to_dict()


def reset_bet_settlement(
    bet_id: str,
    path: Union[str, Path] = BET_LOG_PATH,
) -> dict:
    df = load_bet_log(path)
    matches = df.index[df["bet_id"] == bet_id].tolist()
    if not matches:
        raise ValueError(f"Bet {bet_id} was not found.")

    index = matches[0]
    df.loc[index, "result"] = "pending"
    df.loc[index, "profit_loss_dkk"] = 0.0
    df.loc[index, "settled"] = False
    save_bet_log(df, path)

    bet = df.loc[index].to_dict()
    bet["warning"] = "Bankroll was not reversed. Correct bankroll manually if needed."
    return bet


def calculate_bet_summary(path: Union[str, Path] = BET_LOG_PATH) -> dict:
    df = load_bet_log(path)
    if df.empty:
        return {
            "total_bets": 0,
            "pending_bets": 0,
            "settled_bets": 0,
            "won_bets": 0,
            "lost_bets": 0,
            "void_bets": 0,
            "total_staked": 0.0,
            "total_profit_loss": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_odds": 0.0,
            "average_edge": 0.0,
        }

    settled_mask = df["settled"].apply(_normalize_settled)
    settled_df = df[settled_mask]
    valid_results = df["result"].isin(ALLOWED_BET_RESULTS)
    df = df[valid_results].copy()
    if df.empty:
        return {
            "total_bets": 0,
            "pending_bets": 0,
            "settled_bets": 0,
            "won_bets": 0,
            "lost_bets": 0,
            "void_bets": 0,
            "total_staked": 0.0,
            "total_profit_loss": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_odds": 0.0,
            "average_edge": 0.0,
        }
    won_bets = int((df["result"] == "won").sum())
    lost_bets = int((df["result"] == "lost").sum())
    void_bets = int((df["result"] == "void").sum())
    pending_bets = int((df["result"] == "pending").sum())
    total_staked = float(df["stake_dkk"].sum())
    settled_staked = float(settled_df["stake_dkk"].sum())
    total_profit_loss = float(settled_df["profit_loss_dkk"].sum())
    win_rate_denominator = won_bets + lost_bets

    return {
        "total_bets": int(len(df)),
        "pending_bets": pending_bets,
        "settled_bets": int(settled_mask.sum()),
        "won_bets": won_bets,
        "lost_bets": lost_bets,
        "void_bets": void_bets,
        "total_staked": total_staked,
        "total_profit_loss": total_profit_loss,
        "roi": total_profit_loss / settled_staked if settled_staked else 0.0,
        "win_rate": won_bets / win_rate_denominator if win_rate_denominator else 0.0,
        "average_odds": float(df["odds"].mean()),
        "average_edge": float(df["edge"].mean()),
    }


def add_bet_from_recommendation(
    match_id: str,
    match: str,
    bookmaker: str,
    outcome: str,
    recommendation: dict,
    model_probability: float,
    market_prefix: str = "best",
    kickoff_time_dk: str = "",
    path: Union[str, Path] = BET_LOG_PATH,
) -> dict:
    return add_bet(
        match_id=match_id,
        match=match,
        bookmaker=bookmaker,
        outcome=outcome,
        odds=recommendation[f"recommended_odds_{market_prefix}"],
        model_probability=model_probability,
        edge=recommendation[f"recommended_edge_{market_prefix}"],
        full_kelly=recommendation[f"recommended_full_kelly_{market_prefix}"],
        fractional_kelly=recommendation[f"recommended_fractional_kelly_{market_prefix}"],
        stake_dkk=recommendation[f"recommended_stake_{market_prefix}"],
        kickoff_time_dk=kickoff_time_dk,
        path=path,
    )
