import pandas as pd
import pytest

from bankroll import get_current_bankroll, reset_bankroll
from bet_log import (
    BET_LOG_COLUMNS,
    add_bet,
    calculate_bet_summary,
    ensure_bet_log_exists,
    load_bet_log,
    reset_bet_settlement,
    settle_bet,
)


def _paths(tmp_path):
    return {
        "bet_log": str(tmp_path / "bet_log.csv"),
        "state": str(tmp_path / "bankroll_state.json"),
        "history": str(tmp_path / "bankroll_history.csv"),
    }


def _add_sample_bet(path, odds=2.0, stake=25.0):
    return add_bet(
        match_id="M001",
        match="Mexico vs South Africa",
        bookmaker="Danske Spil",
        outcome="Home",
        odds=odds,
        model_probability=0.55,
        edge=0.10,
        full_kelly=0.10,
        fractional_kelly=0.025,
        stake_dkk=stake,
        path=path,
    )


def test_bet_log_file_is_created_with_correct_headers(tmp_path):
    bet_log_path = tmp_path / "bet_log.csv"

    ensure_bet_log_exists(str(bet_log_path))

    df = pd.read_csv(bet_log_path)
    assert list(df.columns) == BET_LOG_COLUMNS


def test_adding_a_bet_creates_pending_unsettled_bet(tmp_path):
    paths = _paths(tmp_path)

    bet = _add_sample_bet(paths["bet_log"])
    df = load_bet_log(paths["bet_log"])

    assert bet["result"] == "pending"
    assert bet["settled"] is False
    assert len(df) == 1
    assert df.iloc[0]["result"] == "pending"
    assert bool(df.iloc[0]["settled"]) is False


def test_settling_won_bet_updates_bet_and_bankroll(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)

    settled = settle_bet(
        bet["bet_id"],
        "won",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    assert settled["result"] == "won"
    assert settled["profit_loss_dkk"] == pytest.approx(25.0)
    assert settled["settled"] is True
    assert get_current_bankroll(paths["state"]) == pytest.approx(1025.0)


def test_settling_lost_bet_updates_bet_and_bankroll(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)

    settled = settle_bet(
        bet["bet_id"],
        "lost",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    assert settled["result"] == "lost"
    assert settled["profit_loss_dkk"] == pytest.approx(-25.0)
    assert settled["settled"] is True
    assert get_current_bankroll(paths["state"]) == pytest.approx(975.0)


def test_settling_void_bet_updates_bet_and_leaves_bankroll_unchanged(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)

    settled = settle_bet(
        bet["bet_id"],
        "void",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    assert settled["result"] == "void"
    assert settled["profit_loss_dkk"] == pytest.approx(0.0)
    assert settled["settled"] is True
    assert get_current_bankroll(paths["state"]) == pytest.approx(1000.0)


def test_double_settlement_is_prevented(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)
    settle_bet(
        bet["bet_id"],
        "won",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    with pytest.raises(ValueError, match="already been settled"):
        settle_bet(
            bet["bet_id"],
            "lost",
            bet_log_path=paths["bet_log"],
            bankroll_state_path=paths["state"],
            bankroll_history_path=paths["history"],
        )

    assert get_current_bankroll(paths["state"]) == pytest.approx(1025.0)


def test_reset_bet_settlement_does_not_reverse_bankroll(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)
    settle_bet(
        bet["bet_id"],
        "won",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    reset_bet = reset_bet_settlement(bet["bet_id"], path=paths["bet_log"])

    assert reset_bet["result"] == "pending"
    assert reset_bet["profit_loss_dkk"] == pytest.approx(0.0)
    assert reset_bet["settled"] is False
    assert "Bankroll was not reversed" in reset_bet["warning"]
    assert get_current_bankroll(paths["state"]) == pytest.approx(1025.0)


def test_bet_summary_calculations_work(tmp_path):
    paths = _paths(tmp_path)
    reset_bankroll(1000.0, state_path=paths["state"], history_path=paths["history"])
    won_bet = _add_sample_bet(paths["bet_log"], odds=2.0, stake=25.0)
    lost_bet = _add_sample_bet(paths["bet_log"], odds=3.0, stake=10.0)
    _add_sample_bet(paths["bet_log"], odds=4.0, stake=5.0)

    settle_bet(
        won_bet["bet_id"],
        "won",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )
    settle_bet(
        lost_bet["bet_id"],
        "lost",
        bet_log_path=paths["bet_log"],
        bankroll_state_path=paths["state"],
        bankroll_history_path=paths["history"],
    )

    summary = calculate_bet_summary(paths["bet_log"])

    assert summary["total_bets"] == 3
    assert summary["pending_bets"] == 1
    assert summary["settled_bets"] == 2
    assert summary["won_bets"] == 1
    assert summary["lost_bets"] == 1
    assert summary["void_bets"] == 0
    assert summary["total_staked"] == pytest.approx(40.0)
    assert summary["total_profit_loss"] == pytest.approx(15.0)
    assert summary["roi"] == pytest.approx(15.0 / 35.0)
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["average_odds"] == pytest.approx(3.0)
    assert summary["average_edge"] == pytest.approx(0.10)

