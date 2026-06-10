import pandas as pd

from bankroll import (
    add_bankroll_history_entry,
    load_bankroll_history,
    load_bankroll_state,
    reset_bankroll,
    save_bankroll_state,
    update_bankroll,
)


def test_loading_default_bankroll_state(tmp_path):
    state_path = tmp_path / "bankroll_state.json"

    state = load_bankroll_state(str(state_path))

    assert state == {"starting_bankroll": 1000.0, "current_bankroll": 1000.0}


def test_saving_bankroll_state(tmp_path):
    state_path = tmp_path / "bankroll_state.json"

    save_bankroll_state(
        {"starting_bankroll": 1500.0, "current_bankroll": 1250.0},
        str(state_path),
    )

    assert load_bankroll_state(str(state_path)) == {
        "starting_bankroll": 1500.0,
        "current_bankroll": 1250.0,
    }


def test_updating_bankroll_with_deposit(tmp_path):
    state_path = tmp_path / "bankroll_state.json"
    history_path = tmp_path / "bankroll_history.csv"

    state = update_bankroll(250.0, "deposit", state_path=str(state_path), history_path=str(history_path))

    assert state["starting_bankroll"] == 1000.0
    assert state["current_bankroll"] == 1250.0


def test_updating_bankroll_with_withdrawal(tmp_path):
    state_path = tmp_path / "bankroll_state.json"
    history_path = tmp_path / "bankroll_history.csv"

    state = update_bankroll(
        -150.0,
        "withdrawal",
        state_path=str(state_path),
        history_path=str(history_path),
    )

    assert state["current_bankroll"] == 850.0


def test_updating_bankroll_with_manual_correction(tmp_path):
    state_path = tmp_path / "bankroll_state.json"
    history_path = tmp_path / "bankroll_history.csv"

    state = update_bankroll(
        -25.5,
        "manual correction",
        note="Correction",
        state_path=str(state_path),
        history_path=str(history_path),
    )

    assert state["current_bankroll"] == 974.5


def test_resetting_bankroll(tmp_path):
    state_path = tmp_path / "bankroll_state.json"
    history_path = tmp_path / "bankroll_history.csv"

    update_bankroll(500.0, "deposit", state_path=str(state_path), history_path=str(history_path))
    state = reset_bankroll(2000.0, state_path=str(state_path), history_path=str(history_path))

    assert state == {"starting_bankroll": 2000.0, "current_bankroll": 2000.0}
    history = load_bankroll_history(str(history_path))
    assert history.iloc[-1]["transaction_type"] == "reset"


def test_bankroll_history_entry_is_created(tmp_path):
    history_path = tmp_path / "bankroll_history.csv"

    add_bankroll_history_entry(
        transaction_type="deposit",
        amount=100.0,
        bankroll_before=1000.0,
        bankroll_after=1100.0,
        note="Initial top-up",
        path=str(history_path),
    )

    history = pd.read_csv(history_path)
    assert list(history.columns) == [
        "timestamp",
        "transaction_type",
        "amount",
        "bankroll_before",
        "bankroll_after",
        "note",
    ]
    assert len(history) == 1
    assert history.iloc[0]["bankroll_after"] == 1100.0

