# FIFA World Cup 2026 Prediction and Kelly Staking App

Streamlit MVP for a World Cup 2026 prediction and Kelly staking workflow.

The app uses sample data only. It is ready for local use and Streamlit Community Cloud testing, but it does not yet include live odds, live APIs, a historical ML model or backtesting.

## Project Overview

- Core odds and edge calculations
- Kelly staking with configurable profiles
- Danske Spil vs best market recommendation comparison
- Local bankroll tracking
- Bet log, settlement and double-settlement protection
- Streamlit dashboard with sample World Cup 2026 group-stage data

## MVP Limitations

- `data/sample_predictions.csv` is the only match source.
- Model probabilities, Danske Spil odds and best market odds are sample inputs.
- Draw-context is explanatory only in this MVP.
- Bankroll and bet log are stored as local CSV/JSON files.
- The tool is for analysis and tracking. It does not guarantee profit.

Bankroll and bet log are stored as local CSV/JSON files in this MVP. On Streamlit Community Cloud this is suitable for testing only. For production, use persistent storage such as Supabase, Postgres, SQLite with mounted storage, or another database.

## Local Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

or, if using the included virtual environment:

```bash
.venv/bin/python -m pytest
```

## Running Streamlit Locally

```bash
streamlit run app.py
```

## Pages

- Overview: upcoming matches, probabilities, odds, edge, recommendations and quick bet logging
- Match Detail: probability comparison, odds comparison, Kelly table, draw-context and add-bet actions
- Bankroll: current bankroll, bankroll history, manual updates and reset
- Bet Log: bet table, manual bet entry, settlement and settlement reset
- Analytics: bet-log summaries and simple breakdowns
- Settings: Kelly profiles, manual staking overrides, preferred bookmaker and data mode
- About: methodology, limitations, storage warning and health check

## GitHub Setup

1. Create a GitHub repository.
2. Make sure the project root contains `app.py`, `requirements.txt` and this README.
3. Commit the code.
4. Push the `main` branch to GitHub.

Do not commit `.env` or `.streamlit/secrets.toml`.

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select:
   - repository
   - branch: `main`
   - main file path: `app.py`
5. Deploy.
6. No secrets are needed for this MVP.
7. For future live odds, add `ODDS_API_KEY` via Streamlit secrets.

The MVP uses sample data only and does not require API keys.

## Runtime Data Files

These local runtime files are created automatically if missing:

- `data/bankroll_state.json`
- `data/bankroll_history.csv`
- `data/bet_log.csv`

They are intentionally ignored by git. Deployment-safe example files are committed:

- `data/bankroll_state.example.json`
- `data/bankroll_history.example.csv`
- `data/bet_log.example.csv`

If an example file is missing, the app creates safe defaults with valid headers and a 1000 DKK starting bankroll.

## Streamlit Secrets

No secrets are required for the MVP.

`.streamlit/secrets.toml.example` documents the future live-odds placeholder:

```toml
ODDS_API_KEY = "your_api_key_here"
```

Do not commit real secrets.

## Kelly Profiles

The default profile is Standard:

- Fractional Kelly: 0.25
- Max stake: 2.5% of current bankroll
- Minimum edge: 2.5%
- Minimum stake: 0.25%

The app also includes Conservative, Offensive and Aggressive profiles. Manual overrides are validated in the Settings page.

## Bankroll and Bet Log

The current bankroll is used for all future stake calculations. The starting bankroll is kept for return tracking and is not changed by deposits, withdrawals, corrections or bet settlement unless the bankroll is explicitly reset.

Won bets add profit only, not the full return. Lost bets subtract the stake. Void bets do not change the bankroll. A settled bet cannot be settled again, which prevents accidental double bankroll updates.

Example:

- Starting bankroll: 1000 DKK
- Stake: 25 DKK
- Odds: 2.00
- Won bet profit: 25 DKK
- New bankroll: 1025 DKK

## Troubleshooting

- If Streamlit Cloud says the app is not connected to GitHub, push the current branch to a GitHub repository and select that repository in Streamlit Cloud.
- If the app cannot find `sample_predictions.csv`, confirm that `data/sample_predictions.csv` is committed.
- If bankroll or bet-log files are missing, restart the app; runtime files are recreated automatically.
- If a bet cannot be settled, check that it is still pending. Double settlement is intentionally blocked.
- If the app shows sample-data validation errors, fix `data/sample_predictions.csv` and redeploy.

