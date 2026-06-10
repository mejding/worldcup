# FIFA World Cup 2026 Prediction and Kelly Staking App

Streamlit MVP for a World Cup 2026 prediction and Kelly staking workflow.

This MVP intentionally excludes:

- Live API integrations
- Historical machine-learning model
- Live odds ingestion

## Modules

- `config.py`: default staking profile
- `data_loader.py`: sample prediction CSV loader
- `odds_utils.py`: odds, implied probability, overround, and edge utilities
- `kelly.py`: full Kelly, fractional Kelly, cap, and stake calculations
- `recommendations.py`: Danske Spil and best-market recommendation selection
- `bankroll.py`: bankroll state, updates, reset, and bankroll history
- `bet_log.py`: bet logging, settlement, double-settlement protection, and summaries
- `app.py`: Streamlit dashboard

## Bankroll and Bet Log

The current bankroll is used for all future stake calculations. The starting bankroll is kept for return tracking and is not changed by deposits, withdrawals, corrections, or bet settlement unless the bankroll is explicitly reset.

Won bets add profit only, not the full return. Lost bets subtract the stake. Void bets do not change the bankroll. A settled bet cannot be settled again, which prevents accidental double bankroll updates.

Example:

- Starting bankroll: 1000 DKK
- Stake: 25 DKK
- Odds: 2.00
- Won bet profit: 25 DKK
- New bankroll: 1025 DKK

## Pages

- Overview: upcoming matches, model probabilities, odds, edge, recommendations and quick bet logging
- Match Detail: probability comparison, odds comparison, Kelly tables, draw-context and add-bet actions
- Bankroll: current bankroll, history, manual updates and reset
- Bet Log: bet table, manual bet entry, settlement and settlement reset
- Analytics: bet-log based performance summaries and simple breakdowns
- Settings: Kelly profiles, manual staking overrides, preferred bookmaker and data mode
- About: methodology, MVP limitations and storage notes

## Sample Data Mode

The MVP uses `data/sample_predictions.csv` only. Model probabilities, Danske Spil odds and best market odds are sample inputs in this version. Recommendations compare Danske Spil against best market odds while keeping the two recommendation tracks separate.

Draw-context is an explanatory indicator only in the MVP. It is not a manual draw bonus.

Bankroll and bet log state are stored locally as CSV/JSON files. On Streamlit Community Cloud this is suitable for testing only. For production, use persistent storage such as Supabase, Postgres, SQLite with mounted storage, or another database.

This tool is for analysis and tracking. It does not guarantee profit.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repo.
4. Select the repository, branch and main file path: `app.py`.
5. Deploy.
6. For this MVP, no secrets are required.
7. For live odds later, add API keys via Streamlit secrets.

The MVP uses sample data only and does not require API keys.

## Run Tests

```bash
.venv/bin/python -m pytest
```
