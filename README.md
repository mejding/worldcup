# FIFA World Cup 2026 Prediction and Kelly Staking App

Streamlit MVP for a World Cup 2026 prediction and Kelly staking workflow.

## Project Overview

- Core odds and edge calculations
- Kelly staking with configurable profiles
- Danske Spil vs best market recommendation comparison
- Optional live odds ingestion via The Odds API or a compatible h2h odds provider
- Local bankroll tracking
- Bet log, settlement and double-settlement protection
- Streamlit dashboard with sample World Cup 2026 group-stage data
- Historical model training and time-based backtesting

## MVP Limitations

- Sample mode uses `data/sample_predictions.csv`; live mode can generate `data/processed/live_predictions.csv` from odds API data.
- In sample mode, model probabilities, Danske Spil odds and best market odds are sample inputs.
- In live mode, model probabilities can use market-implied probabilities or saved historical model predictions.
- Draw-context is explanatory only in this MVP.
- Backtesting evaluates model probability quality only. It does not evaluate historical betting P/L because historical odds are not included yet.
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
- Backtest & Metrics: walk-forward model evaluation, calibration tables, segments and report
- Settings: Kelly profiles, manual staking overrides, preferred bookmaker and data mode
- About: methodology, limitations, storage warning and health check

## Sprint 5B UI Polish

The dashboard layout has been polished with KPI cards, cleaner recommendation cards, a more compact match overview, better bankroll and bet log sections, and a clearer draw-context visual indicator. Core model, odds, Kelly and recommendation logic are unchanged.

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

Live odds mode also writes runtime files that are ignored by git:

- `data/raw/odds_snapshots.csv`
- `data/raw/fixtures_snapshots.csv`
- `data/processed/live_predictions.csv`

Raw odds snapshots are appended. Processed live predictions are regenerated into the same schema as `data/sample_predictions.csv`.

## Streamlit Secrets

No secrets are required for sample data mode.

Live odds mode requires `ODDS_API_KEY`.

Local environment variable:

```bash
export ODDS_API_KEY="your_key"
```

Or create a local `.streamlit/secrets.toml` file:

```toml
ODDS_API_KEY = "your_key"
```

For Streamlit Community Cloud, add this in app secrets:

```toml
ODDS_API_KEY = "your_key"
```

Do not commit real secrets. `.streamlit/secrets.toml.example` is committed only as documentation.

## Live Odds Mode

Sample data mode works without API keys. Live odds mode requires an API key from The Odds API or a compatible provider that supports football 1X2/h2h odds.

Live mode currently does this:

- Fetches h2h odds for `soccer_fifa_world_cup`
- Normalizes bookmaker/outcome rows
- Detects draw outcomes such as `Draw`, `Tie`, `X` or `Uafgjort`
- Identifies Danske Spil odds when available
- Identifies best home/draw/away odds across bookmakers
- Calculates consensus fair market probabilities by removing overround per bookmaker, averaging fair probabilities and normalizing again
- Builds `data/processed/live_predictions.csv`

Because the historical ML model has not been added yet, live mode sets:

- `model_home_prob = market_home_prob`
- `model_draw_prob = market_draw_prob`
- `model_away_prob = market_away_prob`

Odds APIs may not include Danske Spil or all World Cup fixtures at all times. The app will still work with best market odds where possible, and it falls back to sample data if live data is unavailable or invalid.

## Historical Model

Sprint 6 adds the first historical international football model. It is intentionally simple: a multinomial logistic regression predicting home win, draw and away win.

Place historical data here:

```text
data/historical/international_results.csv
```

Required columns:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`

Strongly preferred columns:

- `tournament`
- `neutral`

The model uses broader international football data rather than World Cup-only data, because World Cup-only results are too sparse. Supported match types can include World Cup, Euros, Copa América, AFCON, Asian Cup, Gold Cup, qualifiers, Nations League and friendlies.

Current features include tournament flags, team historical win/draw/loss rates, recent form, goals for/against, relative differences and simple Elo features. Features are generated using only matches before the match being predicted.

Train from the Streamlit Settings page with `Train/update historical model`, then apply it to current matches with `Apply model to current matches`.

Command-line training is also available:

```bash
python train_model.py --input data/historical/international_results.csv
```

Metrics shown:

- Accuracy
- Log loss
- Multiclass Brier score
- Actual draw rate
- Predicted draw rate

Market probabilities remain an important benchmark and are preserved in the app. Model probabilities replace only `model_home_prob`, `model_draw_prob` and `model_away_prob` when the historical model is selected and applied.

## Backtesting and Metrics

Sprint 7 adds time-based model backtesting. The backtest currently evaluates the historical model only. Market-aware ensemble will be added in a later sprint.

The app uses walk-forward backtesting because football prediction is a chronological problem: a model should only train on matches that happened before the matches it predicts. A random split is not appropriate for reporting because it can mix future information into training, especially through team form, Elo and tournament context features.

Run the backtest from the `Backtest & Metrics` page:

1. Add historical data at `data/historical/international_results.csv`.
2. Choose an initial train end date, test window, step size and minimum training matches.
3. Click `Run walk-forward backtest`.
4. Review the KPI cards, fold chart, segment table, draw calibration, confidence calibration and generated report.

Backtest output files are written as runtime artifacts:

- `data/processed/backtest_predictions.csv`
- `data/processed/backtest_summary.csv`
- `data/processed/backtest_by_segment.csv`
- `data/processed/backtest_draw_calibration.csv`
- `data/processed/backtest_calibration_bins.csv`
- `data/reports/backtest_report.md`

Metrics:

- Accuracy: share of matches where the highest probability outcome matched the result.
- Log loss: probability quality metric; lower is better.
- Brier score: squared probability error; lower is better.
- ECE: expected calibration error; lower is better.
- Draw calibration gap: predicted draw rate minus actual draw rate. A positive value means the model overpredicts draws.

The overall international backtest is the broadest and most stable view of model quality. The major tournament segment isolates World Cup, Euros, Copa América, AFCON, Asian Cup and Gold Cup-style matches inside the same walk-forward run. The World Cup-only sanity check trains before each selected World Cup year and tests only that tournament, but those samples are small and should not be treated as definitive evidence.

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
- If live odds mode says `ODDS_API_KEY` is missing, add it via environment variable or Streamlit secrets.
- If the odds API returns no events, rate limits or omits draw odds, the app keeps sample/current data and shows a warning.
- If a bet cannot be settled, check that it is still pending. Double settlement is intentionally blocked.
- If the app shows sample-data validation errors, fix `data/sample_predictions.csv` and redeploy.
