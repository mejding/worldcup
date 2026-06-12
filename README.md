# FIFA World Cup 2026 Prediction and Kelly Staking App

Streamlit MVP for a World Cup 2026 prediction and Kelly staking workflow.

## Project Overview

- Core odds and edge calculations
- Kelly staking with configurable profiles
- Danske Spil vs best market recommendation comparison
- Optional live odds ingestion via The Odds API or a compatible h2h odds provider
- Local bankroll tracking
- Bet log, settlement and double-settlement protection
- Streamlit dashboard with an official-reference fixture layer, explicit sample/demo mode and optional live odds
- Historical model training and time-based backtesting

## MVP Limitations

- Official mode is the default and reads fixtures from `data/reference/worldcup_2026_fixtures.csv`.
- The bundled reference currently includes all 72 known group-stage fixtures. It is still marked incomplete against the full 104-match tournament because knockout participants are not known until the group stage is played.
- Sample mode uses `data/sample_predictions.csv` only when explicitly selected in `Admin / Settings`; it is demo data and not an official fixture schedule.
- Live mode can generate `data/processed/live_predictions.csv` from the official fixture reference plus odds API data. It does not silently fall back to sample data if live predictions are missing.
- In sample mode, model probabilities, Danske Spil odds and best market odds are sample inputs.
- In live/official mode, model probabilities can use market-implied probabilities or saved historical model predictions when they match the fixture reference.
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

- Match Overview: the default landing page with upcoming matches, favorites, probabilities, betting decision, stake and reason.
- Betting Center: value bets, Danske Spil view, best-market view, bet slip, bankroll and bet history in horizontal tabs.
- My Bets: pending/settled bets, bankroll and betting performance without model metrics.
- Match Detail: simple match decision first, with probability and Kelly details tucked into expanders.
- Model & Data: prediction engine, data readiness, model status, backtest, ensemble, draw-context and fixture data.
- Settings: user-facing data mode, Kelly profile and preferred bookmaker settings.
- Advanced / Admin: odds refresh, model training/apply actions, backtest, draw hypothesis, ensemble actions and fixture validation.

## End-to-End Flow

1. Open `Match Overview` and scan upcoming matches, favorite, probabilities and recommendation status.
2. Select a match to open `Match Detail`.
3. Compare market, model and active probabilities, then inspect Danske Spil and best-market recommendations separately.
4. Add a recommended bet to the `Bet slip` only if the card is not `No bet`.
5. Review total stake and exposure in `Betting Center` -> `Bet slip`, then add selected bets to Bet Log.
6. Settle a pending bet as `won`, `lost` or `void` from `My Bets` or `Betting Center` -> `Bet history`.
7. Bankroll updates exactly once when the bet is settled. Wins add profit only, losses subtract stake, void bets do not change bankroll.
8. Review `My Bets` for bankroll and performance tracking.

## Interpreting Recommendations

- `Playable at Danske Spil`: the active probability source finds value at Danske Spil odds after edge and Kelly thresholds.
- `Better elsewhere`: Danske Spil does not qualify, but the best tracked market odds do.
- `No bet`: no outcome passes the configured edge and Kelly thresholds, odds are missing, or the final stake is below the minimum.

Each recommendation uses `edge = active_probability * odds - 1`. Kelly stake is calculated from the current bankroll, not starting bankroll, because stake sizing should reflect the money currently available.

The normal UI shows one label: `Best available prediction`. Internally this can use a validated ensemble, the bundled pre-trained model or market probabilities as fallback.

## Pre-Trained Prediction Flow

The app is designed so normal users do not need to train or apply models manually.

- Bundled model artifacts live in `data/models/model.pkl`, `data/models/model_metadata.json` and `data/models/feature_columns.json`.
- On startup, the app checks for model artifacts and automatically prepares upcoming-match predictions when possible.
- The primary UI shows the decision, not the modelling machinery: favorite, probabilities, No bet/Playable/Better elsewhere, bookmaker and stake.
- A bundled model is production-ready only when metadata proves it was trained on a substantial historical international dataset: at least 1000 training rows, 200 test rows and 10 features.
- Small generated/sample artifacts are marked as demo models. Demo models can be inspected in Advanced / Admin but are not used as `Best available prediction` in official/live mode.
- If model files are missing or cannot be applied, the app falls back to market probabilities and keeps the normal user flow working.
- `data/historical/international_results.csv` is developer training input only. It is not required at runtime when pre-trained artifacts are bundled.

Manual retraining, backtests, draw-context analysis, ensemble comparison and odds refresh are admin/developer actions under `Advanced / Admin`.

## Fixture Data

Fixture data has a separate source-of-truth file:

```text
data/reference/worldcup_2026_fixtures.csv
```

Required fixture columns include `match_id`, `match_number`, `kickoff_utc`, `kickoff_local`, `kickoff_timezone`, teams, group/stage/matchday, city/stadium and source metadata. The validator checks required columns, duplicate match IDs, parseable kickoff times, completeness against 104 expected matches and known Group B corrections.

The app intentionally keeps knockout matches out of user-facing predictions until the teams are known. This avoids showing placeholder fixtures such as `Winner Group A vs 3rd Group C/E/F/H/I` as if they were real betting matches.

Known correction covered by tests: Canada vs Bosnia and Herzegovina is on 2026-06-12, Group B, Toronto Stadium. Canada vs Switzerland must not appear on 2026-06-12.

## Draw-Context

Draw-context is not draw probability and not a betting recommendation. It is a contextual signal for whether a draw may be strategically plausible. High draw-context should prompt a closer look at draw probability, draw odds and draw edge, but never a draw bet by itself.

## Danish Time

Raw `kickoff_time` values are preserved for data compatibility, but visible kickoff times are shown as `kickoff_time_dk` in `Europe/Copenhagen`, for example `14. juni 2026, 22:00 dansk tid`.

## If the App Feels Incomplete

Check `About` -> `Health check`:

- whether official, sample or live data is active
- whether the fixture source is complete or explicitly marked incomplete
- whether live odds have been fetched
- whether a historical model is trained and applied
- whether ensemble predictions exist
- whether bankroll and bet log runtime files are loaded

## Sprint 5B UI Polish

The dashboard layout has been polished with KPI cards, cleaner recommendation cards, a more compact match overview, better bankroll and bet log sections, and a clearer draw-context visual indicator. Core model, odds, Kelly and recommendation logic are unchanged.

## Sprint 10 Premium Dashboard UI

Sprint 10 adds a compact premium dashboard layer without changing the core model, odds, Kelly, bankroll or bet settlement logic.

- Overview cards now show active probabilities directly under each match.
- Recommendations are displayed as compact betting lines with outcome, odds, bookmaker, edge, Kelly and stake.
- Kickoff times are preserved as raw `kickoff_time` and displayed as derived `kickoff_time_dk` in `Europe/Copenhagen`.
- Tooltips explain edge, Kelly, active probability source, recommendation statuses and draw-context fields.
- Draw-context is shown as an explanatory signal only. It does not trigger bets by itself.
- Streamlit theme settings and custom CSS make the app more readable on laptop-width screens.

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
6. No secrets are needed for official fixture mode or sample/demo mode.
7. For API-based live odds, add `ODDS_API_KEY` via Streamlit secrets. If the API is missing coverage, use `data/reference/manual_odds.csv` and import it from Advanced / Admin.

The default deployed mode uses the fixture reference file and does not require API keys.

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

Raw odds snapshots are appended. Processed live predictions are regenerated into the prediction schema plus fixture provenance columns such as `kickoff_utc` and `fixture_source`.

Manual odds can be entered in:

- `data/reference/manual_odds.csv`

Add one row per match/bookmaker. The app expands `home_odds`, `draw_odds` and `away_odds` into normalized 1X2 odds internally. `data/reference/manual_odds.example.csv` shows the format, but is never loaded automatically.

## Streamlit Secrets

No secrets are required for official fixture mode or sample/demo mode.

API-based live odds mode requires `ODDS_API_KEY`. Manual odds CSV and cached snapshots do not require secrets.

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

## Odds Setup

The app supports three odds sources:

1. The Odds API
2. Manual odds CSV
3. Cached odds snapshots

The app never scrapes bookmaker websites, never commits API keys, and never fabricates Danske Spil odds. Sample/demo odds are only used in sample mode.

### The Odds API

In Streamlit Community Cloud, open app settings, then Secrets, and add:

```toml
ODDS_API_KEY = "your_api_key_here"
```

For local development:

```bash
export ODDS_API_KEY="your_api_key_here"
```

The default API settings are:

- sport key: `soccer_fifa_world_cup`
- regions: `eu`
- markets: `h2h`
- odds format: `decimal`
- date format: `iso`

### Manual Odds CSV

Manual fallback file:

```text
data/reference/manual_odds.csv
```

Required columns:

- `match_id`
- `home_team`
- `away_team`
- `kickoff_utc`
- `bookmaker`
- `home_odds`
- `draw_odds`
- `away_odds`
- `odds_last_updated_utc`

Optional columns:

- `bookmaker_key`
- `is_danske_spil`
- `source`
- `notes`

Manual odds are not live odds. They should be updated by the user/admin and are labeled in the UI as manual CSV odds.

### Cached Odds

Successful API fetches are appended to:

```text
data/raw/odds_snapshots.csv
```

If the API is unavailable later, the app can use the latest cached snapshot. Cached odds are clearly labeled as cached and are not presented as freshly fetched live odds.

## Live Odds Mode

Sample data mode works without API keys. Live odds can come from The Odds API, the manual odds CSV, or the latest cached odds snapshot.

Live mode currently does this:

- Fetches h2h odds for `soccer_fifa_world_cup`
- Imports manual h2h odds from `data/reference/manual_odds.csv`
- Falls back to cached odds snapshots when live API odds are unavailable
- Normalizes bookmaker/outcome rows
- Matches odds to official fixtures by match ID first, then by normalized teams and kickoff tolerance
- Detects draw outcomes such as `Draw`, `Tie`, `X` or `Uafgjort`
- Identifies Danske Spil odds when available
- Identifies best home/draw/away odds across bookmakers
- Calculates consensus fair market probabilities by removing overround per bookmaker, averaging fair probabilities and normalizing again
- Builds `data/processed/latest_odds.csv`
- Builds `data/processed/live_predictions.csv`

When live odds are the only available prediction source, live mode can use market probabilities as the model baseline:

- `model_home_prob = market_home_prob`
- `model_draw_prob = market_draw_prob`
- `model_away_prob = market_away_prob`

Odds APIs may not include Danske Spil or all World Cup fixtures at all times. The app keeps official fixtures visible where possible and shows missing odds as `No bet`; it does not silently replace live/official data with sample fixtures. Add an API key or valid manual odds, then click `Refresh odds now` under Advanced / Admin.

## Historical Model

Sprint 6 added the first historical international football model. It is intentionally simple: a multinomial logistic regression predicting home win, draw and away win.

Normal users do not need historical data. The deployed app should ship with production-ready pre-trained artifacts in `data/models/`. The current bundled baseline artifact is marked as a demo model because it was trained on too little generated/sample data.

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

Developer training workflow:

1. Add historical training data at `data/historical/international_results.csv`.
2. Run the export script.
3. Commit or deploy the generated artifacts in `data/models/`.
4. Open the app. It loads pre-trained artifacts automatically.

Production training expectations:

- thousands of historical international matches
- at least 1000 training rows and 200 test rows after chronological split
- multiple years of World Cup, continental tournament, qualifier and friendly data
- Elo-style strength features, recent-form features, tournament context and neutral venue
- metadata fields such as `training_rows`, `test_rows`, `feature_count`, `training_data_source`, performance metrics and `is_demo_model`

Command-line training is also available:

```bash
python scripts/train_and_export_model.py --input data/historical/international_results.csv
```

For deliberately small development runs, pass `--allow-demo`. Demo exports are marked with `is_demo_model = true` and are not treated as production-ready.

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

Run the backtest from `Advanced / Admin`:

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

## Draw Hypothesis and Draw-Context Features

Sprint 8 adds empirical testing for the draw hypothesis:

> In World Cup and major tournament group-stage matches, some teams may be strategically satisfied with a draw. This can increase the probability of a draw, depending on matchday, group state, qualification situation and relative team strength.

The app tests this hypothesis instead of assuming it is true. It calculates draw rates by tournament category, major tournament flag, World Cup flag, neutral venue, group-stage metadata, group matchday, strength buckets and strategic draw-context fields.

Group-state features are generated where historical data contains enough metadata such as `group`, `stage`, `matchday` or `group_matchday`. When metadata is missing, the app keeps matches in the dataset and uses neutral defaults. Because historical group-state metadata may be incomplete, results should be interpreted carefully.

The group-state logic is intentionally conservative. It reconstructs pre-match group points, goal difference, goals for, group position and matches played where possible. Matchday 3 approximations are:

- teams with 0 or 1 point are marked as must-win
- teams with 4 or more points are marked as draw-sufficient
- both teams draw-satisfied is true only when both teams are draw-sufficient

These are not exact qualification simulations.

Draw-context features include major tournament group flags, World Cup group flags, group matchday flags, even-match and heavy-favorite indicators, must-win indicators, draw-sufficient indicators, mutual draw acceptance and an explainable `draw_context_score` from 0 to 100. The score is explanatory and may be used as a model feature, but it does not manually override probabilities.

Run this from `Advanced / Admin`:

1. Click `Run draw hypothesis analysis` to create draw-rate segment outputs and a report.
2. Click `Compare baseline vs draw-context model` to run both walk-forward variants.
3. Review the decision card before enabling draw-context features in developer settings.

Output files:

- `data/processed/draw_hypothesis_summary.csv`
- `data/processed/draw_hypothesis_by_segment.csv`
- `data/processed/draw_feature_comparison.csv`
- `data/processed/backtest_predictions_with_draw_features.csv`
- `data/processed/backtest_summary_with_draw_features.csv`
- `data/processed/group_state_features.csv`
- `data/reports/draw_hypothesis_report.md`

Draw-context features are included only when the user enables them for future training/apply-model runs. The app does not automatically switch the production model based on one comparison run.

## Market-Aware Ensemble

Sprint 9 adds a probability-source layer and a market-aware ensemble. Market probabilities matter because bookmaker markets are strong baselines. A model-only probability set may add context, but it should not be assumed to beat the market.

The ensemble combines probabilities like this:

```text
final_home_prob = 0.8 * market_home_prob + 0.2 * model_home_prob
```

The same formula is applied to home, draw and away probabilities, then normalized so they sum to 1.0.

Supported probability sources:

- Market only
- Historical model
- Draw-context model
- Market-aware ensemble
- Best validated source

The active probability source controls only the probabilities used for edge and Kelly. The app never changes bookmaker odds. Edge remains:

```text
active_probability * odds - 1
```

Ensemble weights can be tested from 100% market / 0% model through 0% market / 100% model. Selection prioritizes log loss, then Brier score, ECE and draw calibration. Accuracy is shown but is not the primary selection metric because probability quality matters more for staking.

Run ensemble work from the `Ensemble` page:

1. `Run ensemble comparison` evaluates saved backtest predictions if historical market probabilities are available.
2. `Use recommended probability source` saves the best validated source.
3. `Apply manual ensemble to current matches` creates `data/processed/ensemble_predictions.csv` for upcoming matches.

Output files:

- `data/processed/ensemble_comparison.csv`
- `data/processed/ensemble_predictions.csv`
- `data/processed/ensemble_backtest_summary.csv`
- `data/processed/ensemble_backtest_by_segment.csv`
- `data/processed/active_probability_source.json`
- `data/reports/ensemble_report.md`

If historical market probabilities are unavailable, the app will say so and will not pretend the ensemble has been fully validated. Current-match ensemble probabilities can still be created from live/sample market probabilities, but should be treated as experimental until validated.

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
- If the app cannot find official fixtures, confirm that `data/reference/worldcup_2026_fixtures.csv` is committed.
- If sample/demo mode fails, confirm that `data/sample_predictions.csv` is committed.
- If bankroll or bet-log files are missing, restart the app; runtime files are recreated automatically.
- If the app says `Odds data source missing`, add `ODDS_API_KEY` via environment variable/Streamlit secrets, create `data/reference/manual_odds.csv` with valid odds, or refresh odds after adding the key.
- If the odds API returns no events, rate limits or omits draw odds, the app keeps official fixture rows where possible and shows a warning.
- If a bet cannot be settled, check that it is still pending. Double settlement is intentionally blocked.
- If the app shows sample-data validation errors, fix `data/sample_predictions.csv` before using demo mode.
