# FIFA Ranking Feature Report

- FIFA ranking data available: no
- Selected variant: elo_only
- Recommendation: Kept elo_only because FIFA ranking did not improve both log loss and Brier score versus Elo.
- Warning: FIFA ranking and Elo are correlated team-strength signals. FIFA ranking is only used if backtesting shows it adds predictive value.

## Coverage
- Missing rate: 100.00%
- Teams covered: 0

## Variants Tested
- baseline_no_strength: n=9882, log_loss=0.9209, brier=0.5437, ece=0.0124
- elo_only: n=9882, log_loss=0.8763, brier=0.5155, ece=0.0148
- fifa_only: n=9882, log_loss=0.9209, brier=0.5437, ece=0.0124
- elo_plus_fifa: n=9882, log_loss=0.8763, brier=0.5155, ece=0.0148
