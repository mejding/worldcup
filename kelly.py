from config import DEFAULT_STAKING_PROFILE


def calculate_full_kelly(model_probability: float, odds: float) -> float:
    if odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    if not 0 <= model_probability <= 1:
        raise ValueError("Model probability must be between 0 and 1.")

    b = odds - 1
    p = model_probability
    q = 1 - p
    full_kelly = (b * p - q) / b

    return max(full_kelly, 0)


def calculate_fractional_kelly(
    full_kelly: float,
    fractional_kelly_multiplier: float = DEFAULT_STAKING_PROFILE["fractional_kelly_multiplier"],
) -> float:
    if full_kelly < 0:
        raise ValueError("Full Kelly cannot be negative.")
    return full_kelly * fractional_kelly_multiplier


def apply_stake_cap(
    stake_fraction: float,
    max_stake_pct_of_bankroll: float = DEFAULT_STAKING_PROFILE["max_stake_pct_of_bankroll"],
) -> float:
    if stake_fraction < 0:
        raise ValueError("Stake fraction cannot be negative.")
    return min(stake_fraction, max_stake_pct_of_bankroll)


def calculate_final_stake_fraction(
    model_probability: float,
    odds: float,
    fractional_kelly_multiplier: float = DEFAULT_STAKING_PROFILE["fractional_kelly_multiplier"],
    max_stake_pct_of_bankroll: float = DEFAULT_STAKING_PROFILE["max_stake_pct_of_bankroll"],
) -> dict[str, float]:
    full_kelly = calculate_full_kelly(model_probability, odds)
    fractional_kelly = calculate_fractional_kelly(full_kelly, fractional_kelly_multiplier)
    final_stake_fraction = apply_stake_cap(fractional_kelly, max_stake_pct_of_bankroll)

    return {
        "full_kelly": full_kelly,
        "fractional_kelly": fractional_kelly,
        "final_stake_fraction": final_stake_fraction,
    }


def calculate_suggested_stake(current_bankroll: float, final_stake_fraction: float) -> float:
    if current_bankroll < 0:
        raise ValueError("Current bankroll cannot be negative.")
    if final_stake_fraction < 0:
        raise ValueError("Final stake fraction cannot be negative.")
    return current_bankroll * final_stake_fraction

