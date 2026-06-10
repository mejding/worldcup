from typing import Sequence


def decimal_odds_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1 / decimal_odds


def remove_overround_proportional(probabilities: Sequence[float]) -> list[float]:
    total_probability = sum(probabilities)
    if total_probability <= 0:
        raise ValueError("Probability total must be positive.")
    return [probability / total_probability for probability in probabilities]


def calculate_edge(model_probability: float, odds: float) -> float:
    return model_probability * odds - 1
