"""Positive D-correctness weights; pure CPU math, no training RNG consumed."""
import math

WEIGHTS_VERSION = 'search-e1-opsd-correctness-weights-v1'


def correctness_weights(rewards):
    values = list(rewards)
    if len(values) != 5 or any(type(x) is not int or x not in (0, 1) for x in values):
        raise ValueError('Exactly five native strict binary rewards are required')
    mu = sum(values) / 5.0
    sigma = math.sqrt(sum((x - mu) ** 2 for x in values) / 4.0)
    advantages = [(x - mu) / (sigma + 1e-6) for x in values] if sigma > 0 else [0.0] * 5
    largest = max(advantages)
    exponentials = [math.exp(x - largest) for x in advantages]
    total = sum(exponentials)
    weights = [5.0 * x / total for x in exponentials]
    if not all(math.isfinite(x) and x > 0 for x in weights) or abs(sum(weights) - 5.0) > 1e-12:
        raise ValueError('Correctness weight normalization is nonfinite or invalid')
    return dict(version=WEIGHTS_VERSION, rewards=values, mu=mu, sigma=sigma,
                correction=1, epsilon=1e-6, temperature=1.0,
                advantages=advantages, weights=weights, mean_weight=sum(weights) / 5.0)
