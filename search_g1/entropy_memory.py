"""Keep exact entropy metrics without a zero-coefficient backward graph."""
import torch


def entropy_metric(logits, coefficient, implementation):
    # With coefficient exactly zero, entropy has no contribution to the policy
    # gradient. Preserve the original forward, value and finite checks; avoid
    # retaining its full-vocabulary softmax tensors solely for a zero gradient.
    with torch.set_grad_enabled(torch.is_grad_enabled() and float(coefficient) != 0.0):
        return implementation(logits)
