"""Human sparse detail-recording schedule; other experiments keep legacy logging."""
import os
STEPS = frozenset((5, 50, 100, 150, 200))
def record_details(step):
    return not os.environ.get('RECURSIVE_B_CONTINUATION_BINDING') or int(step) in STEPS
