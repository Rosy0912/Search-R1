"""Human sparse detail-recording schedule; other experiments keep legacy logging."""
import os
STEPS = frozenset((5, 50, 100, 150, 200))
def record_details(step):
    sparse = bool(os.environ.get('RECURSIVE_A_CONTINUATION_BINDING')) or os.environ.get('RECURSIVE_A_SPARSE_DETAILS')=='1'
    return not sparse or int(step) in STEPS
