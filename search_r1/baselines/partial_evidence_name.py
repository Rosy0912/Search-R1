"""Shared driver/worker evidence naming; no model, RNG or filesystem writes."""
import re

def evidence_name(identity, continuing, environ):
    outer = identity['attempted_outer']
    name = (f"partial_attempt_{identity['attempt_id']:06d}_outer_{outer:06d}"
            if continuing else f"partial_outer_{outer:06d}")
    if environ.get('RECURSIVE_B_CONTINUATION_BINDING'):
        run = identity['run_id']
        if not isinstance(run,str) or not re.fullmatch('[0-9a-f]{32}',run):
            raise ValueError('B partial evidence requires an exact immutable run ID')
        name += '_run_' + run
    return name
