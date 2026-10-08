"""Run four existing actual finite-clip test bodies without optional verl imports.

The full pytest collection is preserved as a failure (tensordict unavailable).
No production module replacements and no rewritten test assertions are used.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'tree-behavior-mu-ppo'
sys.path.insert(0, str(ROOT))
from search_r1.baselines.gradient_norm import clip_grad_norm64, _norm64

path = ROOT / 'tests/baselines/test_gradient_numerics.py'
names = (
    'test_finite_gradients_with_overflowing_fp32_norm_clip_correctly',
    'test_normal_clipping_matches_native_and_does_not_change_rng',
    'test_nonfinite_gradients_are_refused_before_mutation',
    'test_chunked_norm_matches_full_and_empty_norm',
)
tree = ast.parse(path.read_bytes())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == len(names)
scope = dict(copy=copy, pytest=pytest, torch=torch, clip_grad_norm64=clip_grad_norm64, _norm64=_norm64)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
for name in names:
    scope[name]()
    print(name, 'PASS', flush=True)
print(json.dumps(dict(actual_test_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    passed=4, scope='Four unchanged actual finite-clip CPU test function AST bodies; full pytest collection unavailable due missing tensordict; no GPU/model'), sort_keys=True))
