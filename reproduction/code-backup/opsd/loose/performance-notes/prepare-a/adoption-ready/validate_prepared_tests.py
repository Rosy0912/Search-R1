"""Preparation only: isolated Python process validates tests with pinned A.
Never writes the repository; no GPU or scheduler operation.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest
HERE = Path(__file__).resolve().parent
candidate = HERE.parent / 'proposal_distillation_a.py'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == '2131315b6770779f0c7579bf2a6139e742cf75de03b7ee9f44662f8c32a870cf'
# A complete isolated module replacement, only in this validation process.
# Adopted tests themselves always import the installed production module.
name = 'search_r1.baselines.distillation'
assert (Path.cwd() / 'search_r1/baselines/distillation.py').is_file(), 'Run from repository root'
sys.path.insert(0, str(Path.cwd()))
import search_r1.baselines
spec = importlib.util.spec_from_file_location(name, candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules[name] = module
search_r1.baselines.distillation = module
raise SystemExit(pytest.main([str(HERE / 'tests/baselines/test_opsd_empty_chunks.py'), '-q']))
