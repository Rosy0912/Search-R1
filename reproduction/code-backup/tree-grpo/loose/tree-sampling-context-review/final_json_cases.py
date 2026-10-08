"""Fixed-candidate JSON byte/hash and file-identity counterexamples."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

CANDIDATE = Path(__file__).parent/'candidate-28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac.py'
spec=importlib.util.spec_from_file_location('fixed',CANDIDATE)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
checks=[]
with tempfile.TemporaryDirectory(prefix='repaired-json-case-',dir=Path(__file__).parent) as temp:
    path=Path(temp)/'bound.json'; original=b'{"bound": true}'
    path.write_bytes(original);expected=hashlib.sha256(original).hexdigest()
    assert A.json_bound(path,expected)=={'bound':True}
    checks.append('unchanged_json_returns_exact_hash_bound_payload')
    original_sha=A.hashlib.sha256
    def mutate_on_hash(payload=b''):
        observed=original_sha(payload)
        if payload==original:path.write_bytes(b'{"bound": false}')
        return observed
    A.hashlib.sha256=mutate_on_hash
    try:
        try:A.json_bound(path,expected)
        except ValueError as exc:assert 'JSON changed while loading' in str(exc)
        else:raise AssertionError('Mutation was accepted')
    finally:A.hashlib.sha256=original_sha
    checks.append('same_descriptor_hash_then_metadata_detects_content_mutation_before_parse')
    path.write_bytes(original)
    def replace_on_hash(payload=b''):
        observed=original_sha(payload)
        if payload==original:
            sibling=path.with_suffix('.replacement');sibling.write_bytes(original);sibling.replace(path)
        return observed
    A.hashlib.sha256=replace_on_hash
    try:
        try:A.json_bound(path,expected)
        except ValueError as exc:assert 'JSON changed while loading' in str(exc)
        else:raise AssertionError('Same-content inode replacement was accepted')
    finally:A.hashlib.sha256=original_sha
    checks.append('same_content_different_inode_replacement_refused')
    path.write_bytes(original)
    original_loads=A.json.loads
    def mutate_at_parse(payload):
        path.write_bytes(b'{"bound": false}')
        return original_loads(payload)
    A.json.loads=mutate_at_parse
    try:assert A.json_bound(path,expected)=={'bound':True}
    finally:A.json.loads=original_loads
    checks.append('post_read_change_does_not_replace_parsed_hash_bound_payload')
result=dict(status='REPAIRED_JSON_BINDING_PASS',candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    checks=checks,scope='Small temporary JSON fixtures only, no actual assets or remote execution')
print(json.dumps(result,sort_keys=True,allow_nan=False))
