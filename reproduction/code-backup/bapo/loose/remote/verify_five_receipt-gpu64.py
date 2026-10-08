#!/usr/bin/env python3
"""Fail closed before fresh acceptance training; never creates review receipts."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo')
SOURCE = 'b20f371972bb1bc41b735c6cd350b9a779d8820d'
STAGE = '51a401e2f2a275dce502d58fa67b8d85d342362194d98efcf24f77d798ca1c39'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(job_id):
    if not str(job_id).isascii() or not str(job_id).isdigit():
        raise ValueError('Invalid job id')
    run = ROOT / 'runs' / ('bapo-accept5-' + str(job_id))
    receipt = json.loads((ROOT / 'logs' / ('accept5-' + str(job_id) + '-acceptance.json')).read_text())
    if receipt.get('status') != 'PASS_INDEPENDENT_FIVE_ITERATION_REVIEW':
        raise ValueError('Independent five-iteration review not passed')
    expected = {'run_root': str(run), 'actual_outer_updates': 5,
                'source_commit': SOURCE, 'launch_stage_sha256': STAGE}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError('Receipt identity mismatch: ' + key)
    if not (run / 'TRAIN_RETURNED_SUCCESSFULLY').is_file():
        raise ValueError('Training success marker missing')
    source = subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'status', '--porcelain'], text=True)
    if source != SOURCE or dirty:
        raise ValueError('Current source changed')
    if digest(ROOT / 'tools' / 'stage.sh') != STAGE:
        raise ValueError('Current launch stage changed')
    evidence = receipt.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Missing evidence bindings')
    review = receipt.get('independent_review')
    if not isinstance(review, dict):
        raise ValueError('Missing independent review binding')
    seen = set()
    for item in evidence + [review]:
        path = Path(item['path'])
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Evidence missing: ' + str(path))
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise ValueError('Evidence outside owned root')
        if digest(path) != item['sha256']:
            raise ValueError('Evidence hash mismatch: ' + str(path))
        seen.add(str(path))
    if str(run / 'train.log') not in seen:
        raise ValueError('Authoritative train log not bound')
    if not any('/actor/global_step_5/' in p and p.endswith('.safetensors') for p in seen):
        raise ValueError('Checkpoint weights not bound')
    if not any('/algorithm/global_step_5/' in p and p.endswith('/state.json') for p in seen):
        raise ValueError('Algorithm state not bound')
    extraction_path = receipt.get('extraction_path')
    if extraction_path not in seen:
        raise ValueError('Extraction JSON not bound')
    extraction = json.loads(Path(extraction_path).read_text())
    if extraction.get('run_root') != str(run) or extraction.get('expected_outer_updates') != 5:
        raise ValueError('Extraction run or iteration identity mismatch')
    checks = extraction.get('outer_step_checks', {})
    if set(checks) != {str(i) for i in range(1, 6)} or any(v.get('status') != 'NUMERIC_SUMMARY_PRESENT' or v.get('issues') for v in checks.values()):
        raise ValueError('Five complete numeric outer updates missing')
    if extraction.get('stage_observed_steps') != [1, 2, 3, 4, 5]:
        raise ValueError('Full stage observations missing')
    # Recheck concrete proof fields, not only a self-described status string.
    checkpoint = extraction.get('checkpoint', {})
    if checkpoint.get('hf_load', {}).get('status') != 'HF_LOAD_AND_FORWARD_OK':
        raise ValueError('Real HF CPU load/forward proof missing')
    if checkpoint.get('tokenizer_compatibility', {}).get('status') != 'TOKENIZER_MATCH':
        raise ValueError('Tokenizer identity mismatch')
    hf = checkpoint.get('hf_load', {})
    if hf.get('forward', {}).get('logits_finite') is not True or hf.get('forward', {}).get('device') != 'cpu' or hf.get('nonfinite_parameters') != []:
        raise ValueError('HF finite CPU forward/parameters missing')
    loading = hf.get('loading_info')
    if not isinstance(loading, dict) or any(loading.get(k) != [] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Concrete clean HF loading information missing')
    if hf.get('tied_input_output_storage') is not True:
        raise ValueError('Actual HF tied storage proof missing')
    tokens = checkpoint.get('tokenizer_compatibility', {})
    if not all(tokens.get(k) is True for k in ('full_vocabulary_equal','bos_equal','eos_equal')):
        raise ValueError('Concrete tokenizer checks missing')
    delta = checkpoint.get('parameter_delta', {})
    if delta.get('changed') is not True or delta.get('comparison') != 'base_cast_to_checkpoint_dtype_then_float64':
        raise ValueError('Dtype-aligned changed weight proof missing')
    alias = delta.get('tied_weight_resolution', {})
    expected_alias = dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED', base_logical_key='lm_head.weight',
        base_storage_key='model.embed_tokens.weight', checkpoint_extra_key='lm_head.weight',
        checkpoint_equal_key='model.embed_tokens.weight', shape=[151936,2048], elements_compared=311164928, exact_equal=True)
    if any(alias.get(k) != v for k,v in expected_alias.items()):
        raise ValueError('Complete finite exact observed Qwen2 tied alias proof missing')
    value = delta.get('max_abs_delta')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite aligned parameter delta missing')
    checkpoint_path = Path(checkpoint.get('path', ''))
    if not checkpoint_path.is_absolute() or run not in checkpoint_path.parents or checkpoint_path.name != 'global_step_5':
        raise ValueError('Checkpoint path identity mismatch')
    files = checkpoint.get('files', [])
    if not files:
        raise ValueError('Checkpoint file manifest missing')
    disk_files = {str(p.relative_to(checkpoint_path)) for p in checkpoint_path.rglob('*') if p.is_file()}
    listed_files = [item['path'] for item in files]
    if len(set(listed_files)) != len(listed_files) or set(listed_files) != disk_files:
        raise ValueError('Checkpoint manifest does not cover every current file')
    for item in files:
        path = checkpoint_path / item['path']
        if checkpoint_path not in path.resolve().parents:
            raise ValueError('Invalid checkpoint relative path')
        if str(path) not in seen or digest(path) != item.get('sha256'):
            raise ValueError('Exported checkpoint file not completely bound')
    # Actual configured output nesting is permitted only inside this exact run.
    candidates = [p for p in run.rglob('state.json') if p.parent.name == 'global_step_5' and p.parent.parent.name == 'algorithm']
    if len(candidates) != 1 or str(candidates[0]) not in seen:
        raise ValueError('Exact run algorithm state not bound')
    state = json.loads(candidates[0].read_text())
    if state.get('method') != 'bapo' or state.get('global_step') != 5 or state.get('actual_outer_updates') != 5 or state.get('algorithm_source_commit') != SOURCE:
        raise ValueError('Algorithm state identity mismatch')
    for filename in ('own-process-cleanup.json','resource-watch.jsonl','cpu-process-monitor.jsonl','gpus-before.csv','gpus-exit.csv'):
        if str(run / filename) not in seen:
            raise ValueError('Resource/cleanup evidence not bound: '+filename)
    cleanup = json.loads((run/'own-process-cleanup.json').read_text())
    if cleanup.get('own_marker') != str(run) or cleanup.get('remaining') != []:
        raise ValueError('Own cleanup not complete')
    input_hashes = {item['file']: item['sha256'] for item in extraction.get('inputs', [])}
    if input_hashes.get(str(run/'train.log')) != digest(run/'train.log'):
        raise ValueError('Extraction authoritative log hash mismatch')
    print('PASS independent five-iteration receipt; formal500 submission permitted')

if __name__ == '__main__':
    verify(sys.argv[1])
