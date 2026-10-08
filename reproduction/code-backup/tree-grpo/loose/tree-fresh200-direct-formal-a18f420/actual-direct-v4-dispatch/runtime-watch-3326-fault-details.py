import datetime
import glob
import hashlib
import json
import os
import re

BASE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fresh200-train-3326'

def safe_description(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [safe_description(v) for v in value if not isinstance(v, (dict, list))]
    if not isinstance(value, str):
        return {'type': type(value).__name__}
    if re.search(r'<(?:/?search|/?answer|/?information)\b|\b(?:questions?|contexts?|prompts?)\s*[:=]|\b(?:tensor|array)\s*\(|\b(?:token_ids|input_ids|log_probs|logps)\s*[:=]', value, re.I):
        return {'exception_classes': sorted(set(re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Refusal))\b', value))), 'description_omitted': True, 'description_length': len(value), 'description_sha256': hashlib.sha256(value.encode()).hexdigest()}
    cleaned = re.sub(r'\[[\s\d.,eE+\-]+\]', '[numeric array omitted]', value)
    if len(cleaned) > 1600:
        return {'exception_classes': sorted(set(re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Refusal))\b', value))), 'description_omitted': True, 'description_length': len(value), 'description_sha256': hashlib.sha256(value.encode()).hexdigest()}
    return cleaned

def numeric_tree(value):
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items() if isinstance(v, (dict, bool, int, float)) or v is None}
    return value if isinstance(value, (bool, int, float)) or value is None else None

result = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': 3326, 'partial_manifests': [], 'policy_safety': [], 'traceback_files': [], 'exception_classes': [], 'terminal_metadata_files': []}
for name in ['run-result.json', 'cost.json', 'termination-cost.json']:
    path = os.path.join(BASE, name)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            contents = f.read()
        obj = json.loads(contents)
        result['terminal_metadata_files'].append({'name': name, 'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest(), 'mtime': os.path.getmtime(path), 'top_field_types': {k: type(v).__name__ for k, v in obj.items()}, 'numeric_fields': numeric_tree(obj) if name != 'run-result.json' else None})
for path in sorted(glob.glob(os.path.join(BASE, 'checkpoints/training_state/partial_attempt_*/manifest.json'))):
    with open(path) as f:
        obj = json.load(f)
    outcome = obj.get('worker_outcome', {})
    states = []
    for state in outcome.get('worker_states', []):
        keys = ['rank', 'refusal', 'safety_guard', 'error', 'state_finite_error', 'model_adam_finite', 'gradients_finite', 'committed_optimizer_calls']
        record = {k: safe_description(state.get(k)) for k in keys if k in state}
        record['ledger'] = [{k: safe_description(entry[k]) for k in ['guard_reason', 'status', 'actual_writes'] if k in entry} for entry in state.get('optimizer_opportunity_ledger', []) if isinstance(entry, dict)]
        states.append(record)
    resets = []
    for origin, reset in [('gradient_reset', obj.get('gradient_reset')), ('gradient_reset_after_raw_persistence', outcome.get('gradient_reset_after_raw_persistence'))]:
        if reset is None:
            resets.append({'origin': origin, 'value': None})
            continue
        entries = reset if isinstance(reset, list) else reset.get('worker_states', reset.get('ranks', [reset])) if isinstance(reset, dict) else []
        if isinstance(entries, dict):
            entries = list(entries.values())
        keys = ['rank', 'error', 'normal_zero_grad', 'public_gradients_after_reset_finite', 'private_gradients_after_reset_finite', 'private_gradient_attributes_still_present']
        resets.append({'origin': origin, 'entries': [{k: safe_description(entry[k]) for k in keys if k in entry} for entry in entries if isinstance(entry, dict)]})
    result['partial_manifests'].append({'name': os.path.relpath(path, BASE), 'status': obj.get('status'), 'continuation_decision': obj.get('continuation_decision'), 'workers': states, 'gradient_reset': resets})

for path in sorted(glob.glob(os.path.join(BASE, 'output/tree-rank-audit/policy-safety/*'))):
    if not os.path.isfile(path) or not path.endswith('.json'):
        continue
    try:
        with open(path) as f:
            obj = json.load(f)
    except json.JSONDecodeError as error:
        with open(path) as f:
            text = f.read()
        field_keys = re.findall(r'"([A-Za-z_][A-Za-z_0-9]*)"\s*:', text[:error.pos + 1])
        result['policy_safety'].append({'name': os.path.relpath(path, BASE), 'mtime': os.path.getmtime(path), 'bytes': os.path.getsize(path), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'read_error_type': type(error).__name__, 'error_line': error.lineno, 'error_column': error.colno, 'error_position': error.pos, 'last_parsed_field_name': field_keys[-1] if field_keys else None, 'error_at_eof': error.pos == len(text)})
        continue
    decision = obj.get('decision', {})
    if not isinstance(decision, dict):
        decision = {}
    result['policy_safety'].append({'name': os.path.relpath(path, BASE), 'mtime': os.path.getmtime(path), 'status': obj.get('status') if isinstance(obj.get('status'), str) and re.fullmatch(r'[A-Z_0-9]+', obj['status']) else None, 'decision_reasons': safe_description(decision.get('reasons')), 'decision_numeric': numeric_tree(decision), 'top_numeric': {k:v for k,v in obj.items() if isinstance(v, (bool, int, float))}, 'metrics_numeric': numeric_tree(obj.get('metrics', {}))})

stage = os.path.join(BASE, 'stage.log')
if os.path.isfile(stage):
    with open(stage, errors='replace') as f:
        for line in f:
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            match = re.search(r'File "(/[^"\n]+)", line (\d+)', clean)
            if match:
                record = {'file': match.group(1), 'line': int(match.group(2))}
                if record not in result['traceback_files']:
                    result['traceback_files'].append(record)
            for exception in re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Refusal|Stop))\b', clean):
                if exception not in result['exception_classes']:
                    result['exception_classes'].append(exception)
print(json.dumps(result, sort_keys=True))
