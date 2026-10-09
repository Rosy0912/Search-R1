"""Bind original closed native records on HKUST; no Mac prediction copy."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
EVAL = BASE / 'eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851'


def proof(path):
    raw = path.read_bytes()
    return raw, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.is_absolute() or not out.resolve().is_relative_to(BASE / 'analysis') or out.exists():
        raise ValueError('New independent remote analysis output required')
    models, closed, pending = {}, {}, []
    for group in ('zero', 'C0', 'C1', 'M', 'R', 'W', 'P', 'J'):
        root = EVAL / group / 'runs'
        summary_path, terminal_path = root / 'summary.json', root / 'controller-complete.json'
        if not summary_path.exists() or not terminal_path.exists():
            pending.append(group)
            continue
        raw, summary_ref = proof(summary_path)
        summary = json.loads(raw)
        terminal_raw, terminal_ref = proof(terminal_path)
        terminal = json.loads(terminal_raw)
        if summary.get('schema') != 'opsd-ablation-native512-summary-v1' or not summary.get('accepted') or not terminal.get('accepted'):
            raise ValueError('Unaccepted original native summary or controller terminal')
        if terminal['run_id'] != summary['run_id'] or terminal['summary']['sha256'] != summary_ref['sha256']:
            raise ValueError('Original terminal does not bind this exact summary')
        ref = summary['private_primary_records']
        path = Path(ref['path'])
        if not path.resolve().is_relative_to(EVAL):
            raise ValueError('Private primary records escaped the original remote study')
        _, record_ref = proof(path)
        if any(record_ref[k] != ref[k] for k in ('sha256', 'bytes')):
            raise ValueError('Native record SHA/size mismatch')
        labels = [('bf16_zero', 'bf16_zero'), ('fp16_zero', 'fp16_zero')] if group == 'zero' else [(group, 'raw10_bf16')]
        if group in ('P', 'J'):
            labels.append((group + '_fp16', 'raw10_fp16'))
        if set(summary['models']) != {version for _, version in labels}:
            raise ValueError('Original native complete version coverage differs')
        for label, version in labels:
            models[label] = dict(summary=summary_ref, records=record_ref, version=version)
        closed[group] = dict(summary=summary_ref, controller_complete=terminal_ref)
    if not models:
        raise ValueError('No original closed native source available')
    value = dict(schema='opsd-ablation-stratified-native-inputs-v1', models=models,
                 remote_original_records=True, closed_native_sources=closed, pending_groups=pending)
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(raw)
    print(json.dumps(dict(accepted=True, manifest=dict(path=str(out), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                         complete_model_count=len(models), pending_groups=pending, raw_records_exported=False)))


if __name__ == '__main__':
    main()
