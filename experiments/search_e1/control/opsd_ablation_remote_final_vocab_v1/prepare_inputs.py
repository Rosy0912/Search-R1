"""Bind completed native CPU evidence on HKUST without exporting private data."""
import argparse
import hashlib
import json
import os
from pathlib import Path

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
DRIVER = BASE / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
PRODUCER = BASE / 'source/training-871dc46/opsd_ablation_v1'


def ref(p):
    p = Path(p)
    if not p.is_absolute() or BASE not in p.parents or any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('Canonical HKUST study file required')
    before = p.stat(); raw = p.read_bytes(); after = p.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Input changed during binding')
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)), raw


def value(r):
    actual, raw = ref(r['path'])
    if any(actual[k] != r[k] for k in ('sha256', 'bytes')):
        raise ValueError('Existing evidence pin differs')
    return json.loads(raw)


def save(p, d):
    with p.open('x') as f:
        f.write(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return ref(p)[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ('debug-root', 'debug-complete-sha256', 'source-root', 'evidence-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if os.getuid() != 1005 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Original HKUST CPU environment required')
    debug, source, evidence, out = map(Path, (args.debug_root, args.source_root, args.evidence_root, args.output))
    complete_ref, raw = ref(debug / 'complete.json')
    if complete_ref['sha256'] != args.debug_complete_sha256:
        raise ValueError('Explicit actual seven-reader completion SHA required')
    done = json.loads(raw)
    if not done['accepted'] or done['groups'] != list(ORDER):
        raise ValueError('Full seven original recorded-reader completion required')
    for r in done['executions'].values():
        if value(r)['actual_wait_returncode'] != 0:
            raise ValueError('Original recorded child actual wait0 required')
    facts_ref, raw = ref(debug / 'facts.private.json'); facts = json.loads(raw)
    if not facts['accepted'] or not facts['coverage']['all70_complete']:
        raise ValueError('Actual all70 native facts required')
    batches = done['batch_proofs']; copied = {}; group_batch = {}
    for i, b in enumerate(batches):
        d = value(b['copies']); w = value(b['actual_wait'])
        if not d['accepted'] or w['actual_ssh_returncode'] != 0 or w['output'] != str(Path(b['copies']['path']).parent):
            raise ValueError('Actual independent-copy batch wait differs')
        for variant, r in d['groups'].items():
            if variant in copied:
                raise ValueError('Repeated copied group')
            copied[variant] = value(r); group_batch[variant] = i
    if set(copied) != set(ORDER):
        raise ValueError('Seven accepted copied groups required')
    cov = dict(publication=ref(debug / 'coverage/debug-coverage-publication.json')[0],
               details=ref(debug / 'coverage/details.private.json')[0],
               derived_manifest=ref(debug / 'coverage/derived-manifest.json')[0])
    groups = []
    for i, variant in enumerate(ORDER):
        c = copied[variant]['config']
        groups.append(dict(variant=variant, config=c,
            evaluation_config=ref(DRIVER / ('%02d-%s' % (i, variant)) / 'evaluation-config.json')[0],
            coverage=dict(kind='recorded', config_sha256=c['sha256'], backups=copied[variant]['backups'], **cov)))
    study_id = value(groups[0]['config'])['study_id']
    ledger = dict(schema='opsd-ablation-remote-final-vocab-inputs-v1', ready=True,
        study_id=study_id, facts=facts_ref, groups=groups,
        copy_batches=[dict(complete=b['copies'], actual_wait=b['actual_wait']) for b in batches],
        sources=dict(collector=ref(source / 'collect.py')[0], consumer=ref(source / 'close_ledger.py')[0],
            aggregate=ref(source / 'aggregate_clipmass.py')[0], producer=ref(PRODUCER / 'vocabulary_dump.py')[0],
            producer_manifest=ref(PRODUCER / 'source-manifest.json')[0]))
    p_groups = []
    for variant in ('P', 'J'):
        label = 'six' if group_batch[variant] == 0 else 'J'
        group = next(g for g in groups if g['variant'] == variant)
        repairs = []
        for p in sorted((BASE / ('group-' + variant) / 'receipts/dtype-recovery').rglob('trainer-rank-*-pid-*.json')):
            r, raw = ref(p)
            if json.loads(raw).get('schema') == 'opsd-ablation-dtype-repair-execution-v1':
                repairs.append(r)
        p_groups.append(dict(variant=variant, config=group['config'], backups=copied[variant]['backups'],
            remote_copy=dict(deployment=ref(evidence / 'six-deployment.json')[0],
                launch=ref(evidence / (label + '-launch.json'))[0],
                actual_wait=ref(evidence / (label + '-wait.json'))[0],
                complete=batches[group_batch[variant]]['copies']), dtype_repair_receipts=repairs))
    precision = dict(schema='opsd-ablation-recorded-precision-history-remote-inputs-v2', ready=True,
        facts=facts_ref, groups=p_groups, sources=dict(aggregate=ledger['sources']['aggregate'],
            trainer=ref(PRODUCER / 'trainer.py')[0], precision=ref(PRODUCER / 'precision.py')[0],
            retry_state=ref(PRODUCER / 'retry_state.py')[0],
            dtype_launcher=ref(BASE / 'source/dtype-538ac99/opsd_ablation_dtype_recovery_v1/trainer_launcher.py')[0]))
    if not out.is_absolute() or BASE / 'analysis' not in out.parents or out.exists():
        raise ValueError('New explicit HKUST input directory required')
    out.mkdir(parents=True, exist_ok=False)
    print(json.dumps(dict(accepted=True, all70_facts=facts_ref,
        ledger_inputs=save(out / 'ledger-inputs.private.json', ledger),
        precision_inputs=save(out / 'precision-inputs.private.json', precision),
        raw_or_tensor_payloads_exported=False)))


if __name__ == '__main__':
    main()
