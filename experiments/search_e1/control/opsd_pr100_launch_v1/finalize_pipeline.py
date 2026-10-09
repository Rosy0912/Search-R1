"""Bind published PR100 periodic templates; no model imports or GPU work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def proof(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return proof(path)


def main():
    p = argparse.ArgumentParser()
    for name in ('template', 'template-sha256', 'publication-evidence', 'previous-terminal',
                 'previous-terminal-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    if os.getuid() != 1005 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('HKUST UID1005 CPU-only binding required')
    if proof(a.template)['sha256'] != a.template_sha256 or proof(a.previous_terminal)['sha256'] != a.previous_terminal_sha256:
        raise ValueError('Bound template or previous foreground terminal changed')
    pipeline = json.loads(Path(a.template).read_text())
    train = json.loads(Path(pipeline['training_config']['path']).read_text())
    original_storage = json.loads(Path(train['gates']['storage']['path']).read_text())
    publication = json.loads(Path(a.publication_evidence).read_text())
    helper = Path(pipeline['helper_root'])
    manifest = proof(helper / 'source-manifest.json')
    receipt_root = Path(a.output).parent
    commit = publication['source_commit']
    if publication.get('accepted') is not True or publication['remote_readback_sha'] != commit:
        raise ValueError('Actual source push/readback required')
    pipeline.update(source_commit=commit, helper_source_manifest=manifest,
                    startup_previous_driver_terminal=proof(a.previous_terminal))
    pipeline['mirror_root'] = original_storage['backup_policy']['durable_target']
    pipeline['publication'] = save(receipt_root / 'pipeline-publication.json', dict(
        publication, run_id=pipeline['run_id'], source_manifest_sha256=manifest['sha256']))
    published_templates = []
    for item in pipeline['evaluations']:
        original = item['template']
        if proof(original['path'])['sha256'] != original['sha256']:
            raise ValueError('Evaluation template changed')
        cfg = json.loads(Path(original['path']).read_text())
        n = item['completed_outers']
        cfg.update(source_commit=commit, helper_source_manifest=manifest)
        cfg['gates']['publication'] = save(receipt_root / ('eval-%03d-publication.json' % n), dict(
            publication, run_id=cfg['run_id'], source_manifest_sha256=manifest['sha256']))
        storage = dict(original_storage, run_id=cfg['run_id'], output_root=cfg['paths']['output_root'],
            scope=dict(study_id=train['study_id'], variant='PR', run_id=cfg['run_id'], output_root=cfg['paths']['output_root']))
        cfg['gates']['storage'] = save(receipt_root / ('eval-%03d-storage.json' % n), storage)
        cfg['state'] = 'PUBLISHED_TEMPLATE: checkpoint and actual training terminal bound at its own future boundary'
        ref = save(receipt_root / ('eval-%03d-published-template.json' % n), cfg)
        published_templates.append(dict(completed_outers=n, template=ref))
    pipeline['evaluations'] = published_templates
    baseline = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/zero/runs/summary.json')
    pipeline['baseline_summary'] = proof(baseline)
    pipeline['baseline_version'] = 'bf16_zero'
    pipeline['state'] = 'READY'
    pipeline['created_unix'] = time.time()
    sys.path.insert(0, str(helper))
    from supervisor import validate_manifest
    validate_manifest(pipeline)
    final = save(a.output, pipeline)
    print(json.dumps(dict(accepted=True, pipeline_config=final, pipeline_run_id=pipeline['run_id'],
        training_run_id=train['run_id'], source_commit=commit, outer_iterations=100, evaluate_every=10,
        evaluation_count=10, large_payloads_on_Mac=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
