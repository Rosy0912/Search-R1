"""Collect existing native evaluations without launching or changing training.

Missing milestones remain null. This report verifies evaluation evidence only;
it cannot replace native-state, first-five, terminal-job or reuse acceptance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .compare import compare, load_evaluation
from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_run(run, group):
    run = Path(run).resolve()
    required = [5, 50, 100, 150] + ([200] if group == 'original' else [])
    result = dict(group=group, run=str(run), required_dev_rounds=required,
        dev_curve={str(step): None for step in required}, final51713=None,
        pending_evaluation_files=[], all_planned_evaluations_present=False)
    receipt = run / 'receipts/configuration.json'
    if not receipt.exists():
        result['evidence_state'] = 'NO_CONFIGURATION_RECEIPT'
        return result
    config = json.loads(receipt.read_text())
    if (config.get('group', config.get('arm')) != group
            or sha(config['config']) != config['config_sha256']):
        raise ValueError('Run group or frozen configuration bytes disagree')
    result.update(evidence_state='CONFIGURED', run_id=config['run_id'],
        configuration_receipt_sha256=sha(receipt), source_commit=config['source_commit'])
    paths = list((run / 'output/evaluations').glob('*.jsonl'))
    paths += list((run / 'output').glob('attempt-*/evaluations/*.jsonl'))
    seen = set()
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError('Evaluation must remain inside its declared run')
        if not path.with_suffix('.json').exists():
            result['pending_evaluation_files'].append(str(path))
            continue
        meta, rows, files = load_evaluation(path)
        expected = dict(run_id=config['run_id'], code_commit=config['source_commit'],
            config_sha256=config['config_sha256'], data_identity=config['data_identity'])
        if any(meta[k] != v for k, v in expected.items()):
            raise ValueError('Evaluation identity differs from the frozen run')
        initialization = path.parent.parent / 'fresh-initialization.json'
        physical = json.loads(initialization.read_text())
        if (str(physical['job_id']) != str(meta['job'])
                or any(physical[k] != v for k, v in expected.items())):
            raise ValueError('Evaluation identity differs from its physical attempt')
        step, split = meta['completed_outer'], meta['split']
        if path.stem != f'{split}-outer-{step:06d}':
            raise ValueError('Evaluation filename differs from its recorded split or round')
        if (split == 'dev512' and step not in required) or (
                split == 'final51713' and (group != 'original' or step != 200)):
            raise ValueError('Evaluation is outside the requested milestone schedule')
        key = (split, step)
        if key in seen:
            raise ValueError('Ambiguous duplicate evaluation across physical attempts')
        seen.add(key)
        by_source = defaultdict(list)
        for row in rows.values():
            by_source[row['source']].append(row['em'])
        point = dict(outer=step, split=split, em=meta['em'], count=meta['count'],
            by_source={s:dict(count=len(v), em=sum(v)/len(v)) for s,v in sorted(by_source.items())},
            metadata=meta, files=files, initialization_sha256=sha(initialization))
        if split == 'dev512':
            result['dev_curve'][str(step)] = point
        else:
            result['final51713'] = point
        result['evidence_state'] = 'VERIFIED_EVALUATIONS_PRESENT'
    result['all_planned_evaluations_present'] = (all(result['dev_curve'].values())
        and (group != 'original' or result['final51713'] is not None))
    return result


def collect(base, cost_receipt=None):
    conditions = planned_conditions()
    groups = ['original', 'curriculum'] + [r['group'] for r in conditions if not r.get('reuse')]
    runs = {g:collect_run(Path(base)/'runs'/g, g) for g in groups}
    baseline = runs['original']['dev_curve']
    paired = {}
    for group, run in runs.items():
        if group == 'original':
            continue
        paired[group] = {}
        for step, point in run['dev_curve'].items():
            left = baseline.get(step)
            paired[group][step] = (compare(left['files']['rows_path'], point['files']['rows_path'])
                if left is not None and point is not None else None)
    costs = None
    if cost_receipt is not None:
        path = Path(cost_receipt)
        costs = dict(path=str(path.resolve()), sha256=sha(path), evidence=json.loads(path.read_text()))
    return dict(at=time.time(), status='PARTIAL_EVALUATION_EVIDENCE_REPORT', runs=runs,
        paired_against_original=paired, costs=costs,
        planned_reuse={r['group']:r['reuse'] for r in conditions if r.get('reuse')},
        goal_completion_assessed=False,
        limitations=[
            'Evaluation evidence does not prove training completion or native recovery acceptance.',
            'Planned reuse still requires matching-condition verification; it is not a new trained run.',
            'Null points mean unavailable evidence, never zero EM or interpolated performance.',
            'Cost evidence is a separately timestamped snapshot and includes its own accounting limits.',
            'Single-seed comparisons use different hardware histories and unequal compute.',
            'Original A reconstructs the author recipe; unavailable resolved author bytes were not reproduced.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--cost-receipt')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = collect(args.base, args.cost_receipt)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(output=args.output, available_dev_points={g:sum(p is not None
        for p in run['dev_curve'].values()) for g,run in result['runs'].items()},
        goal_completion_assessed=False)))


if __name__ == '__main__':
    main()
