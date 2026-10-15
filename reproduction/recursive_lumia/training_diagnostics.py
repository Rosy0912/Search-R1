"""Compare completed main-arm training groups and recorded costs at equal rounds.

Reads immutable question event streams and a closed prefix of each live log.
These are training-distribution diagnostics, never held-out accuracy estimates.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


def question(path, config, outer):
    raw = path.read_bytes()
    assert raw.endswith(b'\n'), path
    events = [json.loads(line) for line in raw.splitlines()]
    first, last = events[0], events[-1]
    assert first['event'] == 'question_start' and last['event'] == 'question_end'
    assert [e['event_seq'] for e in events] == list(range(1, len(events) + 1))
    for e in events:
        assert e['outer'] == outer and e['question_id'] == first['question_id']
        for key in ('run_id', 'source_commit', 'config_sha256'):
            assert e[key] == config[key], (path, key)
    root = last['final_ids'][0]
    rewards, generations = {root: first['initial_reward']}, {root: 0}
    for e in events:
        if e['event'] == 'branch_end' and e['status'] == 'complete':
            rewards[e['id']] = e['reward']
            generations[e['id']] = e['generation']
    final = [rewards[k] for k in last['final_ids']]
    assert len(final) == 5 and all(x in (0, 1) for x in final)
    return dict(question_id=first['question_id'], correct=int(sum(final)),
                root_correct=int(first['initial_reward']), attempts=last['attempts'],
                accepted=last['accepted_branches'], fills=last['fill_count'],
                max_candidate_generation=max(generations.values()),
                sha256=hashlib.sha256(raw).hexdigest())


def log_prefix_costs(path, end_outer):
    costs = Counter()
    metrics = {}
    digest = hashlib.sha256()
    actions = 0
    found = False
    with path.open() as stream:
        for line in stream:
            assert line.endswith('\n'), 'Reached unfinished log before requested boundary'
            digest.update(line.encode())
            if 'TREE_ACTION_TIMING=' in line:
                row = json.JSONDecoder().raw_decode(line.split('TREE_ACTION_TIMING=', 1)[1])[0]
                assert row['successful'], 'Failed action needs separate accounting'
                actions += 1
                for key, value in row['cost_delta'].items():
                    assert isinstance(value, (float, int)) and math.isfinite(value) and value >= 0
                    costs[key] += value
            match = re.search(r'(?:^|\s)step:(\d+) - ', line)
            if match:
                outer = int(match.group(1))
                actual = re.search(r'training/actual_outer_updates:([0-9.eE+-]+)', line)
                if actual and float(actual.group(1)) == outer:
                    values = {k: float(v) for k, v in re.findall(
                        r'([A-Za-z0-9_/.-]+):([-+0-9.eE]+|nan|inf)', line)}
                    assert outer not in metrics, 'Duplicate completed-round metrics'
                    counts = {key: values['tree_variance/' + field] for key, field in
                              [('questions', 'question_count'), ('mixed_groups', 'mixed_question_count'),
                               ('all_correct_groups', 'exact_reward_one_question_count'),
                               ('all_wrong_groups', 'exact_reward_zero_question_count')]}
                    assert all(math.isfinite(x) and x == int(x) and x >= 0 for x in counts.values())
                    counts = {key: int(value) for key, value in counts.items()}
                    assert counts['questions'] == 512
                    assert counts['mixed_groups'] + counts['all_correct_groups'] + counts['all_wrong_groups'] == 512
                    fraction = counts['mixed_groups'] / 512
                    assert abs(fraction - values['tree_variance/mixed_reward_question_fraction']) < 1e-12
                    assert math.isfinite(values['actor/pg_loss'])
                    metrics[outer] = dict(outer=outer, **counts, mixed_group_fraction=fraction,
                                          pg_loss=values['actor/pg_loss'])
                    if outer == end_outer:
                        found = True
                        break
    assert found and actions, 'Requested completed-round log boundary missing'
    assert set(metrics) == set(range(1, end_outer + 1))
    return dict(log=str(path), prefix_sha256=digest.hexdigest(), actions=actions, round_metrics=metrics,
                costs=dict(costs), scope='All recorded action deltas from process start through '
                'the requested completed-round metrics line, including screening, evaluation, '
                'padded rows and discarded emissions in that prefix. No shared-prefix token double counting. '
                'No GPU-time efficiency claim; arms use different hardware.')


def collect_arm(base, arm, job, end_outer):
    run = base / 'runs' / arm
    out = run / 'output'
    config = json.loads((run / 'receipts/configuration.json').read_text())
    completed = {}
    for p in (out / 'attempt-results').glob('attempt-*.json'):
        value = json.loads(p.read_text())
        outer = value['accepted_outer']
        if outer > end_outer:
            continue
        assert outer not in completed
        assert value['all_rank_result']['identity']['run_id'] == config['run_id']
        completed[outer] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert set(completed) == set(range(1, end_outer + 1))
    rounds = {}
    for batch in sorted((out / 'recursive-events').iterdir()):
        if not batch.is_dir():
            continue
        first_path = batch / '0/events.jsonl'
        if not first_path.exists():
            continue
        with first_path.open() as f:
            first = json.loads(f.readline())
        outer = first['outer']
        if not 1 <= outer <= end_outer:
            continue
        assert outer not in rounds, 'Multiple batches require explicit retry accounting'
        paths = sorted(batch.glob('*/events.jsonl'))
        rows = [question(p, config, outer) for p in paths]
        assert len(rows) == 512 and len({r['question_id'] for r in rows}) == 512
        compositions = Counter(r['correct'] for r in rows)
        mixed = sum(0 < r['correct'] < 5 for r in rows)
        evidence = [{'path': str(p.relative_to(out)), 'sha256': r['sha256']}
                    for p, r in zip(paths, rows)]
        rounds[outer] = dict(outer=outer, questions=512, mixed_groups=mixed,
            mixed_group_fraction=mixed / 512, root_correct=sum(r['root_correct'] for r in rows),
            final_correct=sum(r['correct'] for r in rows), final_leaves=2560,
            reward_composition=dict(sorted(compositions.items())),
            continuation_attempts=sum(r['attempts'] for r in rows),
            accepted_branches=sum(r['accepted'] for r in rows), fills=sum(r['fills'] for r in rows),
            max_candidate_generation_counts=dict(sorted(Counter(r['max_candidate_generation'] for r in rows).items())),
            batch=str(batch), attempt_sha256=completed[outer],
            ordered_event_manifest_sha256=hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest())
    ledger = log_prefix_costs(base / 'logs' / f'{arm}-gpu-r1-{job}.log', end_outer)
    metrics = ledger.pop('round_metrics')
    # Full tree events are intentionally retained only at configured milestones.
    # Cross-check those exact raw records against the all-round training metrics.
    assert set(rounds) == set(range(5, min(end_outer, 5) + 1)) | {
        i for i in (50, 100, 150, 200) if i <= end_outer}
    for outer, raw in rounds.items():
        logged = metrics[outer]
        assert logged['mixed_groups'] == raw['mixed_groups']
        assert logged['all_wrong_groups'] == raw['reward_composition'].get(0, 0)
        assert logged['all_correct_groups'] == raw['reward_composition'].get(5, 0)
    for outer, row in metrics.items():
        row['attempt_sha256'] = completed[outer]
    summary = {key: sum(r[key] for r in metrics.values()) for key in
               ('questions', 'mixed_groups', 'all_correct_groups', 'all_wrong_groups')}
    summary['mixed_group_fraction'] = summary['mixed_groups'] / summary['questions']
    return dict(arm=arm, run_id=config['run_id'], source_commit=config['source_commit'],
                config_sha256=config['config_sha256'], summary=summary,
                rounds=[metrics[i] for i in sorted(metrics)],
                raw_event_verified_rounds=[rounds[i] for i in sorted(rounds)],
                evidence_scope='All-round counts come from completed training metrics. Only listed '
                'milestones additionally have all 512 raw question event streams cross-checked.',
                recorded_costs=ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--end-outer', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.end_outer > 0
    arms = {arm: collect_arm(args.base, arm, job, args.end_outer)
            for arm, job in [('original', 138650), ('curriculum', 138651)]}
    result = dict(at=time.time(), end_outer=args.end_outer, arms=arms,
                  scope='Equal completed-round training diagnostics. Different selected questions '
                  'and diverging policies; neither a paired held-out evaluation nor a causal '
                  'estimate of selector benefit. Filtered final rewards cannot estimate independent success probability.',
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), end_outer=args.end_outer,
          arms={a: dict(summary=r['summary'], costs=r['recorded_costs']['costs']) for a, r in arms.items()})))


if __name__ == '__main__':
    main()
