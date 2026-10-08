"""Local seven-group curves from already accepted committed scalar facts.

No Node, model, optimizer, scoring, sampling, or training operations.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
METRICS = ('raw_loss', 'post_loss', 'weighted_post_loss',
           'student_to_teacher_raw_kl', 'student_to_outer_old_raw_kl',
           'student_to_mixture_raw_kl')


def ref(p):
    raw = p.read_bytes()
    return dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def observed_mean(field, k):
    assert field['expected_trajectories'] == k
    observed = field['observed_trajectories']
    assert type(observed) is int and 0 <= observed <= k
    if field['value'] is None:
        return None
    assert observed == k and k > 0 and math.isfinite(field['value'])
    return field['value']


def derive(facts):
    assert facts['schema'] == 'opsd-ablation-outer-facts-v1' and facts['accepted']
    index = {(o['variant'], o['outer']): o for o in facts['outers']}
    assert len(index) == len(facts['outers']) == 70
    assert set(index) == {(v, i) for v in VARIANTS for i in range(10)}
    rows, groups, seen = [], {}, set()
    for v in VARIANTS:
        finished = [index[v, i] for i in range(10) if index[v, i]['status'] == 'COMPLETE_FACTS']
        assert [o['outer'] for o in finished] == list(range(len(finished)))
        cumulative_tokens = 0
        previous_update = previous_k = 0
        for i in range(10):
            o = index[v, i]
            row = dict(variant=v, outer_index=i, outer_number=i + 1, status=o['status'])
            if o['status'] == 'COMPLETE_FACTS':
                c, s, m = o['counts'], o['sampling'], o['metrics']
                k = c['K']
                assert c['optimizer_updates_before'] == previous_update
                assert c['cumulative_pair_exposure'] == previous_k + k
                assert c['optimizer_updates_delta'] == len(o['updates']) == math.ceil(k / 32)
                assert sum(u['global_actual_pairs'] for u in o['updates']) == k
                assert sum(u['global_valid_tokens'] for u in o['updates']) == c['valid_tokens']
                for u in o['updates']:
                    ident = (v, u['update'])
                    assert ident not in seen
                    seen.add(ident)
                assert [u['update'] for u in o['updates']] == list(range(previous_update + 1, c['optimizer_updates'] + 1))
                assert m['trajectory_denominator'] == k
                assert s['questions'] == 512 and s['trajectories'] == 2560
                contract = o['contract']
                assert contract['variant'] == v and contract['direction'] == 'student_to_target_reverse'
                assert contract['mixture_alpha'] == (.1 if v in ('M', 'J') else 0.)
                cumulative_tokens += c['valid_tokens']
                norms = [u['preclip_grad_norm'] for u in o['updates']]
                norm_complete = bool(norms) and all(n is not None and math.isfinite(n) for n in norms)
                row.update(run_id=o['run_id'], precision=o['precision'], mixture_alpha=contract['mixture_alpha'],
                    direction=contract['direction'], train_questions=512, sampled_trajectories=2560,
                    train_strict_reward_hits=s['strict_reward_hits'], train_strict_reward=s['strict_reward_mean'],
                    eligible_P=c['P'], usable_K=k, skipped=sum(c['skipped'].values()),
                    skip_detail=json.dumps(c['skipped'], sort_keys=True),
                    all_wrong_questions=s['classes']['all_wrong'], all_correct_questions=s['classes']['all_correct'],
                    mixed_questions=s['classes']['mixed'], eligible_CC_pairs=s['pair_classes']['CC'],
                    eligible_CW_pairs=s['pair_classes']['CW'], sibling_self_pair=s['self_pair'],
                    response_text_equal=s['response_text_equal'], eligible_pair_weight_sum=s['eligible_pair_weight_sum'],
                    actual_weighted_pair_exposure=m['weighted_pair_exposure'], actual_weight_coverage=m['weight_coverage'],
                    actual_adam_before=c['optimizer_updates_before'], actual_adam_this_outer=c['optimizer_updates_delta'],
                    actual_adam_cumulative=c['optimizer_updates'], cumulative_pair_exposure=c['cumulative_pair_exposure'],
                    valid_policy_tokens=c['valid_tokens'], cumulative_valid_policy_tokens=cumulative_tokens,
                    clipped_vocabulary_count=m['clipped_vocabulary_count'], clipped_count_coverage=m['clipped_count_coverage'],
                    clipped_probability_mass=m['clipped_probability_mass'], clipped_vocabulary_fraction=m['clipped_vocabulary_fraction'],
                    preclip_grad_norm_min=min(norms) if norm_complete else None,
                    preclip_grad_norm_max=max(norms) if norm_complete else None,
                    gradnorm_recorded_update_count=sum(n is not None for n in norms),
                    committed_overflow_count=sum(u['committed_overflow'] for u in o['updates'])
                        if all(u['committed_overflow'] is not None for u in o['updates']) else None,
                    committed_recorded_retries=sum(u['recorded_retries'] for u in o['updates'])
                        if all(u['recorded_retries'] is not None for u in o['updates']) else None,
                    checkpoint_manifest_sha256=o['sources']['checkpoint']['sha256'])
                for name in METRICS:
                    field = m['equal_trajectory_means'][name]
                    row[name] = observed_mean(field, k)
                    row[name + '_trajectory_coverage'] = field['observed_trajectories']
                previous_update, previous_k = c['optimizer_updates'], c['cumulative_pair_exposure']
            else:
                assert o['status'] == 'NOT_COMPLETE'
            row['scope'] = 'Different TRAIN cohort per outer; not fixed heldout512. Native committed facts, not tensor or historical-retry audit.'
            rows.append(row)
        completed = [r for r in rows if r['variant'] == v and r['status'] == 'COMPLETE_FACTS']
        ksum = sum(r['usable_K'] for r in completed)
        means = {}
        for name in METRICS:
            coverage = sum(r[name + '_trajectory_coverage'] for r in completed)
            values_complete = all(r[name] is not None or r['usable_K'] == 0 for r in completed)
            means[name] = dict(value=sum((r[name] or 0) * r['usable_K'] for r in completed) / ksum
                if ksum and coverage == ksum and values_complete else None,
                observed_trajectories=coverage, expected_trajectories=ksum)
        groups[v] = dict(completed_outers=len(completed), expected_outers=10, complete=len(completed) == 10,
            actual_adam=previous_update, raw_pair_exposure=ksum, valid_tokens=cumulative_tokens,
            actual_weighted_pair_exposure=sum(r['actual_weighted_pair_exposure'] for r in completed)
                if all(r['actual_weighted_pair_exposure'] is not None for r in completed) and completed else None,
            equal_pair_means=means, precision=completed[0]['precision'] if completed else None,
            mixture_alpha=.1 if v in ('M', 'J') else 0.)
    assert sum(g['completed_outers'] for g in groups.values()) == facts['coverage']['completed_outers']
    assert len(seen) == facts['totals']['committed_unique_updates']
    assert sum(g['raw_pair_exposure'] for g in groups.values()) == facts['totals']['raw_pair_exposure']
    assert sum(g['valid_tokens'] for g in groups.values()) == facts['totals']['valid_tokens']
    return rows, groups


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = [('train_strict_reward', 'Strict TRAIN reward (different cohorts)'),
              ('usable_K', 'Usable pairs K'), ('raw_loss', 'Raw reverse KL to target'),
              ('weighted_post_loss', 'Weighted clipped objective'),
              ('student_to_outer_old_raw_kl', 'Student-to-frozen-outer-old KL'),
              ('actual_adam_cumulative', 'Actual cumulative Adam updates')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (field, title) in zip(axes.flat, fields):
        for v in VARIANTS:
            rs = [r for r in rows if r['variant'] == v and r['status'] == 'COMPLETE_FACTS']
            if not rs:
                continue
            ys = [float('nan') if r.get(field) is None else r[field] for r in rs]
            ax.plot([r['outer_number'] for r in rs], ys, marker='o', markersize=4,
                    linestyle='-' if len(rs) == 10 else '--', label=v)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xticks(range(1, 11)); ax.set_xlim(.7, 10.3)
        ax.set_xlabel('Online outer (1-10)'); ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=4)
    fig.suptitle('Seven OPSD groups: committed TRAIN records', x=.055, ha='left', fontsize=16)
    fig.text(.055, .92, 'TRAIN cohorts differ by outer. These are not held-out512 effectiveness curves. Dashed = incomplete group.', fontsize=10)
    fig.text(.055, .02, 'Loss: token mean within trajectory, then actual pair mean. Weighted objective is not normalized by weight sum. Missing = NA.', fontsize=9)
    fig.tight_layout(rect=(.03, .055, 1, .90))
    paths = []
    for suffix in ('png', 'svg'):
        p = out / ('seven-group-recorded-curves.' + suffix)
        fig.savefig(p, dpi=160, facecolor='white'); paths.append(p)
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--facts', required=True); p.add_argument('--facts-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); src = Path(a.facts); raw = src.read_bytes()
    source = dict(path=str(src), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert source['sha256'] == a.facts_sha256
    facts = json.loads(raw); rows, groups = derive(facts)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    csvfile = out / 'seven-group-70-slots.csv'
    with csvfile.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = out / 'seven-group-recorded-summary.json'
    summary.write_text(json.dumps(dict(accepted=True, groups=groups, coverage=facts['coverage'], totals=facts['totals'],
        source=source, source_configs=facts['configs'], fixed_heldout_curve=False,
        all70_producer_records_complete=facts['coverage']['all70_complete'],
        training_exit_or_effect_acceptance=False, scope='Only recorded committed scalars; pending slots stay blank. No rescoring or numerical tensor re-audit.'),
        indent=2, sort_keys=True, allow_nan=False) + '\n')
    plots = plot(rows, out)
    manifest = out / 'derived-manifest.json'
    manifest.write_text(json.dumps(dict(schema='seven-group-recorded-curves-v1', accepted=True,
        source=source, source_code=ref(Path(__file__)), coverage=facts['coverage'],
        totals=facts['totals'], artifacts=[ref(q) for q in [csvfile, summary] + plots],
        no_remote_io=True, no_new_scientific_test=True, no_model_or_gpu=True, observed_unix=time.time()),
        indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(accepted=True, coverage=facts['coverage'], totals=facts['totals'], manifest=ref(manifest))))


if __name__ == '__main__':
    main()
