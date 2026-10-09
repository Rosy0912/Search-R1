"""Strict heldout paired endpoints and actual preupdate TRAIN associations, CPU only."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
try:
    from .checkpoint import require, checked_json, checked_file, fileproof, exclusive_json, object_sha
    from .training_metrics import VARIANTS, rows_from_ref
except ImportError:
    from checkpoint import require, checked_json, checked_file, fileproof, exclusive_json, object_sha
    from training_metrics import VARIANTS, rows_from_ref


def exact_mcnemar(gained, lost):
    require(type(gained) is int and type(lost) is int and gained >= 0 and lost >= 0, 'Invalid discordance counts')
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / (1 << n))


def holm(pvalues, alpha=.05):
    require(0 < alpha < 1 and pvalues and all(math.isfinite(x) and 0 <= x <= 1 for x in pvalues.values()), 'Invalid p-values')
    ordered = sorted(pvalues, key=lambda k: (pvalues[k], k)); adjusted = {}; running = 0.
    for i, key in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - i) * pvalues[key]))
        adjusted[key] = dict(raw_p=pvalues[key], holm_p=running, reject_at_family_alpha=running <= alpha)
    return adjusted


def percentile(values, q):
    values = sorted(values); pos = (len(values) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return values[lo] * (hi - pos) + values[hi] * (pos - lo) if hi != lo else values[lo]


def paired_bootstrap(differences, datasets, repetitions=10000, seed=42):
    """One shared stratified question draw per replicate for every model/timepoint."""
    require(type(repetitions) is int and repetitions > 0, 'Invalid bootstrap repetitions')
    n = len(datasets)
    require(n > 0 and all(len(v) == n for v in differences.values()), 'Unpaired bootstrap input')
    strata = {d: [i for i, x in enumerate(datasets) if x == d] for d in sorted(set(datasets))}
    rng = random.Random(seed); draws = {key: [] for key in differences}
    for _ in range(repetitions):
        sample = [ids[rng.randrange(len(ids))] for ids in strata.values() for _ in ids]
        for key, diff in differences.items():
            draws[key].append(sum(diff[i] for i in sample) * 100 / n)
    return {key: dict(low=percentile(v, .025), high=percentile(v, .975), confidence=.95,
                     interval='single_comparison_percentile', unit='percentage_points') for key, v in draws.items()}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in order[start:end]:
            result[i] = rank
        start = end
    return result


def pearson(x, y):
    if len(x) < 2:
        return None
    mx = sum(x) / len(x); my = sum(y) / len(y)
    a = sum((v - mx) ** 2 for v in x); b = sum((v - my) ** 2 for v in y)
    if a == 0 or b == 0:
        return None
    return sum((u - mx) * (v - my) for u, v in zip(x, y)) / math.sqrt(a * b)


def auc(x, binary):
    require(all(type(y) is int and y in (0, 1) for y in binary), 'Invalid AUC labels')
    positive = sum(binary); negative = len(binary) - positive
    if positive == 0 or negative == 0:
        return None
    rr = ranks(x)
    return (sum(r for r, y in zip(rr, binary) if y) - positive * (positive + 1) / 2) / (positive * negative)


def relationships(rows):
    """Actual forward scalars vs sampled outcomes, not heldout joins or predictions."""
    features = ('gap', 'entropy', 'weight', 'raw_loss', 'post_loss', 'weighted_post_loss', 'student_to_teacher_raw_kl')
    buckets = {'pooled': list(rows)}
    for group in VARIANTS:
        selected = [r for r in rows if r['variant'] == group]
        buckets[group] = selected
        for outer in sorted({r['outer_index'] for r in selected}):
            buckets[group + '/outer-' + str(outer)] = [r for r in selected if r['outer_index'] == outer]
    result = {}
    for name, bucket in buckets.items():
        summary = {}
        for feature in features:
            complete = [r for r in bucket if r.get(feature) is not None]
            require(all(type(r[feature]) in (int, float) and math.isfinite(r[feature]) for r in complete), 'Nonfinite relation feature')
            x = [float(r[feature]) for r in complete]; em = [r['student_reward'] for r in complete]
            require(all(type(y) is int and y in (0, 1) for y in em), 'Invalid sampled student reward')
            p = [r['success_count'] / 5 for r in complete]
            require(all(type(r['success_count']) is int and 1 <= r['success_count'] <= 5 for r in complete), 'Only actual successful-reference pairs have training loss')
            summary[feature] = dict(rows=len(complete), missing=len(bucket) - len(complete),
                em_pearson=pearson(x, em), s_over_5_pearson=pearson(x, p),
                s_over_5_spearman=pearson(ranks(x), ranks(p)) if complete else None,
                original_increasing_direction_em_auc=auc(x, em),
                correct_mean=sum(v for v, y in zip(x, em) if y) / sum(em) if sum(em) else None,
                wrong_mean=sum(v for v, y in zip(x, em) if not y) / (len(em) - sum(em)) if len(em) > sum(em) else None,
                zero_variance_or_one_label_is_NA=True)
        result[name] = dict(rows=len(bucket), features=summary)
    return dict(schema='opsd-signal-weight-actual-train-relations-v1', buckets=result,
        sampling_outcomes_precede_current_preupdate_forward=True,
        population='Selected uniform D with successful B and usable encoded pair; excludes all5wrong/encoding skips',
        prediction_or_causal_claim=False, heldout_rows_joined=False, pass5_constant_in_eligible_subset=True,
        intervals_not_computed_for_exploratory_associations=True,
        units='One selected trajectory per question/group/outer, never tokens as independent observations',
        feature_scope='Gap/entropy are full-history diagnostics; privileged B contains answer information; raw model probabilities are not processed generation scores')


def analyze(records, repetitions=10000, seed=42, relation_rows=()):
    expected_keys = {('source100', 0)} | {(v, t) for v in VARIANTS for t in (15, 25)}
    require(len(records) == 4608, 'Expected4608 formal heldout predictions')
    index = {}
    for row in records:
        key = (row['version'], row['completed_outer'], row['validation_index'])
        require(key[:2] in expected_keys and type(key[2]) is int and 0 <= key[2] < 512 and key not in index,
                'Duplicate/foreign heldout endpoint slot')
        require(row.get('primary') is True and row['protocol_em'] in (0, 1) and type(row['protocol_em']) is int,
                'Invalid formal strict row')
        require(row['failure_class'] == (None if row['protocol_em'] else 'final_invalid' if not row['protocol_valid_final_action'] else 'valid_gold_mismatch'),
                'Failure classes must be mutually exclusive')
        index[key] = row
    require(set(index) == {(v, t, i) for v, t in expected_keys for i in range(512)}, 'Missing formal endpoint slots')
    base = [index['source100', 0, i] for i in range(512)]
    require(len({r['question_id'] for r in base}) == 512, 'Duplicate validation question')
    datasets = [r['dataset'] for r in base]
    require(Counter(datasets) == {'nq': 256, 'hotpotqa': 256}, 'Original fixed512 dataset mixture changed')
    for key in expected_keys:
        require(all(index[key + (i,)]['question_id'] == base[i]['question_id'] and
                    index[key + (i,)]['dataset'] == base[i]['dataset'] and
                    index[key + (i,)]['row_semantic_sha256'] == base[i]['row_semantic_sha256'] for i in range(512)),
                'Cross-model heldout identity changed')
    def counts(items):
        return dict(questions=len(items), hits=sum(x['protocol_em'] for x in items),
            em=sum(x['protocol_em'] for x in items) / len(items),
            final_invalid=sum(x['failure_class'] == 'final_invalid' for x in items),
            valid_gold_mismatch=sum(x['failure_class'] == 'valid_gold_mismatch' for x in items),
            no_executed_search=sum(x['search_calls'] == 0 for x in items), termination=dict(Counter(x['termination'] for x in items)))
    models = {}
    for version, time in sorted(expected_keys):
        items = [index[version, time, i] for i in range(512)]
        models[version + '/' + str(time)] = dict(overall=counts(items), datasets={d: counts([r for r in items if r['dataset'] == d]) for d in sorted(set(datasets))})
    # Every arrow reports RIGHT minus LEFT: orient the two primary deltas
    # exactly as the goal's G_POS-G_NEG / H_POS-H_NEG, not the reverse sign.
    primary = [('G_NEG', 25, 'G_POS', 25), ('H_NEG', 25, 'H_POS', 25)]
    comparisons = primary + [('source100', 0, v, t) for v in VARIANTS for t in (15, 25)] + [(v, 15, v, 25) for v in VARIANTS]
    pairs = {}; differences = {}; pvalues = {}
    for a, ta, b, tb in comparisons:
        name = a + '/' + str(ta) + '->' + b + '/' + str(tb)
        outcomes = [(index[a, ta, i]['protocol_em'], index[b, tb, i]['protocol_em']) for i in range(512)]
        c = Counter(outcomes); lost = [index[b, tb, i] for i, x in enumerate(outcomes) if x == (1, 0)]
        differences[name] = [y - x for x, y in outcomes]
        is_primary = (a, ta, b, tb) in primary
        pairs[name] = dict(questions=512, gained=c[0, 1], lost=c[1, 0], preserved=c[1, 1], still_wrong=c[0, 0],
            net_hits=c[0, 1] - c[1, 0], delta_percentage_points=(c[0, 1] - c[1, 0]) * 100 / 512,
            lost_final_invalid=sum(r['failure_class'] == 'final_invalid' for r in lost),
            lost_valid_gold_mismatch=sum(r['failure_class'] == 'valid_gold_mismatch' for r in lost), primary_comparison=is_primary,
            delta_direction=b+'/'+str(tb)+' minus '+a+'/'+str(ta))
        if is_primary:
            pvalues[name] = exact_mcnemar(c[0, 1], c[1, 0])
    intervals = paired_bootstrap(differences, datasets, repetitions, seed)
    adjusted = holm(pvalues)
    for name in pairs:
        pairs[name]['ci95'] = intervals[name]
        if name in adjusted:
            pairs[name]['exact_mcnemar_holm'] = adjusted[name]
    return dict(schema='opsd-signal-weight-final-effects-v1', accepted=True, formal_predictions=4608,
        models=models, paired=pairs, primary_family_size=2, family_alpha=.05, primary_time=25,
        bootstrap=dict(seed=seed, repetitions=repetitions, strata=['nq', 'hotpotqa'], sampling_unit='question_block', shared_draws_all_models_and_times=True),
        train_relations=relationships(relation_rows),
        scope='Reused training-independent512 validation, fixed seed. No E25 control, unseen-test or cross-seed improvement proof; exploratory association is not per-update causality',
        pass_sampling_not_executed=True, best_checkpoint_selection=False)


def extract_committed_relations(update_refs, config_by_variant):
    rows = []; seen = set()
    for ref in update_refs:
        value, proof = checked_json(ref)
        require(value.get('schema') == 'opsd-online-real-update-v1' and value.get('accepted') is True, 'Uncommitted/foreign update')
        variant = value.get('variant', value.get('contract', {}).get('variant')); cfg = config_by_variant[variant]
        try:
            from .checkpoint import science_contract
        except ImportError:
            from checkpoint import science_contract
        require(value['run_id'] == cfg['run_id'] and value['contract'] == science_contract(cfg)
                and value.get('config_sha256', cfg['_config_sha256']) == cfg['_config_sha256'], 'Relation update contract/run differs')
        update = value['optimizer_updates']; outer = value['outer_index']
        ranks = value['all_rank_actual_token_metrics']
        require(len(ranks) == 4 and {r['rank'] for r in ranks} == set(range(4)), 'Four rank actual metrics required')
        for rank in ranks:
            for metric in rank['pairs']:
                key = (variant, update, metric['pair_index'])
                require(key not in seen and 'signal_score' in metric and 'signal_scalar' in metric, 'Duplicate/missing current signal')
                seen.add(key); checked_json(metric['signal_score']); checked_file(metric['token_log'])
                score, _ = checked_json(metric['signal_score'])
                require(score['accepted'] is True and metric['score_training_match']['accepted'] is True,
                        'Current sibling score did not match actual forward')
                scalar = dict(metric['signal_scalar'])
                strict = scalar.get('student_reward', scalar.get('strict_reward'))
                count = scalar.get('success_count')
                if count is None:
                    fraction = scalar['success_fraction']
                    require(type(fraction) in (int,float) and math.isfinite(fraction) and 0 < fraction <= 1
                            and abs(fraction*5-round(fraction*5)) < 1e-12, 'Actual success fraction is not s/5')
                    count = round(fraction*5)
                scalar.update(student_reward=strict, success_count=count,
                              gap=score['selected_score']['answer_gap'], entropy=score['selected_score']['student_entropy'])
                scalar.update(variant=variant, outer_index=outer, optimizer_updates=update, actual_update=proof,
                    raw_loss=metric['raw_loss'], post_loss=metric['post_loss'], weighted_post_loss=metric['weighted_post_loss'],
                    student_to_teacher_raw_kl=metric['student_to_teacher_raw_kl'])
                rows.append(scalar)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True); p.add_argument('--request-sha256', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); request, rp = checked_json(dict(path=a.request, sha256=a.request_sha256))
    records = []; proofs = []
    for ref in request['evaluation_receipts']:
        receipt, ep = checked_json(ref)
        require(receipt.get('accepted') is True and receipt['primary_slots'] == 512 and receipt['actual_child_exit_and_release'] is True,
                'Evaluation has not actually completed/released')
        value, result_proof = rows_from_ref(receipt['results']); records.extend(value); proofs.extend([ep, result_proof])
    relation_rows = []
    if request.get('committed_update_refs'):
        configs = {}
        for ref in request['configs']:
            cfg, cp = checked_json(ref); cfg['_config_sha256'] = cp['sha256']; configs[cfg['variant']] = cfg
        require(set(configs) == set(VARIANTS), 'Four relation configs required')
        relation_rows = extract_committed_relations(request['committed_update_refs'], configs)
    require(relation_rows, 'Final effect report requires actual weight/loss/probability relationship inputs')
    result = analyze(records, 10000, 42, relation_rows)
    result.update(request=rp, inputs=proofs, relation_selected_trajectories=len(relation_rows))
    out = exclusive_json(a.output, result); print(json.dumps(dict(accepted=True, output=out, models=result['models'], paired=result['paired'])))


if __name__ == '__main__':
    main()
