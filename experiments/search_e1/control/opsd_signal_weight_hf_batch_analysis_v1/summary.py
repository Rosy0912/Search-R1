"""Question-level strict EM/probability associations with explicit coverage."""
import argparse
from collections import Counter
import json
import math
try:
    from .common import require, read_json, read_rows, exclusive_json, ENDPOINTS
    from .replay_math import FEATURES
except ImportError:
    from common import require, read_json, read_rows, exclusive_json, ENDPOINTS
    from replay_math import FEATURES


def ranks(values):
    indices = sorted(range(len(values)), key=lambda i: values[i]); result = [0.] * len(values)
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and values[indices[start]] == values[indices[end]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in indices[start:end]: result[i] = rank
        start = end
    return result


def pearson(x, y):
    if len(x) < 2 or len(x) != len(y): return None
    mx = math.fsum(x) / len(x); my = math.fsum(y) / len(y)
    dx = [v - mx for v in x]; dy = [v - my for v in y]
    xx = math.fsum(v*v for v in dx); yy = math.fsum(v*v for v in dy)
    return None if xx == 0 or yy == 0 else math.fsum(a*b for a, b in zip(dx, dy)) / math.sqrt(xx * yy)


def residualize(values, covariates):
    """Stable Gram-Schmidt projection; collinear/constant controls are dropped.

    An exploratory linear adjustment, not logistic success probabilities and
    not proof that conditioning removes all behavioral confounding.
    """
    n = len(values); basis = []; dropped = []
    columns = [('intercept', [1.] * n)] + list(covariates.items())
    for name, original in columns:
        require(len(original) == n, 'Control length differs')
        v = [float(x) for x in original]
        for _ in range(2):
            for q in basis:
                a = math.fsum(x*y for x, y in zip(v, q)); v = [x-a*y for x, y in zip(v, q)]
        norm = math.sqrt(math.fsum(x*x for x in v))
        if norm <= 1e-10 * max(1., math.sqrt(math.fsum(float(x)**2 for x in original))):
            dropped.append(name)
        else:
            basis.append([x/norm for x in v])
    residual = [float(x) for x in values]
    for q in basis:
        a = math.fsum(x*y for x, y in zip(residual, q)); residual = [x-a*y for x, y in zip(residual, q)]
    return residual, dropped


def association(rows, feature):
    available = [r for r in rows if r['replay']['metrics'][feature] is not None]
    x = [r['replay']['metrics'][feature] for r in available]; y = [r['protocol_em'] for r in available]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in x)
        and all(type(v) is int and v in (0, 1) for v in y), 'Association needs finite features and binary strict EM')
    missing = Counter(r['replay']['missing_reasons'][feature] for r in rows if r['replay']['metrics'][feature] is None)
    reason = ('fewer_than_two_available_questions' if len(x) < 2 else 'only_one_EM_class_in_metric_subset'
        if len(set(y)) < 2 else 'zero_feature_variance' if len(set(x)) < 2 else None)
    positives = sum(y); negatives = len(y) - positives
    auc = None if reason else (math.fsum(v for v, label in zip(ranks(x), y) if label)
        - positives*(positives+1)/2) / (positives * negatives)
    controls = dict(log1p_valid_policy_tokens=[math.log1p(r['replay']['valid_policy_tokens']) for r in available],
        executed_search_calls=[r['search_calls'] for r in available],
        dataset_hotpot=[int(r['dataset'] == 'hotpotqa') for r in available])
    partial = None; dropped = []
    if reason is None:
        rx, dropped = residualize(x, controls); ry, _ = residualize(y, controls)
        # Floating residuals of exactly explained labels/features are not data.
        if math.fsum(v*v for v in rx) > 1e-18 and math.fsum(v*v for v in ry) > 1e-18:
            partial = pearson(rx, ry)
    return dict(questions=len(rows), metric_available=len(x), metric_unavailable=len(rows)-len(x),
        missing_reasons=dict(missing), correct_available=positives, wrong_available=negatives,
        subset_em=positives / len(y) if y else None,
        pearson=None if reason else pearson(x, y),
        spearman=None if reason else pearson(ranks(x), ranks(y)),
        original_increasing_direction_EM_AUC=auc, association_NA_reason=reason,
        correct_mean=math.fsum(v for v, label in zip(x, y) if label) / positives if positives else None,
        wrong_mean=math.fsum(v for v, label in zip(x, y) if not label) / negatives if negatives else None,
        adjusted_partial_pearson=partial, controls=list(controls), dropped_collinear_controls=dropped,
        adjusted_scope='Exploratory linear residual association; no causal or calibrated probability claim',
        association_intervals_or_pvalues_not_claimed=True)


def counts(rows):
    require(rows, 'Empty population counts')
    return dict(questions=len(rows), hits=sum(r['protocol_em'] for r in rows),
        em=sum(r['protocol_em'] for r in rows) / len(rows),
        final_invalid=sum(r['failure_class'] == 'final_invalid' for r in rows),
        valid_gold_mismatch=sum(r['failure_class'] == 'valid_gold_mismatch' for r in rows),
        no_executed_search=sum(r['search_calls'] == 0 for r in rows))


def regional_summary(rows):
    result = {}
    for metric in ('raw_teacher_reverse_kl', 'post_clip_teacher_loss', 'source100_ordinary_reverse_kl', 'student_entropy'):
        available = [r for r in rows if r['replay']['metrics'][metric] is not None]
        if not available:
            result[metric] = dict(available=0, regions={}, total_equal_question_mean=None); continue
        labels = sorted({label for r in available for label in r['replay']['regions']})
        contributions = {label: math.fsum(r['replay']['regions'].get(label, {}).get('contributions', {})
            .get(metric, {}).get('trajectory_mean_contribution', 0.) for r in available) / len(available) for label in labels}
        total = math.fsum(r['replay']['metrics'][metric] for r in available) / len(available)
        require(math.isclose(math.fsum(contributions.values()), total, rel_tol=2e-5, abs_tol=2e-6),
            'Region contributions do not reconstruct full effective trajectory mean')
        result[metric] = dict(available=len(available), total_equal_question_mean=total,
            regions={label: dict(equal_question_mean_contribution=value,
                fraction_of_signed_total=value/total if total != 0 else None) for label, value in contributions.items()},
            absent_region_contribution_zero=True, unavailable_whole_metric_not_zero=True,
            contribution_not_gradient_or_causality=True)
    return result


def paired_cases(left, right):
    require([r['validation_index'] for r in left] == [r['validation_index'] for r in right]
        and all(a.get('question_id') == b.get('question_id') and a['dataset'] == b['dataset']
            and a.get('row_semantic_sha256') == b.get('row_semantic_sha256') for a,b in zip(left,right)), 'Unpaired fixed questions')
    outcome = Counter((a['protocol_em'], b['protocol_em']) for a, b in zip(left, right))
    strata = {'lost': (1, 0), 'gained': (0, 1), 'preserved_correct': (1, 1), 'still_wrong': (0, 0)}
    groups = {}
    for name, labels in strata.items():
        selected = [(a, b) for a, b in zip(left, right) if (a['protocol_em'], b['protocol_em']) == labels]
        changes = {}
        for feature in FEATURES:
            complete = [(a['replay']['metrics'][feature], b['replay']['metrics'][feature]) for a, b in selected
                if a['replay']['metrics'][feature] is not None and b['replay']['metrics'][feature] is not None]
            changes[feature] = dict(paired_metric_available=len(complete), missing=len(selected)-len(complete),
                mean_difference_right_minus_left=math.fsum(b-a for a, b in complete) / len(complete) if complete else None)
        groups[name] = dict(questions=len(selected), metric_changes=changes)
    datasets={}
    for dataset in sorted({r['dataset'] for r in left}):
        selected=[(a,b) for a,b in zip(left,right) if a['dataset']==dataset]
        counts=Counter((a['protocol_em'],b['protocol_em']) for a,b in selected)
        datasets[dataset]=dict(questions=len(selected),left_hits=sum(a['protocol_em'] for a,b in selected),
            right_hits=sum(b['protocol_em'] for a,b in selected),gained=counts[0,1],lost=counts[1,0],
            preserved_correct=counts[1,1],still_wrong=counts[0,0],
            delta_percentage_points=(counts[0,1]-counts[1,0])*100/len(selected))
    return dict(questions=len(left),left_hits=sum(r['protocol_em'] for r in left),right_hits=sum(r['protocol_em'] for r in right),
        gained=outcome[0, 1], lost=outcome[1, 0], preserved_correct=outcome[1, 1], still_wrong=outcome[0, 0],
        delta_percentage_points=(outcome[0,1]-outcome[1,0])*100/len(left) if left else None,datasets=datasets,
        outcome_groups=groups, unit='same question; each checkpoint own saved autonomous response',
        different_response_tokens_not_position_paired=True, changed_histories_are_endogenous=True)


def summarize(records, bank, endpoint_identities=None):
    require(len(records) == 4608 and bank.get('accepted') is True and bank['questions'] == 512,
        'Final replay relationship report requires all9 endpoints/all512 EM')
    index = {}; bank_index = {e['validation_index']: e for e in bank['entries']}
    require(set(bank_index) == set(range(512)), 'Bank coverage incomplete')
    for row in records:
        key = (row['version'], row['completed_outer'], row['validation_index'])
        require(key[:2] in ENDPOINTS and key[2] in range(512) and key not in index,
            'Duplicate/foreign replay endpoint slot')
        b = bank_index[key[2]]
        require(row['question_id'] == b['question_id'] and row['dataset'] == b['dataset']
            and row['row_semantic_sha256'] == b['validation_row_semantic_sha256']
            and row['bank_available'] is b['available'], 'Fixed bank/replay question identity differs')
        require(type(row['protocol_em']) is int and row['protocol_em'] in (0, 1)
            and row['failure_class'] == (None if row['protocol_em'] else 'final_invalid'
                if not row['protocol_valid_final_action'] else 'valid_gold_mismatch'), 'Strict outcome classes differ')
        require(row['replay']['original_training_objective'] is False and row['replay']['actual_generation_scores'] is False,
            'Posthoc metrics must not masquerade as online forward/generation scores')
        index[key] = row
    require(set(index) == {(v, t, i) for v, t in ENDPOINTS for i in range(512)}, 'Incomplete final replay endpoints')
    if endpoint_identities is not None:
        require(set(endpoint_identities)=={v+'/'+str(t) for v,t in ENDPOINTS},'All9 exact endpoint checkpoint identities required')
        for version,t in ENDPOINTS:
            identity=endpoint_identities[version+'/'+str(t)]
            require(identity['version']==version and identity['completed_outer']==t
                and all(index[version,t,i]['current_adapter_tensor_sha256']==identity['adapter_tensor_sha256'] for i in range(512)),
                'Actual replay adapter digest differs from exact saved checkpoint identity')
    base = [index['source100', 0, i] for i in range(512)]
    require(all(r['protocol_em'] == bank_index[i]['source_zero_em'] for i, r in enumerate(base)), 'Common-zero EM/bank changed')
    models = {}
    for version, t in sorted(ENDPOINTS):
        rows = [index[version, t, i] for i in range(512)]; eligible = [r for r in rows if r['bank_available']]
        models[version + '/' + str(t)] = dict(all512=counts(rows),
            datasets={d: counts([r for r in rows if r['dataset'] == d]) for d in ('nq', 'hotpotqa')},
            bank_subset=counts(eligible) if eligible else None, bank_coverage=len(eligible),
            features={feature: association(rows, feature) for feature in FEATURES},
            dataset_feature_relations={d: {f: association([r for r in rows if r['dataset'] == d], f) for f in FEATURES}
                for d in ('nq', 'hotpotqa')}, regions=regional_summary(rows),
            checkpoint_identity=endpoint_identities.get(version+'/'+str(t)) if endpoint_identities is not None else None)
    paired = {}
    for v, t in sorted(ENDPOINTS - {('source100', 0)}):
        paired['source100/0->' + v + '/' + str(t)] = paired_cases(base, [index[v, t, i] for i in range(512)])
    for v in ('G_POS', 'G_NEG', 'H_POS', 'H_NEG'):
        paired[v + '/15->' + v + '/25'] = paired_cases([index[v, 15, i] for i in range(512)], [index[v, 25, i] for i in range(512)])
    cross={}
    for t in (15,25):
        for left in ('G_POS','G_NEG'):
            for right in ('H_POS','H_NEG'):
                key=left+'/'+str(t)+'->'+right+'/'+str(t)
                pair_result=paired_cases([index[left,t,i] for i in range(512)],[index[right,t,i] for i in range(512)])
                require(pair_result['questions']==512 and {d:v['questions'] for d,v in pair_result['datasets'].items()}=={'nq':256,'hotpotqa':256},
                    'Cross-signal descriptive pair cannot filter by feature availability')
                cross[key]=dict(pair_result,descriptive_only=True,no_significance_test_or_new_family=True,
                    delta_direction=right+'/'+str(t)+' minus '+left+'/'+str(t),
                    left_checkpoint_identity=models[left+'/'+str(t)]['checkpoint_identity'],
                    right_checkpoint_identity=models[right+'/'+str(t)]['checkpoint_identity'])
    return dict(schema='opsd-signal-heldout-response-relations-v1', accepted=True, formal_EM_rows=4608,
        models=models, paired=paired, descriptive_cross_signal_pairs=cross, new_samples_or_training=0, pass_probability_not_measured=True,
        reference_bank_gold_selected=True, baseline_bank_subset_only_one_class_NA=True,
        all512_EM_not_filtered_by_metric_coverage=True,
        scope='Fixed seed reused validation. Privileged posthoc conditional associations are not calibrated success probabilities, training gradients, or causal effects. No E25 control.',
        no_fine_grained_degradation_time_inference=True)


def validate_endpoint_identity(endpoint, replay_request, endpoint_ref):
    version=endpoint['version'];t=endpoint['completed_outer']
    require((version,t) in ENDPOINTS and endpoint['accepted'] is True
        and endpoint['schema']=='opsd-signal-batched-canonical-endpoint-v1'
        and endpoint['primary_slots']==512 and endpoint['actual_worker_wait_returncodes']==[0]*8
        and replay_request['endpoint']['sha256']==endpoint_ref['sha256']
        and replay_request['input_identity']['expected_adapter_tensor_sha256']==endpoint['input_identity']['expected_adapter_tensor_sha256']
        and replay_request['input_identity']['raw_checkpoint']==endpoint['raw_checkpoint'],
        'Replay exact batched endpoint identity changed')
    return version+'/'+str(t)


def main():
    p = argparse.ArgumentParser()
    for name in ('request', 'request-sha256', 'output'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); request, rp = read_json(dict(path=a.request, sha256=a.request_sha256)); bank, bp = read_json(request['bank'])
    rows = []; inputs = []; identities={}
    for ref in request['replay_receipts']:
        value, proof = read_json(ref)
        require(value.get('accepted') is True and value['actual_workers_exit_and_release'] is True
            and value['bank']['sha256'] == bp['sha256'], 'Replay complete/release/bank proof missing')
        endpoint,ep=read_json(value['endpoint']);replay_request,request_proof=read_json(value['request'])
        version=endpoint['version'];t=endpoint['completed_outer'];key=version+'/'+str(t)
        require(key not in identities and validate_endpoint_identity(endpoint,replay_request,ep)==key, 'Duplicate endpoint')
        identities[key]=dict(version=version,completed_outer=t,evaluation=ep,raw_checkpoint=endpoint['raw_checkpoint'],
            adapter_tensor_sha256=endpoint['input_identity']['expected_adapter_tensor_sha256'],
            source_model_manifest=endpoint['input_identity']['source_model_manifest'],
            runtime_manifest=endpoint['input_identity']['runtime']['manifest'],validation=endpoint['input_identity']['validation'])
        rr, rr_proof = read_rows(value['results']); rows.extend(rr); inputs.extend([proof, rr_proof])
    result = summarize(rows, bank,identities); result.update(request=rp, bank=bp, inputs=inputs)
    output = exclusive_json(a.output, result)
    print(json.dumps(dict(accepted=True, output=output, formal_EM_rows=4608)))


if __name__ == '__main__': main()
