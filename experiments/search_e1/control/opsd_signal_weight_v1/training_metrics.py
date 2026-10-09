"""Complete, unfiltered sampling-preupdate TRAIN EM; no model imports."""
import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
try:
    from .checkpoint import require, object_sha, checked_json, checked_file, fileproof, exclusive_json
    from .scoring import verify_scoring
except ImportError:
    from checkpoint import require, object_sha, checked_json, checked_file, fileproof, exclusive_json
    from scoring import verify_scoring

VARIANTS = ('G_POS', 'G_NEG', 'H_POS', 'H_NEG')


def rows_from_ref(ref):
    proof = checked_file(ref)
    raw = Path(proof['path']).read_bytes()
    require(raw.endswith(b'\n'), 'Incomplete JSONL input')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(fileproof(proof['path']) == proof, 'Input changed while reading rows')
    return rows, proof


def protocol(core, qa, result, answers):
    spans = result['policy_spans']
    response = result['response']
    require(isinstance(response, str), 'Response must remain complete native text')
    end = 0
    for a, b in spans:
        require(type(a) is int and type(b) is int and end <= a <= b <= len(response), 'Invalid authored policy spans')
        end = b
    last = response[spans[-1][0]:spans[-1][1]] if spans else None
    answer = qa.extract_final_answer([last] if last is not None else [])
    reward = core.reward(response, answers, spans)
    require(type(reward) in (int, float) and reward in (0, 1), 'Strict scorer is not binary')
    hit = int(reward)
    require(not hit or answer is not None, 'Strict hit without valid final action')
    return dict(protocol_em=hit, protocol_valid_final_action=answer is not None,
                failure_class=None if hit else 'final_invalid' if answer is None else 'valid_gold_mismatch',
                extracted_final_answer=answer)


def calculate(rows, questions, variant, outer_index, sampling_model_identity, core):
    """The score denominator is 2560, never P/K or successful-reference eligibility."""
    require(variant in VARIANTS and type(outer_index) is int and 0 <= outer_index < 25, 'Wrong group/outer')
    require(isinstance(sampling_model_identity, dict) and len(sampling_model_identity.get('sha256', '')) == 64,
            'Actual outer-start sampling model identity required')
    rows = list(rows); questions = list(questions)
    expected = {q['question_id']: q for q in questions}
    require(len(questions) == len(expected) == 512 and len(rows) == 2560, 'Full512x5 required, not partial/mined data')
    slots = {}; groups = {}
    for row in rows:
        qid = row['question_id']; sibling = row['sibling']
        require(qid in expected and type(sibling) is int and 0 <= sibling < 5 and (qid, sibling) not in slots,
                'Duplicate/foreign/invalid TRAIN slot')
        q = expected[qid]
        require(row['question'] == q['question'] and row['answers'] == q['answers'], 'TRAIN question/gold changed')
        require(type(row['reward']) is int and row['reward'] in (0, 1), 'Stored reward is not integer strict EM')
        value = core.reward(row['response'], row['answers'], row['policy_spans'])
        require(value in (0, 1) and row['reward'] == value, 'Stored reward differs from frozen strict scorer')
        require(type(row['search_calls']) is int and 0 <= row['search_calls'] <= 4, 'Invalid executed search count')
        slots[qid, sibling] = row; groups.setdefault(qid, []).append(row)
    require(set(slots) == {(q, s) for q in expected for s in range(5)}, 'Missing TRAIN question/sibling')
    def counts(items):
        hits = sum(r['reward'] for r in items)
        return dict(trajectories=len(items), strict_correct=hits, denominator=len(items), em=hits / len(items),
                    no_executed_search=sum(r['search_calls'] == 0 for r in items),
                    termination=dict(Counter(r.get('termination', 'not_recorded') for r in items)))
    per_question = []
    for q in questions:
        g = sorted(groups[q['question_id']], key=lambda r: r['sibling'])
        success = sum(r['reward'] for r in g)
        per_question.append(dict(question_id=q['question_id'], dataset=q['data_source'], success_count=success,
            s_over_5=success / 5, rewards=[r['reward'] for r in g], row_sha256=[object_sha(r) for r in g]))
    datasets = {}
    for dataset in sorted({q['data_source'] for q in questions}):
        qids = {q['question_id'] for q in questions if q['data_source'] == dataset}
        datasets[dataset] = dict(counts([r for r in rows if r['question_id'] in qids]), questions=len(qids))
    overall = counts(rows)
    require(overall['denominator'] == 2560 and abs(overall['em'] - sum(q['s_over_5'] for q in per_question) / 512) < 1e-12,
            'Trajectory EM and equal-question s/5 differ')
    return dict(schema='opsd-signal-weight-train-em-v1', accepted=True, variant=variant, outer_index=outer_index,
        displayed_iteration=outer_index + 1, sampling_preupdate_raw_model=sampling_model_identity,
        measurement_time='outer_start_sampling_before_any_this_outer_update',
        overall=overall, datasets=datasets, questions=512, unique_slots=2560,
        all5_wrong=sum(q['success_count'] == 0 for q in per_question),
        all5_correct=sum(q['success_count'] == 5 for q in per_question), per_question=per_question,
        rollouts_semantic_sha256=object_sha(rows), cohort_semantic_sha256=object_sha(questions),
        eligible_pairs_not_used_as_denominator=True, post_training_em=False,
        curve_scope='Different TRAIN questions per outer; this is not fixed-heldout or postupdate performance')


def complete_outer_summary(sampling_summary, completion):
    """One equal-pair mean over ALL committed updates; zero-K is explicit NA."""
    require(sampling_summary['accepted'] is True and completion['accepted'] is True and
            sampling_summary['outer_index'] == completion['outer_index'], 'Sampling/training outer changed')
    K=completion['K']; P=completion['P']
    require(type(K) is int and type(P) is int and 0 <= K <= P <= 512 and
            completion['optimizer_updates_delta'] == math.ceil(K/32), 'Outer K/Adam accounting differs')
    cp, cp_proof=checked_json(completion['checkpoint']); refs=cp['outer_update_receipts']
    require(len(refs)==math.ceil(K/32) and cp['cursor']['next_pair_cursor']==cp['cursor']['usable_pair_count']==K,
            'Raw checkpoint does not commit all actual outer updates')
    metrics=[]; updates=[]; norms=[]; seen=set()
    for ref in refs:
        value, proof=checked_json(ref)
        require(value['schema']=='opsd-online-real-update-v1' and value['accepted'] is True and
                value['outer_index']==completion['outer_index'] and value['run_id']==completion['run_id'], 'Update outside current committed outer')
        ranks=value['all_rank_actual_token_metrics']
        require(len(ranks)==4 and {r['rank'] for r in ranks}==set(range(4)), 'Missing actual four-rank metrics')
        pairs=[p for r in ranks for p in r['pairs']]
        require(len(pairs)==value['global_actual_pairs'] and 1 <= len(pairs) <= 32, 'Actual batch/tail denominator differs')
        for metric in pairs:
            require(metric['pair_index'] not in seen, 'Repeated committed pair exposure')
            seen.add(metric['pair_index']);checked_file(metric['token_log'])
            require(metric['valid_tokens']>0 and all(math.isfinite(metric[k]) for k in ('raw_loss','post_loss','weighted_post_loss','pair_weight')),
                    'Nonfinite/empty actual pair metrics')
        metrics.extend(pairs);updates.append(value['optimizer_updates']);norms.append(value['scaler']['grad_norm'])
    require(len(metrics)==K and (not updates or updates==list(range(completion['optimizer_updates']-len(updates)+1,completion['optimizer_updates']+1))),
            'Outer real update/exposure chain is discontinuous')
    weights=[m['pair_weight'] for m in metrics];total=sum(weights);squares=sum(w*w for w in weights)
    def mean(name):return sum(m[name] for m in metrics)/K if K else None
    return dict(schema='opsd-signal-outer-summary-v1',accepted=True,variant=sampling_summary['variant'],
        outer_index=completion['outer_index'],displayed_iteration=completion['outer_index']+1,
        train_em=sampling_summary['overall']['em'],train_correct=sampling_summary['overall']['strict_correct'],
        train_denominator=2560,train_datasets=sampling_summary['datasets'],all5_wrong=sampling_summary['all5_wrong'],
        train_sampling_preupdate_model=sampling_summary['sampling_preupdate_raw_model'],train_post_update=False,
        P=P,K=K,optimizer_updates_delta=completion['optimizer_updates_delta'],optimizer_updates=completion['optimizer_updates'],
        cumulative_pair_exposure=completion['cumulative_pair_exposure'],raw_checkpoint=cp_proof,
        actual_outer_unweighted_post_loss=mean('post_loss'),actual_outer_weighted_post_loss=mean('weighted_post_loss'),
        actual_outer_raw_loss=mean('raw_loss'),loss_mean_unit='equal_selected_trajectory_over_all_committed_outer_batches',
        weight=dict(sum=total,sumsq=squares,mean=total/K if K else None,
                    ess=total*total/squares if squares else None,min=min(weights) if weights else None,max=max(weights) if weights else None),
        actual_preclip_grad_norms=norms,global_valid_tokens=sum(m['valid_tokens'] for m in metrics),
        empty_outer_no_adam=K==0,source_update_refs=refs)


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'rollouts', 'rollouts-sha256', 'questions', 'questions-sha256',
                 'sampling-model', 'sampling-model-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--outer-index', type=int, required=True); a = p.parse_args()
    cfg, cp = checked_json(dict(path=a.config, sha256=a.config_sha256))
    proof = verify_scoring(cfg); core = importlib.import_module('search_e1.core')
    rows, rp = rows_from_ref(dict(path=a.rollouts, sha256=a.rollouts_sha256))
    questions, qp = rows_from_ref(dict(path=a.questions, sha256=a.questions_sha256))
    model, mp = checked_json(dict(path=a.sampling_model, sha256=a.sampling_model_sha256))
    result = calculate(rows, questions, cfg['variant'], a.outer_index, model, core)
    result.update(config=cp, rollouts=rp, cohort=qp, sampling_model=mp, scoring=proof, run_id=cfg['run_id'])
    out = exclusive_json(a.output, result)
    print(json.dumps(dict(accepted=True, output=out, em=result['overall']['em'], strict_correct=result['overall']['strict_correct'])))


if __name__ == '__main__':
    main()
