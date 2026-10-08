"""Audit the full fresh common8 run; structural completion and accuracy are separate."""
import argparse
import json
import math
import time
from pathlib import Path
from collections import Counter
from reproduction.common8.runtime import budget_rows, verify_scheduler_receipts
from reproduction.common8.worker_evidence import read_console_form
from reproduction.common8.verify_data import digest, verify
from reproduction.common8.validation_identity import expected
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.formal_contract import (
    TARGET_UPDATES, SAVE_STEPS, PROBE_ROUNDS, FORMAL_COUNTING, assert_formal_run, verify_formal_config,
)


def verify_five_scheduler_prefix(five, full):
    """Compare the actual complete five interval, independent of row counts."""
    selected=[row for row in full['selected_rows'] if 1<=row['logical_update']<=5]
    completed=[row for row in selected if row['kind']=='COMPLETE']
    assert [row['logical_update'] for row in completed]==list(range(1,6))
    assert all(row['opportunities_processed']==20 and 1<=row['actual_optimizer_writes']<=20 for row in completed)
    assert five['selected_rows']==selected, 'Five scheduler evidence differs from the final actual interval'
    return selected


def verify_full(base):
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base=Path(base);run=json.loads((base/'logs/run-manifest.json').read_text())
    assert_formal_run(run)
    resolved=json.loads((base/'logs/resolved-training-config.json').read_text())
    assert resolved['run_id']==run['run_id']
    config_proof=verify_formal_config(resolved['resolved_config'],run,base)
    assert resolved['resolved_config_sha256']==config_proof['resolved_config_sha256']
    data=verify(base/'data')
    records=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==list(range(1,TARGET_UPDATES+1))
    assert all(r['run_id']==run['run_id'] and all(math.isfinite(v) for v in r['metrics'].values()) for r in records)
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    behavior_gates = {str(r['step']): read_gate_receipt(base, run, r['step'], update_record=r) for r in records}
    verify_write_receipt_links(base, behavior_gates, 1, TARGET_UPDATES)
    budgets=budget_rows(read_console_form(base,run['run_id']),1,TARGET_UPDATES,complete_records=records)
    scheduler_receipts=verify_scheduler_receipts(base,run,1,TARGET_UPDATES,behavior_gates)
    five=json.loads((base/'logs/five-updates-acceptance.json').read_text())
    assert five['accepted'] is True and five['run_id']==run['run_id']
    from reproduction.common8.formal_inline_safety import verify_plan,verify_first_five
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,inline_plan_sha=verify_plan(base,artifact,run)
    assert five['policy_safety_mode']==run['policy_safety_mode'] and five['formal_inline_plan_sha256']==inline_plan_sha
    inline_first_five=verify_first_five(base,run,{str(step):behavior_gates[str(step)] for step in range(1,6)},
                                      verify_scheduler_receipts(base,run,1,5,behavior_gates))
    assert inline_first_five['selected_evidence_sha256']==five['formal_inline_first_five']['selected_evidence_sha256'], \
        'Actual first-five immutable attempt/state/monitor selection changed'
    from reproduction.common8.formal_inline_evaluation import verify_pair
    paired_development=verify_pair(base,run)
    assert five['paired_development']==paired_development, 'Actual paired5120/5 measurements changed after acceptance'
    assert [r['step'] for r in five['actual_updates']]==list(range(1,6))
    for saved,actual in zip(five['actual_updates'],records[:5]):
        assert saved['probe_version']==actual['probe_version']
        assert all(float(saved['metrics'][key])==value for key,value in actual['metrics'].items())
    for rank in range(8):
        verify_five_scheduler_prefix(five['scheduler_receipts'][str(rank)],scheduler_receipts[str(rank)])
    snapshots={}
    for step in SAVE_STEPS:
        checkpoint=Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
        proof=json.loads((checkpoint/'complete.json').read_text())
        assert proof['run_id']==run['run_id'] and proof['step']==step and proof['finite_all_saved_tensors'] is True
        assert Path(proof['checkpoint']).resolve()==checkpoint.resolve()
        assert all(digest(checkpoint/name)==sha for name,sha in proof['files_sha256'].items())
        snapshots[str(step)]=proof
    final_checkpoint=Path(snapshots[str(TARGET_UPDATES)]['checkpoint'])
    count=validate_saved_checkpoint(final_checkpoint,Path(run['initial_model']),run['started_unix'])
    assert all(digest(Path(run['initial_model'])/name)==sha for name,sha in run['initial_model_sha256'].items())
    # Match every frozen evaluator interval to its genuinely archived calibration.
    archived={};archived_paths={}
    paths=list((base/'artifacts').glob('round_*-attempt-*/version-manifest.json'))
    paths+=list((base/'artifacts').glob('round_0-adopted-*/version-manifest.json'))
    for path in paths:
        live=json.loads(path.read_text())
        if live['run_id']==run['run_id']:
            assert live['version'] not in archived, 'Duplicate archived UUID'
            archived[live['version']]=live
            archived_paths[live['version']]=path
    versions={}
    for round_id in PROBE_ROUNDS:
        adopted={r['probe_version'] for r in records if round_id<r['step']<=round_id+50}
        assert len(adopted)==1
        live=archived[adopted.pop()];assert live['round']==round_id
        versions[round_id]=live
    for round_id,live in versions.items():
        encoder=Path(live['encoder'])
        if round_id==0:
            expected_encoder={n:h for n,h in run['initial_model_sha256'].items()
                              if n not in ('vocab.json','merges.txt','generation_config.json')}
            assert encoder.resolve()==Path(run['initial_model']).resolve()
            archive_path=archived_paths[live['version']]
            assert archive_path.parent.name.startswith('round_0-attempt-'), 'Formal initial0 is freshly fitted in this namespace'
            initial_probe=json.loads((base/'logs/formal-inline-initial-probe.json').read_text())
            assert initial_probe['schema_version']==1 and initial_probe['role']=='fresh200_actual_initial_probe'
            assert initial_probe['run_id']==run['run_id'] and initial_probe['source_map_sha256']==document_sha256(run['source_sha256'])
            assert initial_probe['version']==live['version']==run['initial_probe_version']
            assert initial_probe['model_revision']==run['model_revision'] and initial_probe['initial_model_sha256']==run['initial_model_sha256']
            assert initial_probe['input_sha256']==run['input_sha256']
            assert initial_probe['archive']==str(archive_path.relative_to(base)) and initial_probe['archive_sha256']==digest(archive_path)
            assert 'adoption_origin' not in live, 'Failed old diagnostic state must not be inherited'
        else:
            snapshot=snapshots[str(round_id)]
            expected_encoder=dict(snapshot['files_sha256'])
            expected_encoder.update({n:run['initial_model_sha256'][n] for n in ('tokenizer.json','tokenizer_config.json')})
            assert encoder.resolve()==Path(snapshot['checkpoint']).resolve()
        assert live['encoder_sha256']==expected_encoder
        assert all(digest(encoder/name)==sha for name,sha in live['encoder_sha256'].items())
        heads=Path(live['commit']).parent;dump=heads.parent/'dumps'
        assert all(digest(heads/name)==sha for name,sha in live['head_sha256'].items())
        assert digest(heads/'calibration-report.json')==live['calibration_report_sha256']
        assert {p.name:digest(p) for p in dump.iterdir() if p.is_file()}==live['dump_sha256']
        assert json.loads((heads/'calibration-report.json').read_text())==live['calibration_report']
        assert json.loads((dump/'calibration-label-proof.json').read_text())==live['label_proof']
        assert set(live['label_proof']['question_ids'])==set(run['calibration_question_ids'])
        report=live['calibration_report']
        assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
        assert all(math.isfinite(report[key+'_cv_auc']) and 0<=report[key+'_cv_auc']<=1 for key in ('commit','boundary'))
    for r in records:
        round_id=((r['step']-1)//50)*50
        assert r['probe_version']==versions[round_id]['version']
    evaluation=json.loads((base/'logs/final-identity.json').read_text())
    assert evaluation['run_id']==run['run_id'] and evaluation['step']==TARGET_UPDATES and evaluation['final'] is True
    canonical,frame,ids=expected(base,'final')
    assert evaluation['validation_sha256']==canonical['sha256']
    assert evaluation['question_ids']==ids and evaluation['data_sources']==frame['data_source'].tolist()
    scores=evaluation['em'];assert len(scores)==51713 and all(x in (0,1) for x in scores)
    f1=evaluation['f1'];assert len(f1)==51713 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in f1)
    assert all(em<=score for em,score in zip(scores,f1))
    assert evaluation['weighted_f1']==math.fsum(f1)/len(f1)
    assert evaluation['counts']==canonical['source_counts']
    assert evaluation['correct']==sum(scores) and evaluation['weighted_em']==sum(scores)/len(scores)
    counts=Counter(evaluation['data_sources']);correct={s:0 for s in counts}
    assert len(counts)==7 and counts['nq']==3610
    source_f1={s:[] for s in counts}
    for source,score in zip(evaluation['data_sources'],scores):correct[source]+=score
    for source,score in zip(evaluation['data_sources'],f1):source_f1[source].append(score)
    proof=dict(run_id=run['run_id'],contract_kind=run['contract_kind'],structural_accepted=True,actual_outer_updates=TARGET_UPDATES,
        optimizer_completion_rule=FORMAL_COUNTING,per_rank_optimizer_opportunities=TARGET_UPDATES*20,
        per_rank_actual_optimizer_writes={rank:receipt['cumulative_actual_optimizer_writes']
                                        for rank,receipt in scheduler_receipts.items()},
        final_saved_tensor_count=count,
        shared_data=data,checkpoint_proofs=snapshots,final_counts=dict(counts),
        behavior_probability_gates=behavior_gates,
        scheduler_receipts=scheduler_receipts,formal_config=config_proof,
        policy_safety_mode=run['policy_safety_mode'],formal_inline_plan_sha256=inline_plan_sha,
        formal_inline_first_five=inline_first_five,paired_development=paired_development,
        final_correct=correct,final_weighted_em=evaluation['weighted_em'],
        final_source_em={s:correct[s]/counts[s] for s in counts},
        final_source_f1={s:math.fsum(source_f1[s])/counts[s] for s in counts},
        final_weighted_f1=evaluation['weighted_f1'],
        semantics='full training/data/numerical structure verified; accuracy is the measured result, not automatically reproduction success',
        checked_unix=time.time())
    with (base/'logs/full-run-acceptance.json').open('x') as f:json.dump(proof,f,indent=2)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(verify_full(a.base)))
