"""Audit the full fresh common8 run; structural completion and accuracy are separate."""
import argparse
import json
import math
import re
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
from reproduction.common8.topology import ranks


def verify_five_scheduler_prefix(five, full):
    """Compare the actual complete five interval, independent of row counts."""
    selected=[row for row in full['selected_rows'] if 1<=row['logical_update']<=5]
    completed=[row for row in selected if row['kind']=='COMPLETE']
    assert [row['logical_update'] for row in completed]==list(range(1,6))
    assert all(row['opportunities_processed']==20 and 1<=row['actual_optimizer_writes']<=20 for row in completed)
    assert five['selected_rows']==selected, 'Five scheduler evidence differs from the final actual interval'
    return selected


def verify_a800_adopted_initial_probe(base, run, live, archive_path):
    """Read the genuine official initial0 adoption chain after this run trains."""
    from reproduction.a800_common import adopt_initial_probes as adoption
    from reproduction.common8.policy_calibration_plan import canonical_bytes
    base=adoption._namespace(base)
    assert re.fullmatch(r'/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(base))
    source=adoption._namespace(run['initial_probe_adoption_source_base'])
    assert source!=base and re.fullmatch(r'/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}',str(source))
    receipt=adoption._read(base/'logs/initial-probe-adoption.json',base)
    claim=adoption._read(base/'logs/initial-probe-adoption-claim.json',base)
    assert receipt['schema_version']==1 and receipt['role']=='a800_initial_probe_adoption'
    assert receipt['run_id']==run['run_id'] and receipt['execution_id']==run['execution_id']
    assert receipt['source_commit']==run['source_commit'] and receipt['version']==live['version']
    for key,value in dict(adopted=True,finite_artifacts_revalidated=True,new_calibration_generated=False,
        new_discovery_performed=False,inherited_training_state=False,production_admission=False,
        child_http_admission_required=True).items():assert receipt[key] is value
    assert type(receipt['actual_original_calibration_exit_code']) is int and receipt['actual_original_calibration_exit_code']==0
    origin=receipt['origin'];assert canonical_bytes(live['adoption_origin'])==canonical_bytes(origin)
    assert origin['base']==str(source)
    old_path=adoption.artifacts._owned_file(source/'logs/run-manifest.json',source)
    old=adoption.artifacts._read(old_path)
    assert digest(old_path)==origin['run_manifest_sha256']
    for candidate in (run,old):
        assert_formal_run(candidate)
        assert candidate['execution_backend']=='ssh_local' and candidate['training_world_size']==4
        assert all(candidate.get(k) is None for k in ('actor_checkpoint','reference_checkpoint','resume_path',
            'inherited_policy_run_id','inherited_optimizer_run_id','inherited_scheduler_run_id'))
    assert run['run_id']!=old['run_id'] and run['execution_id']!=old['execution_id']
    for key in ('initial_model','reference_model','initial_model_sha256','model_revision','input_sha256',
        'calibration_question_ids','budget','retrieval_asset_paths','retrieval_encoder'):
        assert canonical_bytes(run[key])==canonical_bytes(old[key]), 'Adopted initial0 scientific identity changed: '+key
    model=Path(run['initial_model']);assert model==model.resolve()==Path('/data/search-g1/assets/qwen2.5-3b-instruct')
    official=adoption.artifacts._owned_file(base/'preparation/official-model-manifest-original.json',base)
    assert adoption.verify_model(model,official)==run['initial_model_sha256']
    for root,candidate in ((base,run),(source,old)):
        for name,sha in candidate['input_sha256'].items():
            assert digest(adoption.artifacts._owned_file(root/'data'/name,root))==sha
    original_source_proof={}
    previous=adoption._source_map(source,old,original_source_proof)
    assert original_source_proof==origin['source_fingerprint_validation']
    # Target Hydra outputs are actual run products. Recheck every frozen source
    # byte without rebuilding its pre-training snapshot from those new products.
    current=run['source_sha256']
    for name,sha in current.items():assert digest(adoption.artifacts._owned_file(base/'repo'/name,base))==sha
    changes=[]
    for name in sorted(set(previous)|set(current)):
        if previous.get(name)==current.get(name):continue
        if name==adoption.RAY_ENTRY:
            before=adoption.artifacts._owned_file(source/'repo'/name,source).read_bytes()
            after=adoption.artifacts._owned_file(base/'repo'/name,base).read_bytes()
            assert adoption._outside_ray_bootstrap(before)==adoption._outside_ray_bootstrap(after)
        else:assert name in adoption.LOCAL_HELPERS, 'Adopted scientific source changed: '+name
        changes.append(name)
    assert receipt['allowed_startup_source_changes']==changes
    original,original_archive,heads,dump,files,job,calibration_log=adoption._original_artifacts(source,old)
    expected_origin=dict(run_id=old['run_id'],source_commit=old['source_commit'],execution_id=old['execution_id'],
        base=str(source),version=original['version'],run_manifest_sha256=digest(old_path),
        archive_sha256=digest(original_archive),calibration_job=adoption._calibration_job_projection(job),calibration_job_sha256=document_sha256(job),
        calibration_log_sha256=digest(calibration_log),head_sha256=original['head_sha256'],
        dump_sha256=original['dump_sha256'],encoder_sha256=original['encoder_sha256'],
        calibration_report_sha256=original['calibration_report_sha256'],source_fingerprint_validation=original_source_proof)
    assert canonical_bytes(origin)==canonical_bytes(expected_origin)
    version=re.fullmatch(re.escape(run['run_id'])+r':0:([0-9a-f]{32})',live['version']);assert version
    directory=base/'artifacts'/('round_0-adopted-'+version.group(1))
    archive_path=adoption.artifacts._owned_file(archive_path,base)
    assert archive_path==directory/'version-manifest.json'
    assert receipt['target_archive']==str(archive_path) and receipt['target_archive_sha256']==digest(archive_path)
    assert claim['schema_version']==1 and claim['role']=='a800_initial_probe_adoption_claim'
    assert claim['run_id']==run['run_id'] and claim['source_run_id']==old['run_id']
    assert claim['source_base']==str(source) and claim['target_round_dir']==str(directory)
    expected=dict(original,run_id=run['run_id'],version=live['version'],encoder=run['initial_model'],
        commit=str(directory/'heads/commit_probe.pkl'),boundary=str(directory/'heads/boundary_probe.pkl'),
        published_unix=live['published_unix'],adoption_origin=origin,target_source_commit=run['source_commit'],
        target_source_sha256=current,new_calibration_generated=False)
    assert canonical_bytes(live)==canonical_bytes(expected)
    adoption.calibrated_version(run,live)
    provenance=directory/'provenance'
    for name,sha in {'original-version-manifest.json':origin['archive_sha256'],
        'original-run-manifest.json':origin['run_manifest_sha256'],
        'original-calibration-job.json':document_sha256(adoption._calibration_job_projection(job)),
        'original-calibration.log':origin['calibration_log_sha256']}.items():
        assert digest(adoption.artifacts._owned_file(provenance/name,base))==sha
    for name,hashes in (('heads',dict(original['head_sha256'],**{'calibration-report.json':original['calibration_report_sha256']})),
                        ('dumps',original['dump_sha256'])):
        adoption.artifacts._files(directory/name,hashes)
    adoption.artifacts._calibration_arrays(directory/'dumps',live['calibration_report'],live['label_proof'])
    adoption.artifacts._fitted_heads(directory/'heads',live['calibration_report'])
    return dict(role='verified_a800_official_initial0_adoption',version=live['version'],
        source_run_id=old['run_id'],source_commit=old['source_commit'],
        adoption_receipt_sha256=digest(base/'logs/initial-probe-adoption.json'),
        new_calibration_generated=False,inherited_training_state=False)


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
    budgets=budget_rows(read_console_form(base,run['run_id']),1,TARGET_UPDATES,complete_records=records,run=run)
    scheduler_receipts=verify_scheduler_receipts(base,run,1,TARGET_UPDATES,behavior_gates)
    five=json.loads((base/'logs/five-updates-acceptance.json').read_text())
    assert five['accepted'] is True and five['run_id']==run['run_id']
    from reproduction.common8.formal_inline_safety import verify_plan,verify_first_five
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,inline_plan_sha=verify_plan(base,artifact,run)
    from reproduction.common8.formal_attempt_audit import verify_attempt_history
    inline_all_attempts=verify_attempt_history(base,run,records,scheduler_receipts,
                                              last=TARGET_UPDATES,plan_sha256=inline_plan_sha)
    assert five['policy_safety_mode']==run['policy_safety_mode'] and five['formal_inline_plan_sha256']==inline_plan_sha
    inline_first_five=verify_first_five(base,run,{str(step):behavior_gates[str(step)] for step in range(1,6)},
                                      verify_scheduler_receipts(base,run,1,5,behavior_gates))
    assert inline_first_five['selected_evidence_sha256']==five['formal_inline_first_five']['selected_evidence_sha256'], \
        'Actual first-five immutable attempt/state/monitor selection changed'
    from reproduction.common8.formal_inline_evaluation import verify_pair
    paired_development=verify_pair(base,run)
    assert five['paired_development']==paired_development, 'Authenticated baseline/current512 comparison changed after acceptance'
    assert [r['step'] for r in five['actual_updates']]==list(range(1,6))
    for saved,actual in zip(five['actual_updates'],records[:5]):
        assert saved['probe_version']==actual['probe_version']
        assert all(float(saved['metrics'][key])==value for key,value in actual['metrics'].items())
    for rank in ranks(run):
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
    initial_probe_adoption=None
    for round_id,live in versions.items():
        encoder=Path(live['encoder'])
        if round_id==0:
            expected_encoder={n:h for n,h in run['initial_model_sha256'].items()
                              if n not in ('vocab.json','merges.txt','generation_config.json')}
            assert encoder.resolve()==Path(run['initial_model']).resolve()
            archive_path=archived_paths[live['version']]
            adopted_initial='adoption_origin' in live
            if adopted_initial:
                initial_probe_adoption=verify_a800_adopted_initial_probe(base,run,live,archive_path)
            else:
                assert archive_path.parent.name.startswith('round_0-attempt-'), 'Formal initial0 is freshly fitted in this namespace'
            initial_probe=json.loads((base/'logs/formal-inline-initial-probe.json').read_text())
            assert initial_probe['schema_version']==1 and initial_probe['role']=='fresh200_actual_initial_probe'
            assert initial_probe['run_id']==run['run_id'] and initial_probe['source_map_sha256']==document_sha256(run['source_sha256'])
            assert initial_probe['version']==live['version']==run['initial_probe_version']
            assert initial_probe['model_revision']==run['model_revision'] and initial_probe['initial_model_sha256']==run['initial_model_sha256']
            assert initial_probe['input_sha256']==run['input_sha256']
            assert initial_probe['archive']==str(archive_path.relative_to(base)) and initial_probe['archive_sha256']==digest(archive_path)
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
        formal_inline_first_five=inline_first_five,formal_inline_all_attempts=inline_all_attempts,
        paired_development=paired_development,
        final_correct=correct,final_weighted_em=evaluation['weighted_em'],
        final_source_em={s:correct[s]/counts[s] for s in counts},
        final_source_f1={s:math.fsum(source_f1[s])/counts[s] for s in counts},
        final_weighted_f1=evaluation['weighted_f1'],
        semantics='full training/data/numerical structure verified; accuracy is the measured result, not automatically reproduction success',
        checked_unix=time.time())
    if initial_probe_adoption is not None:proof['initial_probe_adoption']=initial_probe_adoption
    from reproduction.common8.calibration_runtime import write_once
    write_once(base/'logs/full-run-acceptance.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(verify_full(a.base)))
