"""Read-only final-candidate review with small fabricated CPU bundles."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import torch

HERE=Path(__file__).resolve().parent
CANDIDATE=HERE/'candidate-28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac.py'
spec=importlib.util.spec_from_file_location('final_candidate',CANDIDATE)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def tensors(*, raw=False, old=-2., sampled=-2., weight=1., rows=1):
    out=dict(input_ids=torch.tensor([[1,2,3]]).repeat(rows,1),responses=torch.tensor([[3]]).repeat(rows,1),
             position_ids=torch.tensor([[0,1,2]]).repeat(rows,1),attention_mask=torch.ones(rows,3,dtype=torch.int64),
             old_log_probs=torch.full((rows,1),old,dtype=torch.float64))
    if raw:
        out.update(sampled_log_probs=torch.full((rows,1),sampled,dtype=torch.float64),distribution={'temperature':1.})
    else:out.update(loss_mask=torch.ones(rows,1,dtype=torch.int64),policy_weights=torch.full((rows,1),weight,dtype=torch.float64))
    return out

def fails(fn,message=None):
    try:fn()
    except (ValueError,KeyError,RuntimeError) as exc:
        if message:assert message in str(exc),(message,str(exc))
        return str(exc)
    raise AssertionError('Invalid fixture accepted')

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return A.file_sha(path)

checks=[]
# Same-context final probability subtraction of finite extremes is invalid,
# explicitly counted and never reported as a numeric policy comparison.
idx=A.RawIndex();idx.add_snapshot(tensors(raw=True,old=-1.7e308,sampled=-1.7e308),'extreme-raw')
r=A.analyze_targets(idx,[('extreme-final',tensors(old=1.7e308))])
assert r['finite_numeric_evidence']['invalid_sampled_difference_occurrences']==1
assert r['finite_numeric_evidence']['invalid_actor_difference_occurrences']==1
assert r['finite_numeric_evidence']['matched_tokens_with_finite_raw_sampled_and_final_old']==0
assert r['matched_examples'][0]['final_old_minus_raw_sampled_range'] is None
json.dumps(r,allow_nan=False)
checks.append('finite_operand_difference_overflow_counted_null_and_json_serializable')

# Raw paired subtraction overflow is also recorded as invalid before aggregating.
idx=A.RawIndex();idx.add_snapshot(tensors(raw=True,old=1.7e308,sampled=-1.7e308),'raw-pair-overflow')
assert idx.report()['raw_same_turn_actor_minus_sampled']['invalid_paired_occurrences']==1
assert idx.report()['raw_same_turn_actor_minus_sampled']['occurrence_mean_signed'] is None
json.dumps(idx.report(),allow_nan=False)
checks.append('raw_same_turn_difference_overflow_counted_as_invalid_pair')

# Valid individual weights can sum to infinity; no fake zero coverage/mean.
idx=A.RawIndex();idx.add_snapshot(tensors(raw=True),'raw-normal')
r=A.analyze_targets(idx,[('sum-overflow',tensors(old=-3.,weight=1.7e308,rows=2))])
assert 'total_weight' in r['finite_numeric_evidence']['nonfinite_aggregate_fields']
assert r['tree_weights']['weighted_exact_context_coverage'] is None
assert r['final_leaf_old_minus_actual_raw_sampled']['matched_weight_mean_signed_bounds']==[None,None]
json.dumps(r,allow_nan=False)
checks.append('aggregate_weight_and_numerator_overflow_null_no_infinite_denominator_zero')

with tempfile.TemporaryDirectory(prefix='final-context-review-',dir=HERE) as temporary:
    root=Path(temporary);run=root/'tree-safety300-999999';package_dir=root/'package'
    ready_path=run/'output/probability-calibration/ready.json'
    cfg=run/'resolved.yaml';cfg.parent.mkdir(parents=True);cfg.write_text('fixture config')
    entry_cfg=run/'output/entry-audit/entry-config.yaml';entry_cfg.parent.mkdir(parents=True);entry_cfg.write_bytes(cfg.read_bytes())
    parent=root/'parent.json';parent_identity={'completed_step':110};parent_sha=write_json(parent,{'identity':parent_identity})
    package=dict(stage='calibration110',probability_calibration={'environment':{'TREE_PPO_CALIBRATION_ONLY':'1'}},
                 code_commit='a'*40,common_data_identity='d'*64,model_identity={'sha256':'e'*64},files={},
                 parent_manifest={'path':str(parent),'sha256':parent_sha,'identity':parent_identity})
    for name in A.SOURCE_FILES:
        path=package_dir/'source/repo'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture source')
        package['files']['source/repo/'+name]=A.file_sha(path)
    package_sha=write_json(package_dir/'manifest.json',package)
    identity=dict(job_id='999999',run_id=run.name,code_commit='a'*40,config_sha256=A.file_sha(cfg),data_identity='d'*64,model_identity='e'*64)
    ready=dict(status='CALIBRATION_OBSERVED_ZERO_WRITES',parent_completed_outer=110,observed_outer=111,new_completed_outer=0,
               optimizer_committed=0,scheduler_advanced=False,workers=[],**identity)
    recovery_path=run/'output/recovery-loaded.json'
    write_json(recovery_path,{'provenance':{'parent_identity':parent_identity,'parent_cursor':57600,'exact_executor_resume':False}})
    for rank in range(8):
        raw_path=run/'output/sampling-consistency'/f'rank-{rank:02d}-call-00001.pt';raw_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(tensors(raw=True),raw_path)
        record_path=raw_path.with_suffix('.json')
        record=dict(rank=rank,world_size=8,scheduler_step=110,optimizer_committed=0,saved_rows=1,local_rows=1,
                    enforced=False,generation_call=1,snapshot_path=str(raw_path),snapshot_sha256=A.file_sha(raw_path),
                    distribution={'temperature':1.},decision={'allowed':True},**identity)
        record_sha=write_json(record_path,record)
        monitor_path=run/'output/probability-calibration'/f'rank-{rank:02d}-monitor.pt';monitor_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(tensors(rows=32),monitor_path)
        ready['workers'].append(dict(rank=rank,world_size=8,optimizer_committed=0,scheduler_advanced=False,scheduler_step=110,
            restored_optimizer_steps=[2500],diagnostic_records=[str(record_path)],monitor_path=str(monitor_path),monitor_sha256=A.file_sha(monitor_path),
            diagnostic_entries=[dict(path=str(record_path),sha256=record_sha,snapshot_path=str(raw_path),snapshot_sha256=A.file_sha(raw_path))],**identity))
    ready_sha=write_json(ready_path,ready)
    actor_path=run/'output/trajectories/outer-000111/actor-inputs.pt';actor_path.parent.mkdir(parents=True)
    actor_identity=dict(outer_step=111,job_id='999999',source=identity['code_commit'],config=identity['config_sha256'],data=identity['data_identity'])
    torch.save(dict(identity=actor_identity,rows=3840,tensors=tensors(rows=3840)),actor_path)
    actor_manifest=actor_path.parent/'manifest.json'
    write_json(actor_manifest,dict(status='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED',identity=actor_identity,rows=3840,payloads={'actor-inputs.pt':A.file_sha(actor_path)}))
    def analyze():
        verified=A.verify_bundle(run,package_dir,package_sha,ready_sha,require_node=False)
        return A.analyze_bundle(*verified,package_sha,ready_sha,2)
    r=analyze();assert r['actual_optimizer_writes']==r['actual_forward_passes']==0
    assert r['complete_actor_inputs']['leaf_rows_denominator']==3840
    assert r['monitor_subset']['leaf_rows_denominator']==256
    assert next(e for e in r['evidence'] if e['kind']=='complete_final_actor_inputs')['manifest_sha256']==A.file_sha(actor_manifest)
    assert next(e for e in r['diagnostic_source_files'] if e.get('kind')=='recovery_loaded_metadata')['sha256']==A.file_sha(recovery_path)
    checks.append('complete_eight_rank_bundle_fixed_actor_manifest_and_recovery_hashes_retained')
    worker=ready['workers'][0];entry=worker['diagnostic_entries'][0];raw_path=Path(entry['snapshot_path']);record_path=Path(entry['path'])
    raw_original=raw_path.read_bytes();record_original=record_path.read_bytes();original_entry=copy.deepcopy(entry)
    mismatched=tensors(raw=True);mismatched['distribution']={'temperature':.5};torch.save(mismatched,raw_path)
    entry['snapshot_sha256']=A.file_sha(raw_path);record=json.loads(record_path.read_bytes());record['snapshot_sha256']=entry['snapshot_sha256']
    entry['sha256']=write_json(record_path,record);ready_sha=write_json(ready_path,ready)
    fails(analyze,'Raw snapshot sampling distribution differs')
    checks.append('sha_valid_snapshot_distribution_mismatch_with_json_refused')
    raw_path.write_bytes(raw_original);record_path.write_bytes(record_original);entry.update(original_entry);ready_sha=write_json(ready_path,ready)
    # Refuse exactly hash-bound but wrong actual rank, rather than relying solely on filenames.
    record=json.loads(record_path.read_bytes());record['rank']=1;entry['sha256']=write_json(record_path,record);ready_sha=write_json(ready_path,ready)
    fails(analyze,'Raw diagnostic identity changed')
    checks.append('sha_valid_wrong_rank_record_refused')
    record_path.write_bytes(record_original);entry.update(original_entry);ready_sha=write_json(ready_path,ready)
    # An extra independent record file cannot silently be dropped from accounting.
    extra=record_path.with_name('rank-00-call-00002.json');extra.write_bytes(record_original)
    fails(analyze,'Raw diagnostic file set differs')
    checks.append('extra_unpinned_raw_file_refused')
    extra.unlink()
    # Actual config mismatch with a self-consistent ready cannot be accepted.
    entry_cfg.write_text('changed fixture config')
    fails(analyze,'Actual config identity changed')
    checks.append('entry_config_mismatch_refused')
    entry_cfg.write_bytes(cfg.read_bytes())
    # New actor optimizer writes disqualify this zero-write evidence scope.
    audit=run/'output/tree-rank-audit/step-000111-rank-00.json';audit.parent.mkdir(parents=True);audit.write_text('{}')
    fails(analyze,'actual actor optimizer audit files')
    checks.append('actual_optimizer_audit_file_disqualifies_zero_write_scope')
    audit.unlink()
    assert analyze()['complete_actor_inputs']['exact_context_matched_token_occurrences']==3840
    checks.append('final_restored_fixture_still_exact_context_complete')

result=dict(status='FINAL_CONTEXT_HELPER_INDEPENDENT_CPU_PASS',candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    checks=checks,check_count=len(checks),actual_model_assets_loaded=False,remote_calls=False,cuda_initialized=torch.cuda.is_initialized(),
    scope='Fabricated CPU tensors and metadata only; source review permits read-only same-node CPU analysis, no policy/training acceptance')
print(json.dumps(result,sort_keys=True,allow_nan=False))
