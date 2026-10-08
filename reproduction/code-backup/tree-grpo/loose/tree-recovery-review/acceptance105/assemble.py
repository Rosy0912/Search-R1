from pathlib import Path
import json,hashlib,re,math
p=Path(__file__).parent
load=lambda f:json.loads((p/f).read_text())
state=load('FULLSTATE_CPU_PROOF.json');hf=load('HF105_BINDING_CPU_PROOF.json');guard=load('LIVE_GUARD_FINAL.json');ready=state['ready']
assert state['status']=='PASS_INDEPENDENT_SERIALIZED_FULLSTATE105_CPU_ONLY' and hf['status']=='PASS_HF105_FILE_BYTES_AND_CPU_REPORT_BINDING' and guard['status']=='PASS_LIVE_GUARD_PREFIX_ONLY'
assert state['ready']==hf['ready'] and state['sampler_order_sha256']=='3e0101af98996d517a4a344de01199d6e91fd9cb433a88b3e1ad5481d7d9748a'
identity=state['manifest']['identity'];assert all(ready[k]==v for k,v in identity.items())
proofs=[json.loads(Path('work/tree-recovery-review/prefix101102/REMOTE_CPU_PROOF.json').read_text()),load('RAW103104_CPU_PROOF.json'),load('RAW105_CPU_PROOF.json')]
profiles=[json.loads(Path('work/tree-recovery-review/prefix101102/PROFILES_CPU_PROOF.json').read_text()),load('PROFILES103104_CPU_PROOF.json'),load('PROFILES105_CPU_PROOF.json')]
records=[x for b in proofs for x in b['records']];assert [x['step'] for x in records]==list(range(101,106))
assert all(x['mean_reward']>=.1 for x in records)
assert sum(x['ratio_microbatches'] for x in records)==9600
assert sum(len(b['records']) for b in profiles)==40
metrics=[]
for line in guard['actual_outer_log_lines']:
 m={k:float(v) for k,v in re.findall(r'([\w/]+):([-+\deE.]+)',line)}
 assert all(math.isfinite(v) for v in m.values()) and m['actor/optimizer_steps']==10 and m['actor/lr_before_outer']==m['actor/lr']==1e-6
 metrics.append(m)
assert [int(m['training/actual_outer_updates']) for m in metrics]==list(range(101,106))
slurmfields=dict(re.findall(r'(\S+?)=(\S+)',guard['slurm']));assert slurmfields['JobId']=='3234' and slurmfields['JobState']=='RUNNING' and slurmfields['NodeList']==slurmfields['BatchHost']=='h100-9' and slurmfields['Restarts']=='0'
evidence={}
for b in proofs+[state,hf]:
 for item in b['evidence']:
  assert item['path'] not in evidence or evidence[item['path']]['sha256']==item['sha256'];evidence[item['path']]=item
for b in profiles:
 for item in b['records']:evidence[item['path']]={k:item[k] for k in ['path','sha256']}
readysha=next(x['sha256'] for x in hf['evidence'] if x['path'].endswith('/ready.json'))
restored=Path('work/tree-recovery-review/restored512/RESTORED512_RECEIPT.json');restoredvalue=json.loads(restored.read_text());assert restoredvalue['restored_fixed512_quality_accepted'] is True
r=dict(status='PASS',stage='recovery105',scope='Independent actual recovery101..105 acceptance for same live job3234 release106..300 only; not final300/51713 acceptance or final successful Slurm cleanup.',**{k:identity[k] for k in ['job_id','code_commit','config_sha256','data_identity','model_identity']},ready_sha256=readysha,completed_outer_updates=105,next_outer_update=106,actual_update_limit=300,recovery_parent_completed_step=100,recovery_parent_manifest_sha256='c20922f74f7dc77e7a233fcdcf329f2914bfaff335ed34d02209e1e6d09a3e86',actual_recovery_outer_updates=5,optimizer_calls_per_rank_per_outer=10,full_state_reload_accepted=True,full_hf_reload_accepted=True,rank_tree_numerics_accepted=True,complete_trajectory_logging_accepted=True,restored_fixed512_quality_accepted=True,full_state_manifest=ready['full_state_manifest'],full_state_manifest_sha256=ready['full_state_manifest_sha256'],hf_checkpoint=ready['hf_checkpoint'],hf_identity_sha256=ready['hf_identity']['sha256'],direct_rank_records=40,memory_profile_records=40,tree_audit_files=10,ratio_tail_records=400,ratio_microbatches=9600,complete_retained_leaf_trajectory_rows=19200,actual_full_actor_input_payloads_cpu_loaded=5,sampler_cursor=54400,sampler_epoch=0,sampler_order_sha256=state['sampler_order_sha256'],adam_steps_per_rank=2450,scheduler_last_epoch=105,lr=1e-6,records=records,slurm_live_identity=slurmfields,guard_snapshot_epoch=guard['epoch'],guard_samples=guard['samples'],guard_foreign_memory_max=guard['foreign_memory_max'],guard_foreign_sm_max=guard['foreign_sm_max'],all_original_background_pid_start_identities_preserved=True,protected_processes=guard['protected_proc'],live_guard_proof_sha256=hashlib.sha256((p/'LIVE_GUARD_FINAL.json').read_bytes()).hexdigest(),restored512_receipt_sha256=hashlib.sha256(restored.read_bytes()).hexdigest(),restored512_f1=restoredvalue['f1_all'],restored512_em=restoredvalue['em_all'],hf_cpu_changed_tensors=hf['hf_report']['changed_tensors'],hf_cpu_changed_parameters=hf['hf_report']['changed_parameters'],hf_cpu_forward_finite=True,fullstate_cpu_cost={k:state[k] for k in ['wall_seconds','process_cpu_seconds','peak_rss_kib','torch_threads','cuda_initialized']},training_outer_wall_seconds=sum(m['timing_s/step'] for m in metrics),training_generated_nonpad_token_events=sum(int(m['rollout/generated_tokens']) for m in metrics),training_prefill_token_events=sum(int(m['rollout/prefill_tokens']) for m in metrics),evidence=list(evidence.values()),independent_proof_sha256={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['RAW103104_CPU_PROOF.json','RAW105_CPU_PROOF.json','PROFILES103104_CPU_PROOF.json','PROFILES105_CPU_PROOF.json','FULLSTATE_CPU_PROOF.json','HF105_BINDING_CPU_PROOF.json','LIVE_GUARD_FINAL.json']},approved_file_published_by_reviewer=False,actual_recovery300_accepted=False,limits=['Complete serialized checkpoint floating state finite; per-outer numeric evidence covers existing global norm plus8192 fixed coordinates, not all live parameters at every optimizer.','HF105 full CPU reload/report and file bytes accepted; same-live object fullstate reload receipts accepted. Not exact original executor replay after batch/RNG migration.','Raw trajectories preserve retained all leaves and preoptimizer inputs/tool text; shared prefix action/search counts repeat across leaves, HTTP JSON scores are outside logger scope.','Small finite logratio tails >2 exist and are counted; no claim historical collapse root cause or future stability explained.','Resource snapshot is live samejob gate evidence; no final job exit or cleanup claim. Outer timing sum excludes startup, initial512 and paused independent-review waiting.'])
(p/'ACTUAL105_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS_RECEIPT_SHA256',hashlib.sha256((p/'ACTUAL105_RECEIPT.json').read_bytes()).hexdigest())
