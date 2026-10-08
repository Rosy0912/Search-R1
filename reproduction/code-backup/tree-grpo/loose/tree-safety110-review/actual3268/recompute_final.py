import json,hashlib,math
from pathlib import Path
O=Path('work/tree-safety110-review/actual3268');v=json.loads((O/'FINAL_METADATA.json').read_bytes());p=json.loads((O/'POLL8.json').read_bytes());f=v['files'];h=lambda s:hashlib.sha256(s.encode()).hexdigest()
for x in f.values():assert h(x['text'])==x['sha256']
r=json.loads(f['run-result.json']['text']);cost=json.loads(f['cost.json']['text']);assert r['status']=='CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED' and r['training_body_exit_code']==0 and r['entry_config_verified'] is True and r['final_guard_healthy'] is True and r['cleanup_residual']=={} and r['original_eight_pid_start_times_preserved'] is True
assert (r['parent_completed_outer'],r['observed_outer'],r['new_completed_outer'],r['optimizer_committed'],r['scheduler_advanced'])==(110,111,0,0,False)
assert cost['exit_code']==0 and cost['allocated_gpus']==8 and cost['allocated_cpus']==192 and cost['requested_cpus']==32 and cost['end_epoch']>cost['start_epoch']
s=p['slurm']['stdout'];assert p['slurm']['rc']==0 and 'JobId=3268 ' in s and 'JobState=COMPLETED' in s and 'ExitCode=0:0' in s and 'Restarts=0' in s and 'NodeList=h100-9' in s and 'BatchHost=h100-9' in s
samples=[json.loads(x) for x in f['resource-samples.jsonl']['text'].splitlines()];assert samples and all(x['job_id']=='3268' and not x.get('event') and len(x['gpus'])==8 for x in samples)
for x in samples:
 for g in x['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
protected={**{str(x):'1502352181' for x in (1680782,1680783,1680784,1680785)},**{str(x):'1502398762' for x in (1696443,1696444,1696445,1696446)}}
for pid,start in protected.items():
 fields=v['protected'][pid].rsplit(') ',1)[1].split();assert fields[19]==start and fields[0] not in ('Z','X')
assert {x.split(',')[1].strip() for x in v['gpu_processes_now'].splitlines()}==set(protected)
assert all(int(x.split(',')[2])==1131 and int(x.split(',')[4])==0 for x in v['gpu_now'].splitlines())
assert not any(':18082' in x or ':16382' in x for x in v['ports'].splitlines())
counts=v['output_file_counts'];assert counts['direct_rank']==counts['ratio']==counts['checkpoints']==0 and counts['trajectory_manifests']==1 and counts['tree_audits']==2 and counts['monitor_pt']==8 and counts['sampling_json']==64
ready=json.loads(p['files']['output/probability-calibration/ready.json']['text']);assert len(ready['workers'])==8
common=ready['workers'][0]['decision'];assert all(w['decision']==common and w['optimizer_committed']==0 and w['restored_optimizer_steps']==[2500] and w['scheduler_step']==110 for w in ready['workers']);assert not common['allowed'] and 'initial_old_current_probability_inconsistency' in common['reasons']
out=dict(status='ACTUAL_TERMINAL_ZERO_WRITE_METADATA_PASS_PROBABILITY_DIFFERENCES_NOT_ACCEPTED',job='3268',slurm_state='COMPLETED',slurm_exit='0:0',slurm_wall_seconds=1181,slurm_requested_cpu=32,slurm_allocated_cpu=192,allocated_gpu_hours=1181*8/3600,allocated_cpu_hours=1181*192/3600,supervisor_wall_seconds=cost['end_epoch']-cost['start_epoch'],guard_samples=len(samples),last_guard_epoch=samples[-1]['epoch'],sampled_gpu_peak_GiB=max(g['total_used_memory_bytes'] for x in samples for g in x['gpus'])/2**30,foreign_mem_fraction_peak=max(g['foreign_memory_fraction'] for x in samples for g in x['gpus']),foreign_sm_peak=max(g['foreign_sm_upper'] for x in samples for g in x['gpus']),background_starttimes=protected,ports_closed=True,artifact_counts=counts,ready_sha256=p['files']['output/probability-calibration/ready.json']['sha256'],final_monitor_global_decision=common,actual_new_completed_outers=0,actual_3b_optimizer_writes=0,actual_3b_training_accepted=False,long_training_ready=False,complete_raw_PT_and_trajectory_CPU_proofs_pending=True)
(O/'TERMINAL_METADATA_PROOF.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
