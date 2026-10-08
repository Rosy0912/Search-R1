"""Observe the one saved discovery handle; no GPU kernels or job relaunch."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

PREP = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control


def remote(host, code, payload):
    result = subprocess.run(control.route(host, 'python3 -c ' + shlex.quote(code)),
                            input=json.dumps(payload), capture_output=True, text=True, timeout=55)
    assert result.returncode == 0, 'Observation failed; unknown is not stopped: ' + result.stderr[-2000:]
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--observe', action='store_true')
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    launch = json.loads((PREP / 'policy-discovery-v3-native-launch.json').read_text())
    observation = remote('h100-8', r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);job=p['job_id']
s=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
out=dict(base=str(b),parent_base=p['parent_base'],job_id=job,checked_unix=time.time(),job_stdout=s.stdout,job_stderr=s.stderr,job_returncode=s.returncode,actual_production_updates=0)
out['source_hashes_match']=all(hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==sha for name,sha in p['source_sha256'].items())
for phase in ('initial','coexistence'):
 path=b/'logs'/('model-smoke-'+phase)/'complete.json'
 if path.exists():
  content=path.read_bytes();proof=json.loads(content)
  out['model_gate_'+phase]=dict(accepted=proof['accepted'],run_id=proof['run_id'],slurm_job_id=proof['slurm_job_id'],phase=proof['phase'],proof_sha256=hashlib.sha256(content).hexdigest(),rank_count=len(proof['ranks']),actual_grpo_updates=proof['actual_grpo_updates'],max_actor_reference_error=max(row['parity']['actor_reference_max_abs'] for row in proof['ranks']),max_actor_vllm_error=max(row['parity']['actor_vllm_max_abs'] for row in proof['ranks']),tolerances=[row['parity']['tolerance'] for row in proof['ranks']])
for name in ('run-manifest.json','pipeline-status.json','process-ledger.json','cleanup-status.json','retrieval-staging-status.json','corpus-cache-claim.json','probe-coexistence-admission.json','policy-calibration-segment.json'):
 path=b/'logs'/name
 if path.exists():
  raw=path.read_bytes();row=json.loads(raw)
  if name=='run-manifest.json':out['child_manifest_sha256']=hashlib.sha256(raw).hexdigest();out['child_manifest_slurm_job_id']=row['policy_calibration_slurm_job_id']
  if name=='run-manifest.json':row={k:row[k] for k in ('run_id','completed_updates','source_commit','train_log','checkpoint_dir','policy_calibration_role','parent_production_run_id')}
  if name=='policy-calibration-segment.json':row={k:row[k] for k in ('run_id','role','slurm_job_id','completed_outer_updates','optimizer_writes_per_rank','production_admission','healthy_checkpoint_accepted','evaluation_before','evaluation_after')}
  out[name]=row
staging=out.get('retrieval-staging-status.json',{})
if staging.get('root'):
 root=Path(staging['root']);expected=Path('/dev/shm')/('search-g1-common8-'+out['run-manifest.json']['run_id']+'-'+job)
 assert root==expected and not root.is_symlink()
 if root.exists():out['staging_file_sizes']={name:(root/name).stat().st_size for name in ('e5_Flat.index','e5_Flat.index.partial','wiki-18.jsonl','wiki-18.jsonl.partial') if (root/name).exists() and not (root/name).is_symlink()}
cache=out.get('corpus-cache-claim.json',{})
if cache.get('corpus_extraction',{}).get('phase')=='READY':
 record=cache['corpus_extraction'];target=Path(record['target']);expected=Path('/dev/shm')/('search-g1-common8-'+out['run-manifest.json']['run_id']+'-'+job)/'datasets-cache/wiki_dump.jsonl'
 assert target==expected and not target.is_symlink()
 if target.exists():
  st=target.stat();out['actual_corpus_file_stat']=dict(uid=st.st_uid,gid=st.st_gid,size=st.st_size,device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,uid_matches_claim=st.st_uid==cache['uid'],output_inode_matches=st.st_ino==record['output_inode'] and st.st_dev==record['output_device'],scope='Read-only actual metadata; content SHA remains the native extraction receipt')
ledger=out.get('process-ledger.json',{});actual={}
for child in ledger.get('jobs',{}).values():
 for record in child['ledger'].values():
  pid=record['pid'];path=Path('/proc')/str(pid);v=dict(expected_birth=record['start_ticks'],present=path.exists())
  if path.exists():
   try:
    stat=(path/'stat').read_text().rsplit(')',1)[1].split()
    v.update(actual_birth=int(stat[19]),actual_state=stat[0],birth_matches=int(stat[19])==record['start_ticks'],uid=path.stat().st_uid)
   except (FileNotFoundError,ProcessLookupError):v['present']=False
  actual[str(pid)]=v
out['actual_processes']=actual
out['active_owned_pids']=[int(pid) for pid,row in actual.items() if row.get('birth_matches') and row.get('actual_state')!='Z']
namespace=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  environment=(path/'environ').read_bytes().split(b'\0')
  if marker not in environment:continue
  stat=(path/'stat').read_text().rsplit(')',1)[1].split()
  namespace.append(dict(pid=int(path.name),start_ticks=int(stat[19]),sid=int(stat[3]),uid=path.stat().st_uid,state=stat[0],native_job_matches=('SLURM_JOB_ID='+job).encode() in environment))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
out['actual_namespace_processes']=namespace
parent=Path(p['parent_base']);mp=parent/'logs/run-manifest.json'
if mp.exists():
 content=mp.read_bytes();m=json.loads(content)
 out['parent_run_manifest']=dict(run_id=m['run_id'],completed_updates=m['completed_updates'],source_commit=m['source_commit'],manifest_sha256=hashlib.sha256(content).hexdigest())
 claims=[str(x.relative_to(parent)) for pattern in ('optimizer-rank-*.jsonl','policy-calibration-optimizer-rank-*.json') for x in (parent/'logs').glob(pattern)]
 checkpoint=Path(m['checkpoint_dir'])
 out['parent_actual_unstarted']=dict(training_claim_exists=(parent/'logs/training-launch-claim.json').exists(),optimizer_evidence_paths=claims,train_log_exists=Path(m['train_log']).exists(),checkpoint_entries=[] if not checkpoint.exists() else [x.name for x in checkpoint.iterdir()],actual_updates_exists=(parent/'logs/actual-updates.jsonl').exists(),segment_exists=(parent/'logs/policy-calibration-segment.json').exists(),segment_complete_exists=(parent/'logs/policy-calibration-segment-complete.json').exists())
 evidence=out['parent_actual_unstarted'];out['parent_actual_unstarted_verified']=not any((evidence['training_claim_exists'],evidence['optimizer_evidence_paths'],evidence['train_log_exists'],evidence['checkpoint_entries'],evidence['actual_updates_exists'],evidence['segment_exists'],evidence['segment_complete_exists']))
path=b/'artifacts/live/manifest.json'
if path.exists():
 content=path.read_bytes();live=json.loads(content);report=live['calibration_report'];labels=live['label_proof']
 out['actual_initial_probe']=dict(run_id=live['run_id'],round=live['round'],version=live['version'],live_manifest_sha256=hashlib.sha256(content).hexdigest(),calibration_report_sha256=live['calibration_report_sha256'],encoder_sha256=live['encoder_sha256'],head_sha256=live['head_sha256'],published_unix=live['published_unix'],question_count=labels['question_count'],unique_question_count=report['unique_question_count'],commit_cv_auc=report['commit_cv_auc'],boundary_cv_auc=report['boundary_cv_auc'],commit_class_counts=report['commit_class_counts'],boundary_class_counts=report['boundary_class_counts'],require_all_cv_folds=report['require_all_cv_folds'],configured_cv_folds=report['configured_cv_folds'],calibration_protocol=labels['calibration_protocol'],evidence_deletion=labels['evidence_deletion'])
path=b/'logs'/('gpu-samples-'+out.get('run-manifest.json',{}).get('run_id','')+'.jsonl')
if path.exists():
 samples=[]
 for line in path.read_text().splitlines():
  try:samples.append(json.loads(line))
  except json.JSONDecodeError:pass # a concurrently appended final line is incomplete
 if samples:
  peaks={};totals={}
  for row in samples:
   for g in row['gpus']:
    i=str(g['index']);peaks[i]=max(peaks.get(i,0),g['used_mib']);totals[i]=g['total_mib']
  out['gpu_sampling']=dict(sample_count=len(samples),first_unix=samples[0]['unix'],last_unix=samples[-1]['unix'],max_gap_seconds=max([b['unix']-a['unix'] for a,b in zip(samples,samples[1:])] or [0]),cumulative_peak_mib=peaks,total_mib=totals,scope='This discovery pipeline including model gates and probe calibration; not production training peak')
out['child_actual_unstarted']={}
child_manifest=b/'logs/run-manifest.json'
if child_manifest.exists():
 m=json.loads(child_manifest.read_text());checkpoint=Path(m['checkpoint_dir'])
 evidence=dict(training_claim_exists=(b/'logs/training-launch-claim.json').exists(),optimizer_evidence_paths=[str(x.relative_to(b)) for pattern in ('optimizer-rank-*.jsonl','policy-calibration-optimizer-rank-*.json') for x in (b/'logs').glob(pattern)],train_log_exists=Path(m['train_log']).exists(),checkpoint_entries=[] if not checkpoint.exists() else [x.name for x in checkpoint.iterdir()],actual_updates_exists=(b/'logs/actual-updates.jsonl').exists(),segment_exists=(b/'logs/policy-calibration-segment.json').exists(),segment_complete_exists=(b/'logs/policy-calibration-segment-complete.json').exists())
 out['child_actual_unstarted']=evidence
 out['child_actual_unstarted_verified']=not any(evidence.values())
out['actual_behavior_gates']=[]
for path in sorted((b/'logs').glob('behavior-probability-outer-*.json')):
 raw=path.read_bytes();gate=json.loads(raw)
 out['actual_behavior_gates'].append({k:gate[k] for k in ('run_id','slurm_job_id','logical_update','accepted','trajectory_count','included_actor_token_count','max_abs_discrepancy','tolerance','denominator') }|dict(receipt_sha256=hashlib.sha256(raw).hexdigest()))
out['actual_updates']=[]
path=b/'logs/actual-updates.jsonl'
if path.exists():out['actual_updates']=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for name in ('native-discovery-v3-'+job+'.log','trainer.log','initial-calibration.log','retrieval-stage.log','retriever.log'):
 path=b/'logs'/name
 if path.exists():out[name+'_tail']=path.read_text(errors='replace')[-5000:]
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
out['gpu_current']=dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr)
out['observation_finished_unix']=time.time();print(json.dumps(out))
''', launch)
    encoded=json.dumps(observation, indent=2)
    (PREP / ('policy-discovery-v3-native-observation-'+str(int(observation['checked_unix']*1000000))+'.json')).write_text(encoded)
    (PREP / 'policy-discovery-v3-native-observation.json').write_text(encoded)
    summary = {key: value for key, value in observation.items()
               if key not in ('actual_processes', 'process-ledger.json', 'policy-calibration-segment.json') and not key.endswith('_tail')}
    if args.brief:
        import re
        state = re.search(r'JobState=(\S+)', observation['job_stdout'])
        summary = dict(job_id=launch['job_id'], checked_unix=observation['checked_unix'],
                       job_state=None if state is None else state.group(1),
                       job_observation_returncode=observation['job_returncode'],
                       stage=observation.get('pipeline-status.json'),
                       source_hashes_match=observation['source_hashes_match'],
                       actual_completed_child_updates=len(observation['actual_updates']),
                       actual_production_updates=observation['actual_production_updates'],
                       live_namespace_process_count=sum(r['state'] != 'Z' for r in observation['actual_namespace_processes']),
                       live_recorded_pid_count=len(observation['active_owned_pids']),
                       first_live_recorded_pids=observation['active_owned_pids'][:8],
                       gpu_current=observation['gpu_current'])
        for name in ('model_gate_initial', 'model_gate_coexistence', 'actual_initial_probe', 'gpu_sampling', 'actual_behavior_gates', 'child_actual_unstarted_verified', 'parent_actual_unstarted_verified'):
            if name in observation: summary[name] = observation[name]
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
