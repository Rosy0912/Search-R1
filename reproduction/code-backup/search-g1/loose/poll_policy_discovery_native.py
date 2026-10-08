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
    parser = argparse.ArgumentParser(); parser.add_argument('--cleanup-source', action='store_true')
    parser.add_argument('--brief', action='store_true')
    args = parser.parse_args()
    launch = json.loads((PREP / 'policy-discovery-native-launch.json').read_text())
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
for name in ('run-manifest.json','pipeline-status.json','process-ledger.json','cleanup-status.json','retrieval-staging-status.json','probe-coexistence-admission.json','policy-calibration-segment.json'):
 path=b/'logs'/name
 if path.exists():
  row=json.loads(path.read_text())
  if name=='run-manifest.json':row={k:row[k] for k in ('run_id','completed_updates','source_commit','train_log','checkpoint_dir','policy_calibration_role','parent_production_run_id')}
  if name=='policy-calibration-segment.json':row={k:row[k] for k in ('run_id','role','slurm_job_id','completed_outer_updates','optimizer_writes_per_rank','production_admission','healthy_checkpoint_accepted','evaluation_before','evaluation_after')}
  out[name]=row
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
out['actual_updates']=[]
path=b/'logs/actual-updates.jsonl'
if path.exists():out['actual_updates']=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for name in ('native-discovery-'+job+'.log','trainer.log','initial-calibration.log','retrieval-stage.log','retriever.log'):
 path=b/'logs'/name
 if path.exists():out[name+'_tail']=path.read_text(errors='replace')[-5000:]
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
out['gpu_current']=dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr)
print(json.dumps(out))
''', launch)
    (PREP / 'policy-discovery-native-observation.json').write_text(json.dumps(observation, indent=2))
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
                       live_recorded_pids=observation['active_owned_pids'],
                       gpu_current=observation['gpu_current'])
        for name in ('model_gate_initial', 'model_gate_coexistence', 'actual_initial_probe', 'gpu_sampling'):
            if name in observation: summary[name] = observation[name]
    print(json.dumps(summary, indent=2))
    if not args.cleanup_source:
        return
    staging = observation.get('retrieval-staging-status.json', {})
    pipeline = observation.get('pipeline-status.json', {})
    terminal = any('JobState=' + value in observation['job_stdout'] for value in
                   ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'))
    eligible = (staging.get('stage') == 'COMPLETE' and staging.get('accepted') is True) or (
        terminal and not observation['active_owned_pids'] and not [r for r in observation['actual_namespace_processes'] if r['state']!='Z']) or (
        pipeline.get('stage') == 'FAILED' and observation.get('cleanup-status.json', {}).get('cleaned') is True
        and not observation['active_owned_pids'] and not [r for r in observation['actual_namespace_processes'] if r['state']!='Z'])
    assert eligible, 'Original stream still needed or terminal/owned cleanup not verified'
    handle = json.loads((PREP / 'policy-discovery-retrieval-export-handle.json').read_text())
    proof = remote('h100-9', r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['export']);source=b/'serve_original_retrieval.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==h['source_sha256']
path=Path('/proc')/str(h['pid'])
if path.exists():
 # Bind the process handle before identity reads. A later PID reuse can never
 # redirect the signal to a different process. Any identity race fails closed.
 descriptor=os.pidfd_open(h['pid'])
 try:
  stat=(path/'stat').read_text().rsplit(')',1)[1].split()
  assert (int(stat[19]),int(stat[3]),path.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid'])
  assert str(source).encode() in (path/'cmdline').read_bytes()
  if stat[0]!='Z':
   signal.pidfd_send_signal(descriptor,signal.SIGTERM)
   for _ in range(50):
    if not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.1)
  assert not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 finally:os.close(descriptor)
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),scope='Only this source-SHA/PID-birth/SID/UID stream service stopped')
(b/'cleanup.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
''', handle)
    (PREP / 'policy-discovery-retrieval-source-cleanup.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
