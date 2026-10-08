"""Read-only archived3311 terminal replay. No network, GPU, signals or training."""
import argparse,hashlib,importlib.util,json,math,os,re,sys
from pathlib import Path
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006/numeric-diagnostic-3310-20261006')
JOB=3311;NONCE='2a9da77d45febc79ca59183205e24e9e';PHASE='numeric-zero-adam-replay'
RUN=ROOT/'runs'/f'{PHASE}-{NONCE}-{JOB}'
TOOLS_SHA='8a79b017d526de2cb9ff78f3e701e2572a5da0029545956d985394b72d492830'
PROBE_SHA='9b80627933fd88ca029c5d84d6e779e1669d68604f52d3c8de72bf4fdf0e502d'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def pairs(items):
  d={}
  for key,value in items:require(key not in d,'Duplicate JSON key');d[key]=value
  return d
 return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def ordinary(p):
 p=Path(p);require(p.is_absolute()and p.is_file()and p.resolve()==p and not p.is_symlink(),'Canonical archived ordinary file required');return p

def terminal(value):
 require(type(value.get('job_id'))is int and value['job_id']==JOB and type(value.get('returncode'))is int and value['returncode']==0,'Actual3311 SlurmJSON command success required')
 text=value.get('stdout');require(type(text)is str,'Actual unmodified raw scontrol stdout required')
 fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(\S+)',text))
 expected=dict(JobId=str(JOB),JobState='COMPLETED',ExitCode='0:0',Restarts='0',Requeue='0',NodeList='h100-5',ReqNodeList='h100-5',NumNodes='1',Comment=f'BAPO_ZERO_ADAM:{PHASE}:{SOURCE}:{NONCE}',Command=str(ROOT/'tools'/f'{PHASE}.sbatch'),TresPerNode='gres:gpu:h100:8')
 for key,expected_value in expected.items():require(fields.get(key)==expected_value,'Actual terminal mismatch '+key)
 require(fields.get('UserId','').startswith('root('),'Actual root allocation owner required')
 duration=fields.get('RunTime','');require(re.fullmatch(r'(?:\d+-)?\d+:\d\d:\d\d',duration)is not None,'Actual elapsed scheduler time required')
 days,clock=(duration.split('-',1)if '-'in duration else ('0',duration));h,m,s=map(int,clock.split(':'));seconds=int(days)*86400+h*3600+m*60+s
 return text,fields,seconds

def replay_journal(path,watcher_path,expected_watch):
 require(sha(watcher_path)==expected_watch,'Exact fixed watcher bytes required')
 spec=importlib.util.spec_from_file_location('archived_numeric_watcher',watcher_path);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 require(w.DIAGNOSTIC_ROOT==ROOT and w.SOURCE==SOURCE,'Fixed numeric watcher root/source required')
 before=path.read_bytes();rows=[read_line(line)for line in before.decode().splitlines()];require(rows,'Actual resource samples absent')
 uuids=None;uid=None;previous=-1.;samples=0;phase=None;epochs={}
 for row in rows:
  require(row.get('run')==str(RUN)and row.get('slurm_job_id')==str(JOB),'Actual journal run/job differs')
  require(type(row.get('time'))in(int,float)and math.isfinite(row['time'])and row['time']>=previous,'Monotonic finite journal timestamps required');previous=row['time']
  require(row.get('event')is None,'Collector supports normal samples only; recovery requires independently reviewed full replay adapter')
  require(type(row.get('pause'))is bool and row['pause']is False and row.get('reason')=='within_threshold','Unsafe or unknown raw resource sample cannot pass')
  observed_phase=row.get('phase_pid');require(observed_phase is None or(type(observed_phase)is int and observed_phase>0),'Actual phase PID required')
  if observed_phase is not None:
   if phase is None:phase=observed_phase
   require(phase==observed_phase,'Actual phase PID changed')
  allocated=row['allocated_gpus'];observed=w.inventory(row['raw_inventory'],list(allocated))
  for entry in observed.values():
   n=row['raw_nvml'].get(entry['uuid'],{})
   if 'whole_gpu_sm_percent'in n:entry.update(whole_gpu_sm_percent=w.number(n['whole_gpu_sm_percent'],'whole SM'),whole_gpu_sm_source='nvml_query',whole_gpu_sm_query_us=n.get('whole_gpu_sm_query_us'))
   else:entry['whole_gpu_sm_source']='nvidia_smi_inventory_query'
  require(observed==allocated,'Declared resource inventory differs from actual raw inventory')
  selected={x['uuid']for x in observed.values()};require(len(selected)==8,'Eight actual unique allocated GPUs required')
  if uuids is None:uuids=selected
  require(selected==uuids,'Actual allocation changed')
  proofs=row['owned_before_query'];require(isinstance(proofs,dict)and proofs,'Actual own process proof required')
  for pid,proof in proofs.items():
   require(pid.isdecimal()and type(proof.get('start_ticks'))is int and proof['start_ticks']>0 and type(proof.get('uid'))is int and proof['uid']>=0,'Actual PID birth/UID required')
   require(proof.get('run')==str(RUN)and proof.get('slurm_job_id')==str(JOB)and set(proof.get('lease_uuids',[]))==uuids and len(proof['lease_uuids'])==8,'Actual own UUID/run/job proof required')
   if pid in epochs:require(epochs[pid]==proof['start_ticks'],'Archived own PID epoch reused')
   epochs[pid]=proof['start_ticks']
   if uid is None:uid=proof['uid']
   require(uid==proof['uid'],'Actual own UID changed')
  require(type(row.get('window_start_us'))is int and type(row.get('window_end_us'))is int and 0<row['window_start_us']<=row['window_end_us'],'Actual sample time window invalid')
  sm=w.parse_pmon(row['raw_pmon'],observed);mem=w.parse_memory(row['raw_compute'],observed);fresh,_=w.fresh_nvml(observed,row['raw_nvml'],row['window_start_us'])
  for _,pid in set(sm)|set(mem)|set(fresh):
   ticks=row['pid_start_ticks'].get(str(pid),{});require(type(ticks.get('before'))is int and ticks['before']>0 and ticks.get('after')==ticks['before'],'Actual GPU PID lifecycle unstable')
   if str(pid)in proofs:require(proofs[str(pid)]['start_ticks']==ticks['before'],'Actual own GPU PID birth differs')
  assessed=w.assess_with_bounds(observed,sm,mem,lambda pid:str(pid)in proofs,row['raw_nvml'],row['window_start_us'])
  for key in ('external_processes','external_totals','pause','exceeded'):require(assessed[key]==row[key],'Actual fixed watcher numerical replay differs '+key)
  require(assessed['pause']is False,'Actual resource thresholds not satisfied');samples+=1
 require(path.read_bytes()==before and sha(watcher_path)==expected_watch,'Archived inputs changed during replay')
 return dict(status='PASS_ACTUAL_GUARD_JOURNAL_REPLAY',scope='NORMAL_SAMPLES_ONLY_FIXED_WATCHER_FUNCTION_REPLAY_NOT_NATIVE_ACCEPTANCE',source_commit=SOURCE,job_id=JOB,run_root=str(RUN),raw_sha256=hashlib.sha256(before).hexdigest(),watcher_sha256=expected_watch,allocated_uuids=sorted(uuids),own_uid=uid,samples=samples,unfinished_recovery=False,native_acceptance=False)
def read_line(line):return json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSONL')))
def immutable(p,value):
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def collect(args):
 archive=Path(args.archive_root).resolve();output=Path(args.output).absolute();require(archive.is_dir()and output.parent.is_dir()and not output.exists(),'Existing archive and fresh evidence output required')
 slurm_path=ordinary(args.slurm_json);slurm=read(slurm_path);text,fields,elapsed=terminal(slurm)
 run=archive/'runs'/RUN.name;require(run.is_dir(),'Exact archived3311 ownrun required')
 paths={name:ordinary(run/name)for name in ['numeric-probe','numeric-probe.selected-logits.pt','phase.log','resource-watch.jsonl','own-process-cleanup.json']}
 require((run/'source-commit.txt').read_text().strip()==SOURCE and (run/'original-exit-code.txt').read_text().strip()=='0'and (run/'cleanup-exit-code.txt').read_text().strip()=='0','Actual source/phase/cleanup exit codes required')
 manifest=read(ordinary(archive/'logs/tools-manifest.json'));require(sha(archive/'logs/tools-manifest.json')==TOOLS_SHA and manifest['source_commit']==SOURCE,'Actual new numeric tools manifest required')
 result=read(paths['numeric-probe']);require(result.get('status')=='ACTUAL_ZERO_ADAM_NUMERIC_DIAGNOSTIC_NOT_ADMISSION'and result.get('source_commit')==SOURCE and result.get('numeric_probe_sha256')==PROBE_SHA and result.get('optimizer_writes')==0 and result.get('native_acceptance')is False and result.get('production_release')is False,'Actual final numerical result required')
 side=result['selected_logit_sidecar'];require(side['sha256']==sha(paths['numeric-probe.selected-logits.pt'])and side['bytes']==paths['numeric-probe.selected-logits.pt'].stat().st_size and side['path']==str(RUN/'numeric-probe.selected-logits.pt'),'Actual selected logits sidecar bytes/path differ')
 import torch
 saved=torch.load(paths['numeric-probe.selected-logits.pt'],map_location='cpu',weights_only=True)
 require(saved.get('source_commit')==SOURCE and saved.get('numeric_probe_sha256')==PROBE_SHA and len(saved.get('records',[]))==36,'Actual36selected-logit records required')
 receipt=ordinary(archive/'scheduler-head'/f'{PHASE}-receipt.json');worker=read(receipt);require(type(worker.get('job_id'))is int and worker['job_id']==JOB and worker.get('source')==SOURCE and worker.get('nonce')==NONCE and worker.get('phase')==PHASE and worker.get('comment')==fields['Comment'],'Actual worker receipt required')
 head_origin_path=ordinary(args.head_origin);origin=read(head_origin_path)
 require(origin.get('origin')=='ACTUAL_H10016_HEAD_READ'and origin.get('path')==str(ROOT/'scheduler-head'/f'{PHASE}-receipt.json')and type(origin.get('text'))is str and origin.get('sha256')==hashlib.sha256(origin['text'].encode()).hexdigest()and origin['text'].encode()==receipt.read_bytes(),'Actual independently head-read exact receipt bytes required; worker copy cannot certify head origin')
 cleanup=read(paths['own-process-cleanup.json']);require(cleanup.get('own_marker')==str(RUN)and cleanup.get('remaining')==[],'Actual owncleanup exactmarker/no remaining process required')
 output.mkdir();logical=ROOT/'logs'/output.name
 replay=replay_journal(paths['resource-watch.jsonl'],ordinary(args.watcher),manifest['files']['resource_watch.py']);replay.update(cost_wall_seconds=elapsed,reserved_gpu_hours=elapsed*8/3600,cost_source='actual terminal Slurm RunTime including all8reserved');immutable(output/'resource-replay.json',replay)
 (output/'slurm-terminal.raw.log').write_text(text);immutable(output/'slurm-terminal.actual.json',slurm);immutable(output/'head-receipt-origin.actual.json',origin)
 with (output/'head-receipt.actual.json').open('xb')as f:f.write(origin['text'].encode());f.flush();os.fsync(f.fileno())
 def bind(path,remote):return dict(path=str(remote),bytes=path.stat().st_size,sha256=sha(path))
 artifacts={'numeric_result':bind(paths['numeric-probe'],RUN/'numeric-probe'),'selected_logit_sidecar':bind(paths['numeric-probe.selected-logits.pt'],RUN/'numeric-probe.selected-logits.pt'),'phase_log':bind(paths['phase.log'],RUN/'phase.log'),'resource_journal':bind(paths['resource-watch.jsonl'],RUN/'resource-watch.jsonl'),'resource_replay':bind(output/'resource-replay.json',logical/'resource-replay.json'),'own_cleanup':bind(paths['own-process-cleanup.json'],RUN/'own-process-cleanup.json'),'worker_receipt':bind(receipt,ROOT/'scheduler-head'/receipt.name),'head_receipt':bind(output/'head-receipt.actual.json',logical/'head-receipt.actual.json'),'head_receipt_origin':bind(output/'head-receipt-origin.actual.json',logical/'head-receipt-origin.actual.json'),'approval':bind(ordinary(archive/'logs/diagnostic-approval.json'),ROOT/'logs/diagnostic-approval.json'),'slurm_terminal':bind(output/'slurm-terminal.raw.log',logical/'slurm-terminal.raw.log'),'slurm_json':bind(output/'slurm-terminal.actual.json',logical/'slurm-terminal.actual.json')}
 evidence=dict(schema='ACTUAL_ZERO_ADAM_NUMERIC_TERMINAL_EVIDENCE_V1',root=str(ROOT),source_commit=SOURCE,job_id=JOB,nonce=NONCE,run_root=str(RUN),scheduler_state='COMPLETED',exit_code=0,restarts=0,requeue=0,node='h100-5',protocol='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1',phase=PHASE,allocated_gpu_count=8,computed_gpu_count=1,optimizer_writes=0,formal_outer_updates=0,scheduler_updates=0,numeric_probe_sha256=PROBE_SHA,native_acceptance=False,production_release=False,artifacts=artifacts,actual_scheduler_elapsed_seconds=elapsed,reserved_gpu_hours=elapsed*8/3600,collector_sha256=sha(__file__),scope='READ_ONLY_ARCHIVE_REPLAY_REQUIRES_INDEPENDENT_INTERPRETATION_NOT_NATIVE')
 immutable(output/'numeric-terminal-evidence.json',evidence);return evidence
if __name__=='__main__':
 require(not sys.flags.optimize,'Optimized collector forbidden')
 p=argparse.ArgumentParser();p.add_argument('--archive-root',required=True);p.add_argument('--slurm-json',required=True);p.add_argument('--head-origin',required=True);p.add_argument('--output',required=True);p.add_argument('--watcher',default=str(Path(__file__).parent.parent/'fresh200-numeric-diagnostic-3310/tools/resource_watch.py'));args=p.parse_args();print(json.dumps(collect(args)))
