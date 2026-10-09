import os,json,subprocess,signal,hashlib,tarfile,datetime
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009');run=root/'runs/formal-aligned-em200-ecbddc0404f041babd8fef3b3a9d7a1e-3358';job='3358';pid=1571283
manifest=run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json';raw=manifest.read_bytes();assert hashlib.sha256(raw).hexdigest()=='5b152978f124e5e09ed9ac5607783a9deae0d0e66876468f2c8a64c8acb6e836'
m=json.loads(raw);assert m['global_step']==m['actual_outer_updates']==5
log=(run/'train.log').read_text();assert 'FRESH200_STEP5_PAUSED_JSON='in log and not any(x in log for x in ['FRESH200_STEP5_RELEASED_JSON=','FRESH200_ATTEMPT_JSON=','BAPO_TRAIN_METRICS_JSON='])
witnesses=[json.loads(p.read_text())for p in (run/'complete5-resume/rank-witnesses').glob('rank_*.json')];assert len(witnesses)==8 and {v['rank']for v in witnesses}==set(range(8))
proc=Path('/proc')/str(pid);env=(proc/'environ').read_bytes().split(b'\0');assert ('BAPO_RUN_ID='+str(run)).encode()in env
stat=(proc/'stat').read_text();fd=os.pidfd_open(pid,0)
assert (proc/'stat').read_text()==stat
s=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=15);assert s.returncode==0 and 'JobState=RUNNING'in s.stdout and 'BAPO_ALIGNED_EM200:formal:4090e4e8f961ce4ef2057e85dd5a3e6a9506094a:ecbddc0404f041babd8fef3b3a9d7a1e'in s.stdout
out=root/'evidence/3358-parser-repair-retirement';out.mkdir(exist_ok=False)
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Current user authorized own-task fault repair and exact resume; this is technical retirement of unreleasable paused runtime, not manual pause goal','scope':'ONLY_OWN_PAUSED_DRIVER_1571283_NO_NEW_WRITES_PRESERVE_COMPLETE5','slurm':s.stdout,'pid_stat':stat,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'native_witnesses':witnesses,'actual_outer':5,'preserved_adam_total':800,'new_writes':0}
(out/'before.json').write_text(json.dumps(v,indent=2))
with tarfile.open(out/'raw-small.tar','x')as t:
 for p in run.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.log','.err','.out','.json','.jsonl','.txt','.csv'] and p.stat().st_size<16000000:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
signal.pidfd_send_signal(fd,signal.SIGTERM,None,0);os.close(fd)
v['signal']='SIGTERM_OWN_DRIVER_PIDFD';(out/'action.json').write_text(json.dumps(v,indent=2));print(json.dumps({'scope':v['scope'],'signal':v['signal'],'archive':str(out/'raw-small.tar'),'archive_sha256':hashlib.sha256((out/'raw-small.tar').read_bytes()).hexdigest()}))
