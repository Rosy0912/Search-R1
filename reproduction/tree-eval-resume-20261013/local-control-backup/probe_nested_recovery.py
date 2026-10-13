"""Small actual bootstrap/runtime probe for the native1 recovery instance."""
import argparse,json,shlex
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
code='''import json,subprocess,time
from pathlib import Path
source='ba9f6627ec9a4e9749830d7b6acca53f6feec6b4';root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');unit='sst-tree-nested-branch-fresh200-'+source[:8]+'.service';run=root/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+source)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p,n=3000):
 if not p.is_file():return None
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-n));return f.read().decode(errors='replace')
receipt=read(run/'output/nested-native1-loaded.json')
r=dict(epoch=time.time(),source_commit=source,node='h100-10',unit=unit,run=str(run),systemd=subprocess.check_output(['systemctl','show',unit,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,MemoryCurrent,MemoryPeak,MemoryMax,CPUUsageNSec'],text=True),journal=subprocess.check_output(['journalctl','-u',unit,'-n','16','--no-pager','-o','cat'],text=True)[-4500:],startup=[dict(path=str(p),value=read(p)) for p in (root/'instruct300/startup-audits').glob('*'+source+'*') if p.is_file()],native_loaded=receipt,run_result=read(run/'run-result.json'),stage_tail=tail(run/'stage.log'),ray_tail=tail(run/'ray.log'),guard_tail=tail(run/'guard.log',1500),rank_direct_count=len(list((run/'output/tree-rank-audit').glob('step-*-attempt-*-rank-*.json'))),attempts=[dict(path=str(p),value=read(p)) for p in sorted((run/'checkpoints/training_state/attempt-results').glob('attempt-*.json'))[-2:]],native_manifests=[str(p) for p in (run/'checkpoints/training_state/rolling-completed').glob('*/manifest.json')])
print(json.dumps(r))
'''
x=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=60);x.check_returncode();r=json.loads(x.stdout);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('startup','native_loaded','attempts')}));print('native_loaded',None if r['native_loaded'] is None else r['native_loaded'].get('status'))
