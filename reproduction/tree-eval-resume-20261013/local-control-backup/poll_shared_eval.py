import importlib.util,json,shlex,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import json,time,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');unit='sst-tree-eval-shared-1c15c49.service';run=r/'runs'/unit.removesuffix('.service');s=r/'runs/tree-branpo-fresh200-3360/output/eval-resume/scores'
def tail(p,n=4500):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-n));return f.read().decode(errors='replace')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
b=list((s/'batches').glob('*.done.json'));done=[read(p) for p in b]
x=dict(epoch=time.time(),unit=unit,systemd=subprocess.run(['systemctl','show',unit,'--property=ActiveState,SubState,ExecMainStatus,MainPID,MemoryCurrent'],capture_output=True,text=True).stdout,queue=subprocess.run(['squeue','--local','-h','-j','3391,3392','-o','%i %j %T %N'],capture_output=True,text=True).stdout,done_batches=len(b),done_receipts=done[-2:],rank_provenance=[p.name for p in s.glob('*rank*')],gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,logs={n:tail(run/n) for n in ['guard.log','retriever.log','eval.log','resource-samples.jsonl']},run_result=read(run/'run-result.json'),final_metrics=read(s/'final-metrics.json'),journal=subprocess.run(['journalctl','-u',unit,'-n','15','--no-pager','-o','cat'],capture_output=True,text=True).stdout)
print(json.dumps(x))
'''
x=ssh.run('python3 -c '+shlex.quote(code),timeout=60)
if x.returncode:print(x.stderr);sys.exit(x.returncode)
d=json.loads(x.stdout)
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
