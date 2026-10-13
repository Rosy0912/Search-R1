"""Compact CPU verification receipts; model/numerical arrays stay remote."""
import importlib.util,json,shlex,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'))
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import hashlib,json,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=r/'runs/tree-branpo-fresh200-3360';out=run/'output/eval-handoff-hf-acceptance';inputs=run/'output/eval-handoff-acceptance-inputs'
raw=subprocess.check_output(['systemctl','show','sst-tree-handoff-hf-440b8c6.service','--property=ActiveState,SubState,ExecMainStatus,MainPID,MemoryCurrent,CPUUsageNSec'],text=True)
state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
pid=state.get('MainPID','0');proc=Path('/proc')/pid
result=dict(epoch=time.time(),unit='sst-tree-handoff-hf-440b8c6.service',systemd=state,proc_io=(proc/'io').read_text() if (proc/'io').exists() else None,files={p.name:p.stat().st_size for p in out.glob('*') if p.is_file()},reports={})
for name in ['full-HF-cold-reload.json','tokenizer-cold-reload.json','training-numerical-json-proof.json','identity-bound-handoff-hf.json']:
 p=out/name
 if p.exists():
  data=p.read_bytes();d=json.loads(data)
  skip={'records','input_hashes_before','input_hashes_after','original_partial_cost_files','checkpoint_file_sha256'}
  result['reports'][name]=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),summary={k:v for k,v in d.items() if k not in skip})
p=inputs/'cpu-wrapper.log'
if p.exists():
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-1500));result['log_tail']=f.read().decode(errors='replace')
p=out/'verifier.log'
if p.exists():
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-1500));result['verifier_log_tail']=f.read().decode(errors='replace')
print(json.dumps(result))
'''
r=ssh.run('python3 -c '+shlex.quote(code),timeout=60)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
result=json.loads(r.stdout)
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
