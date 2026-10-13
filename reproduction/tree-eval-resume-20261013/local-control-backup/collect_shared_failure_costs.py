import importlib.util,json,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');rows=[]
for short in ('1c15c49','b316a77','993109d','13a5388','7d28b1b'):
 unit='sst-tree-eval-shared-'+short+'.service';p=root/'runs'/unit.removesuffix('.service')/'run-result.json'
 rows.append(dict(unit=unit,result=json.loads(p.read_text()) if p.exists() else None,actual_systemd=subprocess.check_output(['systemctl','show',unit,'--property=ActiveState,MainPID,ExecMainStatus,CPUUsageNSec,MemoryPeak'],text=True)))
print(json.dumps(dict(epoch=time.time(),attempts=rows,cost_class='shared_evaluation_repair_attempts; separate from successful training and original slow final evaluation')))
'''
x=ssh.run('python3 -c '+shlex.quote(code),timeout=60)
if x.returncode:raise RuntimeError(x.stderr)
Path('outputs/tree-eval-resume-20261013/SHARED_FAILED_ATTEMPT_COSTS.json').write_text(json.dumps(json.loads(x.stdout),indent=2)+'\n');print('saved small failure cost receipt')
