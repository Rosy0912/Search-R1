import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');launch=json.loads((prep/'safety-native-launch.json').read_text())
code='import json,subprocess,time;from pathlib import Path\nb=Path('+repr(launch['base'])+');job='+repr(launch['job_id'])+'\n'+'''
x=dict(checked_unix=time.time(),job=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),component_result=json.loads((b/'logs/component-result.json').read_text()) if (b/'logs/component-result.json').exists() else None,proofs=[json.loads(p.read_text()) for p in sorted((b/'logs').glob('*-rank-*.json'))],log_tail=(b/'logs'/('slurm-'+job+'.log')).read_text()[-11000:] if (b/'logs'/('slurm-'+job+'.log')).exists() else '')
print(json.dumps(x))
'''
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);(prep/'safety-native-observation.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
