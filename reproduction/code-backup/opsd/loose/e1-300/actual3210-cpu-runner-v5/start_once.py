"""Trigger only after actual PAUSED, exactly once; preserves unknown outcomes."""
from pathlib import Path
import hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;E=H.parent
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
D='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210/evidence/first5-cpu-inspector-v5-preparation';PIN=hashlib.sha256((H/'inspect_first5_cpu.py').read_bytes()).hexdigest()
code='''import pathlib,importlib.util,hashlib,json,subprocess,os,datetime
script=pathlib.Path(SCRIPT);assert script.resolve(strict=True)==script and hashlib.sha256(script.read_bytes()).hexdigest()==PIN
spec=importlib.util.spec_from_file_location('actual3210_inspector',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:manifest,mp,pin,before=m.context()
except FileNotFoundError as e:print(json.dumps({'started':False,'reason':str(e)}));raise SystemExit(3)
run=m.RUN;marker=run/'evidence/first5-cpu-inspection-v5-launch.json'
# Existence includes unknown previous SSH outcome: never implicitly repeat.
assert not marker.exists() and not marker.is_symlink(),'Existing launch; inspect its original outcome, never repeat'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=run/'evidence'/('first5-cpu-inspection-v5-'+stamp);out.mkdir(mode=0o700)
cmd=['srun','--jobid=3210','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(m.REPO),str(m.PYTHON),str(script),'--output',str(out)]
with marker.open('x') as f:json.dump({'job_id':'3210','source_commit':m.SOURCE,'output':str(out),'manifest_sha256':pin,'runner_sha256':PIN,'command':cmd,'scope':'CPU inspection launch only, not acceptance/signature/release'},f,sort_keys=True)
controller="import subprocess,json,datetime,pathlib; p=subprocess.run("+repr(cmd)+"); f=pathlib.Path("+repr(str(out/'srun-exit-original.json'))+").open('x'); json.dump({'returncode':p.returncode,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f); f.close()"
with (out/'srun-stdout-original.txt').open('xb') as stdout,(out/'srun-stderr-original.txt').open('xb') as stderr:p=subprocess.Popen(['python3','-c',controller],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
m.write(out/'srun-launch-original.json',{'durable_cpu_controller_pid':p.pid,'launcher_pid':os.getpid(),'stdout':str(out/'srun-stdout-original.txt'),'stderr':str(out/'srun-stderr-original.txt'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':before})
print(json.dumps({'started':True,'durable_cpu_controller_pid':p.pid,'output':str(out),'marker':str(marker),'manifest_sha256':pin}))
'''.replace('SCRIPT',repr(D+'/inspect_first5_cpu.py')).replace('PIN',repr(PIN))
# No timeout is assigned to actual CPU children. SSH only captures the durable
# launch acknowledgement; a timeout preserves marker and is never auto-retried.
p=subprocess.run(ns['route']('python3 -c '+shlex.quote(code)),capture_output=True,timeout=60)
(H/'launch-stdout-original.txt').write_bytes(p.stdout);(H/'launch-stderr-original.txt').write_bytes(p.stderr);(H/'launch-exit-original.json').write_text(json.dumps({'returncode':p.returncode})+'\n');print(p.stdout.decode());print('launcher_exit',p.returncode)
