import sys,json,shlex,subprocess,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');base='/dataset/shaoshitong/Search-G1-common8'
commit=json.loads((prep/'production-source-deployment.json').read_text())['source_commit']
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=3-00:00:00','--dependency=afterok:3235','--job-name=search-g1-common8-full','--chdir='+base+'/repo','--output='+base+'/logs/native-full-%j.log','--export=ALL,SEARCH_G1_BASE='+base+',SEARCH_G1_SOURCE_COMMIT='+commit,base+'/repo/reproduction/common8/launch_native.sh']
code="import json,os,subprocess,time;from pathlib import Path\nbase=Path("+repr(base)+");args=json.loads("+repr(json.dumps(args))+");commit="+repr(commit)+"\n"+"""assert not (base/'logs/run-manifest.json').exists()
assert not (base/'preparation/full-native-launch.json').exists()
with (base/'preparation/full-native-submit-claim.json').open('x') as f:json.dump(dict(submitted_unix=time.time(),source_commit=commit,dependency='afterok:3235'),f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(source_commit=commit,argv=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (base/'preparation/full-native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof));assert r.returncode==0
"""
compile(code,'remote-submit','exec')
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);(prep/'full-native-launch.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
