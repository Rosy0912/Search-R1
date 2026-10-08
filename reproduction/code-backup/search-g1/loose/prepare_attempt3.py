import sys,json,shlex,subprocess,time,uuid,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
def remote(host,code,python='python3',timeout=55):
 compile(code,'remote','exec')
 r=subprocess.run(control.route(host,python+' -c '+shlex.quote(code)),capture_output=True,text=True,timeout=timeout)
 assert r.returncode==0,r.stderr
 return json.loads(r.stdout)
base='/dataset/shaoshitong/Search-G1-common8-attempt3-'+uuid.uuid4().hex[:12]
code='import json,os,shutil,hashlib,subprocess,time;from pathlib import Path\nb=Path('+repr(base)+')\n'+r'''
old=Path('/dataset/shaoshitong/Search-G1-common8');b.mkdir(exist_ok=False)
for n in ('preparation','data','logs','tmp','ray','cache','artifacts'):(b/n).mkdir()
for n in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):(b/n).symlink_to(old/n,target_is_directory=True)
shutil.copytree(Path('/dataset/shaoshitong/g1-common8-device-fix-cpu-d5a7a2d82f65'),b/'repo',ignore=shutil.ignore_patterns('__pycache__','.git','.pytest_cache'))
commit='1e19d6d60e69d273aef53d72874f8d13da8d25f5'
run=json.loads((old/'logs/run-manifest.json').read_text())
assert run['source_commit']=='4b1a06243e915c73b439fca8c62ba5f928ca812b'
expected=dict(run['source_sha256']);expected.update({'verl/workers/fsdp_workers.py': '34add4fa0f63eeda732cd0c813095730a825128f75087005b0304f782470abcf', 'verl/workers/actor/dp_actor.py': '7e42710cbe6b9f86bb2f337628b32dbaf4930ca03228c306d4365958e3b36161', 'reproduction/common8/model_smoke.py': 'dfc6dd95a59277faf1590c2018643a931fbb587ab2cf2da8f5b0b1fec7541b3f', 'reproduction/common8/test_model_smoke.py': '0d78717e1915e5dc176c35ae5d93781603fdc64ca39028482ecee507a04c7859'})
for n,h in expected.items():assert hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest()==h
for n in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):os.link(old/'data'/n,b/'data'/n)
shutil.copy2(old/'preparation/official-model-manifest-original.json',b/'preparation/official-model-manifest-original.json')
assert not (b/'data/calibration-300.parquet').exists() and not (b/'logs/run-manifest.json').exists()
proof=dict(base=str(b),source_commit=commit,parent_attempt_run_id=run['run_id'],actual_updates=0,shared_asset_semantics='same-cluster original assets and repaired installed env only; no prior model checkpoint/probe/counters',canonical_data_links='only six original shared products; separate derivatives',free_bytes=shutil.disk_usage(b).free,checked_unix=time.time())
(b/'preparation/attempt-preparation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
p=remote('h100-8',code);(prep/'attempt3-preparation.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
