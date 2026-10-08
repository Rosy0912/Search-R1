import os,json,base64,hashlib,subprocess,sys
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
assert r.resolve()==r and r.is_dir()
for name in ['repo','tools','logs','runs']:
 p=r/name;assert p.is_dir() and not p.is_symlink() and p.resolve()==p
assert not list((r/'runs').iterdir()) and not list((r/'tools').iterdir())
assert (r/'source-expected.txt').read_text().strip()==source==subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True).strip()
payload=json.load(sys.stdin);manifest_raw=base64.b64decode(payload['manifest_base64'],validate=True);manifest=json.loads(manifest_raw);assert manifest['source_commit']==source and set(manifest['files'])==set(payload['files'])
assert hashlib.sha256(manifest_raw).hexdigest()==payload['manifest_sha256']
for name,item in payload['files'].items():
 assert Path(name).name==name and name not in ['.','..'];raw=base64.b64decode(item['base64'],validate=True);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==manifest['files'][name]==item['sha256']
 p=r/'tools'/name
 with p.open('xb') as f:f.write(raw)
 p.chmod(0o755)
with(r/'logs/tools-manifest.json').open('xb') as f:f.write(manifest_raw)
print(json.dumps(dict(status='NEW_REVIEWED_TOOL_BYTES_DEPLOYED_NOT_JOB',source_commit=source,tools_count=len(manifest['files']),manifest_sha256=payload['manifest_sha256'])),flush=True)
env=dict(os.environ,PYTHONPATH=str(r/'repo'),PYTHONDONTWRITEBYTECODE='1',SPT_NOENV='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
with (r/'logs/runtime-capture.log').open('x') as log:
 subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/capture-runtime-executable.py'),'a2568d0df306ead6e059760e28b5bd53ee3f6876'],cwd=r/'repo',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
with (r/'logs/title-dependency-preflight.log').open('x') as log:
 c=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/actual-title-dependency-preflight.py'),'--output',str(r/'logs/title-dependency-preflight.json')],cwd=r/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
with (r/'logs/title-dependency-preflight-exit.json').open('x') as f:json.dump({'exit_code':c.returncode,'log_sha256':hashlib.sha256((r/'logs/title-dependency-preflight.log').read_bytes()).hexdigest()},f)
print('ACTUAL_TITLE_PREFLIGHT_EXIT',c.returncode,flush=True)
if c.returncode:raise SystemExit(c.returncode)
for kind,args in [('cpu-gate-durable',[str(r/'tools/cpu-tests.py')]),('guardian-tests-durable',['-m','unittest','discover','-s',str(r/'tools'),'-p','test*.py'])]:
 p=r/'logs'/(kind+'.log')
 with p.open('x') as log:c=subprocess.run([str(r/'env/bin/python'),'-B',*args],cwd=r/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
 record={'exit_code':c.returncode,'source_commit':'a2568d0df306ead6e059760e28b5bd53ee3f6876','log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE','training_submitted':False,'tools_manifest_sha256':hashlib.sha256((r/'logs/tools-manifest.json').read_bytes()).hexdigest()}
 with (r/'logs'/(kind+'-exit.json')).open('x') as f:f.write(json.dumps(record)+'\n')
 print(json.dumps(dict(kind=kind,**record)),flush=True)
 if c.returncode:raise SystemExit(c.returncode)
