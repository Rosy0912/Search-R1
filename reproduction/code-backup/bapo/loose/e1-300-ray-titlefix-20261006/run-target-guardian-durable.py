import os,subprocess,json,hashlib
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006');p=r/'logs/guardian-tests-durable.log'
with p.open('x') as f:
 c=subprocess.run([str(r/'env/bin/python'),'-B','-m','unittest','discover','-s',str(r/'tools'),'-p','test*.py'],cwd=r/'repo',env=dict(os.environ,PYTHONPATH=str(r/'repo'),PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
record={'exit_code':c.returncode,'log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tools_manifest_sha256':hashlib.sha256((r/'logs/tools-manifest.json').read_bytes()).hexdigest(),'source_commit':'484c0b30fa73fa5354bdb3452f007134ec5db13f','scope':'ACTUAL_LINUX_GUARDIAN_TESTS_NOT_GPU_ACCEPTANCE'}
with (r/'logs/guardian-tests-durable-exit.json').open('x') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record));print(p.read_text()[-700:]);raise SystemExit(c.returncode)
