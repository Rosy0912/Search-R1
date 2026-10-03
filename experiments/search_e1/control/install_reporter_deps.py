"""Install only new Ray reporter dependencies; refuse changing existing packages."""
import json,os,hashlib,subprocess,importlib.metadata as md
from pathlib import Path
import requests
base=Path(os.environ['E1_BASE']);root=base/'ray-reporter-wheels';root.mkdir(exist_ok=True)
url='http://10.120.17.101:18283/'
r=requests.get(url+'manifest.json',timeout=30);r.raise_for_status();manifest=r.json()
r=requests.get(url+'pins.txt',timeout=30);r.raise_for_status();pins=r.text
canonical=lambda n:n.lower().replace('_','-').replace('.','-')
before={canonical(d.metadata['Name']):d.version for d in md.distributions()}
for row in manifest:
 name=canonical(row['package'])
 if name in before and before[name]!=row['version']:raise RuntimeError('Refuse replacing existing dependency: '+name)
 p=root/row['name']
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
  response=requests.get(url+row['name'],timeout=120);response.raise_for_status();temporary=p.with_suffix('.part');temporary.write_bytes(response.content)
  if len(response.content)!=row['size'] or hashlib.sha256(response.content).hexdigest()!=row['sha256']:raise RuntimeError('Reporter wheel verification failed')
  temporary.rename(p)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'pins.txt').write_text(pins)
subprocess.run([str(base/'env/bin/python'),'-m','pip','install','--no-deps','--no-index','--find-links',str(root),'-r',str(root/'pins.txt')],check=True)
after={canonical(d.metadata['Name']):d.version for d in md.distributions()}
assert all(after.get(name)==version for name,version in before.items()),'Existing package changed'
from ray.dashboard.modules.reporter.reporter_agent import ReporterAgent
from ray.dashboard import optional_deps
(base/'receipts/reporter-dependency-install.json').write_text(json.dumps(dict(existing_unchanged=True,before_count=len(before),after_count=len(after),new_wheels=manifest),indent=2))
print('REPORTER_IMPORT_VALIDATED_WITHOUT_CORE_UPGRADES',len(manifest),flush=True)
