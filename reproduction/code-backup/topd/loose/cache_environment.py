import importlib.util,subprocess,hashlib,json
from pathlib import Path
p='/Users/shaoshitong/Documents/remote_codex_control/control.py'
s=importlib.util.spec_from_file_location('rc',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
root=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-env-20261003')
out=root/'pinned-site-packages.tar.gz.partial'
command="set -o pipefail; nice -n 19 tar --exclude='__pycache__' --exclude='*.pyc' -C /data/search-r1/env/lib/python3.10/site-packages -cf - . | nice -n 19 gzip -1"
print('SNAPSHOT_READ_ONLY_PINNED_ENV_START',flush=True)
with out.open('wb') as f:r=subprocess.run(m.ssh('jump_h20-1')+[command],stdout=f,stderr=subprocess.PIPE)
if r.returncode:print(r.stderr.decode(errors='replace')[-1000:]);raise SystemExit(r.returncode)
h=hashlib.sha256()
with out.open('rb') as f:
 for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
final=out.with_suffix('');out.rename(final)
record={'source':'jump_h20-1:/data/search-r1/env/lib/python3.10/site-packages','target':'h100-11:/dataset/shaoshitong/search-r1-rl/topd/env','architecture':'Linux x86_64','python':'3.10.12','sha256':h.hexdigest(),'size':final.stat().st_size,'snapshot_excludes':['__pycache__','*.pyc'],'archive':str(final),'validated_target':False}
(root/'snapshot-manifest.json').write_text(json.dumps(record,indent=2));print('ENV_SNAPSHOT_SHA256',h.hexdigest(),'bytes',record['size'],flush=True)
