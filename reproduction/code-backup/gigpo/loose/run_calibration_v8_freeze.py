"""Freeze only the independently reviewed v8 diagnostic deployment."""
import base64, gzip, hashlib, json, re, subprocess, sys
from pathlib import Path

review_sha = sys.argv[1]
assert re.fullmatch('[0-9a-f]{64}', review_sha)
script = """export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python - <<'REMOTE'
import base64,gzip,hashlib,json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
helper=root/'tools/policy-safety-calibration-controls-52befe9-v8/freeze_ready_v8.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='2ef4a885a0ee19fb93f36919b4df92c73456b61430540c388194eb5cbd709641'
out=root/'logs/policy-safety-calibration-52befe9-reviewed-binding-v8.json'
assert not out.exists(),'Preserve existing READY; do not overwrite or retry'
cmd=[str(root/'env/bin/python'),str(helper),str(root/'logs/CALIBRATION-CANDIDATE-52BEFE9-v8.json'),
 'e18615c998a69e0212a8aac99fb2a5db32556b97a98cdcfd05fcb87418b618d1',
 str(root/'logs/policy-safety-calibration-52befe9-physical-v8.json'),
 '4efc150b08a44359f1372a7110b17456c21ca6076a675aa6a9e61a7f9a8574d7',
 str(root/'logs/CALIBRATION-V8-ACTUAL-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'),
 'REVIEW_SHA',str(root/'logs/NATIVE3267-CALIBRATION-ACCEPTANCE-SPEC-v1.json'),str(out)]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,returncode=p.returncode,
       stdout=p.stdout,stderr=p.stderr)
if out.exists():
 raw=out.read_bytes();r.update(path=str(out),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                             gzip_base64=base64.b64encode(gzip.compress(raw)).decode())
print(json.dumps(r))
REMOTE
""".replace('REVIEW_SHA',review_sha)
p=subprocess.run(['/usr/bin/python3','work/remote.py','h100-8'],input=script,text=True,capture_output=True,timeout=300)
r={'transport_returncode':p.returncode,'transport_stderr':p.stderr}
if not p.returncode:
    actual=json.loads(p.stdout);encoded=actual.pop('gzip_base64',None)
    if encoded:
        raw=gzip.decompress(base64.b64decode(encoded))
        assert hashlib.sha256(raw).hexdigest()==actual['sha256'] and len(raw)==actual['bytes']
        original=Path('outputs/gigpo-e1-policy-safety/policy-safety-calibration-52befe9-reviewed-binding-v8.json')
        with original.open('xb') as f:f.write(raw)
        actual['local_original']=str(original.resolve())
    r['actual']=actual
else:
    r['raw_stdout']=p.stdout
out=Path('outputs/gigpo-e1-policy-safety/CALIBRATION-V8-ACTUAL-FREEZE-v1.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                  'transport_returncode':p.returncode,'actual':r.get('actual')}))
