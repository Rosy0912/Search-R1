"""Copy frozen owned metadata/code bytes exclusively; execute no project code."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

base = Path.cwd()
root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006'
stop_path = base / 'outputs/gigpo-e1-policy-safety/FRESH200-DEPLOYMENT-CONTROLS-AUTHOR-STOP-v8.json'
stop_raw = stop_path.read_bytes()
assert len(stop_raw) == 8872
assert hashlib.sha256(stop_raw).hexdigest() == 'a39f19e0dc3e46861d38e8f900179586894f644867dc6ca6a41c7330493461a7'
stop = json.loads(stop_raw)
assert stop['all_code_and_metadata_author_writes_stopped'] is True
items = []

def add(record, target):
    path = Path(record['path'])
    assert path.is_relative_to(base) and not path.is_symlink()
    raw = path.read_bytes()
    assert len(raw) == record['bytes']
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    items.append(dict(path=target, bytes=len(raw), sha256=record['sha256'], b64=base64.b64encode(raw).decode()))

assert len(stop['six_deployment_files']) == len(stop['canonical_six']) == 6
for local, remote in zip(stop['six_deployment_files'], stop['canonical_six']):
    assert (local['bytes'], local['sha256']) == (remote['bytes'], remote['sha256'])
    assert remote['path'] == root + '/tools/fresh200-deployment-controls-v8/' + local['relative_path']
    add(local, remote['path'])
add(stop['separate_helper9_local'], stop['separate_helper9_canonical']['path'])
for record in stop['frozen_author_bundle_records']:
    add(record, root + '/logs/' + Path(record['path']).name)
add(dict(path=str(stop_path), bytes=len(stop_raw), sha256=hashlib.sha256(stop_raw).hexdigest()), root + '/logs/' + stop_path.name)

remote_code = '''import base64,datetime,hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);records=[]
for row in p['items']:
 path=Path(row['path']);assert path.is_relative_to(root) and not path.is_symlink();path.parent.mkdir(parents=True,exist_ok=True);assert path.parent.resolve()==path.parent
 raw=base64.b64decode(row['b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 data=path.read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];records.append({key:row[key] for key in ('path','bytes','sha256')})
directory=root/'tools/fresh200-deployment-controls-v8'
assert sorted(x.name for x in directory.iterdir())==sorted(Path(r['path']).name for r in records if Path(r['path']).parent==directory)
print(json.dumps(dict(status='EXACT_UNEXECUTED_CONTROLS8_HELPER9_METADATA_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata_original_records=records,separate_management_helper=True,CPU_executed=False,GPU_executed=False,READY=False,submitted=False,formal_iterations=0),indent=2))
'''
inner = '/usr/bin/python3 -c ' + shlex.quote(remote_code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
r = subprocess.run(route, input=json.dumps(dict(root=root, items=items)).encode(), capture_output=True, timeout=120)
if r.returncode:
    raise RuntimeError(r.stderr.decode(errors='replace'))
result = json.loads(r.stdout)
out = base / 'outputs/gigpo-e1-instruct200/FRESH200-CONTROLS8-HELPER9-METADATA-STAGING-RECEIPT-v1.json'
with out.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps(dict(status=result['status'], UTC=result['UTC'], record_count=len(result['metadata_original_records']), receipt=str(out), CPU_executed=False, GPU_executed=False, submitted=False)))
