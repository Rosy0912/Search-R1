"""Read the two actual owned metadata originals; never prepare or replay them."""
import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:
    raise RuntimeError('Use -E -B -S with assertions enabled')
sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

base = Path.cwd()
invocation = json.loads((base / 'outputs/gigpo-e1-instruct200/FRESH200-V8-ACTUAL-PHYSICAL-AFTER-SUPPORT-INVOKE-v1.json').read_bytes())
records = [invocation['candidate_record'], invocation['physical_record']]
state = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/logs/fresh200-live200-submit-v8'
for row, name in zip(records, ('CANDIDATE.json', 'PHYSICAL.json')):
    assert row['path'] == state + '/' + name
code = '''import base64,hashlib,json,sys,zlib
from pathlib import Path
if sys.flags.optimize or not sys.flags.no_site or not sys.flags.ignore_environment:raise RuntimeError('Isolated startup required')
rows=json.load(sys.stdin);out=[]
for row in rows:
 path=Path(row['path']);assert path.resolve(strict=True)==path and not path.is_symlink();raw=path.read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 out.append(dict(record=row,b64=base64.b64encode(zlib.compress(raw, 6)).decode()))
print(json.dumps(out))
'''
inner = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c ' + shlex.quote(code)
mid = 'env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 ' + shlex.quote(inner)
route = control.ssh('mgmt-host') + ['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 ' + shlex.quote(mid)]
done = subprocess.run(route, input=json.dumps(records).encode(), capture_output=True, timeout=300)
if done.returncode:
    raise RuntimeError('Read-only pull failed: ' + done.stderr.decode(errors='replace'))
rows = json.loads(done.stdout)
assert len(rows) == 2 and [row['record'] for row in rows] == records
target = base / 'outputs/gigpo-e1-instruct200/actual-prephysical-v8-originals-v3'
target.mkdir(exist_ok=False)
receipt = []
for row in rows:
    record = row['record']
    raw = zlib.decompress(base64.b64decode(row['b64']))
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    dest = target / Path(record['path']).name
    with dest.open('xb') as f:
        f.write(raw)
    receipt.append(dict(remote_record=record, local_record=dict(path=str(dest), bytes=len(raw), sha256=record['sha256'])))
with (target / 'RAW-PULL-RECEIPT.json').open('x') as f:
    json.dump(dict(status='EXACT_ORIGINAL_BYTES_READ_ONLY', files=receipt, remote_mutation=False, tests_executed=False, GPU_executed=False, READY=False, submitted=False), f, indent=2)
    f.write('\n')
print(json.dumps(dict(directory=str(target), records=records, READY=False, submitted=False)))
