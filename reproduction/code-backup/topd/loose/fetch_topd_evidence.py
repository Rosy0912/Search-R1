"""Read-only hash-verified fetch of this agent's named evidence files."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

names = sys.argv[1:]
assert names and all('/' not in name and name != '..' for name in names)
script = '''import base64,hashlib,json
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/topd/logs')
result={}
for name in NAMES:
 p=root/name
 if p.is_file():
  b=p.read_bytes();result[name]={'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
print(json.dumps(result))
'''.replace('NAMES', repr(names))
command = '/usr/bin/python3.10 -c '+shlex.quote(script)
for attempt in range(4):
    try:
        result = subprocess.run([sys.executable, 'work/remote_topd.py'], input=command, text=True,
                                capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        print('READ_ONLY_FETCH_TIMEOUT_RETRY', attempt, flush=True)
        time.sleep(3)
        continue
    if result.returncode:
        print('READ_ONLY_FETCH_RETRY', attempt, result.stderr.strip(), flush=True)
        time.sleep(3)
        continue
    records = json.loads(result.stdout)
    output = Path('outputs/remote-evidence'); output.mkdir(exist_ok=True)
    for name, record in records.items():
        data = base64.b64decode(record['base64'])
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        (output/name).write_bytes(data)
        print('EVIDENCE_VERIFIED', name, record['sha256'], flush=True)
    break
else:
    raise RuntimeError('Read-only evidence fetch failed; remote originals retained')
