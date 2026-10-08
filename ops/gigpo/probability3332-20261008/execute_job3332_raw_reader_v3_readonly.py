#!/usr/bin/env python3
"""Transport one reviewed CPU-only read of existing3332 evidence via retained18."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
SOURCE = BASE / 'work/inspect_job3332_actual_rank3_call0_readonly_v3.py'
EXPECTED_SHA = '20a5a93c68c17d57b4d6724e46f81e3da121f996dc1922a4c1bb9a6b1a9ff987'
OUT = BASE / 'outputs/gigpo-e1-instruct200/JOB3332-RANK3-ACTUAL-RAW-READONLY-v3-TRANSPORT.json'
INTENT = OUT.with_name('JOB3332-RANK3-ACTUAL-RAW-READONLY-v3-LOCAL-INTENT.json')


def main():
    source = SOURCE.read_bytes()
    if len(source) != 31605 or hashlib.sha256(source).hexdigest() != EXPECTED_SHA:
        raise RuntimeError('Reviewed reader bytes changed')
    receiver = "import base64,hashlib;source=base64.b64decode(" + repr(base64.b64encode(source).decode()) + ");assert len(source)==31605 and hashlib.sha256(source).hexdigest()==" + repr(EXPECTED_SHA) + ";exec(compile(source,'reviewed-job3332-rank3-reader-v3','exec'))"
    worker = shlex.join(['env', '-u', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES=',
                        'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
                        '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python',
                        '-B', '-c', receiver])
    management = '''import datetime,json,shlex,subprocess
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER],capture_output=True,text=True,timeout=120)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),route='retained18 SSH-W h100-8',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
'''.replace('WORKER', repr(worker))
    if OUT.exists():
        raise RuntimeError('Existing diagnostic receipt must be inspected; do not replay')
    with INTENT.open('x') as handle:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       reader_sha256=EXPECTED_SHA, read_only=True, remote_artifact_writes=0,
                       new_model_forward_calls=0, gpu_visible=False), handle, indent=2)
        handle.write('\n')
    outer = ['env', '-u', 'LD_LIBRARY_PATH', '/usr/bin/ssh', '-T', '-o', 'BatchMode=yes',
             '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
             '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
             '-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host']
    result = subprocess.run(outer + ['env -u LD_LIBRARY_PATH /usr/bin/python3 -B -c ' + shlex.quote(management)],
                            capture_output=True, text=True, timeout=140)
    record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  reader_sha256=EXPECTED_SHA, returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr, read_only=True)
    with OUT.open('x') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    if result.returncode != 0:
        print(json.dumps(dict(receipt=str(OUT),returncode=result.returncode,stderr=result.stderr)))
        raise SystemExit(result.returncode)
    observed = json.loads(result.stdout)
    print(json.dumps(dict(receipt=str(OUT),returncode=observed['returncode'],stdout=observed['stdout'],stderr=observed['stderr'])))
    if observed['returncode']:
        raise SystemExit(observed['returncode'])


if __name__ == '__main__':
    main()
