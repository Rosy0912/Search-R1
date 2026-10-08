"""Exactly-once transport for reviewed metadata-only legacy completion repair."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
OUT = BASE / 'outputs/gigpo-e1-instruct200'
SOURCE = BASE / 'work/close_job3329_legacy_controller_once_v2.py'
REMOTE = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/logs/human-preempt-searchg1-3329-for-gigpo3332-20261008-v1/close_job3329_legacy_controller_once_v2.py'


def main():
    if sys.flags.optimize or not sys.flags.ignore_environment or not sys.flags.no_site:
        raise RuntimeError('Use Python -E -B -S without optimization')
    source = SOURCE.read_bytes()
    review_path = OUT / 'LEGACY3329-COMPLETION-V2-INDEPENDENT-REVIEW.json'
    review = json.loads(review_path.read_bytes())
    if review.get('status') != 'PASS' or review.get('independent_review') is not True:
        raise RuntimeError('Missing genuine clean-context review')
    if review.get('source_sha256') != hashlib.sha256(source).hexdigest():
        raise RuntimeError('Reviewed source differs')
    if review.get('transport_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError('Reviewed transport differs')
    receiver = '''import base64,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
if sys.flags.optimize or not sys.flags.ignore_environment or not sys.flags.no_site:raise RuntimeError('Unisolated receiver')
if socket.gethostname()!='h100-8' or os.geteuid()!=0:raise RuntimeError('Wrong host/uid')
path=Path(REMOTE)
raw=base64.b64decode(SOURCE_B64)
if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise RuntimeError('Payload mismatch')
if path.exists():raise RuntimeError('Already staged; NEVER REPLAY')
with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
r=subprocess.run(['/usr/bin/python3','-E','-B','-S',str(path)],capture_output=True,text=True,timeout=105)
print(json.dumps(dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
'''
    receiver = receiver.replace('SOURCE_B64', repr(base64.b64encode(source).decode())).replace('SOURCE_SHA', repr(hashlib.sha256(source).hexdigest())).replace('REMOTE', repr(REMOTE))
    worker = 'env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c ' + shlex.quote(receiver)
    management = '''import json,shlex,subprocess
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
if ip!='10.33.0.8' or gateway!='10.33.0.18' or user!='root' or cfg['management_identity_file']!='/home/lichen/.ssh/id_ed25519':raise RuntimeError('Confirmed route differs')
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix+['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER],capture_output=True,text=True,timeout=125)
print(json.dumps(dict(route='retained18 SSH-W h100-8',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
'''.replace('WORKER', repr(worker))
    command = 'env -u LD_LIBRARY_PATH /usr/bin/python3 -E -B -S -c ' + shlex.quote(management)
    claim = OUT / 'LEGACY3329-COMPLETION-V2-LOCAL-INVOKE.json'
    with claim.open('x') as f:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_sha256=hashlib.sha256(source).hexdigest(),
            review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(), rule='Invoke exactly once; failure requires observation not replay'), f)
        f.flush()
        __import__('os').fsync(f.fileno())
    management_ssh = ['env', '-u', 'LD_LIBRARY_PATH', '/usr/bin/ssh', '-T',
        '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=8', '-o', 'ConnectionAttempts=1',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host']
    result = subprocess.run(management_ssh + [command], capture_output=True, text=True, timeout=150)
    record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr)
    with (OUT / 'LEGACY3329-COMPLETION-V2-ACTUAL-TRANSPORT.json').open('x') as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
