#!/usr/bin/env python3
"""Deploy only this bounded backup entry over the verified management route.

Only code and small metadata cross this route; native/model bytes stay on H100.
An existing launch journal is inspected rather than relaunched.
"""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'tree-speed-repair-20261008/ssh_retained18.py'
REQUEST = 'retained18-native-once-20261008-r2'


def main():
    spec = importlib.util.spec_from_file_location('retained18_ssh', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = (HERE / 'backup_native_once.py').read_bytes()
    code_sha = hashlib.sha256(raw).hexdigest()
    remote = '''import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/.shared-output-backup-20261008')
assert not root.is_symlink()
root.mkdir(exist_ok=True)
assert root.resolve(strict=True)==root
request=REQUEST
entry=root/('backup_native_once-'+CODE_SHA[:16]+'.py')
raw=base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest()==CODE_SHA
if entry.exists():
 assert hashlib.sha256(entry.read_bytes()).hexdigest()==CODE_SHA
else:
 with entry.open('xb') as f:
  f.write(raw); f.flush(); os.fsync(f.fileno())
journal=root/(request+'.launch.json')
status=root/(request+'.status.json')
if journal.exists():
 print(json.dumps({'outcome':'EXISTING_JOURNAL_NO_DUPLICATE','journal':json.loads(journal.read_text()),'status':json.loads(status.read_text()) if status.exists() else None}))
else:
 state={'request_id':request,'entry':str(entry),'entry_sha256':CODE_SHA,'state':'LAUNCH_INTENT','created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 with journal.open('x') as f:
  json.dump(state,f,indent=2); f.flush(); os.fsync(f.fileno())
 with (root/(request+'.log')).open('ab',buffering=0) as output:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
  proc=subprocess.Popen(['nice','-n','19','ionice','-c','3','python3',str(entry),'--execute','--request-id',request],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 state.update(state='LAUNCHED',pid=proc.pid)
 tmp=journal.with_suffix('.launch.tmp')
 with tmp.open('x') as f:
  json.dump(state,f,indent=2); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,journal)
 print(json.dumps({'outcome':'LAUNCHED','journal':state}))
'''
    remote = remote.replace('REQUEST', repr(REQUEST)).replace('CODE_SHA', repr(code_sha)).replace('PAYLOAD', repr(base64.b64encode(raw).decode()))
    result = helper.run('python3 -c ' + shlex.quote(remote), timeout=60)
    record = dict(observed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), request_id=REQUEST,
                  entry_sha256=code_sha, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (HERE / 'metadata').mkdir(exist_ok=True)
    (HERE / 'metadata/deploy-result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
