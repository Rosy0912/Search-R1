#!/usr/bin/env python3
"""Journal one isolated finalize request; existing intent is read, never relaunched."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import time

HERE = Path(__file__).resolve().parent
REQUEST = 'retained18-native-finalize-r2-20261008'


def main():
    spec = importlib.util.spec_from_file_location('route', HERE.parent / 'tree-speed-repair-20261008/ssh_retained18.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = (HERE / 'finalize_existing_backup.py').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    code = '''import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/.shared-output-backup-20261008')
assert root.is_dir() and not root.is_symlink() and root.resolve(strict=True)==root
request=REQUEST
journal=root/(request+'.launch.json')
status=root/(request+'.status.json')
intent=root/(request+'.request.json')
if journal.exists() or intent.exists():
 print(json.dumps({'outcome':'EXISTING_INTENT_NO_DUPLICATE','journal':json.loads(journal.read_text()) if journal.exists() else None,'intent':json.loads(intent.read_text()) if intent.exists() else None,'status':json.loads(status.read_text()) if status.exists() else None}))
else:
 entry=root/('finalize_existing_backup-'+DIGEST[:16]+'.py')
 raw=base64.b64decode(PAYLOAD)
 assert hashlib.sha256(raw).hexdigest()==DIGEST
 if entry.exists():assert hashlib.sha256(entry.read_bytes()).hexdigest()==DIGEST
 else:
  with entry.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 state={'request_id':request,'entry':str(entry),'entry_sha256':DIGEST,'state':'FINALIZE_LAUNCH_INTENT','created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 with journal.open('x') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
 with (root/(request+'.log')).open('ab',buffering=0) as output:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  p=subprocess.Popen(['nice','-n','19','ionice','-c','3','python3',str(entry)],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 state.update(state='FINALIZE_LAUNCHED',pid=p.pid)
 tmp=journal.with_suffix('.writing')
 with tmp.open('x') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,journal)
 print(json.dumps({'outcome':'FINALIZE_LAUNCHED','journal':state}))
'''.replace('REQUEST', repr(REQUEST)).replace('DIGEST', repr(digest)).replace('PAYLOAD', repr(base64.b64encode(raw).decode()))
    result = helper.run('python3 -c ' + shlex.quote(code), timeout=60)
    record = dict(observed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), request_id=REQUEST,
                  entry_sha256=digest, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (HERE / 'metadata/finalize-deploy-result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError('Finalize launch returned failure; inspect its existing journal before retry')


if __name__ == '__main__':
    main()
