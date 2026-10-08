"""Continue the same saved intent after its source-input transfer timed out.

No new intent, namespace, source, test, review, or training recipe is created.
The failed call never reached namespace creation. A compressed input avoids
repeating a large uncompressed source JSON transfer through the SSH chain.
"""
import base64
import importlib.util
import json
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('same_retry', HERE / 'run_direct_retry.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)
native = retry.native
intent = json.loads((HERE / native.NAMES['intent']).read_text())
inputs, files, method = retry.prepare_inputs()

if (HERE / native.NAMES['submitted']).exists():
    raise RuntimeError('Receipt already exists; observe the same job')

def rpc(host, code, payload, timeout=90):
    return native.remote(host, code, payload, timeout=timeout,
                         producer_sha256=intent['cpu_producer_sha256'])

package_path = HERE / native.NAMES['package']
if not package_path.exists():
    payload = dict(intent,
        files={name: base64.b64encode(content).decode() for name, content in files.items()},
        method_bytes=base64.b64encode(method).decode())
    packed = base64.b64encode(zlib.compress(native.canonical(payload), 6)).decode()
    code = native.PREPARE_CODE.replace('import base64,hashlib,json,os,shutil,stat,sys,time',
                                      'import base64,hashlib,json,os,shutil,stat,sys,time,zlib')
    code = code.replace('p=json.load(sys.stdin);anchor=',
        "p=json.loads(zlib.decompress(base64.b64decode(json.load(sys.stdin)['zlib_payload'])));anchor=")
    assert code != native.PREPARE_CODE
    if not (HERE / 'compressed-prepare-input.json').exists():
        native.save(HERE / 'compressed-prepare-input.json', dict(
            base=intent['base'], source_commit=intent['source_commit'],
            uncompressed_bytes=len(native.canonical(payload)), compressed_base64_bytes=len(packed),
            checked_unix=time.time(), same_saved_intent=True, no_new_submission=True))
    # The first call can finish remotely after its local capture times out.
    # Recover its completed package from disk instead of writing it twice.
    recover = native.ARTIFACT_INPUT_CODE + r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base'])
assert b.resolve()==b and b.parent==Path('/mnt/data/shaoshitong')
if not b.exists():
 print(json.dumps({'base_exists':False}))
else:
 owner=json.loads(bounded_artifact_bytes(b/'preparation/namespace-owner.json',256*1024))
 package=json.loads(bounded_artifact_bytes(b/'preparation/source-package-intent.json',256*1024))
 assert owner['base']==package['base']==str(b) and owner['uid']==os.getuid()
 for key in ('source_commit','source_sha256','source_count','method_sha256','created_unix','job_name'):
  assert package[key]==p[key]
 assert not (b/'preparation/native-submit-claim.json').exists() and not (b/'logs/run-manifest.json').exists()
 q=subprocess.run(['squeue','-h','-n',p['job_name'],'-o','%i|%T|%j'],capture_output=True,text=True,timeout=15)
 assert q.returncode==0 and not q.stdout.strip()
 for name,expected in p['source_sha256'].items():
  assert hashlib.sha256(bounded_artifact_bytes(b/'repo'/name,16*1024*1024)).hexdigest()==expected
 data_hashes={}
 anchor=Path('/mnt/data/shaoshitong/g1p-302569b1/data')
 for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
  content=bounded_artifact_bytes(b/'data'/name,64*1024*1024)
  digest=hashlib.sha256(content).hexdigest()
  assert digest==hashlib.sha256(bounded_artifact_bytes(anchor/name,64*1024*1024)).hexdigest()
  data_hashes[name]=digest
 assert hashlib.sha256(bounded_artifact_bytes(b/'preparation/formal-inline-method.json',256*1024)).hexdigest()==p['method_sha256']
 print(json.dumps(dict(base_exists=True,base=p['base'],execution_role=p['execution_role'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],source_count=p['source_count'],data_sha256=data_hashes,method_sha256=p['method_sha256'],checked_unix=time.time(),scope='Recovered completed same-intent package after local transfer timeout; no duplicated writes or sbatch')))
'''
    prepared = rpc('h100-8', recover, intent, timeout=60)
    if prepared['base_exists'] is False:
        prepared = rpc('h100-8', code, {'zlib_payload': packed}, timeout=180)
    native.save(package_path, prepared)

handle_path = HERE / native.NAMES['handle']
if handle_path.exists():
    handle = json.loads(handle_path.read_text())
else:
    source = native.exporter_bytes(files, intent['export_base'], intent['export_port'])
    handle = rpc('h100-9', native.START_SOURCE_CODE,
                 dict(intent, source=base64.b64encode(source).decode()))
    native.save(handle_path, handle)

failed = json.loads((retry.ORIGINAL / native.NAMES['intent']).read_text())
actual = rpc('h100-8', retry.FAILED_READONLY,
             dict(failed_source_sha256=failed['source_sha256']), timeout=60)
native.save(HERE / 'failed3320-pre-sbatch-readonly.json', actual)
submitted = rpc('h100-8', native.SUBMIT_CODE, dict(intent, handle=handle), timeout=90)
native.save(HERE / native.NAMES['submitted'], submitted)
print(json.dumps({key: value for key, value in submitted.items()
                  if key not in ('source_sha256', 'argv')}, indent=2))
