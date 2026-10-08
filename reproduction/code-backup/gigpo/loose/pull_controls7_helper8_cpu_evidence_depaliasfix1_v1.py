"""Read-only retrieval of original owned CPU results, never replay a test."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
base=Path.cwd();role='controls'
name='CONTROLS7';review=json.loads((base/'outputs/gigpo-e1-instruct200/CONTROLS7-HELPER8-DEPALIASFIX1-INDEPENDENT-PREEXEC-REVIEW-v1.json').read_text());directory=review['approved_cpu_policy']['output_directory'];target=base/'outputs/gigpo-e1-instruct200'/('cpu-'+role+'-'+Path(directory).name)
files=['STARTED.json','COLLECTION-RESULT.json','RESULT.json','collection.log','pytest.log','junit.xml']
code='''import base64,hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);directory=Path(p['directory']);assert directory.is_relative_to('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/cpu-tests')
records=[]
for name in p['files']:
 path=directory/name;assert path.is_file() and not path.is_symlink();raw=path.read_bytes();records.append(dict(path=str(path),name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),b64=base64.b64encode(raw).decode()))
print(json.dumps(records))
'''
inner='/usr/bin/python3 -c '+shlex.quote(code);mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner);route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)];r=subprocess.run(route,input=json.dumps(dict(directory=directory,files=files)).encode(),capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
rows=json.loads(r.stdout);target.mkdir(exist_ok=False);receipts=[]
for row in rows:
 raw=base64.b64decode(row.pop('b64'));assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];dest=target/row['name']
 with dest.open('xb') as f:f.write(raw)
 receipts.append(dict(remote_record={k:row[k] for k in ('path','bytes','sha256')},local_record=dict(path=str(dest),bytes=len(raw),sha256=row['sha256'])))
with (target/'RAW-PULL-RECEIPT.json').open('x') as f:json.dump(dict(status='EXACT_ORIGINAL_BYTES_READ_ONLY',files=receipts,tests_reexecuted=False),f,indent=2);f.write('\n')
x=json.loads((target/'RESULT.json').read_text());print(json.dumps(dict(directory=str(target),status=x['status'],actual_cases=x['actual_cases'],exit_code=x['exit_code'],failure_ids=x['failure_ids'],exact_collection_execution_coverage=x['exact_collection_execution_coverage'])))
