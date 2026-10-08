import sys,json,hashlib,base64
from pathlib import Path
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
code=r'''
import pathlib,json,hashlib,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=root/'tools/qwen25-32b-window-safety-candidate-v5'
manifest=tools/'files.json';b=manifest.read_bytes();assert hashlib.sha256(b).hexdigest()=='c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0'
records=json.loads(b);assert len(records)==12
assert {str(p.relative_to(tools)) for p in tools.rglob('*') if p.is_file()}==set(records)|{'files.json'}
installed=[]
for name,sha in records.items():
 p=tools/name;assert not p.is_symlink();data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 installed.append(dict(path=str(p),size=len(data),sha256=sha))
paths=[root/'logs/topd-window-v5-cpu-3194.log',root/'logs/r1-standard32b-sourcecpu-window-v5-submission.json',root/'logs/r1-standard32b-sourcecpu-3194-observations.jsonl',manifest]
rows=[]
for p in paths:
 a=p.stat();data=p.read_bytes();z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 rows.append(dict(path=str(p),size=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
print(json.dumps(dict(rows=rows,installed=installed)))
'''
r=send(code,attempts=3,retry_safe=True,timeout=90);assert r.returncode==0,(r.stdout,r.stderr)
v=json.loads(r.stdout);out=Path('outputs/remote-evidence/window-v5-native3194');out.mkdir(exist_ok=False)
refs=[]
for row in v['rows']:
 b=base64.b64decode(row.pop('data'));assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];(out/Path(row['path']).name).write_bytes(b);refs.append(row)
(out/'original-refs.json').write_text(json.dumps(refs,indent=2)+'\n');(out/'installed-files-verification.json').write_text(json.dumps(v['installed'],indent=2)+'\n')
sub=json.loads((out/'r1-standard32b-sourcecpu-window-v5-submission.json').read_text());term=sub['observation']['terminal'];assert term['state']=='COMPLETED' and term['exit_code']=='0:0' and term['raw']['job_id']==3194
(out/'actual-terminal.json').write_text(json.dumps(term,indent=2)+'\n')
log=(out/'topd-window-v5-cpu-3194.log').read_text();assert '111 passed' in log and 'failed' not in log and 'skipped' not in log
result=dict(status='PASS_NATIVE_CPU_CONTRACTS_NOT_GPU_APPROVAL',scope='DERIVED_FROM_ORIGINAL_NATIVE_LOG_AND_CAPTURED_CONTROLLER_NOT_BATCH_PRODUCED',job_id=3194,passed=111,skipped=0,runtime_source_sha256=records['runtime_safety.py'] if False else '9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6',adapter_source_sha256='65165ac7c8f5c9042ba254fd36d26e222096e6764fa83c167061e25530d5ad39',source_manifest_sha256='c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0',raw_refs=refs,terminal=dict(path=str((out/'actual-terminal.json').resolve()),sha256=hashlib.sha256((out/'actual-terminal.json').read_bytes()).hexdigest()),actual_gpu_approval=False)
(out/'derived-native-cpu-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
