import sys,json,hashlib,base64
from pathlib import Path
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
code=r'''
import pathlib,json,hashlib,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');tools=root/'tools/qwen25-32b-window-safety-candidate-v6'
manifest=tools/'files.json';b=manifest.read_bytes();assert hashlib.sha256(b).hexdigest()=='f129d67d13acabd013ce290c7c1e1302255dd8bfd21af3ad2fd56ceeacdf2cc3'
records=json.loads(b);assert len(records)==14
assert {str(p.relative_to(tools)) for p in tools.rglob('*') if p.is_file()}==set(records)|{'files.json'}
installed=[]
for name,sha in records.items():
 p=tools/name;assert not p.is_symlink();data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 installed.append(dict(path=str(p),size=len(data),sha256=sha))
paths=[root/'logs/topd-window-v6-cpu-3207.log',root/'logs/r1-standard32b-sourcecpu-window-v6-submission.json',root/'logs/r1-standard32b-sourcecpu-3207-observations.jsonl',manifest]
rows=[]
for p in paths:
 a=p.stat();data=p.read_bytes();z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 rows.append(dict(path=str(p),size=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
print(json.dumps(dict(rows=rows,installed=installed)))
'''
r=send(code,attempts=3,retry_safe=True,timeout=90);assert r.returncode==0,(r.stdout,r.stderr)
v=json.loads(r.stdout);out=Path('outputs/remote-evidence/window-v6-native3207');out.mkdir(exist_ok=False)
refs=[]
for row in v['rows']:
 b=base64.b64decode(row.pop('data'));assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];(out/Path(row['path']).name).write_bytes(b);refs.append(row)
(out/'original-refs.json').write_text(json.dumps(refs,indent=2)+'\n');(out/'installed-files-verification.json').write_text(json.dumps(v['installed'],indent=2)+'\n')
sub=json.loads((out/'r1-standard32b-sourcecpu-window-v6-submission.json').read_text());term=sub['observation']['terminal'];assert term['state']=='COMPLETED' and term['exit_code']=='0:0' and term['raw']['job_id']==3207
(out/'actual-terminal.json').write_text(json.dumps(term,indent=2)+'\n')
log=(out/'topd-window-v6-cpu-3207.log').read_text();assert '117 passed' in log and 'failed' not in log and 'skipped' not in log
result=dict(status='PASS_NATIVE_CPU_CONTRACTS_NOT_GPU_APPROVAL',scope='DERIVED_FROM_ORIGINAL_NATIVE_LOG_AND_CAPTURED_CONTROLLER_NOT_BATCH_PRODUCED',job_id=3207,passed=117,skipped=0,runtime_source_sha256=records['runtime_safety.py'] if False else 'd9c495d8f2eb5bf9b1ff5680b6a8c0c2626d1e5375e92ae4971a78a9bf3dac70',adapter_source_sha256='b07dd60f697a612b00b5ac1802138d6ad7093d3998343cbb5e041e0e43de47ff',source_manifest_sha256='f129d67d13acabd013ce290c7c1e1302255dd8bfd21af3ad2fd56ceeacdf2cc3',raw_refs=refs,terminal=dict(path=str((out/'actual-terminal.json').resolve()),sha256=hashlib.sha256((out/'actual-terminal.json').read_bytes()).hexdigest()),actual_gpu_approval=False)
(out/'derived-native-cpu-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
