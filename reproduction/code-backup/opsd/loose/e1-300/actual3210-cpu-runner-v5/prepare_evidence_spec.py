"""Actual completed CPU evidence joins; no review PASS/signature/continuation."""
from pathlib import Path
import hashlib,importlib.util,json,shlex,subprocess
H=Path(__file__).resolve().parent;E=H.parent;SMALL=H/'actual-small-20261004T174500Z';INDEX=json.loads((SMALL/'index.json').read_text());records={Path(r['path']).name:r for r in INDEX['records'] if 'local_path' in r}
for rec in records.values():assert hashlib.sha256(Path(rec['local_path']).read_bytes()).hexdigest()==rec['sha256']
def read(name):return json.loads(Path(records[name]['local_path']).read_text())
for n in ['native-exit-original.json','hf-exit-original.json','srun-exit-original.json']:assert read(n)['returncode']==0
native=read('native.json');hf=read('hf.json');manifest=read('manifest.json');reports=[read('rank_'+str(r)+'.json') for r in range(8)]
producer=E/'first5-tools-a096-raydedup-candidate/receipt_payload.py';ps=importlib.util.spec_from_file_location('actual_unsigned_schema_only',producer);module=importlib.util.module_from_spec(ps);ps.loader.exec_module(module)
module.check_cpu_evidence(native,hf,manifest,INDEX['manifest_sha256'],reports)
assert native['inspector_source_sha256']=='b017a443f1256d9eedf3d4cdfa6aca1073596fd5e5cf416cd41a8150a58a7623' and hf['inspector_source_sha256']=='38bc089be3e35f193d6192e200beb4155f02583fc1993996bf85ff2198d98aec'
partial=E/'formal3210-pause-resource-independent-20261004T171020Z/resource-service-evidence-spec-partial.json';assert hashlib.sha256(partial.read_bytes()).hexdigest()=='3669f283e5090609a8784ff0c2787c4bf98997da99cd11e9bea8bd75d94ce5f9';spec=json.loads(partial.read_text());assert set(spec)=={'resources','retrieval','service'}
for role,name in [('rank_audit','native.json'),('hf_reload','hf.json')]:spec[role]={k:records[name][k] for k in ('path','sha256')}
assert set(spec)=={'resources','retrieval','service','rank_audit','hf_reload'}
remote_path=INDEX['output']+'/actual-five-role-evidence-spec.json';data=json.dumps(spec,sort_keys=True,indent=2).encode()+b'\n';pin=hashlib.sha256(data).hexdigest()
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
code='''import pathlib,json,hashlib,sys,os
p=pathlib.Path(PATH);assert os.getuid()==0 and p.parent.resolve(strict=True)==p.parent
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==PIN
spec=json.loads(raw)
for rec in spec.values():
 f=pathlib.Path(rec['path']);assert f.resolve(strict=True)==f and f.is_relative_to(pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210')) and f.stat().st_uid==0
 assert hashlib.sha256(f.read_bytes()).hexdigest()==rec['sha256']
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(p),'sha256':PIN,'roles':sorted(spec),'scope':'Actual artifact index only; not independent PASS/signing/release'}))
'''.replace('PATH',repr(remote_path)).replace('PIN',repr(pin))
p=subprocess.run(ns['route']('python3 -c '+shlex.quote(code)),input=data,capture_output=True,timeout=60)
(H/'evidence-spec-remote-stdout-original.json').write_bytes(p.stdout);(H/'evidence-spec-remote-stderr-original.txt').write_bytes(p.stderr);assert p.returncode==0,p.stderr
(H/'actual-five-role-evidence-spec.json').write_bytes(data);print(p.stdout.decode())
print(json.dumps({'native_ranks':len(native['ranks']),'native_adam':sorted(set(s for r in native['ranks'] for s in r['optimizer_step_values'])),'hf_changed_elements':hf['total_changed_elements'],'hf_unique_parameters':hf['actual_unique_parameters'],'both_cuda_uninitialized':not native['cuda_is_initialized'] and not hf['cuda_is_initialized'],'scope':'Schema check only; independent/root actual reviews still required'}))
