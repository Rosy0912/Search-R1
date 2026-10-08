from pathlib import Path
import hashlib,json,tarfile
W=Path('work');names=['qwen25-32b-trained-eval64-r1-v5','topd-32b-native-score-gate-r1-v5','topd-32b-shared-owner-r1-v5','topd-32b-pipeline-ipc-r1-v5','topd-shared32b-supervisor-r1-v5'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for n in names for p in sorted((W/n).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[W/'test_standard_teacher_identity_v5_chain.py']
m={'scope':'SOURCE_PIN_UPDATE_ONLY_NO_ACTUAL_PROFILE100_TEACHER_OR_TOPD_APPROVAL','source_manifest_sha256':'9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0','files':{str(p.relative_to(W)):{'size':p.stat().st_size,'sha256':sha(p)} for p in files}}
mp=W/'standard-search-r1-teacher-identity-v5-files.json';mp.write_text(json.dumps(m,indent=2)+'\n');a=W/'standard-search-r1-teacher-identity-v5.tar.gz';assert not a.exists()
with tarfile.open(a,'w:gz') as t:
 for p in files+[mp]:t.add(p,arcname=str(p.relative_to(W)),recursive=False)
with tarfile.open(a) as t:
 for p in files+[mp]:assert t.extractfile(str(p.relative_to(W))).read()==p.read_bytes()
delta={}
for n in names:
 for p in sorted((W/n).rglob('*')):
  if p.is_file():
   old=W/n.replace('-r1-v5','-r1-v4')/p.relative_to(W/n)
   if old.is_file() and p.read_bytes()!=old.read_bytes():delta[str(p.relative_to(W))]={'old_sha256':sha(old),'new_sha256':sha(p)}
record={'scope':m['scope'],'archive':str(a),'archive_sha256':sha(a),'manifest_sha256':sha(mp),'payload_count':len(files),'canonical_gate_sha256':sha(W/names[1]/'score_gate.py'),'new_source_manifest_sha256':m['source_manifest_sha256'],'tool_manifests':{n:sha(W/n/('tool-files.json' if n==names[1] else 'files.json')) for n in names},'delta':delta,'cpu_contracts':{'focused_actual_chain_and_training_schema':25,'log':'outputs/standard-identity-v5-repin-tests.log','native':'PENDING_NOT_REUSING_NATIVE3166'}}
Path('outputs/standard-search-r1-teacher-identity-v5-freeze.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='delta'},indent=2))
p=W/'r1-32b-hf-semantic-review-v4';payload={f.name:{'size':f.stat().st_size,'sha256':sha(f)} for f in sorted(p.iterdir()) if f.is_file() and f.name!='files.json'};(p/'files.json').write_text(json.dumps({'scope':'SOURCE_PREPARATION_ONLY_NO_ACTUAL_HF_SCAN','files':payload},indent=2)+'\n');a=p.with_suffix('.tar.gz');assert not a.exists()
with tarfile.open(a,'w:gz') as t:
 for f in sorted(p.iterdir()):
  if f.is_file():t.add(f,arcname=p.name+'/'+f.name)
with tarfile.open(a) as t:
 for member in t.getmembers():assert member.isfile() and t.extractfile(member).read()==(p/Path(member.name).name).read_bytes()
record={'scope':'SOURCE_PREPARATION_ONLY_NO_ACTUAL_HF_SCAN','archive':str(a),'archive_sha256':sha(a),'manifest_sha256':sha(p/'files.json'),'source_manifest_sha256':m['source_manifest_sha256'],'files':payload,'cpu_contracts':34,'test_log':'outputs/r1-32b-hf-semantic-review-v4-local-tests.log','native':'PENDING','actual_HF_scan':'NOT_RUN'}
Path('outputs/r1-32b-hf-semantic-review-v4-freeze.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
