from pathlib import Path
import hashlib,json,tarfile,io
W=Path('work');names=['qwen25-32b-trained-eval64-r1-v1','topd-32b-native-score-gate-r1-v1','topd-32b-shared-owner-r1-v1','topd-32b-pipeline-ipc-r1-v1','topd-shared32b-supervisor-r1-v1'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for name in names for p in sorted((W/name).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[W/'test_standard_teacher_identity_chain.py']
man={str(p.relative_to(W)):{'size':p.stat().st_size,'sha256':sha(p)} for p in files};payload=json.dumps({'scope':'SOURCE_CPU_ONLY_NO_ACTUAL100_NO_TEACHER_APPROVAL','files':man},indent=2).encode()+b'\n';mp=W/'standard-search-r1-teacher-identity-v1-files.json';mp.write_bytes(payload)
archive=W/'standard-search-r1-teacher-identity-v1.tar.gz'
if archive.exists():raise RuntimeError('Immutable archive already exists')
with tarfile.open(archive,'w:gz') as tf:
 for p in files+[mp]:tf.add(p,arcname=str(p.relative_to(W)),recursive=False)
with tarfile.open(archive) as tf:
 for p in files+[mp]:assert tf.extractfile(str(p.relative_to(W))).read()==p.read_bytes()
record={'scope':'SOURCE_CPU_ONLY_NO_ACTUAL100_NO_TEACHER_APPROVAL','archive':str(archive),'archive_sha256':sha(archive),'manifest_sha256':sha(mp),'payload_count':len(files),'canonical_gate_sha256':sha(W/names[1]/'score_gate.py'),'cpu_contracts':{'chain_and_eval':31,'score':49,'owner':26,'ipc':17,'supervisor':62,'passed':185,'linux_peer_skipped':1},'native_result':'PENDING','actual100':'PENDING','teacher_quality':'PENDING','numerical_coexistence':'PENDING','tool_manifest_sha256':{name:sha(W/name/('tool-files.json' if name==names[1] else 'files.json')) for name in names}}
Path('outputs/standard-search-r1-teacher-identity-v1-freeze.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
