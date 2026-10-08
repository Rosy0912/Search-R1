from pathlib import Path
import shutil,hashlib,json,tarfile
W=Path('work');D=W/'standard-identity-v4-native-cpu';shutil.copytree(W/'standard-identity-v3-native-cpu-r2',D)
for p in (D/'native-cpu.sbatch',D/'native_cpu_runner.py'):
 s=p.read_text().replace('identity-v3','identity-v4').replace('teacher-identity-v4-r2-cpu','teacher-identity-v4-cpu').replace('teacher-identity-v4-native-r2','teacher-identity-v4')
 if p.name=='native_cpu_runner.py':
  s=s.replace("import torch\nassert", "assert not list(P.rglob('__pycache__')) and not list(P.rglob('*.pyc')),'Source bytecode existed before CPU tests'\nimport torch\nassert")
  s=s.replace("text=log.read_text();print", "assert not list(P.rglob('__pycache__')) and not list(P.rglob('*.pyc')),'CPU tests wrote source bytecode'\n text=log.read_text();print")
 p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(p.relative_to(D)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='native-files.json'};(D/'native-files.json').write_text(json.dumps({'scope':'SOURCE_DEPENDENCY_SNAPSHOTS_AND_CPU_ENTRY_ONLY','files':files},indent=2)+'\n')
a=W/'standard-identity-v4-native-cpu.tar.gz'
with tarfile.open(a,'w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(D)),recursive=False)
with tarfile.open(a) as tf:
 for m in tf.getmembers():assert m.isfile() and tf.extractfile(m).read()==(D/m.name).read_bytes()
p=W/'run_standard_identity_v4_native_cpu.py';s=(W/'run_standard_identity_v3_native_cpu_r2.py').read_text().replace('standard-search-r1-teacher-identity-v3-native-r2','standard-search-r1-teacher-identity-v4').replace('standard-search-r1-teacher-identity-v3.tar.gz','standard-search-r1-teacher-identity-v4.tar.gz').replace('88dee56abf81571934a7cb4141528643f0fb468129f04b12ad46e9a1f3bc2b45','1192ade9fd52d16fbd7b0c42f2be56f74d8bd30c44b5336d20ffe0475ce5272d').replace('19e8d6b26d41480c7e53b8365cf41175d2b05d56e04bd2fb9e7098466d91ef2e','f545b2241325b00c16b7c2e20f867befc93b2694051955e1730fef78c8007c95').replace('standard-search-r1-teacher-identity-v3-files.json','standard-search-r1-teacher-identity-v4-files.json').replace('standard-identity-v3-native-cpu-r2.tar.gz','standard-identity-v4-native-cpu.tar.gz').replace('b20b784fd9e29eb738487281d826406a1c61f911cea3be87a3ecdabfadbe2d3a',sha(a)).replace('bcf05b68228ae7144fe2e4689e050b9d557a94d7bcf9fec90d74f7bd3808cc67',sha(D/'native-files.json')).replace('standard-r1-teacher-identity-v3-r2-cpu','standard-r1-teacher-identity-v4-cpu').replace('standard-identity-v3-native-cpu-r2','standard-identity-v4-native-cpu');p.write_text(s)
freeze=W.parent/'outputs/standard-search-r1-teacher-identity-v4-freeze.json';r=json.loads(freeze.read_text());r['cpu_contracts']['baseline_reused_from_v3']=True;r['cpu_contracts']['v4_incremental_passed']=11;r['cpu_contracts']['v4_incremental_log']='outputs/standard-identity-v4-incremental-tests.log';freeze.write_text(json.dumps(r,indent=2)+'\n')
print('ARCHIVE',sha(a),'MANIFEST',sha(D/'native-files.json'),'HELPER',sha(p))
