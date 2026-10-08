from pathlib import Path
import shutil,hashlib,json,tarfile
W=Path('work');D=W/'standard-identity-v3-native-cpu-r2';shutil.copytree(W/'standard-identity-v3-native-cpu',D)
window=W/'qwen25-32b-window-safety-candidate-v4'
for p in sorted(window.iterdir()):
 if p.is_file():shutil.copyfile(p,D/'qwen25-32b-window-safety-candidate-v4'/p.name)
p=D/'native-cpu.sbatch';s=p.read_text().replace('standard-r1-teacher-identity-v3-cpu','standard-r1-teacher-identity-v3-r2-cpu').replace('PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v3"','PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v3-native-r2"');p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(p.relative_to(D)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='native-files.json'}
(D/'native-files.json').write_text(json.dumps({'scope':'SOURCE_DEPENDENCY_SNAPSHOTS_AND_CPU_ENTRY_ONLY','files':files},indent=2)+'\n')
a=W/'standard-identity-v3-native-cpu-r2.tar.gz'
with tarfile.open(a,'w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(D)),recursive=False)
with tarfile.open(a) as tf:
 for m in tf.getmembers():assert m.isfile() and tf.extractfile(m).read()==(D/m.name).read_bytes()
p=W/'run_standard_identity_v3_native_cpu_r2.py';s=(W/'run_standard_identity_v3_native_cpu.py').read_text().replace("NAME='standard-search-r1-teacher-identity-v3'","NAME='standard-search-r1-teacher-identity-v3-native-r2'").replace('standard-identity-v3-native-cpu.tar.gz','standard-identity-v3-native-cpu-r2.tar.gz').replace('f2fb87246bd3fb24b0f65bc1d5eb4e6520f0a2f47b5e8ab0fe6610eb1d01483e',sha(a)).replace('af29ce3be5045467b234182fdc879ca0d21fa66103b87a15dbfe4d1a2d08fda0',sha(D/'native-files.json')).replace("'standard-r1-teacher-identity-v3-cpu'","'standard-r1-teacher-identity-v3-r2-cpu'").replace("('standard-r1-teacher-identity-v3-cpu-'+jid)","('standard-r1-teacher-identity-v3-r2-cpu-'+jid)").replace('standard-identity-v3-native-cpu-local.lock','standard-identity-v3-native-cpu-r2-local.lock').replace('standard-identity-v3-native-cpu-invocation.log','standard-identity-v3-native-cpu-r2-invocation.log');p.write_text(s)
print('ARCHIVE',sha(a),'MANIFEST',sha(D/'native-files.json'),'HELPER',sha(p))
