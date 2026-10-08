from pathlib import Path
import shutil,hashlib,json,tarfile
W=Path('work');D=W/'standard-identity-v4-native-cpu-r2';shutil.copytree(W/'standard-identity-v4-native-cpu',D)
p=D/'native-cpu.sbatch';s=p.read_text().replace('standard-r1-teacher-identity-v4-cpu','standard-r1-teacher-identity-v4-r2-cpu').replace('PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v4"','PACKAGE="$TASK_ROOT/tools/standard-search-r1-teacher-identity-v4-native-r2"');p.write_text(s)
p=D/'native_cpu_runner.py';s=p.read_text().replace('rows=[]',"T=R.parent/('ti'+os.environ['SLURM_JOB_ID']);T.mkdir(mode=0o700)\nrows=[]").replace("str(R/(name+'-tmp'))","str(T/name[0])");p.write_text(s)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads((D/'native-files.json').read_text())
for p in (D/'native-cpu.sbatch',D/'native_cpu_runner.py'):r['files'][p.name]={'size':p.stat().st_size,'sha256':sha(p)}
(D/'native-files.json').write_text(json.dumps(r,indent=2)+'\n')
a=W/'standard-identity-v4-native-cpu-r2.tar.gz'
with tarfile.open(a,'w:gz') as tf:
 for p in sorted(D.rglob('*')):
  if p.is_file():tf.add(p,arcname=str(p.relative_to(D)),recursive=False)
with tarfile.open(a) as tf:
 for m in tf.getmembers():assert m.isfile() and tf.extractfile(m).read()==(D/m.name).read_bytes()
p=W/'run_standard_identity_v4_native_cpu_r2.py';s=(W/'run_standard_identity_v4_native_cpu.py').read_text().replace("NAME='standard-search-r1-teacher-identity-v4'","NAME='standard-search-r1-teacher-identity-v4-native-r2'").replace('standard-identity-v4-native-cpu-r1.tar.gz','standard-identity-v4-native-cpu-r2.tar.gz').replace('a47f57c46d43aab741344066df8a99fa55b82751d8d91f6a88401058ef4be9f9',sha(a)).replace('8d66117cb9f5c0ecbb256839aff6219a79bf0e1b9f9a2cb79da77f8a19dc3572',sha(D/'native-files.json')).replace("'standard-r1-teacher-identity-v4-cpu'","'standard-r1-teacher-identity-v4-r2-cpu'").replace("('standard-r1-teacher-identity-v4-cpu-'+jid)","('standard-r1-teacher-identity-v4-r2-cpu-'+jid)").replace('standard-identity-v4-native-cpu-local.lock','standard-identity-v4-native-cpu-r2-local.lock').replace('standard-identity-v4-native-cpu-invocation.log','standard-identity-v4-native-cpu-r2-invocation.log');p.write_text(s)
print('ARCHIVE',sha(a),'MANIFEST',sha(D/'native-files.json'),'HELPER',sha(p));print('MAX_TEST_SOCKET',len('/dataset/shaoshitong/search-r1-rl/topd/logs/ti3167/i/test_real_linux_eight_sockets_0/r0'))
