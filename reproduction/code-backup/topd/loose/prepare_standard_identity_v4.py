from pathlib import Path
import shutil
W=Path('work');names=['qwen25-32b-trained-eval64-r1','topd-32b-native-score-gate-r1','topd-32b-shared-owner-r1','topd-32b-pipeline-ipc-r1','topd-shared32b-supervisor-r1']
for n in names:
 d=W/(n+'-v4');assert not d.exists();shutil.copytree(W/(n+'-v3'),d,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
 for p in d.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.sbatch'):
   s=p.read_text()
   for old in names:s=s.replace(old+'-v3',old+'-v4')
   p.write_text(s)
for old,new in [('refresh_standard_teacher_identity_v3_pins.py','refresh_standard_teacher_identity_v4_pins.py'),('freeze_standard_teacher_identity_v3.py','freeze_standard_teacher_identity_v4.py'),('test_standard_teacher_identity_v3_chain.py','test_standard_teacher_identity_v4_chain.py')]:
 s=(W/old).read_text()
 for n in names:s=s.replace(n+'-v3',n+'-v4')
 s=s.replace('identity-v3','identity-v4').replace('test_standard_teacher_identity_v3_chain.py','test_standard_teacher_identity_v4_chain.py')
 (W/new).write_text(s)
p=W/'topd-32b-native-score-gate-r1-v4/test_caller_contracts.py';s=p.read_text();assert "[sys.executable,'-I','-c',code]" in s;s=s.replace("[sys.executable,'-I','-c',code]","[sys.executable,'-I','-B','-c',code]");p.write_text(s)
