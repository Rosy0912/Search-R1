import sys,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import pathlib,hashlib,json,subprocess,tarfile,os,time,shutil
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');rec=R/'logs/topd3224-after3228-amendment.json'
if rec.exists():print(rec.read_text());raise SystemExit(0)
s=subprocess.check_output(['scontrol','-o','show','job','3224'],text=True,timeout=15);assert 'JobState=PENDING' in s,s
A=R/'tools/topd-instruct300-launch-v5.tar.gz';assert hashlib.sha256(A.read_bytes()).hexdigest()=='a7c7bee12de2a42c0c5e8ed98f4c7452bf4a7af7663146cc7a543d1398da7f6c'
if not (R/'tools/topd-instruct300-launch-v5').exists():
 with tarfile.open(A) as t:
  for m in t.getmembers():assert not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts and not m.issym() and not m.islnk()
  t.extractall(R/'tools')
for file,sha in [('bind_topd_instruct300_v7.py','66628a48281b5718404c3ed3dab958ae0d12224db5d2550018cfc8bde8d9c978'),('run_topd_instruct300_after_teacher100_v5.py','2465108b30fe96b40315a505f8f6f1fc5169c4ea68732fc62c3284ccd693b618')]:assert hashlib.sha256((R/'tools'/file).read_bytes()).hexdigest()==sha
old=R/'tools/start_topd_after_teacher3218.py';new=R/'tools/start_topd_after_teacher3228.py';backup=R/'tools/start_topd_after_teacher3218-before3228-amendment.py';assert not backup.exists();shutil.copyfile(old,backup);shutil.copyfile(new,old)
x=subprocess.run(['scontrol','update','JobId=3224','Dependency=afterok:3228'],capture_output=True,text=True,timeout=15)
after=subprocess.check_output(['scontrol','-o','show','job','3224'],text=True,timeout=15)
r={'epoch':time.time(),'job_id':3224,'teacher_job_id':3228,'before':s,'after':after,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'dispatcher_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'original_dispatcher_backup':str(backup),'no_new_submission':True}
with rec.open('x') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(r));assert x.returncode==0 and 'afterok:3228' in after
'''
r=send(source,attempts=1,timeout=120);pathlib.Path('outputs/topd3224-after3228-amendment-result.log').write_bytes(r.stdout+r.stderr);print(r.stdout.decode());print(r.stderr.decode());sys.exit(r.returncode)
