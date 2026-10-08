import sys,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
src=r'''
import pathlib,tarfile,hashlib,json,subprocess,os,time,socket
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
T=R/'tools/search-r1-32b-grpo100-tools-direct-v2';D=R/'logs/r1-standard32b-grpo100-direct-v2-r1';record=R/'logs/r1-standard32b-grpo100-direct-v2-r1-submission.json'
if record.exists():
 print(json.dumps({'existing_record':json.loads(record.read_text()),'scope':'NO_DUPLICATE_SUBMISSION'}));raise SystemExit(0)
q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,timeout=15)
assert not any('r1-32b-grpo' in x for x in q.splitlines()),q
A=R/'tools/search-r1-32b-grpo100-tools-direct-v2.tar.gz'
assert hashlib.sha256(A.read_bytes()).hexdigest()=='a9b4c0e89a5135bcef299f370b6516bc1b9ec2cea521eee7e0d22d29c540686a'
if not T.exists():
 with tarfile.open(A) as tf:
  for m in tf.getmembers():assert not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts and not m.issym() and not m.islnk()
  tf.extractall(R/'tools')
assert hashlib.sha256((T/'files.json').read_bytes()).hexdigest()=='e8ab6bcb50e579d4be3e2c62b9681b684cb5a2e7615e9cbea490d817548f5203'
prior=R/'logs/r1-standard32b-profile2-v14/bound-static-config-v14.json'
if not prior.exists():prior=R/'logs/r1-standard32b-profile2-v14/static-config.json'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
if not D.exists():subprocess.run([str(R/'env/bin/python'),str(T/'prepare.py'),'--existing-asset-config',str(prior),'--mode','train100','--directory',str(D)],env=env,check=True)
c=json.loads((D/'static-config.json').read_text())
c['telemetry_policy']['actual_independent_native_review']={'path': '/dataset/shaoshitong/search-r1-rl/topd/logs/window-v6-native3207-independent-cpu-policy-review.json', 'sha256': '7a6f6f828cd13fabeb0a6eab981177f1f07c8ce9b424bef9fc87f75c6a92e75d'}
(D/'static-config.json').write_text(json.dumps(c,indent=2)+'\n')
env.update(R1_CONFIG=str(D/'static-config.json'),R1_TOOLS=str(T),R1_PREFLIGHT=c['preflight_source']['path'])
rec={'status':'UNKNOWN_BEFORE_SUBMISSION','epoch':time.time(),'directory':str(D),'config':env['R1_CONFIG'],'micro_global':32,'micro_rank':4,'expected_outer':100}
with record.open('x') as f:json.dump(rec,f);f.flush();os.fsync(f.fileno())
r=subprocess.run(['sbatch','--parsable',str(T/'production.sbatch')],env=env,capture_output=True,text=True,timeout=30)
rec.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:
 jid=r.stdout.strip().split(';')[0];assert jid.isdecimal();rec.update(status='SUBMITTED',job_id=int(jid))
else:rec['status']='SUBMISSION_FAILED'
p=record.with_suffix('.tmp');p.write_text(json.dumps(rec,indent=2)+'\n');os.replace(p,record)
print(json.dumps(rec));raise SystemExit(r.returncode)
'''
r=send(src,attempts=1,timeout=180)
pathlib.Path('outputs/standard-r1-direct100-v2-r1-submission-result.log').write_bytes(r.stdout+r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
