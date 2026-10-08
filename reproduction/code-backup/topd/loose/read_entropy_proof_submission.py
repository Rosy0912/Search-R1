import sys,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
src="""
import pathlib,json,hashlib,time,subprocess
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/entropy-device-proof-v1-r1-final-submission.json')
d={'epoch':time.time(),'record_exists':p.exists()}
if p.exists():
 b=p.read_bytes();r=json.loads(b);d.update(record=r,record_sha256=hashlib.sha256(b).hexdigest(),record_raw=b.decode())
 if r.get('job_id'):
  s=subprocess.run(['scontrol','show','job','-dd',str(r['job_id'])],capture_output=True,text=True,timeout=5);d['controller']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
print(json.dumps(d))
"""
r=send(src,attempts=3,retry_safe=True,timeout=60)
p=pathlib.Path('outputs/entropy-proof-submission-readonly-response.json');p.write_bytes(r.stdout);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
