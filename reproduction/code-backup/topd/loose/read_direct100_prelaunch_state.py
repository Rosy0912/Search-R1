import sys,pathlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
src=r'''
import pathlib,subprocess,json,time,hashlib,socket
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');out={'scope':'READ_ONLY_CURRENT_CANDIDATE_STATE_NOT_LEASE_ACQUISITION','node':socket.gethostname(),'epoch':time.time()}
queue=subprocess.run(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,capture_output=True,timeout=10);out['queue']={'returncode':queue.returncode,'stdout':queue.stdout,'stderr':queue.stderr}
record=R/'logs/entropy-device-proof-v1-r1-final-submission.json';out['old3212record']={'sha256':hashlib.sha256(record.read_bytes()).hexdigest(),'data':json.loads(record.read_text())}
old=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:cg=(p/'cgroup').read_text()
 except FileNotFoundError:continue
 if '/job_3212/' in cg or cg.rstrip().endswith('/job_3212'):old.append({'pid':int(p.name),'cgroup':cg})
out['actual_old3212_cgroup_members']=old
out['samples']=[]
for unused in range(2):
 start=time.time();g=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);a=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);ids=[]
 for row in a.stdout.splitlines():
  fields=[x.strip() for x in row.split(',')]
  if len(fields)!=3 or not fields[1].isdecimal():continue
  p=pathlib.Path('/proc')/fields[1]
  try:
   st=(p/'stat').read_text();tail=st.rsplit(')',1)[1].split();ids.append({'pid':int(fields[1]),'starttime_ticks':int(tail[19]),'state':tail[0],'cgroup':(p/'cgroup').read_text(),'uid':p.stat().st_uid})
  except FileNotFoundError:ids.append({'pid':int(fields[1]),'identity_error':'PROC_ABSENT_NOT_ZERO'})
 out['samples'].append({'query_started_epoch':start,'query_completed_epoch':time.time(),'gpus':{'returncode':g.returncode,'stdout':g.stdout,'stderr':g.stderr},'applications':{'returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr},'identities':ids})
print(json.dumps(out))
'''
r=send(src,attempts=3,retry_safe=True,timeout=60);r.check_returncode();pathlib.Path('outputs/direct100-current-gpu-candidate-readonly.json').write_bytes(r.stdout);print(r.stdout.decode())
