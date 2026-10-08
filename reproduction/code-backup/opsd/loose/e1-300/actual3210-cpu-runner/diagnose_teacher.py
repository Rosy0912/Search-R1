"""Authorized readonly existing-job rank0 raw/report diagnostic; no check retry."""
from pathlib import Path
import datetime,json,shlex,subprocess
H=Path(__file__).resolve().parent;E=H.parent
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
CHILD=r'''import pathlib,torch,json,hashlib,os,socket,re
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210');mp=root/'full_state/global_step_5/manifest.json';m=json.loads(mp.read_bytes())
assert hashlib.sha256(mp.read_bytes()).hexdigest()=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b'
assert os.environ.get('SLURM_JOB_ID')=='3210' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
cg=pathlib.Path('/proc/self/cgroup').read_text();assert re.search(r'(?:^|/)job_3210(?:/|$)',cg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
r=m['files']['actor/training_state/rank_0.json'];p=m['files']['actor/training_state/rank_0.pt'];rp=pathlib.Path(r['path']);pt=pathlib.Path(p['path']);assert sha(rp)==r['sha256'] and sha(pt)==p['sha256']
raw=torch.load(pt,map_location='cpu',weights_only=False);report=json.loads(rp.read_bytes());teacher={k:v for k,v in raw['teacher'].items() if k!='buffers'}
diffs=[]
def compare(a,b,path):
 if type(a)!=type(b):diffs.append({'path':path,'raw_type':type(a).__name__,'report_type':type(b).__name__})
 if isinstance(a,dict) and isinstance(b,dict):
  for k in a.keys()-b.keys():diffs.append({'path':path,'only_raw_key':repr(k),'key_type':type(k).__name__})
  for k in b.keys()-a.keys():diffs.append({'path':path,'only_report_key':repr(k),'key_type':type(k).__name__})
  for k in a.keys()&b.keys():compare(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
  if len(a)!=len(b):diffs.append({'path':path,'raw_length':len(a),'report_length':len(b)})
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif a!=b:diffs.append({'path':path,'raw_value':repr(a),'report_value':repr(b)})
compare(teacher,report['teacher'],'teacher')
projected=json.loads(json.dumps(teacher,sort_keys=True,allow_nan=False))
assert sha(rp)==r['sha256'] and sha(pt)==p['sha256'] and not torch.cuda.is_initialized()
print(json.dumps({'job_id':'3210','rank':0,'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'torch':torch.__version__,'cgroup':cg,'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'raw_pt_sha256':p['sha256'],'report_sha256':r['sha256'],'raw_teacher_equals_report':teacher==report['teacher'],'strict_json_projection_equals_report':projected==report['teacher'],'differences':diffs,'raw_counter_equals_report':teacher['counters']==report['teacher']['counters'],'cuda_is_initialized':False,'scope':'READONLY_RANK0_ACTUAL_SERIALIZATION_DIAGNOSIS_NOT_INSPECTION_PASS'},sort_keys=True,allow_nan=False))
'''
root='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210';stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=root+'/evidence/teacher-report-diagnostic-'+stamp
cmd=['srun','--jobid=3210','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--nodelist=h100-10','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python','-c',CHILD]
code='''import pathlib,json,subprocess,datetime
out=pathlib.Path(OUT);assert out.parent.resolve(strict=True)==out.parent;out.mkdir(mode=0o700)
cmd=CMD
(out/'command-original.json').write_text(json.dumps({'argv':cmd,'scope':'Own job3210 readonly rank0 diagnostic only'})+'\\n')
controller="import subprocess,json,pathlib; p=subprocess.run("+repr(cmd)+"); f=pathlib.Path("+repr(str(out/'exit-original.json'))+").open('x'); json.dump({'returncode':p.returncode},f); f.close()"
with (out/'stdout-original.txt').open('xb') as so,(out/'stderr-original.txt').open('xb') as se:p=subprocess.Popen(['python3','-c',controller],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
print(json.dumps({'output':str(out),'durable_controller_pid':p.pid}))
'''.replace('OUT',repr(out)).replace('CMD',repr(cmd))
p=subprocess.run(ns['route']('python3 -c '+shlex.quote(code)),capture_output=True,timeout=60);(H/('teacher-diagnostic-launch-'+stamp+'.json')).write_bytes(p.stdout);(H/('teacher-diagnostic-launch-'+stamp+'-stderr.txt')).write_bytes(p.stderr);print(p.stdout.decode());print('launch_exit',p.returncode)
