"""Explicit actual author200 job only, stored-state/CPU HF inspection; never signs/releases."""
import argparse,datetime,hashlib,importlib.util,json,os,re,socket,subprocess,sys
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
ASSET_ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004')
JOB=None;RUN=None
SOURCE='BLOCKED_UNBOUND_SOURCE_COMMIT'
REPO=ROOT/('repo-'+SOURCE[:12]);BASE=ASSET_ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
NATIVE=Path('BLOCKED_NATIVE_TOOLS_PATH');HF=Path(__file__).resolve().parent
TOOLS_MANIFEST_SHA=None
PINS={NATIVE/'inspect_payloads.py':'BLOCKED_NATIVE_INSPECTOR_SHA',NATIVE/'candidate-manifest.json':'BLOCKED_NATIVE_MANIFEST_SHA',BASE/'VERIFIED-MODEL-MANIFEST.json':'7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44',REPO/'search_r1/baselines/checkpoint_gate.py':'BLOCKED_GATE_SOURCE_SHA'}
PYTHON=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python')
def need(x,msg):
 if not x:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def physical(p):
 need(p.is_absolute() and p.resolve(strict=True)==p and p.stat().st_uid==0,'Foreign/nonphysical path '+str(p));return p
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n')
def context():
 need(type(JOB) is str and JOB.isdigit() and re.fullmatch('[0-9a-f]{40}',SOURCE) and re.fullmatch('[0-9a-f]{64}',TOOLS_MANIFEST_SHA or ''),'Actual job/source/tools pin required')
 tm_path=physical(HF/'manifest.json');need(sha(tm_path)==TOOLS_MANIFEST_SHA,'Reviewed receipt tools manifest changed');tm=json.loads(tm_path.read_bytes());need(tm['source_commit']==SOURCE,'Wrong first5 tools source')
 for name,pin in tm['tools'].items():need(sha(physical(HF/name))==pin,'First5 tool bytes changed')
 for name,pin in tm['source_files'].items():need(sha(physical(REPO/name))==pin,'First5 source bytes changed')
 need(os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10','Wrong host/UID')
 physical(RUN);physical(REPO)
 need(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(),'Source differs')
 for p,pin in PINS.items():need(sha(physical(p))==pin,'Tool/source/official pin changed '+str(p))
 spec=importlib.util.spec_from_file_location('stored_checkpoint_gate',REPO/'search_r1/baselines/checkpoint_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 manifest_path=physical(RUN/'full_state/global_step_5/manifest.json');m=g.load_json(manifest_path);pin=sha(manifest_path)
 need(m['source_commit']==SOURCE and m['job_id']==JOB and m['uid']==0 and m['run_root']==str(RUN) and (m['actual_outer_updates'],m['global_step'],m['next_global_step'],m['world_size'],m['horizon'],m['warmup_steps'])==(5,5,6,8,200,0),'Wrong actual checkpoint')
 workers=g.manifest_worker_pids(m,RUN)
 need(not (RUN/'acceptance-5-receipt.json').exists() and not (RUN/'acceptance-5-receipt.json').is_symlink(),'Receipt already exists')
 pauses=[];releases=[]
 for line in physical(RUN/'train.log').read_text().splitlines():
  for marker,items in [('OPSD_E1_STEP5_PAUSED_JSON=',pauses),('OPSD_E1_STEP5_RELEASED_JSON=',releases)]:
   if marker in line:items.append(g.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id=JOB,worker_pids=workers))
 need(not releases and len(pauses)==1,'Actual unique PAUSED required')
 p=pauses[0];need(p['manifest_path']==str(manifest_path) and p['manifest_sha256']==pin and p['pause_nonce']==m['pause_nonce'],'PAUSED checkpoint binding differs')
 driver=Path('/proc')/str(m['driver_pid']);cg=(driver/'cgroup').read_text();status=(driver/'status').read_text()
 need(re.search(r'(?:^|/)job_'+re.escape(JOB)+r'(?:/|$)',cg) and set(map(int,re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()))=={0} and re.search(r'^State:\s+(\S+)',status,re.M).group(1) in ('R','S','D','I'),'Actual paused driver lost')
 mem=Path('/proc/meminfo').read_text();available=int(re.search(r'^MemAvailable:\s+(\d+) kB',mem,re.M).group(1));need(available>=96*1024*1024,'Available host RAM below96GiB')
 return m,manifest_path,pin,{'pins':{str(p):sha(p) for p in PINS},'paused':p,'driver_cgroup':cg,'meminfo':mem,'available_kib':available}
def run(out):
 need(os.environ.get('SLURM_JOB_ID')==JOB and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1','Wrong CPU step environment')
 cg=Path('/proc/self/cgroup').read_text();need(re.search(r'(?:^|/)job_'+re.escape(JOB)+r'(?:/|$)',cg),'CPU inspector not in own job')
 m,mp,pin,before=context();physical(out);out.relative_to(RUN/'evidence')
 import torch
 need(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'Native Torch2.4 CPU only')
 write(out/'cpu-step-original.json',dict(pid=os.getpid(),uid=os.getuid(),hostname=socket.gethostname(),job_id=JOB,cgroup=cg,torch=torch.__version__,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_is_initialized=False,before=before,manifest_sha256=pin))
 common=['--repo',str(REPO),'--output']
 commands=[('native',[str(PYTHON),str(NATIVE/'inspect_payloads.py'),'--manifest',str(mp),'--manifest-sha256',pin,'--expected-source',SOURCE,'--repo',str(REPO),'--base-model',str(BASE),'--base-manifest',str(BASE/'VERIFIED-MODEL-MANIFEST.json'),'--base-manifest-sha256',PINS[BASE/'VERIFIED-MODEL-MANIFEST.json'],'--output',str(out/'native.json')]),('hf',[str(PYTHON),str(HF/'hf_cpu_reload.py'),'--run-root',str(RUN),'--checkpoint-manifest',str(mp),'--checkpoint-manifest-sha256',pin,'--repo',str(REPO),'--output',str(out/'hf.json')])]
 for name,cmd in commands:
  _,_,current_pin,state=context();need(current_pin==pin,'Manifest drifted')
  write(out/(name+'-command.json'),dict(argv=cmd,context=state,start_time=datetime.datetime.now(datetime.timezone.utc).isoformat()))
  with (out/(name+'-stdout-original.txt')).open('xb') as stdout,(out/(name+'-stderr-original.txt')).open('xb') as stderr:p=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
  write(out/(name+'-exit-original.json'),dict(returncode=p.returncode,end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),output_exists=(out/(name+'.json')).exists()))
  need(p.returncode==0,'Actual '+name+' inspection failed; originals retained')
 _,_,after_pin,after=context();need(after_pin==pin and not torch.cuda.is_initialized(),'Final identity/CUDA changed')
 write(out/'inspection-completed-review-required.json',dict(scope='CPU_STORED_STATE_AND_HF_INSPECTION_ONLY_NOT_SIGNATURE_OR_RELEASE',source_commit=SOURCE,job_id=JOB,manifest_sha256=pin,after=after,cuda_is_initialized=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--tools-manifest-sha256',required=True);p.add_argument('--status-only',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 JOB=a.job_id;RUN=ROOT/'runs'/('formal-'+JOB);TOOLS_MANIFEST_SHA=a.tools_manifest_sha256
 if a.status_only:
  try:m,mp,pin,state=context();print(json.dumps(dict(ready=True,manifest_path=str(mp),manifest_sha256=pin,context=state),sort_keys=True))
  except FileNotFoundError as e:print(json.dumps(dict(ready=False,reason=str(e))));sys.exit(3)
 else:
  need(a.output is not None,'Output required');run(a.output)
