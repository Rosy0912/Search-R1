"""Owned-allocation capacity harness. Synthetic two Adam updates, never RL outers."""
import argparse,csv,hashlib,io,json,math,os,re,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004')
OLD=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
SOURCE=os.environ.get('E1_EXPECTED_SOURCE')
ASSET_SOURCE='04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
MODEL=ROOT/'assets/qwen2.5-3b-instruct-aa8e725'
MODEL_MANIFEST=MODEL/'VERIFIED-MODEL-MANIFEST.json'
MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
ASSETS_SHA='87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
ASSET_MANIFEST=ROOT/'tools/tool-assets-full-verify-04f8-d1f36d3d5578/evidence/manifest.json'
PROBE_SHA='bbe157e70f959fb654f0d10836d91f32a1c6af0919eacdec8b2caf5c446e9b2b'
PROBE='reproduction/baselines/e1-300/microbatch_probe_e1.py'
QUERIES=['What is the capital of France?','Who wrote Hamlet?','When was Apollo 11 launched?','Where is Mount Everest?','Who discovered penicillin?','What is the capital of Japan?','Who painted the Mona Lisa?','Where was Albert Einstein born?']*3

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def unique(raw):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate key')
   d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(ValueError('Nonfinite JSON')))
def physical(p,owned=False):
 p=Path(p)
 if not p.is_absolute() or p.resolve(strict=True)!=p or not p.is_file() or (owned and p.stat().st_uid!=os.getuid()):raise ValueError('Physical owned path required')
 return p
def pinned(p,pin):
 p=physical(p)
 if not re.fullmatch('[0-9a-f]{64}',pin) or sha(p)!=pin:raise ValueError('External SHA differs')
 return unique(p.read_bytes())
def emit(path,value):
 with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
def command(args):return subprocess.check_output(args,text=True,timeout=20)
def own(pid):
 p=Path('/proc')/str(pid);c=(p/'cgroup').read_text();status=(p/'status').read_text()
 if not re.search(r'(?:^|/)job_'+re.escape(os.environ['SLURM_JOB_ID'])+r'(?:/|$)',c,re.M):raise ValueError('Foreign process cgroup')
 if {int(x) for x in re.search(r'^Uid:\s+(.*)',status,re.M).group(1).split()}!={os.getuid()}:raise ValueError('Foreign process UIDs')
 return dict(pid=pid,state=re.search(r'^State:\s+(\S)',status,re.M).group(1),start=(p/'stat').read_text().rsplit(')',1)[1].split()[19],cgroup=c,uid=os.getuid())
def same_process(record):
 try:
  current=own(record['pid']);return current['start']==record['start'] and current['state'] not in ('Z','X')
 except (OSError,ValueError):return False
def snapshot():
 inv=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
 apps=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
 rows=[dict(index=int(v[0]),uuid=v[1].strip(),name=v[2].strip(),used=float(v[3]),total=float(v[4]),util=float(v[5])) for v in csv.reader(io.StringIO(inv))]
 procs=[dict(uuid=v[0].strip(),pid=int(v[1]),memory=v[2].strip(),owner=own(int(v[1]))) for v in csv.reader(io.StringIO(apps)) if v]
 return dict(inventory=rows,applications=procs)
def validate_snapshot(value,expected_uuids=None,service_pid=None,idle=False):
 rows=value['inventory'];assert len(rows)==8 and len({r['uuid'] for r in rows})==8 and all('H100' in r['name'] for r in rows)
 if expected_uuids is not None:assert {r['uuid'] for r in rows}==set(expected_uuids)
 if service_pid is None:assert not value['applications']
 else:
  assert {r['pid'] for r in value['applications']}=={service_pid}
  assert {r['uuid'] for r in value['applications']}==set(expected_uuids)
 if idle:assert all(r['used']<=.15*r['total'] and r['util']<=15 for r in rows)
def verify_tool_assets(manifest):
 assert manifest['status']=='FULL_TOOL_ASSET_CONTENT_VERIFIED' and manifest['frozen_source_commit']==ASSET_SOURCE
 assert manifest['assets_root']==str(OLD/'assets-root/assets') and manifest['hostname']=='h100-10'
 r=manifest['results'];assert len(r['e5'])==8
 records=r['e5']+[r['wiki_gzip'],r['index'],r['corpus']]
 for v in records:
  p=physical(v['path']);assert OLD/'assets-root/assets' in p.parents
  before=p.stat();assert before.st_size==v['bytes'] and sha(p)==v['sha256']
  after=p.stat();assert (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 assert r['corpus']['sha256']==r['gzip_decompressed']['sha256'] and r['corpus']['bytes']==r['gzip_decompressed']['bytes']
 assert len(r['index']['segment_sha256'])==2
 return records

def source_tools(run):
 repo=Path(os.environ['SEARCH_R1_REPO']);assert isinstance(SOURCE,str) and re.fullmatch('[0-9a-f]{40}',SOURCE)
 assert repo==ROOT/('repo-'+SOURCE[:12]) and repo.resolve()==repo
 assert command(['git','-C',str(repo),'rev-parse','HEAD']).strip()==SOURCE
 assert not command(['git','-C',str(repo),'status','--porcelain']).strip()
 tools=Path(os.environ['E1_CAPACITY_TOOLS']);assert tools.parent==ROOT/'tools' and tools.resolve()==tools and tools.stat().st_uid==os.getuid()
 m=pinned(tools/'manifest.json',os.environ['E1_CAPACITY_MANIFEST_SHA256']);assert m['source_commit']==SOURCE
 assert set(m['tools'])=={'capacity_job.sh','capacity_entry.py','resource_sample.py','rank_entry.py','retriever_entry.py'}
 for n,pin in m['tools'].items():assert sha(physical(tools/n,True))==pin
 assert set(m['source_files'])=={PROBE,'reproduction/h20/retriever_local.py','search_r1/search/retrieval_server.py','search_r1/baselines/preflight.py','search_r1/baselines/e1_profile.py'}
 for n,pin in m['source_files'].items():assert sha(physical(repo/n))==pin
 assert sha(MODEL_MANIFEST)==MODEL_SHA
 assert Path(os.environ['E1_TOOL_ASSET_MANIFEST'])==ASSET_MANIFEST
 assert m['source_files'][PROBE]==PROBE_SHA
 assert os.environ['E1_TOOL_ASSET_MANIFEST_SHA256']==ASSETS_SHA and m['retrieval_asset_manifest_sha256']==ASSETS_SHA
 assets=pinned(os.environ['E1_TOOL_ASSET_MANIFEST'],ASSETS_SHA);verify_tool_assets(assets)
 return repo,tools,m,assets

def prepare(run):
 repo,tools,m,assets=source_tools(run);job=os.environ['SLURM_JOB_ID']
 assert socket.gethostname().split('.')[0]=='h100-10' and os.environ['SLURM_JOB_NODELIST']=='h100-10' and int(os.environ['SLURM_CPUS_PER_TASK'])>=32
 own(os.getpid());raw=command(['scontrol','show','job',job,'-o'])
 assert 'JobId='+job+' ' in raw and 'JobState=RUNNING' in raw and 'NodeList=h100-10' in raw
 assert re.search(r'UserId=\S+\('+str(os.getuid())+r'\)',raw) and re.search(r'(?:^| )TresPerNode=gres:gpu:h100:8(?: |$)',raw)
 selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(selected)==len(set(selected))==8
 current=snapshot();validate_snapshot(current,idle=True)
 with socket.socket() as s:s.bind(('127.0.0.1',18084))
 emit(run/'allocation.json',dict(source=SOURCE,job_id=job,hostname=socket.gethostname(),uid=os.getuid(),cgroup=Path('/proc/self/cgroup').read_text(),scontrol=raw,gpus=current,model=str(MODEL),model_manifest_sha256=MODEL_SHA,tools_manifest=m,tool_assets_manifest_sha256=os.environ['E1_TOOL_ASSET_MANIFEST_SHA256']))

def http24(run,pid):
 import urllib.request
 body=json.dumps(dict(queries=QUERIES,topk=3,return_scores=True)).encode();start=time.perf_counter()
 request=urllib.request.Request('http://127.0.0.1:18084/retrieve',data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as response:raw=response.read()
 data=unique(raw);assert isinstance(data,dict) and len(data['result'])==24
 for row in data['result']:
  assert len(row)==3
  for item in row:assert isinstance(item['document']['contents'],str) and item['document']['contents'] and math.isfinite(float(item['score']))
 emit(run/'http-top3-24.json',dict(job_id=os.environ['SLURM_JOB_ID'],service=own(pid),queries=QUERIES,topk=3,seconds=time.perf_counter()-start,response=data,scope='Actual live local GPU retrieval API content/finite conformance only; no legacy numerical acceptance'))

def validate_probe(data,job):
 assert data['passed'] is True and data['local_micro']==2 and len(data['ranks'])==8
 assert {r['rank'] for r in data['ranks']}==set(range(8))
 for r in data['ranks']:
  assert r['passed'] is True and r['world_size']==8 and r['job_id']==job and r['source_commit']==SOURCE
  assert r['component_clip'] is None and r['teacher_prompt_mode']=='user_context' and r['official_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
  assert r['student_master_parameter_dtypes']==['torch.float32'] and r['student_mixed_precision_parameter_dtype']=='torch.bfloat16'
  assert r['teacher_active_runtime']['parameter_dtypes']==['torch.float32'] and r['teacher_active_runtime']['attention']=='eager'
  assert r['teacher_returned_runtime']['active_device']=='cpu'
  assert r['local_micro']==2 and r['input_width']==6144 and r['response_width']==4096 and r['optimizer_steps']==2
  assert len(r['optimizer_step_records'])==2 and all(math.isfinite(v['loss']) and math.isfinite(v['grad_norm']) for v in r['optimizer_step_records'])
  assert r['minimum_required_free_fraction']==.15 and r['estimated_peak_device_used_bytes']<=.85*r['gpu_total_bytes']

def validate_resource_samples(path,service_pid,uuids,job):
 rows=[unique(line) for line in Path(path).read_text().splitlines() if line]
 candidates=[]
 for row in rows:
  if row.get('job_id')!=job or 'query_error' in row:raise ValueError('Resource sample job/query error')
  apps=row.get('gpu_processes',[])
  if any(v.get('unavailable') or not v.get('in_slurm_job') or str(v.get('uid'))!=str(os.getuid()) for v in apps):raise ValueError('Resource sampler saw foreign/unavailable GPU application')
  if len({v.get('pid') for v in apps})!=9 or service_pid not in {v.get('pid') for v in apps}:continue
  if any(v.get('unavailable') or not v.get('in_slurm_job') or str(v.get('uid'))!=str(os.getuid()) for v in apps):raise ValueError('Resource sampler saw foreign GPU application')
  if {v['uuid'] for v in apps}!=set(uuids):raise ValueError('Application UUID inventory differs')
  workers={v['pid'] for v in apps}-{service_pid}
  if candidates and candidates[0]['workers']!=sorted(workers):continue
  candidates.append(dict(time=row['time'],workers=sorted(workers),applications=apps))
 if len(candidates)<2:raise ValueError('Need two actual resource samples with eight own GPU worker PIDs plus service')
 return candidates[:2]

def run_capacity(run):
 from search_r1.baselines.preflight import validate_inherited_gpu_leases
 validate_inherited_gpu_leases(8);records=unique(os.environ['SEARCH_R1_GPU_LEASES']);uuids=[r['uuid'] for r in records];fds=tuple([9]+[int(r['fd']) for r in records])
 assert set(uuids)=={r['uuid'] for r in unique((run/'allocation.json').read_bytes())['gpus']['inventory']}
 emit(run/'leases.json',records);repo,tools,m,assets=source_tools(run)
 service=None;sampler=None;proc=None;env=dict(os.environ,OPSD_EVIDENCE=str(run),OPSD_E1_OFFICIAL_MODEL=str(MODEL),OPSD_E1_OFFICIAL_MANIFEST=str(MODEL_MANIFEST),PYTHONPATH=str(repo))
 try:
  sampler=subprocess.Popen([sys.executable,str(tools/'resource_sample.py'),str(run/'resources.ndjson'),str(os.getpid())],env=env,pass_fds=fds);sampler_owner=own(sampler.pid)
  renv=dict(env,PYTHONPATH=str(repo)+':'+str(OLD/'retriever-gpu-lib'))
  log=(run/'retriever.log').open('xb');service=subprocess.Popen([sys.executable,'-u',str(tools/'retriever_entry.py'),'--index_path',str(OLD/'assets-root/assets/wiki/e5_Flat.index'),'--corpus_path',str(OLD/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(OLD/'assets-root/assets/e5-base-v2'),'--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18084','--threads','8','--topk','3'],env=renv,pass_fds=fds,stdout=log,stderr=subprocess.STDOUT)
  owner=own(service.pid);emit(run/'service.json',dict(owner=owner,uuids=uuids,command=service.args,environment_pythonpath=renv['PYTHONPATH']))
  deadline=time.monotonic()+600
  while True:
   if service.poll() is not None:raise RuntimeError('Own retriever exited during initialization')
   raw=command(['ss','-lntp'])
   if re.search(r'127\.0\.0\.1:18084\s',raw):
    lines=[line for line in raw.splitlines() if re.search(r'127\.0\.0\.1:18084\s',line)]
    assert len(lines)==1 and re.search(r'pid='+str(service.pid)+r'[,)]',lines[0]);(run/'service-listener.txt').write_text(raw);break
   if time.monotonic()>deadline:raise TimeoutError('Retriever own listener initialization')
   time.sleep(2)
  http24(run,service.pid)
  # Query is a real GPU activity; bounded idle wait preserves original guard.
  deadline=time.monotonic()+120;checks=[]
  while len(checks)<2:
   v=snapshot();validate_snapshot(v,uuids,service.pid)
   if all(r['used']<=.15*r['total'] and r['util']<=15 for r in v['inventory']):checks.append(v)
   else:checks=[]
   if time.monotonic()>deadline:raise RuntimeError('GPUretrieval does not satisfy double15% before capacity')
   time.sleep(2)
  emit(run/'pre-probe-double-gpu-apps.json',checks)
  with (run/'torchrun.log').open('xb') as out:
   proc=subprocess.Popen(['torchrun','--standalone','--nproc_per_node=8',str(tools/'rank_entry.py')],env=env,pass_fds=fds,stdout=out,stderr=subprocess.STDOUT)
   probe_owner=own(proc.pid)
   rc=proc.wait();(run/'torchrun-exit.txt').write_text(str(rc)+'\n')
  if rc:raise RuntimeError('Actual isolated capacity torchrun failed')
  validate_probe(unique((run/'micro-2.json').read_bytes()),os.environ['SLURM_JOB_ID'])
  assert same_process(owner) and service.poll() is None
  actual_samples=validate_resource_samples(run/'resources.ndjson',service.pid,uuids,os.environ['SLURM_JOB_ID']);emit(run/'actual-nine-gpu-application-samples.json',actual_samples)
  source_tools(run)
  pending_summary=dict(status='ACTUAL_OFFICIAL_3B_MICRO2_GPU_RETRIEVAL_CAPACITY_PASS',rl_outer_updates=0,source=SOURCE,job_id=os.environ['SLURM_JOB_ID'],uuids=uuids,service=owner,model=str(MODEL),model_manifest_sha256=MODEL_SHA,scope='Synthetic6144/4096 all8 full backward/twoAdam/15%headroom only; excludesvLLM/first5/20AdamRL')
 finally:
  descendants=[]
  for q in Path('/proc').iterdir():
   if not q.name.isdigit():continue
   try:
    cmd=(q/'cmdline').read_bytes().split(b'\0')
    if str(tools/'rank_entry.py').encode() in cmd:descendants.append(own(int(q.name)))
   except (OSError,ValueError):continue
  for record in descendants:
   if same_process(record):os.kill(record['pid'],signal.SIGTERM)
  for p,record in [(proc,locals().get('probe_owner')),(service,locals().get('owner')),(sampler,locals().get('sampler_owner'))]:
   if p is not None and p.poll() is None and record and same_process(record):
    p.terminate()
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
     if same_process(record):p.kill();p.wait(timeout=10)
  for record in descendants:
   deadline=time.monotonic()+20
   while same_process(record) and time.monotonic()<deadline:time.sleep(.2)
   if same_process(record):os.kill(record['pid'],signal.SIGKILL)
  post=snapshot();deadline=time.monotonic()+30
  while post['applications'] and time.monotonic()<deadline:
   time.sleep(1);post=snapshot()
  assert not post['applications']
  emit(run/'cleanup.json',dict(owned_descendants=descendants,post_gpu_snapshot=post,all_gpu_processes_zero=True))
 if 'pending_summary' in locals():
  identities=[unique((run/f'worker-identity-{i}.json').read_bytes()) for i in range(8)]
  assert {r['rank'] for r in identities}==set(range(8)) and {r['actual_cuda_uuid'] for r in identities}==set(uuids) and len({r['pid'] for r in identities})==8
  for r in identities:assert r['job_id']==os.environ['SLURM_JOB_ID'] and r['source']==SOURCE and set(map(int,r['uids']))=={os.getuid()}
  for sample in actual_samples:
   for r in identities:
    entries=[a for a in sample['applications'] if a['pid']==r['pid'] and a['uuid']==r['actual_cuda_uuid'] and a['process_start_ticks']==r['start']]
    assert entries
  pending_summary['actual_workers']=identities
  emit(run/'capacity-summary.json',pending_summary)

def main():
 def interrupted(signum,frame):raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args();run=Path(os.environ['E1_CAPACITY_RUN']);assert run.parent==ROOT/'runs' and run.resolve()==run and run.stat().st_uid==os.getuid()
 prepare(run) if a.prepare else run_capacity(run)
if __name__=='__main__':main()
