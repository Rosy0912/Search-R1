"""Admission metadata only, no weights/model/tensor import."""
import hashlib,json,pathlib,re,os
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
MANIFEST='1b4bae153472c5778c5bfd8ab00905491a5b58820dbec62e742a3a377bab17db'
SOURCE='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def job_id(x):
 need(type(x)is int or type(x)is str and re.fullmatch('[0-9]+',x),'Job must be positive ASCIIdecimal')
 n=int(x);need(n>0,'Positive job required');return n
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def own(p):
 p=pathlib.Path(p);need(not p.is_symlink() and ROOT in p.resolve().parents,'Only canonical ownROOT path');return p

def ref(r):
 p=own(r['path']);need(sha(p)==r['sha256'],'Original raw SHA drift');return json.loads(p.read_bytes())
def check_request(q,proc_root=pathlib.Path('/proc')):
 need(q['scope']=='REQUEST_BOUNDED_CPU_HF_SEMANTIC_SCAN_NOT_APPROVAL','Request scope differs')
 job=job_id(q['job_id']);steps={'profile2':2,'train100':100}[q['mode']]
 terminal=ref(q['controller_terminal'])
 need(job_id(terminal['job_id'])==job and terminal['returncode']==0,'Actual captured terminal missing')
 out=terminal['stdout'];need(re.findall(r'\bJobId=([0-9]+)',out)==[str(job)] and re.search(r'\bJobState=COMPLETED\b',out) and re.search(r'\bExitCode=0:0\b',out),'Actual training controller not complete0:0')
 r=ref(q['train_result']);c=ref(q['effective_config'])
 need(r['status']=='COMPLETE_UNREVIEWED' and r['cleanup_passed'] is True and r['error'] is None,'Failed/live result cannot be scanned')
 need(job_id(r['job'])==job_id(c['job'])==job,'Actual source job differs')
 need(r['effective_config_sha256']==q['effective_config']['sha256'] and c['expected_outer_steps']==steps and c['mode']==q['mode'],'Actual config/mode differs')
 need(c['source_manifest']['sha256']==SOURCE,'Only fixed source9040')
 ref(c['source_manifest']);ref(c['tools_manifest'])
 need(r['scope']==('ACTUAL_SEARCH_R1_PROFILE2_UNREVIEWED_NOT_100_NOT_TOPD' if steps==2 else 'ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD'),'Actual algorithm scope differs')
 need(q['review_tool_manifest']=={'path':str(ROOT/'tools/r1-32b-hf-semantic-review-v4/files.json'),'sha256':MANIFEST},'Exact reviewed scanner only')
 outer=ref(r['actual_outer_evidence']);need(outer['actual_completed_outer_steps']==list(range(1,steps+1)) and outer['export_step']==steps,'Actual complete outer2/100 required')
 need(r['export']['directory']==c['final_checkpoint']==outer['checkpoint_directory'],'Export identity differs')
 need(r['export']['tensor_count']==771 and len(r['export']['shards'])>0 and set(r['export']['shards'])=={n for n in r['export']['files'] if n.endswith('.safetensors')},'Actual full HF inventory required')
 cleanup=ref(r['cleanup']);need(cleanup['cleanup']['actual_cgroup_empty'] is True and cleanup['terminal_monitors_drained_before_authorized_exit'] is True and cleanup['parent_cuda_initialized'] is False,'Actual cleanup/drain required')
 for n in ['own_processes_remaining','own_gpu_records_remaining','parent_gpu_records','baseline_identity_missing','original_gpu_baseline_missing','guardian_errors','sampler_errors','failure_history']:need(cleanup[n]==[],'Prior cleanup not complete '+n)
 need(cleanup['error'] is None and cleanup['primary_error'] is None,'Original failure retained')
 # Actual post-cleanup proc census; never signal any PID.
 for p in proc_root.iterdir():
  if not p.name.isdigit():continue
  try:cg=(p/'cgroup').read_text()
  except (FileNotFoundError,ProcessLookupError):continue
  need(not any(re.search(r'/job_'+str(job)+r'(?:/|$)',line) for line in cg.splitlines()),'Training job still has live cgroup processes')
 for pid,old in cleanup['identities'].items():
  try:raw=(proc_root/str(pid)/'stat').read_text()
  except (FileNotFoundError,ProcessLookupError):continue
  fields=raw[raw.rfind(')')+2:].split();need(int(fields[19])!=old['starttime_ticks'],'Training owned PID/start still alive')
 return r,c
