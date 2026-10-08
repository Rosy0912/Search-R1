"""Single reviewed standard Search-R1 allocation; unknown submissions never retry."""
import hashlib,json,pathlib,subprocess,sys
sys.dont_write_bytecode=True
config=pathlib.Path(sys.argv[1]);gate=pathlib.Path(sys.argv[2]);c=json.loads(config.read_text());g=json.loads(gate.read_text())
assert g['status']=='PASS'
p=dict(config=c,gate_name=gate.name,gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),attempt=sys.argv[3])
helper_path=pathlib.Path(__file__).with_name('standard_r1_v5_failed_profile_admission.py')
assert hashlib.sha256(helper_path.read_bytes()).hexdigest()=='8f7543d6bb65cd7ea9a0df5ee39a84559f6261e4fcb6abecd295be1ec8009a8f'
admission_source=helper_path.read_text()
code=r'''
import fcntl,hashlib,json,os,pathlib,re,socket,subprocess,time
P=PAYLOAD
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',P['attempt'])
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,value):
 t=p.with_name(p.name+'.private');assert not t.exists()
 with t.open('x') as out:json.dump(value,out,indent=2);out.flush();os.fsync(out.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
c=P['config'];gate=ROOT/'logs'/P['gate_name'];assert sha(gate)==P['gate_sha256'];g=json.loads(gate.read_text())
canonical=hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='independent_gate'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
expected='SEARCH_R1_32B_PROFILE2_SOURCE_AND_ALLOCATION_GATE' if c['mode']=='profile2' else 'SEARCH_R1_32B_GRPO100_ACTUAL_PROFILE_AND_SOURCE_GATE'
assert g['status']=='PASS' and g['scope']==expected and g['reviewed_static_recipe_sha256']==canonical
if 'allowed_attempt' in g:assert g['allowed_attempt']==P['attempt']
if g.get('prior_source_restoration'):
 ref=g['prior_source_restoration'];restoration=pathlib.Path(ref['path']);assert ROOT/'logs' in restoration.resolve().parents and sha(restoration)==ref['sha256']
 saved=json.loads(restoration.read_text());assert saved['status']=='EXACT_RESTORED' and saved['own3158_processes']==[]
 prior_record=json.loads((ROOT/'logs/r1-standard32b-profile2-v3-submission.json').read_text());assert prior_record['job_id']==3158 and prior_record['status']=='SUBMITTED'
 prior_live=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   cg=(proc/'cgroup').read_text()
   if '/job_3158/' in cg or cg.rstrip().endswith('/job_3158'):prior_live.append(int(proc.name))
  except (FileNotFoundError,ProcessLookupError):pass
 assert prior_live==[],prior_live
for key,root in [('source_manifest',c['candidate']),('tools_manifest',c['tools'])]:
 p=pathlib.Path(c[key]['path']);assert not p.is_symlink() and sha(p)==c[key]['sha256'];m=json.loads(p.read_text())
 for n,r in m['files'].items():
  f=pathlib.Path(root)/n;assert not f.is_symlink() and f.stat().st_size==r['size'] and sha(f)==r['sha256']
if P['attempt']=='profile2-v5' or g.get('prior_failed_job_id')==3163:
 validate_failed3163(c,g,P['attempt'],ROOT)
if g.get('prior_run_cleanup'):
 ref=g['prior_run_cleanup'];cp=pathlib.Path(ref['path']);assert ROOT/'logs' in cp.resolve().parents and sha(cp)==ref['sha256']
 cl=json.loads(cp.read_text());assert cl['cleanup']['actual_cgroup_empty'] is True and cl['own_gpu_records_remaining']==[] and cl['own_processes_remaining']==[] and cl['original_gpu_baseline_missing']==[]
 job=int(g['prior_failed_job_id']);assert job in (3159,3163)
 for pid,saved in cl['identities'].items():
  proc=pathlib.Path('/proc')/pid
  try: raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  except FileNotFoundError:continue
  cg=(proc/'cgroup').read_text();assert '/job_'+str(job)+'/' not in cg
  assert int(fields[19])!=saved.get('starttime_ticks'),'Prior failed profile registered identity remains alive'
if g.get('prior_failed_job_id')==3163:
 live=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cg=(proc/'cgroup').read_text()
  except (FileNotFoundError,ProcessLookupError):continue
  if '/job_3163/' in cg or cg.rstrip().endswith('/job_3163'):live.append(int(proc.name))
 assert live==[],('Prior3163 actual job cgroup is not empty',live)
c['independent_gate']={'path':str(gate),'sha256':P['gate_sha256']}
destination=pathlib.Path(c['evidence_root']).parent/('approved-config-'+P['attempt']+'.json');assert ROOT/'logs' in destination.resolve().parents
if destination.exists():assert json.loads(destination.read_text())==c
else:atomic(destination,c)
entry=pathlib.Path(c['tools'])/'production.sbatch';subprocess.run(['bash','-n',str(entry)],check=True)
preflight=pathlib.Path(c['preflight_source']['path']);assert sha(preflight)==c['preflight_source']['sha256']
record=ROOT/'logs'/('r1-standard32b-'+P['attempt']+'-submission.json')
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['canonical']==canonical and d['gate_sha256']==P['gate_sha256']
 else:
  assert not pathlib.Path(c['evidence_root']).exists(),'Inspect existing evidence before unknown/new submission'
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(row.split('|')[1] in ('r1-32b-grpo','g1-32b-initial300') for row in queue.splitlines()),queue
  old=json.loads((ROOT/'logs/g1-initial3102-r4r10-allocation-evidence/cleanup.json').read_text())
  assert old['cleanup']['actual_cgroup_empty'] is True and old['own_gpu_records_remaining']==[] and old['own_processes_remaining']==[] and old['original_gpu_baseline_missing']==[]
  prior_rows=[]
  for pid,saved in old['identities'].items():
   proc=pathlib.Path('/proc')/pid
   try:
    raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
   except FileNotFoundError:prior_rows.append({'pid':int(pid),'actual_proc_absent':True});continue
   actual={'pid':int(pid),'starttime_ticks':int(fields[19]),'uid':proc.stat().st_uid,'cgroup':(proc/'cgroup').read_text()}
   assert '/job_3144/' not in actual['cgroup'],'Old job still has a live descendant'
   assert actual['starttime_ticks']!=saved.get('starttime_ticks'),'Prior owned identity is still alive; inspect before new allocation'
   prior_rows.append({'pid':int(pid),'actual_reused_unrelated_identity_not_owned':actual})
  assert not pathlib.Path('/proc/2982663').exists(),'Prior supervisor exists; inspect its identity'
  latest=ROOT/'logs/r1-standard32b-profile2-v2/evidence/owned-cleanup.json'
  latest_cleanup=json.loads(latest.read_text())
  assert latest_cleanup['cleanup']['actual_cgroup_empty'] is True
  assert latest_cleanup['own_gpu_records_remaining']==[] and latest_cleanup['own_processes_remaining']==[] and latest_cleanup['original_gpu_baseline_missing']==[]
  latest_rows=[]
  for pid,saved in latest_cleanup['identities'].items():
   proc=pathlib.Path('/proc')/pid
   try: raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
   except FileNotFoundError: latest_rows.append({'pid':int(pid),'actual_proc_absent':True});continue
   actual={'pid':int(pid),'starttime_ticks':int(fields[19]),'uid':proc.stat().st_uid,'cgroup':(proc/'cgroup').read_text()}
   assert '/job_3155/' not in actual['cgroup'],'Prior3155 live descendant remains'
   assert actual['starttime_ticks']!=saved.get('starttime_ticks'),'Prior3155 owned identity remains alive'
   latest_rows.append({'pid':int(pid),'actual_reused_unrelated_identity_not_owned':actual})
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  assert len(gpu.splitlines())==8
  uuids=[]
  for row in gpu.splitlines():
   u,used,total,sm=row.split(',');uuids.append(u.strip());assert float(used)/float(total)<=.15 and float(sm)<=15
  assert len(set(uuids))==8
  assert set(uuids)=={row.split(',')[1].strip() for row in old['gpu_after']['gpus']},'Physical GPU UUID roster changed'
  roster=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',mode=c['mode'],canonical=canonical,gate_sha256=P['gate_sha256'],entry_sha256=sha(entry),epoch=time.time(),queue_before=queue,pre_submit_gpu=gpu,pre_submit_uuid_roster=uuids,pre_submit_compute_roster=roster,prior_owned_identities_fresh=prior_rows,prior3155_owned_identities_fresh=latest_rows,prior3155_cleanup_sha256=sha(latest),config=str(destination));atomic(record,d)
  env=dict(os.environ,R1_CONFIG=str(destination),R1_TOOLS=c['tools'],R1_PREFLIGHT=str(preflight),PYTHONDONTWRITEBYTECODE='1')
  run=subprocess.run(['sbatch','--parsable','--chdir='+str(ROOT),str(entry)],env=env,text=True,capture_output=True)
  d.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr);atomic(record,d);assert run.returncode==0,d
  job=run.stdout.strip().split(';')[0];assert job.isdigit();d.update(status='SUBMITTED',job_id=int(job));atomic(record,d)
  ctl=subprocess.run(['scontrol','show','job','-dd',job],text=True,capture_output=True)
  initial=ROOT/'logs'/('r1-standard32b-'+job+'-scontrol-initial.json');atomic(initial,dict(stdout=ctl.stdout,stderr=ctl.stderr,returncode=ctl.returncode,epoch=time.time(),job_id=int(job)))
  d['initial_scontrol']={'path':str(initial),'sha256':sha(initial)};atomic(record,d)
 print(json.dumps(d))
'''.replace('PAYLOAD',repr(p))
code=admission_source+'\n'+code
compile(code,'standard-r1-single-allocation','exec')
if '--render-only' in sys.argv:
 pathlib.Path('outputs/standard-r1-gpu-submit-v5-diagnostic-v2-render.py').write_text(code);print('SOURCE_ONLY_RENDERED');sys.exit()
r=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'STANDARD_R1_SUBMIT'\n"+code+'\nSTANDARD_R1_SUBMIT\n',text=True,capture_output=True)
pathlib.Path('outputs/standard-r1-gpu-submit-'+p['attempt']+'-invocation.log').write_text(r.stdout+r.stderr)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr);sys.exit(r.returncode)
