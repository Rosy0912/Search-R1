"""Single initial-only native calibration allocation after independent concrete recipe review."""
import base64,hashlib,json,pathlib,subprocess,sys
job=int(sys.argv[1]);gate=pathlib.Path(sys.argv[2]);g=json.loads(gate.read_text());assert g['status']=='PASS' and g['scope']=='G1_32B_INITIAL_CALIBRATION_SOURCE_AND_ALLOCATION_GATE'
payload={'cpu_job':job,'gate_name':gate.name,'gate_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),'attempt':'r4r8-after-failed3132','native_config_path':sys.argv[3] if len(sys.argv)>3 else '/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial-static-cpu-'+str(job)+'/recipe/initial-only-static-config.json'}
code=r'''
import base64,fcntl,hashlib,json,os,pathlib,socket,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
p=json.loads(base64.b64decode('PAYLOAD'));original=pathlib.Path(p['native_config_path']);assert r/'logs' in original.resolve().parents and not original.is_symlink();recipe=original.parent
c=json.loads(original.read_text())
assert c['evidence_root'] not in (str(r/'logs/g1-initial-static-cpu-3102/recipe/initial-allocation-evidence'),str(r/'logs/g1-initial3102-r4r1-allocation-evidence'),str(r/'logs/g1-initial3102-r4r2-allocation-evidence'),str(r/'logs/g1-initial3102-r4r3-allocation-evidence'),str(r/'logs/g1-initial3102-r4r4-allocation-evidence'),str(r/'logs/g1-initial3102-r4r5-allocation-evidence'),str(r/'logs/g1-initial3102-r4r6-allocation-evidence')), 'New attempt must have distinct evidence root'
assert r/'logs' in pathlib.Path(c['evidence_root']).resolve().parents, 'Evidence must remain in owned local logs'
gate=r/'logs'/p['gate_name'];assert hashlib.sha256(gate.read_bytes()).hexdigest()==p['gate_sha256']
g=json.loads(gate.read_text());assert g['status']=='PASS' and g['scope']=='G1_32B_INITIAL_CALIBRATION_SOURCE_AND_ALLOCATION_GATE'
canonical=hashlib.sha256(json.dumps({k:v for k,v in c.items() if k!='independent_gate'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert canonical==g['reviewed_static_recipe_sha256']
c['independent_gate']={'path':str(gate),'sha256':p['gate_sha256']}
final=recipe/('initial-only-approved-config-'+p['attempt']+'.json')
if final.exists():assert json.loads(final.read_text())==c
else:
 with final.open('x') as f:json.dump(c,f,indent=2);f.flush();os.fsync(f.fileno())
pipeline=pathlib.Path(next(x['root'] for x in c['source_bindings'] if x['kind']=='pipeline'))
entry=pipeline/'production.sbatch';manifest=json.loads((pipeline/'files.json').read_text());assert hashlib.sha256(entry.read_bytes()).hexdigest()==manifest['files']['production.sbatch']['sha256']
assert c['preflight_source']['path']==str(r/'repo-terminal-fixed413/search_r1/baselines/preflight.py')
assert hashlib.sha256(pathlib.Path(c['preflight_source']['path']).read_bytes()).hexdigest()==c['preflight_source']['sha256']
subprocess.run(['bash','-n',str(entry)],check=True)
queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
record=r/'logs'/('g1-initial300-gpu-'+p['attempt']+'-submission.json')
def persist(d):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,record);fd=os.open(record.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
  assert d['cpu_job']==p['cpu_job'] and d['recipe_canonical_sha256']==canonical and d['gate_sha256']==p['gate_sha256'] and d['entry_sha256']==hashlib.sha256(entry.read_bytes()).hexdigest(), 'Existing submission belongs to a different immutable recipe; inspect its actual job before any new submission'
 else:
  assert not pathlib.Path(c['evidence_root']).exists(), 'Evidence already exists without known new submission; inspect first'
  previous=json.loads((r/'logs/g1-initial300-gpu-r4r6-after-failed3129-submission.json').read_text());assert previous['job_id']==3132 and previous['status']=='SUBMITTED'
  prior=json.loads((r/'logs/g1-initial3102-r4r6-allocation-evidence/cleanup.json').read_text());assert prior['cleanup']['actual_cgroup_empty'] is True and prior['own_gpu_records_remaining']==[] and prior['own_processes_remaining']==[] and prior['original_gpu_baseline_missing']==[] and prior['baseline_identity_missing']==[]
  assert not pathlib.Path('/proc/2950045').exists(), 'Old owned retriever still exists; inspect identity first'
  assert not any(row.split('|')[0]=='3132' for row in queue.splitlines()),'Old job active'
  assert not any('g1-32b-initial300' in row for row in queue.splitlines()),queue
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  assert len(gpu.splitlines())==8
  for line in gpu.splitlines():
   uuid,used,total,sm=line.split(',');assert float(used)/float(total)<=.15 and float(sm)<=15
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),scope='REAL_G1_INITIAL300_CALIBRATION_NOT_ACTUAL_OUTER_UPDATE',cpu_job=p['cpu_job'],recipe_canonical_sha256=canonical,gate_sha256=p['gate_sha256'],entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),pre_submit_gpu=gpu,queue_before=queue);persist(d)
  env=dict(os.environ,G1_PIPELINE_CONFIG=str(final),G1_PIPELINE_DIR=str(pipeline),G1_PIPELINE_MODE='initial-only',G1_PREFLIGHT_ENTRY=c['preflight_source']['path'])
  proc=subprocess.run(['sbatch','--parsable','--chdir='+str(r),str(entry)],env=env,text=True,capture_output=True)
  d.update(sbatch_returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr);persist(d);assert proc.returncode==0,d
  jid=proc.stdout.strip().split(';')[0];assert jid.isdigit(),d
  d.update(status='SUBMITTED',job_id=int(jid));persist(d)
  ctl=subprocess.run(['scontrol','show','job','-dd',jid],text=True,capture_output=True)
  saved=r/'logs'/('g1-initial300-gpu-'+jid+'-scontrol-initial.txt')
  with saved.open('x') as f:f.write(ctl.stdout+ctl.stderr);f.flush();os.fsync(f.fileno())
  d['initial_scontrol_sha256']=hashlib.sha256(saved.read_bytes()).hexdigest();persist(d)
print(json.dumps(d))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(payload).encode()).decode())
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'SUBMIT_G1_INITIAL'\n"+code+'\nSUBMIT_G1_INITIAL\n',text=True,capture_output=True)
pathlib.Path('outputs/g1-initial300-gpu-r4r8-submission-invocation.log').write_text(p.stdout+p.stderr)
print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr);raise SystemExit(p.returncode)
