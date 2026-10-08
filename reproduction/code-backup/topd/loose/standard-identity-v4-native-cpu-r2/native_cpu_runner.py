"""CPU contracts only; no model load/GPU lifecycle or submission."""
import sys,pathlib,subprocess,json,hashlib,os,re
P,R=map(pathlib.Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for manifest in ('standard-search-r1-teacher-identity-v4-files.json','native-files.json'):
 for name,v in json.loads((P/manifest).read_text())['files'].items():
  p=P/name
  assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['size'] and sha(p)==v['sha256'],name
assert not list(P.rglob('__pycache__')) and not list(P.rglob('*.pyc')),'Source bytecode existed before CPU tests'
import torch
assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==''
print('NATIVE_CPU_CONTRACTS_NO_MODEL_NO_GPU',sys.version,torch.__version__,flush=True)
suites={
'chain':(34,['test_standard_teacher_identity_v4_chain.py','qwen25-32b-trained-eval64-r1-v4/test_training_identity.py','qwen25-32b-trained-eval64-r1-v4/test_trained_assets.py']),
'score':(49,['topd-32b-native-score-gate-r1-v4/test_score_gate.py','topd-32b-native-score-gate-r1-v4/test_caller_contracts.py']),
'owner':(26,['topd-32b-shared-owner-r1-v4/test_owner.py','topd-32b-shared-owner-r1-v4/test_context_policy.py']),
'ipc':(18,['topd-32b-pipeline-ipc-r1-v4/test_backend.py']),
'supervisor':(62,['topd-shared32b-supervisor-r1-v4/test_contracts.py'])}
T=R.parent/('ti'+os.environ['SLURM_JOB_ID']);T.mkdir(mode=0o700)
rows=[]
for name,(count,paths) in suites.items():
 log=R/(name+'.log')
 with log.open('x') as f:
  r=subprocess.run([sys.executable,'-m','pytest','-q',*[str(P/p) for p in paths],'--basetemp',str(T/('p' if name=='supervisor' else name[0])),'-o','cache_dir='+str(R/(name+'-cache'))],cwd=R,stdout=f,stderr=subprocess.STDOUT)
  f.flush();os.fsync(f.fileno())
 assert not list(P.rglob('__pycache__')) and not list(P.rglob('*.pyc')),'CPU tests wrote source bytecode'
 text=log.read_text();print(name,r.returncode,text,flush=True)
 good=r.returncode==0 and len(re.findall(r'(?m)^'+str(count)+r' passed(?:,.*)? in ',text))==1 and not re.search(r'\b[1-9][0-9]* skipped\b',text)
 rows.append(dict(suite=name,passed=count if good else None,returncode=r.returncode,path=str(log),sha256=sha(log),verified=good))
 if not good:break
report=dict(scope='NATIVE_SOURCE_CONTRACT_ONLY_NO_MODEL_NO_GPU_NOT_TEACHER_APPROVAL',status='PASS' if len(rows)==5 and all(x['verified'] for x in rows) else 'FAIL',suites=rows,passed=189 if len(rows)==5 and all(x['verified'] for x in rows) else None,parent_cuda_initialized=torch.cuda.is_initialized(),source_manifest_sha256=sha(P/'standard-search-r1-teacher-identity-v4-files.json'),entry_manifest_sha256=sha(P/'native-files.json'))
with (R/'result.json').open('x') as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
assert report['status']=='PASS',report
