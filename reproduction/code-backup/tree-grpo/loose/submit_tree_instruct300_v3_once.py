import pathlib,hashlib,json,os,subprocess,time,socket
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TOOLS=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161'
PACKAGE='aa9e3161ee8a211e6e311d2a87be9d456371f0f569d0e0c9c1703e3a575ba69f'
REVIEW=os.environ['TREE_INSTRUCT_RETRY_REVIEW_SHA256']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert socket.gethostname().split('.')[0]=='h100-9'
assert sha(TOOLS/'manifest.json')==PACKAGE and sha(TOOLS/'PRE_DISPATCH_RECEIPT.json')==REVIEW
review=json.loads((TOOLS/'PRE_DISPATCH_RECEIPT.json').read_bytes())
assert review['status']=='READY_FOR_SAME_JOB_NATIVE_GATE_THEN_INSTRUCT300'
assert review['manifest_sha256']==PACKAGE and review['requires_native8_fullstate_before3b'] is True and review['requires_independent_formal5_before6'] is True
prior=ROOT/'runs/tree-instruct300-3149'
prior_result=json.loads((prior/'run-result.json').read_bytes())
assert prior_result['status']=='FAILED' and prior_result['error']=='GPU KL backward proof failed' and prior_result['cleanup_residual']=={}
assert not (prior/'stage.log').exists() and not list((prior/'native-fullstate').glob('rank-*.json')) and not list((prior/'output/tree-rank-audit').glob('*.json'))
prior_terminal=ROOT/'instruct300/PRIOR_3149_FAILURE_RECEIPT.json'
assert sha(prior_terminal)=='63e1936ea73bb9adbe3a705d1c7169bdd123449db0670b83c8c3d04ba19d94e9'
failure=json.loads(prior_terminal.read_bytes())
assert failure['status']=='ACTUAL_PRETRAIN_GATE_FAILURE_CONFIRMED' and failure['job_id']=='3149' and failure['scientific_updates']==0 and failure['slurm_exit']=='1:0'
for record in failure['raw_files']:assert sha(pathlib.Path(record['path']))==record['sha256'],record['path']
prior_live=subprocess.run(['scontrol','show','job','3149'],text=True,capture_output=True)
if prior_live.returncode==0:assert 'JobState=FAILED' in prior_live.stdout
else:assert 'Invalid job id specified' in prior_live.stderr,'Unknown scheduler failure; do not submit'

prior2=ROOT/'runs/tree-instruct300-3150'
result2=json.loads((prior2/'run-result.json').read_bytes())
assert result2['status']=='FAILED' and result2['error']=='Native fullstate replay failed before any3B training' and result2['cleanup_residual']=={}
assert not (prior2/'stage.log').exists() and not list((prior2/'output/tree-rank-audit').glob('*.json'))
receipt2=ROOT/'instruct300/PRIOR_3150_FAILURE_RECEIPT.json'
assert sha(receipt2)=='dd4fffbd51ed08de3f808c1007bb33b2ba647dff287679c268a2beee99961bd3'
failure2=json.loads(receipt2.read_bytes())
assert failure2['status']=='ACTUAL_NATIVE_FULLSTATE_GATE_FAILED' and failure2['job_id']=='3150' and failure2['actual_scientific_updates']==0 and failure2['slurm_exit']=='1:0'
for record in failure2['raw_files']:assert sha(pathlib.Path(record['path']))==record['sha256'],record['path']
live2=subprocess.run(['scontrol','show','job','3150'],text=True,capture_output=True)
if live2.returncode==0:assert 'JobState=FAILED' in live2.stdout
else:assert 'Invalid job id specified' in live2.stderr,'Unknown scheduler failure; do not submit'

m=json.loads((TOOLS/'manifest.json').read_bytes())
for file,expected in m['files'].items():assert sha(TOOLS/file)==expected,file
for file,expected in m['external_files'].items():assert sha(pathlib.Path(file))==expected,file
model=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
actual={str(p.relative_to(model)):sha(p) for p in model.rglob('*') if p.is_file()}
assert actual==m['model_identity']['files'],'incomplete or changed official assets'
assert sha(ROOT/'instruct300/common-data/manifest.json')==m['common_data_identity']
active=subprocess.check_output(['squeue','-h','-w','h100-9','-o','%i %j %T'],text=True)
assert 'tree-instruct300' not in active,'already submitted'
journal=ROOT/'instruct300/SUBMISSION_ONCE-aa9e3161.json'
intent={'status':'SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN','epoch':time.time(),'package_sha256':PACKAGE,'review_sha256':REVIEW,'model_identity':m['model_identity']['sha256'],'actual_updates_authorized':300,'prior_startup_failure_jobs':[3149,3150],'prior_actual_training_updates':0}
with journal.open('x') as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
env=os.environ.copy();env.update(TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_PACKAGE_SHA256=PACKAGE,TREE_INSTRUCT_REVIEW_SHA256=REVIEW)
try:
 p=subprocess.run(['sbatch','--parsable','--output='+str(ROOT/'instruct300/slurm-%j.out'),'--error='+str(ROOT/'instruct300/slurm-%j.err'),str(TOOLS/'launch.sbatch')],env=env,text=True,capture_output=True,timeout=60)
 intent.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 if p.returncode==0:
  job=p.stdout.strip().split(';')[0];assert job.isdigit(),p.stdout;intent.update(status='SUBMITTED',job_id=int(job))
 else:intent['status']='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
except BaseException as e:
 intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',exception=repr(e));raise
finally:
 temp=journal.with_suffix('.final.tmp')
 with temp.open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,journal);print(json.dumps(intent),flush=True)
