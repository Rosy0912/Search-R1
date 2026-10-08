import pathlib,hashlib,json,os,subprocess,time,socket
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TOOLS=ROOT/'instruct300/candidates/tree-cbb74d0-a42c2f86'
PACKAGE='a42c2f86d860836aab2919c277653c57a6daa3d659523244c1f8335d373ecc50'
REVIEW='7d8651520a10d3cae01e9b19ff4b8f764c7806ed461000ab0917ec4ae3dacca5'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert socket.gethostname().split('.')[0]=='h100-9'
assert sha(TOOLS/'manifest.json')==PACKAGE and sha(TOOLS/'PRE_DISPATCH_RECEIPT.json')==REVIEW
m=json.loads((TOOLS/'manifest.json').read_bytes())
for file,expected in m['files'].items():assert sha(TOOLS/file)==expected,file
for file,expected in m['external_files'].items():assert sha(pathlib.Path(file))==expected,file
model=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
actual={str(p.relative_to(model)):sha(p) for p in model.rglob('*') if p.is_file()}
assert actual==m['model_identity']['files'],'incomplete or changed official assets'
assert sha(ROOT/'instruct300/common-data/manifest.json')==m['common_data_identity']
active=subprocess.check_output(['squeue','-h','-w','h100-9','-o','%i %j %T'],text=True)
assert 'tree-instruct300' not in active,'already submitted'
journal=ROOT/'instruct300/SUBMISSION_ONCE.json'
intent={'status':'SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN','epoch':time.time(),'package_sha256':PACKAGE,'review_sha256':REVIEW,'model_identity':m['model_identity']['sha256'],'actual_updates_authorized':300}
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
