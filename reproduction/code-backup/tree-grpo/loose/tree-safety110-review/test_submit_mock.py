import ast,json,hashlib,tempfile,os,socket,subprocess,time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
W=Path.cwd();helper=W/'work/submit_tree_safety110_once.py'; H=lambda b:hashlib.sha256(b).hexdigest(); digest=H(helper.read_bytes());results=[]
for case in ['success_duplicate','timeout_duplicate','bad_receipt','bad_payload','bad_parent','active_own','bad_protected','sbatch_nonzero']:
 with tempfile.TemporaryDirectory() as td:
  root=Path(td).resolve();tools=root/'instruct300/candidates/p';tools.mkdir(parents=True);(root/'runs/tree-recovery300-3234').mkdir(parents=True)
  payload=tools/'launch.sbatch';payload.write_text('fixture no execution');parent=root/'parent.json';parent.write_text('{}')
  m=dict(code_commit='a632',stage='calibration110',files={'launch.sbatch':H(payload.read_bytes())},external_files={},parent_manifest=dict(path=str(parent),sha256=H(parent.read_bytes()),identity=dict(completed_step=110)),probability_calibration=dict(environment={'TREE_PPO_CALIBRATION_ONLY':'1'}))
  (tools/'manifest.json').write_text(json.dumps(m));mh=H((tools/'manifest.json').read_bytes())
  r=dict(status='READY_FOR_HEALTHY110_ZERO_WRITE_CALIBRATION',manifest_sha256=mh,code_commit='a632',submit_helper_sha256=digest,requires_zero_optimizer_writes=True,long_training_ready=False)
  if case=='bad_receipt':r['requires_zero_optimizer_writes']=False
  (tools/'PRE_DISPATCH_RECEIPT.json').write_text(json.dumps(r));rh=H((tools/'PRE_DISPATCH_RECEIPT.json').read_bytes())
  prior=root/'runs/tree-recovery300-3234/run-result.json';prior.write_text(json.dumps(dict(status='FAILED',cleanup_residual={},job_id='3234')))
  if case=='bad_parent':parent.write_text('tampered')
  if case=='bad_payload':payload.write_text('tampered')
  env=dict(TREE_SAFETY110_PACKAGE_PATH=str(tools),TREE_SAFETY110_PACKAGE_SHA256=mh,TREE_SAFETY110_REVIEW_SHA256=rh)
  tree=ast.parse(helper.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in n.targets))]; code=compile(tree,str(helper),'exec')
  calls=[];orig=Path.read_text
  def read(p,*a,**kw):
   if str(p).startswith('/proc/'):
    pid=int(p.parts[2]);f=['S']+['0']*18+['1502352181' if pid<1690000 else '1502398762'];
    if case=='bad_protected':f[19]='0'
    return f'{pid} (protected fixture) '+' '.join(f)
   return orig(p,*a,**kw)
  def run(args,**kw):
   calls.append(args);assert args[0]=='sbatch' and args[-1]==str(payload)
   assert kw['env']['TREE_INSTRUCT_PACKAGE_SHA256']==mh
   if case=='timeout_duplicate':raise subprocess.TimeoutExpired(args,60)
   return SimpleNamespace(returncode=1 if case=='sbatch_nonzero' else 0,stdout='' if case=='sbatch_nonzero' else '9999\n',stderr='mock failure' if case=='sbatch_nonzero' else '')
  errors=[]
  with patch.dict(os.environ,env),patch.object(socket,'gethostname',return_value='h100-9'),patch.object(Path,'read_text',read),patch.object(subprocess,'check_output',return_value='777 tree-recovery300 RUNNING' if case=='active_own' else ''),patch.object(subprocess,'run',run):
   for attempt in range(2 if case.endswith('duplicate') else 1):
    try:exec(code,dict(ROOT=root,__file__=str(helper),__name__='__main__'))
    except (AssertionError,FileExistsError,subprocess.TimeoutExpired) as e:
     errors.append(type(e).__name__)
  journal=root/'instruct300'/('SUBMISSION_ONCE-safety110-'+mh[:12]+'.json')
  if case=='success_duplicate':assert len(calls)==1 and errors==['FileExistsError'] and json.loads(journal.read_text())['status']=='SUBMITTED'
  elif case=='timeout_duplicate':assert len(calls)==1 and errors==['TimeoutExpired','FileExistsError'] and json.loads(journal.read_text())['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
  elif case=='sbatch_nonzero':assert len(calls)==1 and json.loads(journal.read_text())['status']=='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
  else:assert not calls and errors==['AssertionError'] and not journal.exists(),(case,calls,errors)
  results.append(dict(case=case,sbatch_calls=len(calls),errors=errors,status='PASS'))
print(json.dumps(dict(status='CPU_MOCK_SUBMIT_PASS',cases=results,submit_helper_sha256=digest,actual_submission=False)))
