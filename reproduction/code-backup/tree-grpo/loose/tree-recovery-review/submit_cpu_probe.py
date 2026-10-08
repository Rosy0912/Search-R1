import pathlib,json,hashlib,ast,tempfile,os,subprocess
from unittest.mock import patch
from types import SimpleNamespace
source=pathlib.Path('work/submit_tree_recovery_once.py').resolve();raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest();out=[]
for case in ('success','active','bad_prior','bad_review','wrong_helper','duplicate','timeout_then_retry'):
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td);tools=root/'tools';tools.mkdir();(root/'instruct300').mkdir();priorpath=root/'runs/tree-instruct300-3154';priorpath.mkdir(parents=True)
  manifest={'files':{},'external_files':{},'parent_manifest':{'path':str(root/'parent.json'),'sha256':'parent-pin'}}
  (tools/'manifest.json').write_text(json.dumps(manifest));(root/'parent.json').write_text('{}')
  review={'status':'READY_FOR_NATIVE_COLD_GATE_THEN_RECOVERY300','manifest_sha256':'f91c4f17a3a3454bb16a81e9ca400f4db07a56a0a27cf6cfac2c1814459c3087','submit_helper_sha256':sha,'requires_native8_fullstate_before3b':True,'requires_independent_recovery105_before106':True}
  if case=='bad_review':review['status']='PASS'
  if case=='wrong_helper':review['submit_helper_sha256']='bad'
  (tools/'PRE_DISPATCH_RECEIPT.json').write_text(json.dumps(review));reviewsha=hashlib.sha256((tools/'PRE_DISPATCH_RECEIPT.json').read_bytes()).hexdigest()
  prior={'job_id':'3154','cleanup_residual':{},'status':'FAILED','code_commit':'f994af33c6c93b6b9844863be912cab2a4020dc1','error':'Own Instruct300 supervisor interrupted 15'}
  if case=='bad_prior':prior['cleanup_residual']={'999':'S'}
  (priorpath/'run-result.json').write_text(json.dumps(prior))
  journal=root/'instruct300/SUBMISSION_ONCE-recovery-f91c4f17.json'
  if case=='duplicate':journal.write_text('{}')
  tree=ast.parse(raw)
  for n in tree.body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
    if n.targets[0].id=='ROOT':n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(root))],keywords=[])
    if n.targets[0].id=='TOOLS':n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(tools))],keywords=[])
  index=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assert));tree.body.insert(index,ast.Assign(targets=[ast.Name(id='sha',ctx=ast.Store())],value=ast.Name(id='fixture_sha',ctx=ast.Load())))
  def fixture_sha(path):
   path=pathlib.Path(path)
   if path==tools/'manifest.json':return review['manifest_sha256']
   if path==root/'parent.json':return 'parent-pin'
   return hashlib.sha256(path.read_bytes()).hexdigest()
  original_exists=pathlib.Path.exists
  def exists(path):return True if str(path).startswith('/proc/') else original_exists(path)
  calls=[]
  def submit(*a,**kw):
   calls.append(a[0]);assert a[0][0]=='sbatch' and '--parsable' in a[0]
   if case=='timeout_then_retry':raise subprocess.TimeoutExpired('sbatch',60)
   return SimpleNamespace(returncode=0,stdout='9999\n',stderr='')
  env={'TREE_RECOVERY_DISPATCH_REVIEW_SHA256':reviewsha}
  namespace={'__name__':'review_fixture','__file__':str(source),'fixture_sha':fixture_sha}
  with patch.dict(os.environ,env),patch('socket.gethostname',return_value='h100-9'),patch('pathlib.Path.exists',exists),patch('subprocess.check_output',return_value='123 tree-recovery300 RUNNING\n' if case=='active' else ''),patch('subprocess.run',side_effect=submit):
   try:exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
   except (AssertionError,FileExistsError,subprocess.TimeoutExpired):assert case!='success'
   else:assert case=='success'
   if case=='timeout_then_retry':
    assert json.loads(journal.read_text())['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
    try:exec(compile(tree,str(source),'exec'),namespace);raise AssertionError('retry allowed')
    except FileExistsError:pass
  assert len(calls)==(1 if case in ('success','timeout_then_retry') else 0)
  out.append({'case':case,'mock_sbatch_calls':len(calls),'passed':True})
pathlib.Path('work/tree-recovery-review/SUBMIT_CPU_PROOF.json').write_text(json.dumps({'status':'CPU_SUBMIT_CONTROLFLOW_FIXTURES_PASS','submit_helper_sha256':sha,'cases':out,'limitations':'Only temporary paths, mocked hostname/PID visibility/scheduler and manifest hash; no actual submission'},indent=2))
