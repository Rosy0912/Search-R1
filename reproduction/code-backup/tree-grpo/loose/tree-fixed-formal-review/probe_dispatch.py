import ast,os,sys,json,subprocess,tempfile,hashlib,types,contextlib,io,shlex
from pathlib import Path
from unittest.mock import patch
R=Path.cwd();O=R/'work/tree-fixed-formal-review';SOURCE=R/'work/dispatch_tree_fixed_formal.py';COMMIT='296217a4b676a26a92fc729f5d9882b785dbfa96';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'READY_FOR_FRESH_FORMAL500_DISPATCH_ONLY','code_commit':COMMIT,'actual_updates_authorized':500,'manifest_sha256':sha(R/'work/tree-fixed-formal-296217a/manifest.json'),'wrapper_sha256':sha(R/'work/start_tree_fixed_formal_296217a.sh'),'cpu_afterany_sha256':sha(R/'work/check_fixed_formal_afterany_296217a.py'),'summary_sha256':sha(R/'work/summarize_tree_fixed_formal_296217a.py')};cases=[]
for case in ['success','GPU_unknown','CPU_submit_failed']:
 with tempfile.TemporaryDirectory() as d:
  temp=Path(d);(temp/'work/tree-fixed-formal-review').mkdir(parents=True);(temp/'outputs/evidence').mkdir(parents=True);(temp/'work/tree-fixed-formal-review/PRE_DISPATCH_RECEIPT.json').write_text(json.dumps(receipt))
  for rel in ['work/tree-fixed-formal-296217a/manifest.json','work/start_tree_fixed_formal_296217a.sh','work/check_fixed_formal_afterany_296217a.py','work/summarize_tree_fixed_formal_296217a.py']:
   p=temp/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((R/rel).read_bytes())
  calls=[]
  def route(cmd,node):assert node=='h100-16';return ['fake',cmd]
  def fake(args,**kwargs):
   intent=temp/'outputs/evidence/tree-fixed-formal-dispatch-intent.json';assert intent.exists() and json.loads(intent.read_text())['status']=='INTENT_BEFORE_SBATCH_DO_NOT_RETRY_AUTOMATICALLY';calls.append(shlex.split(args[1]))
   if case=='GPU_unknown' or case=='CPU_submit_failed' and len(calls)==2:raise subprocess.CalledProcessError(1,args)
   return str(199+len(calls))+'\n'
  mod=types.ModuleType('remote');mod.route=route;code=compile(SOURCE.read_text(),str(SOURCE),'exec');old=Path.cwd();os.chdir(temp)
  try:
   with patch.dict(sys.modules,{'remote':mod}),patch.object(subprocess,'check_output',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
    error=None
    try:exec(code,{'__name__':'__main__'})
    except subprocess.CalledProcessError as e:error=str(e)
    n=len(calls)
    try:exec(code,{'__name__':'__main__'})
    except (AssertionError,FileExistsError):pass
    else:raise AssertionError('Duplicate dispatch allowed')
    assert len(calls)==n
   record=json.loads((temp/'outputs/evidence/tree-fixed-formal-dispatch.json').read_text()) if (temp/'outputs/evidence/tree-fixed-formal-dispatch.json').exists() else None
   if case=='success':
    assert len(calls)==4 and '--gres=gpu:8' in calls[0] and '--exclusive' in calls[0] and '--time=3-00:00:00' in calls[0] and '--dependency=afterany:200' in calls[1] and '--dependency=afterany:201' in calls[2] and '--dependency=afterok:201:202' in calls[3] and record['summary_cpu_job_id']==203
   elif case=='GPU_unknown':assert len(calls)==1 and record is None
   else:assert len(calls)==2 and record['training_job_id']==200 and 'afterany_cpu_job_id' not in record
   cases.append({'case':case,'submit_calls':len(calls),'second_invocation_submits':0,'intent_retained':True,'partial_job_record':record,'error':error})
  finally:os.chdir(old)
(O/'DISPATCH_CASES.json').write_text(json.dumps({'status':'PASS_LOCAL_SUBMISSION_MOCK_ONLY','source_sha256':sha(SOURCE),'cases':cases,'scope':'Actual dispatcher with subprocess/SSH route mocked; no jobs submitted'},indent=2));print('3 exact dispatcher cases passed; durable intent prevents duplicate/unknown retry')
