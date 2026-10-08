import ast,hashlib,json,tempfile,types,shlex
from pathlib import Path
from unittest.mock import patch
source=Path('work/queue_recovery3234_terminal_once.py');tree=ast.parse(source.read_text())
names={'ROOT','DIRECTORY','TOOL','RECEIPT','SHA','RECEIPT_SHA'}
tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
code=compile(tree,str(source),'exec');results=[]
cases=['success','existing_intent','sbatch_error','sbatch_timeout','sbatch_unknown','bad_tool','bad_receipt_sha','wrong_status','wrong_source','wrong_updates','wrong_receipt_job','wrong_parent','node_prefix','batchhost_wrong','restarted','not_running','scheduler_error']
for case in cases:
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);tool=root/'verify.py';tool.write_text('EXECUTED_FIXTURE=True\n');sha=hashlib.sha256(tool.read_bytes()).hexdigest()
  review=dict(tool_sha256=sha,status='READY_FOR_CPU_TERMINAL_EVIDENCE_CAPTURE_ONLY',job_id='3234',expected_new_updates=200,source_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326')
  for c,k,v in [('wrong_status','status','PASS'),('wrong_source','source_commit','old'),('wrong_updates','expected_new_updates',300),('wrong_receipt_job','job_id','3154')]:
   if case==c:review[k]=v
  receipt=root/'READY.json';receipt.write_text(json.dumps(review));rsha=hashlib.sha256(receipt.read_bytes()).hexdigest()
  if case=='bad_receipt_sha':receipt.write_text('{}')
  if case=='existing_intent':(root/'SUBMISSION_ONCE.json').write_text('existing untouched')
  if case=='bad_tool':tool.write_text('changed')
  ns=dict(ROOT=root,DIRECTORY=root,TOOL=tool,RECEIPT=receipt,SHA=sha,RECEIPT_SHA=rsha);calls=[]
  parent='JobId=3234 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 Restarts=0'
  for c,a,b in [('wrong_parent','JobId=3234','JobId=3154'),('node_prefix','h100-9','h100-90'),('batchhost_wrong','BatchHost=h100-9','BatchHost=h100-8'),('restarted','Restarts=0','Restarts=1'),('not_running','RUNNING','COMPLETED')]:
   if case==c:parent=parent.replace(a,b)
  def control(*a,**kw):
   if case=='scheduler_error':raise RuntimeError('Unavailable scheduler')
   return parent
  def submit(args,**kw):
   calls.append(args);assert (root/'SUBMISSION_ONCE.json').exists()
   if case=='sbatch_timeout':raise TimeoutError('Unknown submission status')
   return types.SimpleNamespace(returncode=1 if case=='sbatch_error' else 0,stdout='' if case=='sbatch_error' else 'unknown' if case=='sbatch_unknown' else '3999\n',stderr='fixture')
  err=None
  with patch('subprocess.check_output',control),patch('subprocess.run',submit):
   try:exec(code,ns)
   except Exception as e:err=type(e).__name__
  dispatched=case in ('success','sbatch_error','sbatch_timeout','sbatch_unknown');assert len(calls)==int(dispatched)
  if case=='existing_intent':assert (root/'SUBMISSION_ONCE.json').read_text()=='existing untouched'
  if dispatched:assert (root/'SUBMISSION_ONCE.json').exists()
  if case=='success':
   args=calls[0];wrapped=next(a[len('--wrap='):] for a in args if a.startswith('--wrap='));words=shlex.split(wrapped);expression=words[words.index('-c')+1]
   execution_ns={};exec(expression,execution_ns);assert execution_ns['EXECUTED_FIXTURE'] is True
   tool.write_text('EXECUTED_FIXTURE=False\n')
   try:exec(expression,{})
   except AssertionError:pass
   else:raise AssertionError('Changed delayed tool accepted')
   assert '--dependency=afterany:3234' in args and '--gres=gpu:0' in args and '--mem=32G' in args and '--time=02:00:00' in args and args[args.index('-w')+1]=='h100-9' and '--cpus-per-task=8' in args
  if dispatched:
   before=len(calls)
   with patch('subprocess.check_output',control),patch('subprocess.run',submit):
    try:exec(code,ns)
    except (FileExistsError,AssertionError):pass
    else:raise AssertionError('Repeat unexpectedly accepted')
   assert len(calls)==before
  results.append(dict(case=case,sbatch_calls=len(calls),error=err,intent_exists=(root/'SUBMISSION_ONCE.json').exists(),result_exists=(root/'SUBMISSION_RESULT.json').exists()))
Path(__file__).with_name('CPU_PROOF.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Actual AST control flow with only fixture paths/pins and mocked scheduler; no real submission',cases=results),indent=2)+'\n')
print('PASS',len(results))
