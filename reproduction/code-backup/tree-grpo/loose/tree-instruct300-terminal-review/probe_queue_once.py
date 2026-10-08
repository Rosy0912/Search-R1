import ast,hashlib,json,tempfile,types,shlex
from pathlib import Path
from unittest.mock import patch
source=Path(__file__).resolve().parents[1]/'queue_instruct3154_terminal_once.py';tree=ast.parse(source.read_text())
# Relocate only fixture paths/pins; execute every original control-flow and submission line.
names={'ROOT','DIRECTORY','TOOL','RECEIPT','SHA','RECEIPT_SHA'}
tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
code=compile(tree,str(source),'exec');results=[]
for case in ['success','existing_intent','sbatch_error','sbatch_timeout','bad_tool','wrong_parent','node_prefix']:
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);tool=root/'verify.py';tool.write_text('EXECUTED_FIXTURE=True\n');sha=hashlib.sha256(tool.read_bytes()).hexdigest();receipt=root/'READY.json';receipt.write_text(json.dumps(dict(source_sha256=sha,status='READY_FOR_CPU_TERMINAL_EVIDENCE_COLLECTION_ONLY')))
  if case=='existing_intent':(root/'SUBMISSION_ONCE.json').write_text('existing untouched')
  if case=='bad_tool':tool.write_text('changed')
  ns=dict(ROOT=root,DIRECTORY=root,TOOL=tool,RECEIPT=receipt,SHA=sha,RECEIPT_SHA=hashlib.sha256(receipt.read_bytes()).hexdigest());calls=[]
  parent='JobId=3154 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 Restarts=0'
  if case=='wrong_parent':parent=parent.replace('JobId=3154','JobId=3153')
  if case=='node_prefix':parent=parent.replace('h100-9','h100-90')
  def submit(args,**kw):
   calls.append(args);assert (root/'SUBMISSION_ONCE.json').exists()
   if case=='sbatch_timeout':raise TimeoutError('Unknown submission status')
   return types.SimpleNamespace(returncode=1 if case=='sbatch_error' else 0,stdout='' if case=='sbatch_error' else '3999\n',stderr='fixture')
  err=None
  with patch('subprocess.check_output',lambda *a,**k:parent),patch('subprocess.run',submit):
   try:exec(code,ns)
   except Exception as e:err=type(e).__name__
  if case in ('existing_intent','bad_tool','wrong_parent','node_prefix'):assert not calls
  else:assert len(calls)==1
  if case=='existing_intent':assert (root/'SUBMISSION_ONCE.json').read_text()=='existing untouched'
  if case=='success':
   args=calls[0]
   wrapped=[a for a in args if a.startswith('--wrap=')][0][len('--wrap='):]
   words=shlex.split(wrapped);expression=words[words.index('-c')+1]
   execution_ns={'__name__':'__main__'};exec(expression,execution_ns);assert execution_ns['EXECUTED_FIXTURE'] is True
   tool.write_text('EXECUTED_FIXTURE=\"changed\"\n')
   try:exec(expression,{'__name__':'__main__'})
   except AssertionError:pass
   else:raise AssertionError('Changed delayed tool accepted')
   assert '--dependency=afterany:3154' in args and '--gres=gpu:0' in args and '--mem=32G' in args and '--time=00:30:00' in args and args[args.index('-w')+1]=='h100-9' and '--cpus-per-task=8' in args
  results.append(dict(case=case,sbatch_calls=len(calls),error=err,intent_exists=(root/'SUBMISSION_ONCE.json').exists(),result_exists=(root/'SUBMISSION_RESULT.json').exists()))
Path(__file__).with_name('QUEUE_CPU_PROBES.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Actual AST control flow with fixture paths/pins and mocked scheduler; no actual submission',cases=results),indent=2)+'\n')
