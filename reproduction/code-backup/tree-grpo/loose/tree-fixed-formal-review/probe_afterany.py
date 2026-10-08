import ast,pathlib,json,tempfile,hashlib,subprocess,os,sys,socket
from unittest.mock import patch
ROOT=pathlib.Path.cwd();SOURCE=ROOT/'work/check_fixed_formal_afterany_296217a.py';OUT=ROOT/'work/tree-fixed-formal-review';cases=[]
for label,status,terminalcode,cleanup,expected in [('success','ACTUAL_STAGE_COMPLETED_UNREVIEWED',0,{},True),('failure','FAILED',0,{},False),('preflight','NO_SUPERVISOR_RESULT',0,{},False),('unknownterminal','ACTUAL_STAGE_COMPLETED_UNREVIEWED',1,{},False),('residual','ACTUAL_STAGE_COMPLETED_UNREVIEWED',0,{'12':'R'},False)]:
 with tempfile.TemporaryDirectory() as d:
  temp=pathlib.Path(d);(temp/'runs').mkdir();(temp/'tools').mkdir();(temp/'tools/check_real_checkpoint.py').write_bytes((ROOT/'work/check_real_checkpoint.py').read_bytes());run=temp/'runs/tree-fixed-formal-123'
  if status!='NO_SUPERVISOR_RESULT':run.mkdir();(run/'run-result.json').write_text(json.dumps(dict(status=status,code_commit='296217a4b676a26a92fc729f5d9882b785dbfa96',cleanup_residual=cleanup)))
  class Remap(ast.NodeTransformer):
   def visit_Constant(self,node):
    if node.value=='/dataset/shaoshitong/search-r1-rl/tree-grpo':return ast.copy_location(ast.Constant(str(temp)),node)
    return node
  calls=[]
  def fake_run(args,**kwargs):
   if args[0]=='scontrol':return subprocess.CompletedProcess(args,terminalcode,'JobId=123 JobState=COMPLETED ExitCode=0:0','unknown' if terminalcode else '')
   assert args[args.index('--checkpoint')+1].endswith('actor/global_step_500') and args[args.index('--output')+1].endswith('hf500-full-reload.json') and kwargs['env']['CUDA_VISIBLE_DEVICES']=='';calls.append(args);return subprocess.CompletedProcess(args,0,'','')
  code=compile(ast.fix_missing_locations(Remap().visit(ast.parse(SOURCE.read_text()))),str(SOURCE),'exec');caught=None
  with patch.object(sys,'argv',[str(SOURCE),'123']),patch.dict(os.environ,{'SLURM_JOB_ID':'456','SLURM_JOB_GPUS':''}),patch.object(socket,'gethostname',return_value='h100-9'),patch.object(subprocess,'check_output',return_value='JobState=RUNNING TresPerNode=gres:gpu:0 NumCPUs=8'),patch.object(subprocess,'run',side_effect=fake_run):
   try:exec(code,{'__name__':'__main__','__file__':str(SOURCE)})
   except AssertionError as error:caught=str(error)
  assert bool(calls)==expected,(label,calls);record=json.loads((temp/'runs/tree-fixed-formal-cpu-123-456/result.json').read_text())
  if label in ('unknownterminal','residual'):assert caught is not None and record['exit_code']==1
  if expected:assert record['status']=='HF500_FULL_RELOAD_COMPLETE_PENDING_REVIEW'
  cases.append(dict(case=label,hf_checker_calls=len(calls),caught=caught,record=record))
(OUT/'AFTERANY_CASES.json').write_text(json.dumps(dict(status='PASS_CPU_FIXTURES_ONLY',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),scope='Exact script AST with only ROOT remapped; mocked scontrol and HF subprocess, no actual HF/GPU/Slurm',cases=cases),indent=2)+'\n');print('5 HF500 afterany control flow cases passed')
