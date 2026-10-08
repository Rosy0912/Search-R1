"""Small v2 delta review only; source/metadata and generated CPU fixtures."""
import ast,copy,hashlib,importlib.util,json,os,pathlib,shlex,subprocess,sys,tempfile
from unittest.mock import patch
ROOT=pathlib.Path.cwd();HERE=ROOT/'work/tree-probe-selection-review/v2/child'
F2PATH=ROOT/'work/freeze_tree_probability_selection_v2.py';D2PATH=ROOT/'work/submit_tree_selection_v2_once.py'
FSHA='70d07788f4a372329d9e9c3ecb73cc5ba3bbccf4da0e39d52c6a8679c31f443b';DSHA='9b71ea06183046d7df819a41f7e64cabfeba0d0e42b30752eb5e4053d82fabef';TSHA='8e2e21bd068467e0e12ac68a9096f1619c26a2dc8905ddf79b46c29707095075'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(F2PATH)==FSHA and sha(D2PATH)==DSHA and sha(ROOT/'work/test_freeze_tree_probability_selection_v2.py')==TSHA
sys.path.insert(0,str(ROOT/'work'))
import freeze_tree_probability_selection_v2 as F
spec=importlib.util.spec_from_file_location('selection_v2_review_dispatch',D2PATH);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
cases=[]
def ok(n,b,d=None):
 assert b,(n,d);cases.append(dict(name=n,passed=True,detail=d))
def refused(n,call,phrase=None):
 try:call()
 except (ValueError,KeyError,SystemExit) as e:ok(n,phrase is None or phrase in str(e),type(e).__name__+': '+str(e))
 else:raise AssertionError(n+' unexpectedly accepted')
def dump(x):return ast.dump(x,include_attributes=False)
def parse(p):return ast.parse(pathlib.Path(p).read_text())
for p in [F2PATH,D2PATH,ROOT/'work/test_freeze_tree_probability_selection_v2.py']:
 compile(p.read_bytes(),str(p),'exec')
ok('three_v2_sources_exact_SHA_and_compile',True)

# Normalize only reviewed v2 helper changes, then compare entire AST to v1.
old=parse(ROOT/'work/freeze_tree_probability_selection.py');new=parse(F2PATH)
new.body=[n for n in new.body if not (isinstance(n,ast.FunctionDef) and n.name=='verify_cpu_terminal') and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CPU_TERMINAL_SHA' for t in n.targets))]
om=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main');nm=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
nm.body=[n for n in nm.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ((isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_argument' and any(isinstance(a,ast.Constant) and a.value=='--cpu-terminal' for a in n.value.args)) or (isinstance(n.value.func,ast.Name) and n.value.func.id=='verify_cpu_terminal'))) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='terminal' for t in n.targets))]
oldloop=next(n for n in om.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='sha' for t in n.target.elts))
newloop=next(n for n in nm.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='sha' for t in n.target.elts))
oldrequire=next(n for n in oldloop.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='require')
newloop.body.insert(2,copy.deepcopy(oldrequire))
for n in ast.walk(nm):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict':n.keywords=[k for k in n.keywords if k.arg!='cpu_terminal']
ok('v2_helper_AST_only_explicit_terminal_and_schema_fix_delta',dump(old)==dump(new))

# Normalize v2 dispatcher paths/hashes/terminal argument only, verify all other
# control/resource/shell semantics match the already 44-case-reviewed v1 AST.
mapstrings={
 'instruct300/probe-selection-v2-source':'instruct300/probe-selection-32593d9-source',
 'instruct300/probe-selection-v2-result':'instruct300/probe-selection-32593d9-result',
 'freeze_tree_probability_selection_v2.py':'freeze_tree_probability_selection.py',
 'test_freeze_tree_probability_selection_v2.py':'test_freeze_tree_probability_selection.py',
 FSHA:'32593d9e7dd0744dcd7c8ee764b5ecf516d213d2edcd05e72a240e2aa4ff62a4',
 TSHA:'34341c1f0fa2f1f59cfa25c177830d80556609bcb2b06bab8671ff1ea13a3fa9',
 'tree-probe-selection-3268-v2':'tree-probe-selection-3268',
 '--job-name=tree-probe-selection-3268-v2':'--job-name=tree-probe-selection-3268',
 'instruct300/PROBE_SELECTION_ONCE-3268-v2.json':'instruct300/PROBE_SELECTION_ONCE-3268-32593d9.json'}
class Normalize(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str) and n.value in mapstrings:n.value=mapstrings[n.value]
  return n
nd=Normalize().visit(parse(D2PATH));od=parse(ROOT/'work/submit_tree_selection_once.py')
b=next(n for n in nd.body if isinstance(n,ast.FunctionDef) and n.name=='body')
freeze=next(n for n in b.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='freeze' for t in n.targets))
assert isinstance(freeze.value,ast.List) and freeze.value.elts[-2].value=='--cpu-terminal';freeze.value.elts=freeze.value.elts[:-2]
ok('v2_dispatch_AST_only_new_paths_pins_terminal_argv_delta',dump(nd)==dump(od))
oldreceipt=ROOT/'work/tree-probe-selection-review/dispatch-child/DISPATCH_CHILD_RECEIPT.json'
ok('old_44_case_dispatch_proof_retained_exact',sha(oldreceipt)=='687183fdc3cf2df148d05169e003d6d8dfbf0951f9978bad1e1f8a97094848b8')

terminalpath=ROOT/'work/tree-position-boundary-review/ACTUAL_CPU3273_RECEIPT.json';terminal=json.loads(terminalpath.read_bytes())
ok('actual_cpu3273_receipt_exact_SHA',sha(terminalpath)==F.CPU_TERMINAL_SHA)
F.verify_cpu_terminal(terminal);ok('actual_cpu3273_receipt_schema_passes_v2_verifier',True)
reports={}
for name,expected in F.CPU_REPORTS.items():
 path=ROOT/'outputs/tree-instruct-300/cpu3273'/name;reports[name]=json.loads(path.read_bytes());ok('actual_report_'+name+'_exact_binding',sha(path)==expected and terminal['position_report_sha256' if name.startswith('position') else 'action_report_sha256']==expected)
action=reports['retained-actions-3273.json']
ok('real_action_report_lacks_old_invented_write_field','actual_optimizer_writes' not in action and action.get('actual_optimizer_writes') is None)
for key,value in [('job_id',3271),('parent_calibration_job_id',111),('host','other'),('slurm_state','FAILED'),('slurm_exit_code','1:0'),('gpu_count',1),('ready_sha256','0'*64),('package_manifest_sha256','0'*64),('position_report_sha256','0'*64),('action_report_sha256','0'*64),('actual_optimizer_writes',1),('actual_optimizer_writes',False),('actual_model_forwards',None),('actual_model_forwards',1)]:
 refused('terminal_refuses_'+key+'_'+str(value),lambda k=key,v=value:F.verify_cpu_terminal(dict(terminal,**{k:v})))
refused('terminal_absent_required_data',lambda:F.verify_cpu_terminal({}))

# Real helper main's local pre-data prefix. verify_bundle is a fabricated external
# metadata fixture; actual terminal/reports are byte-bound; PT loads never occur.
with tempfile.TemporaryDirectory(prefix='selection-v2-main-review-') as tmp:
 task=(pathlib.Path(tmp)/'task').resolve();run=task/'runs/run';package=task/'package';out=task/'instruct300/new';reportsdir=task/'reports';reportsdir.mkdir(parents=True);out.parent.mkdir(parents=True)
 for name in F.CPU_REPORTS:(reportsdir/name).write_bytes((ROOT/'outputs/tree-instruct-300/cpu3273'/name).read_bytes())
 term=task/'terminal.json';term.write_bytes(terminalpath.read_bytes())
 parent=dict(config_sha256=terminal['identity']['config_sha256'])
 bundle=(run,dict(parent_manifest={'sha256':F.PARENT_STATE}),dict(workers=[]),parent,[])
 argv=['freeze','--run',str(run),'--package',str(package),'--cpu-reports',str(reportsdir),'--output',str(out),'--cpu-terminal',str(term)]
 with patch.object(F.A,'TASK_ROOT',task),patch.object(F.A,'verify_bundle',return_value=bundle),patch.object(F.A,'load_verified',side_effect=AssertionError('Original PT load forbidden')) as loader,patch.object(F.socket,'gethostname',return_value='h100-9'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(sys,'argv',argv):
  refused('main_passes_actual_receipt_and_both_actual_report_schemas_then_refuses_absent_raw_data',F.main,'Raw complete counts changed')
  ok('pre_data_prefix_no_PT_load_and_no_output',not loader.called and not out.exists())
  term.write_text('{}')
  refused('main_changed_terminal_bytes_refused',F.main,'JSON SHA changed')
  ok('terminal_tamper_no_PT_load_and_no_output',not loader.called and not out.exists())
  with patch.object(sys,'argv',argv[:-2]):refused('main_cpu_terminal_argument_required',F.main)

# Reuse only v1's two fixture/invoke functions, not its 44-case test driver.
prior=parse(ROOT/'work/tree-probe-selection-review/dispatch-child/check_dispatch.py')
functions=[copy.deepcopy(n) for n in prior.body if isinstance(n,ast.FunctionDef) and n.name in ['fixture','invoke','json_write']]
class FixtureRename(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   n.value=n.value.replace('probe-selection-32593d9-source','probe-selection-v2-source').replace('probe-selection-32593d9-result','probe-selection-v2-result').replace('PROBE_SELECTION_ONCE-3268-32593d9.json','PROBE_SELECTION_ONCE-3268-v2.json').replace('freeze_tree_probability_selection.py','freeze_tree_probability_selection_v2.py')
  return n
fixturemodule=FixtureRename().visit(ast.Module(body=functions,type_ignores=[]));ast.fix_missing_locations(fixturemodule)
ns=dict(ROOT=ROOT,HERE=HERE,SOURCE=D2PATH,EXPECTED=DSHA,M=D,tempfile=tempfile,pathlib=pathlib,hashlib=hashlib,json=json,subprocess=subprocess,patch=patch,sys=sys)
exec(compile(fixturemodule,'<only-prior-fixture-functions>','exec'),ns)
for name,q,sbatch in [('success',None,None),('unknown_timeout',None,subprocess.TimeoutExpired(['sbatch'],60)),('own_active_v2',subprocess.CompletedProcess(['squeue'],0,'3 tree-probe-selection-3268-v2 RUNNING\n',''),None)]:
 temp,cfg,review_sha=ns['fixture']()
 try:
  calls,error,j=ns['invoke'](cfg,review_sha,queue=q,sbatch=sbatch)
  if name=='own_active_v2':ok('new_own_prefix_active_refused_before_journal_submit',error is not None and j is None and all(c['args'][0]!='sbatch' for c in calls))
  else:
   ok('v2_dispatch_mock_'+name,j['status']==('SUBMITTED' if name=='success' else 'SUBMISSION_UNKNOWN_DO_NOT_RETRY') and sum(c['args'][0]=='sbatch' for c in calls)==1)
   calls2,error2,j2=ns['invoke'](cfg,review_sha);ok('v2_'+name+'_exclusive_journal_blocks_repeat',isinstance(error2,FileExistsError) and all(c['args'][0]!='sbatch' for c in calls2) and j2==j)
   if name=='success':
    args=calls[-1]['args'];shell=calls[-1]['kwargs']['input'];ok('v2_resource_unchanged_and_new_jobname',all(v in args for v in ['--job-name=tree-probe-selection-3268-v2','--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:15:00','--no-requeue']))
    tokens=shlex.split(shell.splitlines()[-1]);ok('shell_mandatory_terminal_argument_exact',tokens[tokens.index('--cpu-terminal')+1]==str(cfg['TOOLS']/'ACTUAL_CPU3273_RECEIPT.json'))
 finally:temp.cleanup()
canonical=D.body();(HERE/'CANONICAL_V2_BODY.sh').write_text(canonical)
p=subprocess.run(['/bin/bash','-n',str(HERE/'CANONICAL_V2_BODY.sh')],capture_output=True,text=True)
ok('canonical_v2_shell_compiles',p.returncode==0,p.stderr)
# Actual queued inline SHA verifier, with external paths relocated; no phases run.
temp,cfg,review_sha=ns['fixture']()
try:
 with patch.multiple(D,**cfg):shell=D.body()
 verifier=shlex.split(shell.splitlines()[4])[-1]
 exec(compile(verifier,'<actual-inline-v2-verifier>','exec'));ok('queued_inline_verifier_pinned_sources_pass',True)
 (cfg['TOOLS']/'freeze_tree_probability_selection_v2.py').write_text('tampered=1\n')
 try:exec(compile(verifier,'<actual-inline-v2-verifier>','exec'))
 except AssertionError:ok('queued_v2_payload_tamper_refused',True)
 else:raise AssertionError('queued tamper accepted')
finally:temp.cleanup()
failedpath=ROOT/'outputs/tree-instruct-300/PPO_PROBE_SELECTION_CPU3275_PROGRESS1_RAW.json';failed=json.loads(failedpath.read_text())
state=failed['commands']['scontrol show job 3275']['stdout']
ok('historical3275_failed_terminal_retained',failed['job_id']==3275 and 'JobState=FAILED ' in state and 'ExitCode=1:0' in state)
result=dict(status='V2_SMALL_DELTA_SOURCE_DISPATCH_CPU_CASES_PASSED',helper_sha256=FSHA,dispatcher_sha256=DSHA,tests_sha256=TSHA,cases=cases,case_count=len(cases),old_44_tests_reexecuted=False,actual_selection_frozen=False,actual_gpu_accepted=False,actual_PT_assets_loaded=False,actual_model_forwards=0,actual_optimizer_writes=0,actual_remote_calls=0,actual_sbatch_calls=0,scope='Only v2 delta AST/schema/local pre-data main/control mocks and exact inline verifier. Real saved JSON/text metadata read, no PT/model/remote. Existing 3275 failure not overwritten or converted to success.')
(HERE/'V2_CHILD_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],case_count=len(cases))))
