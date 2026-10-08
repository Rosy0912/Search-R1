"""Independent local control-flow simulation; no remote or real Slurm commands."""
import hashlib, importlib.util, json, os, pathlib, shutil, subprocess, sys, tempfile
from unittest.mock import patch
ROOT=pathlib.Path.cwd()
HERE=ROOT/'work/tree-probe-selection-review/dispatch-child'
SOURCE=ROOT/'work/submit_tree_selection_once.py'
EXPECTED='67dc6fac52cbbeee96c4286df2b08e01793354c976c41b976c91dacc8fbc3933'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
spec=importlib.util.spec_from_file_location('review_selection_dispatch', SOURCE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
records=[]
def check(name, condition, detail=None):
    assert condition, (name,detail)
    records.append(dict(name=name,passed=True,detail=detail))
def json_write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture():
    temp=tempfile.TemporaryDirectory(prefix='selection-dispatch-review-')
    root=pathlib.Path(temp.name)/'tree-grpo'
    tools=root/'instruct300/probe-selection-32593d9-source'
    output=root/'instruct300/probe-selection-32593d9-result'
    run=root/'runs/tree-safety300-3268'
    package=root/'instruct300/candidates/safety110-a632ba9-84e3e186'
    tools.mkdir(parents=True);run.mkdir(parents=True);package.mkdir(parents=True)
    for name,sha in M.PAYLOADS.items():
        data=(ROOT/'work'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==sha
        compile(data,str(ROOT/'work'/name),'exec')
        (tools/name).write_bytes(data)
    for src,dest,sha in [
      (ROOT/'work/tree-position-boundary-review/ACTUAL_CPU3273_RECEIPT.json',tools/'ACTUAL_CPU3273_RECEIPT.json',M.TERMINAL_SHA),
      (ROOT/'outputs/tree-instruct-300/cpu3273/position-analysis-3273.json',run/'position-analysis-3273.json',M.POSITION_SHA),
      (ROOT/'outputs/tree-instruct-300/cpu3273/retained-actions-3273.json',run/'retained-actions-3273.json',M.ACTION_SHA),
      (ROOT/'outputs/tree-instruct-300/healthy110-3268/output/probability-calibration/ready.json',run/'output/probability-calibration/ready.json',M.READY_SHA)]:
        data=src.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    review=dict(status='SOURCE_AND_DISPATCH_REVIEWED_FOR_SAME_NODE_CPU_FREEZE_ONLY',helper_sha256=M.PAYLOADS['freeze_tree_probability_selection.py'],dispatcher_sha256=EXPECTED,actual_selection_frozen=False,actual_gpu_accepted=False)
    review_path=tools/'SELECTION_DISPATCH_CPU_RECEIPT.json'
    review_sha=json_write(review_path,review)
    return temp,dict(ROOT=root,TOOLS=tools,OUTPUT=output,RUN=run,PACKAGE=package),review_sha

def invoke(cfg,review_sha,queue=None,sbatch=None,hostname='h100-9',mutate=None):
    calls=[]
    queue=queue if queue is not None else subprocess.CompletedProcess(['squeue'],0,'','')
    sbatch=sbatch if sbatch is not None else subprocess.CompletedProcess(['sbatch'],0,'9999;fixture\n','')
    def fake_run(args,**kw):
        assert args[0] in ('squeue','sbatch'),args
        calls.append(dict(args=args,kwargs=kw))
        obj=queue if args[0]=='squeue' else sbatch
        if isinstance(obj,BaseException):raise obj
        return obj
    patches=[patch.object(M,k,v) for k,v in cfg.items()]
    for p in patches:p.start()
    caught=None
    try:
      with patch.object(M.socket,'gethostname',return_value=hostname),patch.object(M.subprocess,'run',side_effect=fake_run),patch.object(sys,'argv',[str(SOURCE),'--review-sha256',review_sha]),patch('builtins.print'):
        try:M.main()
        except BaseException as e:caught=e
    finally:
      for p in reversed(patches):p.stop()
    journal=cfg['ROOT']/'instruct300/PROBE_SELECTION_ONCE-3268-32593d9.json'
    value=json.loads(journal.read_text()) if journal.exists() else None
    return calls,caught,value

# Original h100-9-path body is syntax-checked locally only.
canonical=M.body();(HERE/'CANONICAL_BODY.sh').write_text(canonical)
cp=subprocess.run(['/bin/bash','-n',str(HERE/'CANONICAL_BODY.sh')],capture_output=True,text=True)
check('canonical_h1009_path_bash_body_compiles',cp.returncode==0,cp.stderr)
# No real scheduler or network process is invoked: subprocess.run is mocked for all main calls.
temp,cfg,sha=fixture()
try:
 calls,error,j=invoke(cfg,sha)
 check('successful_mock_submission_once',error is None and j['status']=='SUBMITTED' and j['job_id']==9999)
 args=calls[-1]['args'];shell=calls[-1]['kwargs']['input']
 check('correct_slurm_resource_and_requeue_arguments', all(x in args for x in ['--nodes=1','--nodelist=h100-9','--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:15:00','--no-requeue']))
 check('no_unrequested_gpu_or_remote_cli',all(c['args'][0] in ['squeue','sbatch'] for c in calls) and sum(a.startswith('--gres=') for a in args)==1)
 check('own_prefix_current_successful_queue_check',calls[0]['args']==['squeue','-h','-w','h100-9','-o','%i %j %T'])
 check('new_tools_results_and_journal_names',all('probe-selection-32593d9' in str(cfg[k]) for k in ['TOOLS','OUTPUT']) and 'PROBE_SELECTION_ONCE-3268-32593d9' in str(cfg['ROOT']/'instruct300/PROBE_SELECTION_ONCE-3268-32593d9.json'))
 check('saved_terminal_scope_recorded',j['terminal_scope']=='SAVED_VERIFIED_CPU3273_SUCCESSFUL_TERMINAL' and j['terminal_receipt_sha256']==M.TERMINAL_SHA)
 check('journal_has_exact_source_review_and_resource_bindings',j['source_sha256']==M.PAYLOADS and j['review_sha256']==sha and all(j[k]==v for k,v in dict(gpu_count=0,cpu_count=4,memory_gib=64,time_minutes=15).items()))
 calls2,error2,j2=invoke(cfg,sha)
 check('existing_journal_blocks_second_submission',isinstance(error2,FileExistsError) and all(c['args'][0]!='sbatch' for c in calls2) and j2==j)
 (HERE/'EXACT_BODY.sh').write_text(shell)
 cp=subprocess.run(['/bin/bash','-n',str(HERE/'EXACT_BODY.sh')],capture_output=True,text=True)
 check('exact_bash_body_compiles',cp.returncode==0,cp.stderr)
 check('shell_disables_cuda_and_limits_cpu_threads', 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4' in shell)
 check('shell_verifier_precedes_tests_and_freeze',shell.index(' -c ') < shell.index('test_freeze_tree_probability_selection.py'))
 check('shell_executes_test_before_freeze',shell.index('test_freeze_tree_probability_selection.py')<shell.index('exec '))
 check('shell_noclobber_and_error_stop',all(s in shell for s in ['set -euo pipefail','set -o noclobber']))
finally:temp.cleanup()

# Reject changed source/receipts/output/queue before any journal or submit.
rejections=[
 ('wrong_hostname',lambda cfg,s: s,dict(hostname='other-node')),
 ('nonhex_review_sha',lambda cfg,s:'z'*64,{}),
 ('changed_payload',lambda cfg,s: ((cfg['TOOLS']/'freeze_tree_probability_selection.py').write_text('x=1\n'),s)[1],{}),
 ('wrong_review_byte_hash',lambda cfg,s:'0'*64,{}),
 ('changed_terminal_receipt',lambda cfg,s: ((cfg['TOOLS']/'ACTUAL_CPU3273_RECEIPT.json').write_text('{}'),s)[1],{}),
 ('changed_position_report',lambda cfg,s: ((cfg['RUN']/'position-analysis-3273.json').write_text('{}'),s)[1],{}),
 ('changed_action_report',lambda cfg,s: ((cfg['RUN']/'retained-actions-3273.json').write_text('{}'),s)[1],{}),
 ('changed_ready_metadata',lambda cfg,s: ((cfg['RUN']/'output/probability-calibration/ready.json').write_text('{}'),s)[1],{}),
 ('preexisting_output',lambda cfg,s: (cfg['OUTPUT'].mkdir(),s)[1],{}),
 ('queue_nonzero',lambda cfg,s:s,dict(queue=subprocess.CompletedProcess(['squeue'],1,'',''))),
 ('queue_warning_stderr',lambda cfg,s:s,dict(queue=subprocess.CompletedProcess(['squeue'],0,'','warning\n'))),
 ('queue_own_active_prefix',lambda cfg,s:s,dict(queue=subprocess.CompletedProcess(['squeue'],0,'9 tree-probe-selection-3268 RUNNING\n',''))),
 ('queue_timeout',lambda cfg,s:s,dict(queue=subprocess.TimeoutExpired(['squeue'],15))),
]
for name,change,kw in rejections:
 temp,cfg,sha=fixture()
 try:
  sha=change(cfg,sha);calls,error,j=invoke(cfg,sha,**kw)
  check(name,error is not None and j is None and all(c['args'][0]!='sbatch' for c in calls),type(error).__name__)
 finally:temp.cleanup()
for field,value in [('status','NOT_REVIEWED'),('helper_sha256','0'*64),('dispatcher_sha256','0'*64),('actual_selection_frozen',True),('actual_gpu_accepted',True)]:
 temp,cfg,sha=fixture()
 try:
  path=cfg['TOOLS']/'SELECTION_DISPATCH_CPU_RECEIPT.json';v=json.loads(path.read_text());v[field]=value;sha=json_write(path,v)
  calls,error,j=invoke(cfg,sha)
  check('reject_review_'+field,isinstance(error,AssertionError) and j is None and all(c['args'][0]!='sbatch' for c in calls))
 finally:temp.cleanup()
for name,obj,status in [
 ('sbatch_nonzero',subprocess.CompletedProcess(['sbatch'],1,'','allocation unavailable'),'SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'),
 ('sbatch_timeout',subprocess.TimeoutExpired(['sbatch'],60),'SUBMISSION_UNKNOWN_DO_NOT_RETRY'),
 ('sbatch_transport_exception',OSError('fixture disconnect'),'SUBMISSION_UNKNOWN_DO_NOT_RETRY'),
 ('sbatch_success_malformed_output',subprocess.CompletedProcess(['sbatch'],0,'not-a-job\n',''),'SUBMISSION_UNKNOWN_DO_NOT_RETRY')]:
 temp,cfg,sha=fixture()
 try:
  calls,error,j=invoke(cfg,sha,sbatch=obj)
  check(name, isinstance(error,SystemExit) and j['status']==status and sum(c['args'][0]=='sbatch' for c in calls)==1)
  calls2,error2,j2=invoke(cfg,sha)
  check(name+'_cannot_repeat',isinstance(error2,FileExistsError) and all(c['args'][0]!='sbatch' for c in calls2) and j2==j)
 finally:temp.cleanup()

# Exact generated shell is executed with a fabricated interpreter facade.
# The facade executes only the tiny inline SHA verifier. Test/freeze phases are simulations,
# never actual helper/test invocation and never original PT/model reads.
temp,cfg,sha=fixture()
try:
 for k,v in cfg.items():setattr(M,k,v)
 envdir=cfg['ROOT']/'env/bin';envdir.mkdir(parents=True)
 fake=envdir/'python'
 facade='''#!/usr/bin/env python3
import hashlib,json,os,pathlib,sys
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ['OMP_NUM_THREADS']==os.environ['MKL_NUM_THREADS']=='4'
a=sys.argv[1:]
assert a.pop(0)=='-B'
if a[0]=='-c':
 exec(compile(a[1],'<exact-inline-sha-verifier>','exec'),{'__builtins__':__builtins__})
elif pathlib.Path(a[0]).name=='test_freeze_tree_probability_selection.py':
 print('SIMULATED_TEST_PHASE')
elif pathlib.Path(a[0]).name=='freeze_tree_probability_selection.py':
 out=pathlib.Path(a[a.index('--output')+1]);out.mkdir()
 (out/'simulated-freeze-call.json').write_text(json.dumps({'simulation_only':True,'arguments':a,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}))
else:raise RuntimeError('Unexpected facade argument')
'''
 fake.write_text(facade);fake.chmod(0o755)
 shell=M.body();script=cfg['TOOLS']/'exact-body-fixture.sh';script.write_text(shell)
 env=os.environ.copy();env['SLURM_JOB_ID']='fixture9999'
 cp=subprocess.run(['/bin/bash',str(script)],capture_output=True,text=True,env=env)
 check('exact_shell_fake_phases_succeed',cp.returncode==0 and (cfg['OUTPUT']/'simulated-freeze-call.json').exists(),cp.stderr)
 log=cfg['TOOLS']/'target-cpu-tests-fixture9999.txt'
 check('exact_shell_preserves_simulated_test_stdout',log.read_text()=='SIMULATED_TEST_PHASE\n')
 shutil.rmtree(cfg['OUTPUT'])
 cp=subprocess.run(['/bin/bash',str(script)],capture_output=True,text=True,env=env)
 check('exact_shell_noclobber_stops_before_simulated_freeze',cp.returncode!=0 and not cfg['OUTPUT'].exists() and log.read_text()=='SIMULATED_TEST_PHASE\n',cp.stderr)
 log.unlink();(cfg['TOOLS']/'analyze_tree_sampling_calibration.py').write_text('changed=1\n')
 cp=subprocess.run(['/bin/bash',str(script)],capture_output=True,text=True,env=env)
 check('queued_payload_tamper_refuses_before_test_and_freeze',cp.returncode!=0 and not log.exists() and not cfg['OUTPUT'].exists(),cp.stderr)
finally:temp.cleanup()
result=dict(status='INDEPENDENT_DISPATCH_CONTROL_FLOW_CPU_FIXTURES_PASSED',dispatcher_sha256=EXPECTED,helper_sha256=M.PAYLOADS['freeze_tree_probability_selection.py'],cases=records,case_count=len(records),scope='Exact local wrapper import/source compilation, saved JSON/text metadata copies and fabricated scheduler/interpreter facade; no actual PT/model/data reads, no scheduler/network/remote commands, no real helper main or freeze.',actual_sbatch_calls=0,actual_remote_calls=0,actual_selection_frozen=False,actual_gpu_accepted=False,actual_model_forwards=0,actual_optimizer_writes=0)
(HERE/'DISPATCH_CHILD_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],case_count=len(records))))
