"""Exact v2 delta and control probes; no shell payload/PT/model/remote rerun."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;WORK=HERE.parent;BASE=WORK.parent/'outputs/tree-instruct-300'
PATH=WORK/'submit_tree_layout_analysis_v2_once.py'
EXPECTED='6d3532a7bc54992f84be375d0a4cdfe4dfd5172d2dfb31db68e7a3b459ea454d'
OLD=HERE/'candidate-b23fa9d3731f3d7dc0b56d34f8c940e5b05eda13cb4c5dd54c8dff2cd5680e18.py'
source=PATH.read_bytes();old_source=OLD.read_bytes()
assert hashlib.sha256(source).hexdigest()==EXPECTED
compile(source,str(PATH),'exec')
snapshot=HERE/('candidate-'+EXPECTED+'.py')
if not snapshot.exists():snapshot.write_bytes(source)
assert snapshot.read_bytes()==source
pins={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
    and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_SHA')}
payloads={'inspect_tree_position_boundaries.py':pins['POSITION_SHA'],'analyze_tree_retained_actions.py':pins['ACTION_SHA'],
          'analyze_tree_sampling_calibration.py':pins['CONTEXT_SHA']}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,pin in payloads.items():
    data=(WORK/name).read_bytes();assert hashlib.sha256(data).hexdigest()==pin;compile(data,str(WORK/name),'exec')
checks=['exact_v2_wrapper_all_three_original_payloads_hash_and_compile']
results=[]
LIVE=json.loads((BASE/'PPO_CONTEXT_CPU_JOB3271_PROGRESS2_RAW.json').read_bytes())['commands']['scontrol show job 3271']['stdout']
INVALID='slurm_load_jobs error: Invalid job id specified\n'

def fixture(root):
    run=root/'runs/tree-safety300-3268';package=root/'instruct300/candidates/safety110-a632ba9-84e3e186'
    tools=root/'instruct300/context-layout-3992-b637-v2';tools.mkdir(parents=True)
    for name in payloads:(tools/name).write_bytes((WORK/name).read_bytes())
    copies={'RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json':WORK/'tree-retained-actions-review/SOURCE_CPU_READY_RECEIPT.json',
        'POSITION_SOURCE_CPU_READY_RECEIPT.json':HERE/'POSITION_SOURCE_CPU_RECEIPT.json',
        'CPU3271_SAVED_TERMINAL_RECEIPT.json':HERE/'CPU3271_SAVED_TERMINAL_RECEIPT.json',
        'CPU3271_SAVED_PROGRESS2_RAW.json':BASE/'PPO_CONTEXT_CPU_JOB3271_PROGRESS2_RAW.json'}
    for name,path in copies.items():(tools/name).write_bytes(path.read_bytes())
    package.mkdir(parents=True);(package/'manifest.json').write_bytes((WORK/'tree-safety110-a632ba9-v2/manifest.json').read_bytes())
    ready=run/'output/probability-calibration/ready.json';ready.parent.mkdir(parents=True)
    ready.write_bytes((BASE/'healthy110-3268/output/probability-calibration/ready.json').read_bytes())
    (run/'run-result.json').write_bytes((BASE/'healthy110-3268/run-result.json').read_bytes())
    for name in ['raw-stats-3271.json','tree-trajectory-3271.json','context-analysis-3271.json','context-analysis-3271.err']:
        (run/name).write_bytes((BASE/'cpu3271'/name).read_bytes())
    assert sha(package/'manifest.json')==pins['PACKAGE_SHA'] and sha(ready)==pins['READY_SHA']
    return dict(root=root,run=run,package=package,tools=tools)

def namespace(f,old=False):
    tree=ast.parse(old_source if old else source)
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ROOT':
            n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(f['root']))],keywords=[])
    ns={'__name__':'independent_layout_v2_fixture'}
    exec(compile(ast.fix_missing_locations(tree),str(OLD if old else PATH),'exec'),ns)
    return ns

def invoke(f,ns,*,rc=1,stdout='',stderr=INVALID,active='',queue_error=False,outcome=None):
    requests=[];scontrol=[];capture=io.StringIO();error=None
    def fake_run(argv,**kwargs):
        if argv[0]=='scontrol':
            scontrol.append((argv,kwargs));return subprocess.CompletedProcess(argv,rc,stdout,stderr)
        assert argv[0]=='sbatch';requests.append((argv,kwargs))
        if isinstance(outcome,BaseException):raise outcome
        return subprocess.CompletedProcess(argv,*outcome) if outcome is not None else subprocess.CompletedProcess(argv,0,'99997\n','')
    def fake_queue(*args,**kwargs):
        if queue_error:raise subprocess.CalledProcessError(1,['squeue'],stderr='scheduler unavailable')
        return active
    with patch.object(socket,'gethostname',return_value='h100-9'),patch.object(subprocess,'run',side_effect=fake_run),\
         patch.object(subprocess,'check_output',side_effect=fake_queue),patch.object(sys,'argv',[str(PATH),'--position-receipt-sha256',ns['POSITION_REVIEW_SHA']]),redirect_stdout(capture):
        try:ns['main']()
        except BaseException as exc:error=type(exc).__name__
    journal=f['root']/'instruct300/POSITION_ACTION_ANALYSIS_ONCE-3268-3992-b637-v2.json'
    if 'context-layout-3992-b637-v2' not in str(ns['TOOLS']):journal=journal.with_name(journal.name.replace('-v2.json','.json'))
    intent=json.loads(journal.read_bytes()) if journal.exists() else None
    return dict(error=error,requests=requests,scontrol_calls=scontrol,intent=intent,stdout=capture.getvalue())

with tempfile.TemporaryDirectory(prefix='layout-v2-delta-',dir=HERE) as directory:
    base=Path(directory)
    cases=[('purged_exact_historical_success',{}),('live_current_success',dict(rc=0,stdout=LIVE,stderr='')),
        ('live_running',dict(rc=0,stdout=LIVE.replace('JobState=COMPLETED','JobState=RUNNING'),stderr='')),
        ('live_failed',dict(rc=0,stdout=LIVE.replace('JobState=COMPLETED','JobState=FAILED').replace('ExitCode=0:0','ExitCode=1:0'),stderr='')),
        ('wrong_live_job',dict(rc=0,stdout=LIVE.replace('JobId=3271','JobId=999'),stderr='')),
        ('other_error',dict(rc=1,stderr='Slurm connection refused\n')),
        ('exit255',dict(rc=255,stderr=INVALID)),('invalid_with_stdout',dict(stdout='unexpected')),
        ('invalid_plus_other_error',dict(stderr=INVALID+'another scheduler error\n')),
        ('missing_squeue_response',dict(queue_error=True)),('active_layout_v2',dict(active='99997 tree-position-actions-3268-v2 RUNNING\n')),
        ('active_prior_cpu',dict(active='3271 tree-context-analysis-3268-v2 RUNNING\n')),
        ('submit_timeout_unknown',dict(outcome=subprocess.TimeoutExpired('sbatch',60))),
        ('submit_malformed_unknown',dict(outcome=(0,'unrecognized',''))),('submit_failure',dict(outcome=(1,'','refused')))]
    for name,options in cases:
        f=fixture(base/name);ns=namespace(f);out=invoke(f,ns,**options)
        if name in ('purged_exact_historical_success','live_current_success'):
            assert out['error'] is None and len(out['requests'])==1 and out['intent']['status']=='SUBMITTED'
            scope='SAVED_VERIFIED_SUCCESSFUL_TERMINAL_LIVE_JOB_RECORD_PURGED' if name.startswith('purged') else 'CURRENT_LIVE_SCONTROL_SUCCESSFUL_TERMINAL'
            assert out['intent']['previous_terminal_evidence_scope']==scope
            assert out['intent']['previous_live_scontrol']==dict(rc=options.get('rc',1),stdout=options.get('stdout',''),stderr=options.get('stderr',INVALID))
            second=invoke(f,ns,**options);assert second['error']=='FileExistsError' and not second['requests'] and second['intent']==out['intent']
            checks.append(name+'_exclusive_journal_refuses_second_submission')
        elif name.startswith('submit_'):
            assert len(out['requests'])==1
            if name=='submit_failure':assert out['intent']['status']=='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY' and out['error'] is None
            else:assert out['intent']['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY' and out['error'] in ('TimeoutExpired','AssertionError')
            second=invoke(f,ns);assert second['error']=='FileExistsError' and not second['requests'] and second['intent']==out['intent']
            checks.append(name+'_journal_preserved_no_retry')
        else:
            assert not out['requests'] and out['intent'] is None
            assert out['error']==('CalledProcessError' if name=='missing_squeue_response' else 'AssertionError')
        checks.append(name+'_control_flow')
        results.append(dict(name=name,error=out['error'],fake_sbatch_calls=len(out['requests']),intent=out['intent']))
    for name,target in [('terminal_receipt_tamper','CPU3271_SAVED_TERMINAL_RECEIPT.json'),('raw_capture_tamper','CPU3271_SAVED_PROGRESS2_RAW.json'),
        ('context_report_tamper','context-analysis-3271.json'),('raw_report_missing','raw-stats-3271.json'),('empty_err_tamper','context-analysis-3271.err')]:
        f=fixture(base/name);ns=namespace(f);p=(f['tools'] if target.startswith('CPU3271') else f['run'])/target
        if name.endswith('_missing'):p.unlink()
        else:p.write_bytes(p.read_bytes()+b'x')
        out=invoke(f,ns);assert not out['requests'] and out['intent'] is None and out['error'] in ('AssertionError','FileNotFoundError')
        checks.append(name+'_refused_no_fallback_submit')
    # A valid live terminal is sufficient without the historical cache; no cache
    # bypass is permitted in the purged branch.
    f=fixture(base/'live_no_historical_cache');ns=namespace(f)
    (f['tools']/'CPU3271_SAVED_TERMINAL_RECEIPT.json').unlink()
    out=invoke(f,ns,rc=0,stdout=LIVE,stderr='');assert out['error'] is None and out['intent']['previous_terminal_evidence_scope']=='CURRENT_LIVE_SCONTROL_SUCCESSFUL_TERMINAL'
    checks.append('live_terminal_path_does_not_mislabel_or_require_historical_cache')

    f=fixture(base/'body_equivalence');ns=namespace(f)
    v2=invoke(f,ns,rc=0,stdout=LIVE,stderr='');assert v2['error'] is None
    oldtools=f['root']/'instruct300/context-layout-3992-b637';shutil.copytree(f['tools'],oldtools)
    oldns=namespace(f,old=True);old=invoke(f,oldns,rc=0,stdout=LIVE,stderr='');assert old['error'] is None
    new_argv,new_kw=v2['requests'][0];old_argv,old_kw=old['requests'][0]
    assert new_kw['input'].replace('/context-layout-3992-b637-v2/','/context-layout-3992-b637/')==old_kw['input']
    assert [s.replace('--job-name=tree-position-actions-3268-v2','--job-name=tree-position-actions-3268') for s in new_argv]==old_argv
    assert {k:v for k,v in new_kw.items() if k!='input'}=={k:v for k,v in old_kw.items() if k!='input'}
    syntax=subprocess.run(['/bin/bash','-n'],input=new_kw['input'],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
    # Extract and independently compile the actual inline Python from the shell
    # argument list instead of treating bash syntax as Python compilation.
    import shlex
    inline=shlex.split(new_kw['input'].splitlines()[4]);assert inline[1:3]==['-B','-c']
    compile(inline[3],'<actual-inline-bound-hash-check>','exec')
    for flag in ['--job-name=tree-position-actions-3268-v2','--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:15:00','--no-requeue']:
        assert flag in new_argv
    (HERE/'LAYOUT_V2_BODY.sh').write_text(new_kw['input'])
    checks.append('generated_v2_body_exact_b23_equivalence_except_tool_path_jobname_and_compiled_inline_bash')
    bodyproof=dict(v2_body_sha256=hashlib.sha256(new_kw['input'].encode()).hexdigest(),b23_fixture_body_sha256=hashlib.sha256(old_kw['input'].encode()).hexdigest(),
        payload_execution_repeated=False,prior_real_merged_shell_receipt_sha256='30a94d46577c88a1886ed18b6704d99daaca80dd705b1085b3f7242d5f892dea')

assert hashlib.sha256(PATH.read_bytes()).hexdigest()==EXPECTED
result=dict(status='LAYOUT_V2_EXACT_DELTA_AND_CONTROL_PROBES_PASS',source_sha256=EXPECTED,payload_sha256=payloads,
    checks=checks,checks_count=len(checks),mock_cases=results,body=bodyproof,actual_sbatch_calls=0,remote_calls=0,actual_pt_reads=0,
    actual_gpu_calls=0,actual_optimizer_writes=0,scope='Exact v2 source and all payload compile; mocked Slurm branches with actual already-saved JSON bytes, source-derived body equivalence to previously executed b23 shell. No PT or actual payload rerun, no remote or real sbatch; never training/probability/threshold acceptance.')
(HERE/'LAYOUT_V2_DELTA_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,sort_keys=True,allow_nan=False))
