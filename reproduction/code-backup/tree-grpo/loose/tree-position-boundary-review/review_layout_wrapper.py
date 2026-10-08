"""Independent exact wrapper compilation, mocked Slurm, genuine shell fixture.

Payload bytes stay intact. External fixture launcher maps only local metadata
expectations (HF source fixtures and actor manifest SHA), never a production job.
"""
import ast
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch

HERE=Path(__file__).resolve().parent;WORK=HERE.parent
CANDIDATE=WORK/'submit_tree_layout_analysis_once.py'
EXPECTED='b23fa9d3731f3d7dc0b56d34f8c940e5b05eda13cb4c5dd54c8dff2cd5680e18'
source=CANDIDATE.read_bytes();assert hashlib.sha256(source).hexdigest()==EXPECTED
compile(source,str(CANDIDATE),'exec')
copy=HERE/('candidate-'+EXPECTED+'.py')
if not copy.exists():copy.write_bytes(source)
assert copy.read_bytes()==source
constants={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
    and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_SHA')}
payloads={'inspect_tree_position_boundaries.py':constants['POSITION_SHA'],'analyze_tree_retained_actions.py':constants['ACTION_SHA'],
    'analyze_tree_sampling_calibration.py':constants['CONTEXT_SHA']}
for name,pin in payloads.items():
    data=(WORK/name).read_bytes();assert hashlib.sha256(data).hexdigest()==pin
    compile(data,str(WORK/name),'exec')
spec=importlib.util.spec_from_file_location('fixture_builder',WORK/'tree-sampling-context-review/dispatch/build_shell_fixture.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
spec=importlib.util.spec_from_file_location('position_fixtures',WORK/'test_inspect_tree_position_boundaries.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
real_run=subprocess.run
checks=['exact_frozen_wrapper_and_three_original_bound_payloads_compiled']
results=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return sha(path)

def basic(root):
    run=root/'runs/tree-safety300-3268';package=root/'instruct300/candidates/safety110-a632ba9-84e3e186'
    tools=root/'instruct300/context-layout-3992-b637';tools.mkdir(parents=True)
    for name in payloads:(tools/name).write_bytes((WORK/name).read_bytes())
    (tools/'RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json').write_bytes((WORK/'tree-retained-actions-review/SOURCE_CPU_READY_RECEIPT.json').read_bytes())
    (tools/'POSITION_SOURCE_CPU_READY_RECEIPT.json').write_bytes((HERE/'POSITION_SOURCE_CPU_RECEIPT.json').read_bytes())
    assert sha(tools/'RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json')==constants['ACTION_REVIEW_SHA']
    assert sha(tools/'POSITION_SOURCE_CPU_READY_RECEIPT.json')==constants['POSITION_REVIEW_SHA']
    package_sha=write_json(package/'manifest.json',{'stage':'calibration110','code_commit':'a632ba91840fb116aacfc0e770681a860647b295'})
    ready_sha=write_json(run/'output/probability-calibration/ready.json',{'status':'CALIBRATION_OBSERVED_ZERO_WRITES','optimizer_committed':0,
        'parent_completed_outer':110,'new_completed_outer':0,'job_id':'3268'})
    write_json(run/'run-result.json',{'status':'CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED','cleanup_residual':{}})
    return dict(root=root,run=run,tools=tools,package=package,package_sha=package_sha,ready_sha=ready_sha)

def namespace(fixture):
    tree=ast.parse(source)
    replaces={'ROOT':ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(fixture['root']))],keywords=[]),
        'PACKAGE_SHA':ast.Constant(fixture['package_sha']),'READY_SHA':ast.Constant(fixture['ready_sha'])}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in replaces:
            n.value=replaces[n.targets[0].id]
    ns={'__name__':'layout_wrapper_fixture'}
    exec(compile(ast.fix_missing_locations(tree),str(CANDIDATE),'exec'),ns)
    return ns

def invoke(ns,fixture,*,active='',terminal='JobId=3271 JobState=COMPLETED ExitCode=0:0 ',host='h100-9',
           outcome=None,argv_receipt=None):
    requests=[];scontrol=[];capture=io.StringIO();error=None
    def fake_run(argv,**kwargs):
        if argv[0]=='scontrol':
            scontrol.append(argv)
            return subprocess.CompletedProcess(argv,0,terminal,'')
        assert argv[0]=='sbatch';requests.append((argv,kwargs))
        if isinstance(outcome,BaseException):raise outcome
        return subprocess.CompletedProcess(argv,*outcome) if outcome is not None else subprocess.CompletedProcess(argv,0,'99998;fixture\n','')
    with patch.object(socket,'gethostname',return_value=host),patch.object(subprocess,'check_output',return_value=active),\
         patch.object(subprocess,'run',side_effect=fake_run),patch.object(sys,'argv',[str(CANDIDATE),'--position-receipt-sha256',argv_receipt or ns['POSITION_REVIEW_SHA']]),redirect_stdout(capture):
        try:ns['main']()
        except BaseException as exc:error=type(exc).__name__
    journal=fixture['root']/'instruct300/POSITION_ACTION_ANALYSIS_ONCE-3268-3992-b637.json'
    intent=json.loads(journal.read_bytes()) if journal.exists() else None
    return dict(error=error,requests=requests,scontrol_calls=scontrol,intent=intent,stdout=capture.getvalue())

with tempfile.TemporaryDirectory(prefix='layout-wrapper-review-',dir=HERE) as directory:
    base=Path(directory)
    cases=[('success',{}),('unknown_timeout',dict(outcome=subprocess.TimeoutExpired('sbatch',60))),
        ('unknown_malformed_success',dict(outcome=(0,'unexpected',''))),('known_submit_failure',dict(outcome=(1,'','slurm refused'))),
        ('active_prior_cpu',dict(active='3271 tree-context-analysis-3268-v2 RUNNING\n')),
        ('active_layout_cpu',dict(active='99998 tree-position-actions-3268 RUNNING\n')),
        ('prior_cpu_failed',dict(terminal='JobId=3271 JobState=FAILED ExitCode=1:0 ')),
        ('prior_cpu_purged_unknown',dict(terminal='')),('wrong_host',dict(host='other-node')),
        ('wrong_receipt_cli',dict(argv_receipt='0'*64))]
    for case,options in cases:
        fixture=basic(base/case);ns=namespace(fixture);out=invoke(ns,fixture,**options)
        if case=='success':
            assert out['error'] is None and out['intent']['status']=='SUBMITTED' and out['intent']['job_id']==99998
            second=invoke(ns,fixture);assert second['error']=='FileExistsError' and not second['requests']
            assert second['intent']==out['intent'];checks.append('existing_exclusive_success_journal_refuses_duplicate')
        elif case.startswith('unknown_'):
            assert out['error'] in ('TimeoutExpired','AssertionError') and out['intent']['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
            second=invoke(ns,fixture);assert second['error']=='FileExistsError' and not second['requests']
            assert second['intent']==out['intent'];checks.append(case+'_journal_blocks_replay')
        elif case=='known_submit_failure':
            assert out['error'] is None and out['intent']['status']=='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
            assert out['intent']['stderr']=='slurm refused'
        else:assert out['error']=='AssertionError' and not out['requests'] and out['intent'] is None
        checks.append(case+'_actual_control_flow')
        results.append(dict(case=case,error=out['error'],fake_sbatch_calls=len(out['requests']),intent=out['intent']))
    for case,field,value in [('action_status','status','UNREVIEWED'),('action_acceptance','actual_3b_training_accepted',True),
        ('action_threshold','thresholds_calibrated',True),('position_real_assets','actual_calibration_assets_loaded',True),
        ('position_model_forward','actual_model_forwards',1),('position_context','context_helper_sha256','0'*64)]:
        fixture=basic(base/case);ns=namespace(fixture)
        action=case.startswith('action_');name='RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json' if action else 'POSITION_SOURCE_CPU_READY_RECEIPT.json'
        receipt=json.loads((fixture['tools']/name).read_bytes());receipt[field]=value
        pin=write_json(fixture['tools']/name,receipt);ns['ACTION_REVIEW_SHA' if action else 'POSITION_REVIEW_SHA']=pin
        out=invoke(ns,fixture);assert out['error']=='AssertionError' and not out['requests'] and out['intent'] is None
        checks.append(case+'_rejects_even_rehashed_receipt_semantics')
    fixture=basic(base/'compile_invalid');ns=namespace(fixture)
    bad=fixture['tools']/'analyze_tree_retained_actions.py';bad.write_bytes(b'print(1))')
    ns['PAYLOADS']['analyze_tree_retained_actions.py']=sha(bad)
    out=invoke(ns,fixture);assert out['error']=='SyntaxError' and not out['requests'] and out['intent'] is None
    checks.append('matching_hash_invalid_payload_compile_refuses_before_journal')

    # Build a synthetic complete actor file and exact 4,237,098 action-token
    # classification count. Original main literals need no count bypass.
    fixture=F.build(base/'merged');root=fixture['root'];run=fixture['run'];package=fixture['package_dir'] if 'package_dir' in fixture else root/'instruct300/candidates/safety110-a632ba9-84e3e186'
    tools=root/'instruct300/context-layout-3992-b637';tools.mkdir()
    for name in payloads:(tools/name).write_bytes((WORK/name).read_bytes())
    (tools/'RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json').write_bytes((WORK/'tree-retained-actions-review/SOURCE_CPU_READY_RECEIPT.json').read_bytes())
    (tools/'POSITION_SOURCE_CPU_READY_RECEIPT.json').write_bytes((HERE/'POSITION_SOURCE_CPU_RECEIPT.json').read_bytes())
    package_record=json.loads((package/'manifest.json').read_bytes())
    for name in T.P.FORWARD_FILES:
        p=package/'source/repo'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((WORK/'tree-safety110-a632ba9-v2/source/repo'/name).read_bytes());package_record['files']['source/repo/'+name]=sha(p)
    fixture['package_sha']=write_json(package/'manifest.json',package_record)
    fixture.update(tools=tools,package=package)
    directory=run/'output/trajectories/outer-000111';manifest=json.loads((directory/'manifest.json').read_bytes());identity=manifest['identity']
    rows=3840;total=4237098;minimum,remainder=divmod(total,rows);width=minimum+1
    ids=torch.full((rows,width+2),3,dtype=torch.int64);ids[:,0]=1;ids[:,1]=2
    ids[remainder:,-1]=0;attention=torch.ones_like(ids);attention[remainder:,-1]=0
    loss=attention[:,2:].clone();counts=loss.sum(1,keepdim=True)
    tensors=dict(input_ids=ids,position_ids=torch.arange(width+2).repeat(rows,1),attention_mask=attention,
        responses=ids[:,2:],old_log_probs=torch.full((rows,width),-2.),loss_mask=loss,policy_weights=loss.float()/counts)
    actor=directory/'actor-inputs.pt';torch.save(dict(identity=identity,rows=rows,tensors=tensors),actor)
    trajectory=directory/'trajectories.jsonl.gz'
    with gzip.open(trajectory,'wt') as stream:
        for i in range(rows):
            n=minimum+(i<remainder);action_ids=[3]*n
            action=dict(action_token_ids=action_ids,observation_token_ids=[],action_tokens=n,observation_tokens=0,
                raw_generation_token_ids=action_ids,generation_context_token_ids=[1,2],scoring_context_matches_generation=True)
            stream.write(json.dumps(dict(row=i,identity=identity,token_ids=[1,2]+action_ids,metadata={'action_records':[action]}))+'\n')
    manifest['payloads']={'actor-inputs.pt':sha(actor),'trajectories.jsonl.gz':sha(trajectory)}
    manifest_sha=write_json(directory/'manifest.json',manifest)
    hf_dir=root/'fixture-hf-source';(hf_dir/'transformers/models/qwen2').mkdir(parents=True)
    hf=hf_dir/'transformers/modeling_flash_attention_utils.py';hf.write_text(T.HF_FIXTURE)
    qwen=hf_dir/'transformers/models/qwen2/modeling_qwen2.py';qwen.write_text(T.QWEN_FIXTURE)
    fixture_metadata=root/'fixture-expectations.json';write_json(fixture_metadata,dict(actor_manifest_sha256=manifest_sha,hf_sha256=sha(hf),qwen_sha256=sha(qwen),hf_root=str(hf_dir)))
    bootstrap=root/'fixture-bootstrap.py'
    bootstrap.write_text('''import importlib.metadata,json,os,pathlib,runpy,sys
args=sys.argv[1:]
if args and args[0]=='-B':args=args[1:]
if args[0]=='-c':
 sys.argv=['-c']+args[2:];exec(compile(args[1],'<original-inline-hash-check>','exec'),{'__name__':'__main__'});raise SystemExit
path=args[0];sys.argv=args
ns=runpy.run_path(path,run_name='_external_fixture_only')
metadata=json.loads(pathlib.Path(os.environ['TREE_LAYOUT_FIXTURE_METADATA']).read_bytes())
globals_=ns['main'].__globals__
if path.endswith('inspect_tree_position_boundaries.py'):
 actual_version=importlib.metadata.version;actual_distribution=importlib.metadata.distribution
 importlib.metadata.version=lambda name:'4.46.3' if name=='transformers' else actual_version(name)
 class Distribution:
  def locate_file(self,name):return pathlib.Path(metadata['hf_root'])/name
 importlib.metadata.distribution=lambda name:Distribution() if name=='transformers' else actual_distribution(name)
 globals_['HF_SHA']=metadata['hf_sha256'];globals_['HF_QWEN_SHA']=metadata['qwen_sha256']
elif path.endswith('analyze_tree_retained_actions.py'):
 globals_['MANIFEST_SHA']=metadata['actor_manifest_sha256']
else:raise AssertionError(path)
globals_['main']()
''')
    compile(bootstrap.read_bytes(),str(bootstrap),'exec')
    env_bin=root/'env/bin';env_bin.mkdir(parents=True)
    launcher=env_bin/'python';launcher.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' -B '+shlex.quote(str(bootstrap))+' "$@"\n');launcher.chmod(0o755)
    ns=namespace(fixture);out=invoke(ns,fixture);assert out['error'] is None
    argv,kwargs=out['requests'][0]
    assert '--job-name=tree-position-actions-3268' in argv
    for flag in ['--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:15:00','--no-requeue','--nodelist=h100-9']:
        assert flag in argv,flag
    assert kwargs['timeout']==60 and kwargs['text'] and kwargs['capture_output']
    body=kwargs['input'];assert body.count('\nexec ')==1 and 'set -o noclobber' in body
    assert body.index('/inspect_tree_position_boundaries.py --run')<body.index('\nexec ')
    syntax=real_run(['/bin/bash','-n'],input=body,capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
    checks.append('actual_generated_shell_inline_compile_bash_jobname_gpu0_cpu4_mem64G_15min_stdio')
    environment=dict(os.environ,PYTHONPATH=str(fixture['shim']),TREE_CONTEXT_REVIEW_FAKE_ROOT=str(root),TREE_LAYOUT_FIXTURE_METADATA=str(fixture_metadata),SLURM_JOB_ID='99998')
    execution=real_run(['/bin/bash'],input=body,capture_output=True,text=True,env=environment,timeout=60)
    assert execution.returncode==0,execution.stderr
    action_result=json.loads(execution.stdout);position_file=run/'position-analysis-99998.json';position_result=json.loads(position_file.read_bytes())
    assert action_result['helper_sha256']==constants['ACTION_SHA'] and action_result['counts']['retained_action_token_occurrences']==total
    assert action_result['counts']['leaf_rows']==3840 and action_result['own_record_equal_leaf_weight_coverage']==1.
    assert position_result['helper_sha256']==constants['POSITION_SHA'] and position_result['actor_input_rows']==3840
    assert position_result['actual_optimizer_writes']==position_result['actual_model_forwards']==0
    assert position_result['analysis_cost']['cuda_initialized'] is False and position_result['analysis_cost']['analysis_slurm_job_id']=='99998'
    checks.append('genuine_generated_shell_executes_two_original_payloads_on_explicit_external_cpu_fixture_metadata')
    position_bytes=position_file.read_bytes()
    second=real_run(['/bin/bash'],input=body,capture_output=True,text=True,env=environment,timeout=60)
    assert second.returncode!=0 and not second.stdout and position_file.read_bytes()==position_bytes
    checks.append('real_second_shell_noclobber_prevents_position_and_action_rerun')
    position_file.unlink();changed=tools/'analyze_tree_retained_actions.py';changed.write_bytes(changed.read_bytes()+b'\n# queue mutation fixture\n')
    third=real_run(['/bin/bash'],input=body,capture_output=True,text=True,env=environment,timeout=60)
    assert third.returncode!=0 and not third.stdout and not position_file.exists()
    checks.append('actual_inline_hash_guard_refuses_queued_source_tamper_before_either_payload')
    (HERE/'LAYOUT_MERGED_BODY.sh').write_text(body)
    (HERE/'LAYOUT_MERGED_POSITION_FIXTURE.json').write_bytes(position_bytes)
    (HERE/'LAYOUT_MERGED_ACTION_FIXTURE.json').write_text(execution.stdout)
    proof=dict(initial_exit_code=execution.returncode,noclobber_exit_code=second.returncode,queued_hash_exit_code=third.returncode,
        synthetic_actor_rows=3840,synthetic_actor_full_shape=list(ids.shape),synthetic_action_token_count=total,
        position_cpu_cost=position_result['analysis_cost'],action_cpu_cost={k:action_result[k] for k in ['wall_seconds','process_cpu_seconds','peak_rss_bytes']},
        fixture_metadata_overrides=['task root and hostname','synthetic package/Ready pins','installed HF metadata/version/source fixtures','synthetic actor manifest SHA'],
        actual_payload_bytes_modified=False,action_main_literal_count_bypassed=False)

assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()==EXPECTED
assert not torch.cuda.is_initialized()
result=dict(status='LAYOUT_CPU_WRAPPER_EXACT_SOURCE_AND_REAL_MERGED_SHELL_FIXTURES_PASS',source_sha256=EXPECTED,payload_sha256=payloads,
    checks=checks,checks_count=len(checks),mock_cases=results,proof=proof,actual_sbatch_calls=0,remote_calls=0,actual_model_assets_loaded=False,
    cuda_initialized=False,fixture_directory_removed=True,
    scope='Original payload bytes, real generated shell and main functions run on synthetic CPU assets with explicit external metadata mapping. This does not execute real hard-bound h100-9 assets or actual installed HF kernel/class trace, nor accept probability consistency or training.')
(HERE/'LAYOUT_WRAPPER_ACTUAL_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,sort_keys=True,allow_nan=False))
