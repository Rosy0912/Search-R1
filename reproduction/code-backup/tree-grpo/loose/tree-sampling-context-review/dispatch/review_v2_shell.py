"""Compile exact v2 payloads and execute the generated shell on tiny CPU fixtures."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--expected-submit-sha256',required=True);args=parser.parse_args()
path=WORK/'submit_tree_context_analysis_v2_once.py';source=path.read_bytes();assert hashlib.sha256(source).hexdigest()==args.expected_submit_sha256
compile(source,str(path),'exec');tree_original=ast.parse(source)
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree_original.body if isinstance(n,ast.Assign) and len(n.targets)==1
    and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'INDEPENDENT_PROBES','HELPER_SHA'}}
spec=importlib.util.spec_from_file_location('build_fixture',HERE/'build_shell_fixture.py');F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
checks=['exact_submit_source_hash_and_compile'];payloads={};proof={};real_run=subprocess.run
with tempfile.TemporaryDirectory(prefix='merged-body-fixture-',dir=HERE) as temporary:
    fixture=F.build(Path(temporary)/'fake-task-root');root=fixture['root'];tools=fixture['tools'];run=fixture['run']
    for name,pin in {'analyze_tree_sampling_calibration.py':constants['HELPER_SHA'],**constants['INDEPENDENT_PROBES']}.items():
        origin=WORK/'analyze_tree_sampling_calibration.py' if name=='analyze_tree_sampling_calibration.py' else WORK/'tree-safety110-review/actual3268'/name
        data=origin.read_bytes();assert hashlib.sha256(data).hexdigest()==pin,(name,pin);compile(data,str(origin),'exec');(tools/name).write_bytes(data)
        payloads[name]=pin
    checks.append('all_three_exact_bound_payload_hashes_ast_and_compile')
    review=json.loads((WORK/'tree-sampling-context-review/CONTEXT_HELPER_RECEIPT.json').read_bytes());review['expected_run']=str(run)
    review_sha=F.json_file(tools/'CONTEXT_HELPER_RECEIPT.json',review)
    env_bin=root/'env/bin';env_bin.mkdir(parents=True);(env_bin/'python').symlink_to(sys.executable)
    tree=ast.parse(source)
    replacements={'ROOT':ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(root))],keywords=[]),
        'REVIEW_SHA':ast.Constant(value=review_sha),'PACKAGE_SHA':ast.Constant(value=fixture['package_sha']),'READY_SHA':ast.Constant(value=fixture['ready_sha'])}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
            node.value=replacements[node.targets[0].id]
    requests=[];fake_submit_out=io.StringIO()
    def fake_submit(argv,**kwargs):requests.append((argv,kwargs));return subprocess.CompletedProcess(argv,0,'99999\n','')
    with patch.object(socket,'gethostname',return_value='h100-9'),patch.object(subprocess,'check_output',return_value=''),patch.object(subprocess,'run',side_effect=fake_submit),redirect_stdout(fake_submit_out):
        exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),{'__name__':'local_shell_review_fixture'})
    assert len(requests)==1;argv,kwargs=requests[0]
    assert '--job-name=tree-context-analysis-3268-v2' in argv
    for flag in ['--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:45:00','--no-requeue','--nodelist=h100-9']:
        assert flag in argv,flag
    assert kwargs['timeout']==60 and kwargs['capture_output'] and kwargs['text']
    intent=json.loads(fake_submit_out.getvalue());assert intent['status']=='SUBMITTED' and intent['job_id']==99999
    body=kwargs['input'];assert body.count('\nexec ')==1 and 'set -o noclobber' in body
    checks.append('final_job_name_resources_stdio_and_single_fake_sbatch_verified')
    shell_syntax=real_run(['/bin/bash','-n'],input=body,text=True,capture_output=True);assert shell_syntax.returncode==0,shell_syntax.stderr
    checks.append('actual_generated_shell_body_bash_syntax_pass')
    environment=dict(os.environ,PYTHONPATH=str(fixture['shim']),TREE_CONTEXT_REVIEW_FAKE_ROOT=str(root),SLURM_JOB_ID='99999')
    execution=real_run(['/bin/bash'],input=body,text=True,capture_output=True,timeout=60,env=environment)
    assert execution.returncode==0,execution.stderr
    helper=json.loads(execution.stdout)
    raw_path=run/'raw-stats-99999.json';trajectory_path=run/'tree-trajectory-99999.json'
    raw=json.loads(raw_path.read_bytes());trajectory=json.loads(trajectory_path.read_bytes())
    assert raw['status']=='RAW_SAME_TURN_PHYSICAL_TOKEN_STATS_INDEPENDENTLY_RECOMPUTED'
    assert raw['all_token_occurrences']==8 and raw['all_max_abs']==0. and raw['cuda_initialized'] is False
    assert trajectory['status']=='ACTUAL_CALIBRATION_3840_TREE_TRAJECTORY_INPUTS_VERIFIED_ZERO_WRITE_ONLY'
    assert trajectory['records'][0]['trajectory_rows']==3840 and trajectory['cuda_initialized'] is False
    assert helper['status']=='EXACT_CONTEXT_SAMPLING_EVIDENCE_ANALYZED_NO_TRAINING_ACCEPTANCE'
    assert helper['complete_actor_inputs']['leaf_rows_denominator']==3840
    assert helper['monitor_subset']['leaf_rows_denominator']==256
    assert helper['complete_actor_inputs']['exact_context_matched_token_occurrences']==3840
    assert helper['actual_optimizer_writes']==helper['actual_forward_passes']==0 and helper['analysis_cost']['cuda_initialized'] is False
    assert helper['analysis_cost']['analysis_slurm_job_id']=='99999'
    checks.append('real_three_original_payloads_serial_shell_execution_small_cpu_assets_complete')
    original_raw=raw_path.read_bytes();original_trajectory=trajectory_path.read_bytes()
    second=real_run(['/bin/bash'],input=body,text=True,capture_output=True,timeout=60,env=environment)
    assert second.returncode!=0 and not second.stdout
    assert raw_path.read_bytes()==original_raw and trajectory_path.read_bytes()==original_trajectory
    checks.append('actual_body_second_execution_noclobber_refuses_without_overwriting_reports')
    # Source corruption after queueing is caught by the actual inline hash
    # verifier before any of the payloads can run.
    raw_path.unlink();trajectory_path.unlink();probe=tools/'raw_stats_cpu_v2.py';probe.write_bytes(probe.read_bytes()+b'\n# changed after queueing\n')
    third=real_run(['/bin/bash'],input=body,text=True,capture_output=True,timeout=60,env=environment)
    assert third.returncode!=0 and not third.stdout and not raw_path.exists() and not trajectory_path.exists()
    checks.append('actual_inline_bound_hash_verifier_refuses_after_queue_source_change')
    proof=dict(raw_rows=8,raw_valid_response_tokens=8,tree_leaf_rows=3840,monitor_rows=256,
        tensor_max_input_shape=[3840,3],helper_actual_optimizer_writes=0,helper_actual_forward_passes=0,
        raw_cpu_wall_seconds=raw['wall_seconds'],trajectory_cpu_wall_seconds=trajectory['wall_seconds'],helper_cpu_wall_seconds=helper['analysis_cost']['wall_seconds'],
        helper_peak_rss_bytes=helper['analysis_cost']['peak_rss_bytes'],initial_shell_exit_code=execution.returncode,
        noclobber_second_exit_code=second.returncode,hash_corruption_third_exit_code=third.returncode,
        artifact_bytes={'raw_stats':len(original_raw),'tree_trajectory':len(original_trajectory),'helper_stdout':len(execution.stdout.encode())})
    (HERE/'MERGED_BODY_FIXTURE_ACTUAL_HELPER.json').write_text(json.dumps(helper,sort_keys=True,allow_nan=False)+'\n')
    (HERE/'MERGED_BODY_FIXTURE_ACTUAL_RAW_STATS.json').write_bytes(original_raw)
    (HERE/'MERGED_BODY_FIXTURE_ACTUAL_TREE_TRAJECTORY.json').write_bytes(original_trajectory)
    (HERE/'MERGED_BODY_FIXTURE_BODY.sh').write_text(body)
result=dict(status='CPU_DISPATCH_V2_EXACT_COMPILE_AND_REAL_SMALL_SHELL_EXECUTION_PASS',source_sha256=args.expected_submit_sha256,
    payload_sha256=payloads,checks=checks,check_count=len(checks),proof=proof,actual_sbatch_calls=0,remote_calls=False,
    real_remote_model_assets_loaded=False,fixture_directory_removed=True,
    scope='Actual original bound payload source and generated shell executed on local fabricated CPU tensors with explicit task-root/hostname fixture mapping. No actual3268 PT or probability/training acceptance.')
print(json.dumps(result,sort_keys=True,allow_nan=False))
