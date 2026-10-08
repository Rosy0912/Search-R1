"""Fixed-source dispatch precheck with local temporary metadata and mocked Slurm."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
CANDIDATE=WORK/'submit_tree_context_analysis_v2_once.py'
EXPECTED='a4bf62e8fcd3df0a1080738c8ad0d5b5600936a27b4a0a85fed13575511940aa'
source=CANDIDATE.read_bytes();assert hashlib.sha256(source).hexdigest()==EXPECTED
checks=[];results=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return sha(path)

def run_case(name,*,rows=1,active='',returncode=0,stdout='3271;cluster\n',exception=None,existing=False,
             terminal='CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED',hostname='h100-9',bad_hash=False,bad_syntax=False):
    with tempfile.TemporaryDirectory(prefix='submit-review-',dir=HERE) as temporary:
        root=Path(temporary);run=root/'runs/tree-safety300-3268';package=root/'instruct300/candidates/safety110-a632ba9-84e3e186'
        tools=root/'instruct300/context-analysis-28e665eb-v2';tools.mkdir(parents=True)
        (tools/'analyze_tree_sampling_calibration.py').write_bytes((WORK/'analyze_tree_sampling_calibration.py').read_bytes())
        for filename in ['raw_stats_cpu_v2.py','tree_trajectory_cpu.py']:
            (tools/filename).write_bytes((WORK/'tree-safety110-review/actual3268'/filename).read_bytes())
        review=json.loads((WORK/'tree-sampling-context-review/CONTEXT_HELPER_RECEIPT.json').read_bytes());review['expected_run']=str(run)
        review_sha=write_json(tools/'CONTEXT_HELPER_RECEIPT.json',review)
        package_sha=write_json(package/'manifest.json',{'stage':'calibration110','code_commit':'a632ba91840fb116aacfc0e770681a860647b295'})
        workers=[]
        for rank in range(8):
            p=run/'output/sampling-consistency'/f'rank-{rank:02d}-call-00001.json'
            record_sha=write_json(p,{'rank':rank,'saved_rows':rows,'local_rows':rows})
            workers.append({'rank':rank,'diagnostic_entries':[{'path':str(p),'sha256':record_sha}]})
        ready_sha=write_json(run/'output/probability-calibration/ready.json',{'status':'CALIBRATION_OBSERVED_ZERO_WRITES',
            'optimizer_committed':0,'parent_completed_outer':110,'observed_outer':111,'new_completed_outer':0,
            'code_commit':'a632ba91840fb116aacfc0e770681a860647b295','job_id':'3268','workers':workers})
        write_json(run/'run-result.json',{'status':terminal,'cleanup_residual':{}})
        journal=root/'instruct300/CPU_ANALYSIS_ONCE-3268-28e665eb-v2.json'
        if existing:journal.write_text('{"status":"SUBMISSION_UNKNOWN_DO_NOT_RETRY"}')
        if bad_hash:(tools/'raw_stats_cpu_v2.py').write_bytes(b'changed source')
        if bad_syntax:(tools/'raw_stats_cpu_v2.py').write_bytes(b'print(1))')
        tree=ast.parse(source)
        replacements={'ROOT':ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(root))],keywords=[]),
            'REVIEW_SHA':ast.Constant(value=review_sha),'PACKAGE_SHA':ast.Constant(value=package_sha),'READY_SHA':ast.Constant(value=ready_sha)}
        if bad_syntax:
            pins=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='INDEPENDENT_PROBES'))
            pins['raw_stats_cpu_v2.py']=sha(tools/'raw_stats_cpu_v2.py')
            replacements['INDEPENDENT_PROBES']=ast.parse(repr(pins),mode='eval').body
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
                node.value=replacements[node.targets[0].id]
        requests=[]
        def submit(args,**kwargs):
            requests.append({'args':args,**kwargs})
            if exception:raise exception
            return subprocess.CompletedProcess(args,returncode,stdout,'fixture stderr' if returncode else '')
        captured=io.StringIO();error=None
        with patch.object(socket,'gethostname',return_value=hostname),patch.object(subprocess,'check_output',return_value=active),patch.object(subprocess,'run',side_effect=submit),redirect_stdout(captured):
            try:exec(compile(ast.fix_missing_locations(tree),str(CANDIDATE),'exec'),{'__name__':'dispatch_cpu_fixture'})
            except BaseException as exc:error=type(exc).__name__
        intent=json.loads(journal.read_bytes()) if journal.exists() else None
        result={'name':name,'error':error,'sbatch_calls':len(requests),'intent':intent}
        if requests:
            request=requests[0]
            assert '--job-name=tree-context-analysis-3268-v2' in request['args']
            for flag in ['--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:45:00','--no-requeue','--nodelist=h100-9']:
                assert flag in request['args'],flag
            assert request['capture_output'] is True and request['text'] is True and request['timeout']==60
            body=request['input'];assert 'set -euo pipefail' in body and 'set -o noclobber' in body
            assert 'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4' in body
            assert body.index('/raw_stats_cpu_v2.py') < body.rindex('/raw_stats_cpu_v2.py') < body.rindex('/tree_trajectory_cpu.py') < body.rindex('/analyze_tree_sampling_calibration.py')
            assert '\nexec ' in body and body.count('\nexec ')==1
            assert '--expected-ready-sha256 '+ready_sha in body
            assert all(str(run) in arg for arg in request['args'] if arg.startswith(('--output=','--error=')))
            result['resource_and_serial_args_verified']=True
        results.append(result)
        return result

r=run_case('success');assert r['error'] is None and r['sbatch_calls']==1 and r['intent']['status']=='SUBMITTED' and r['intent']['job_id']==3271
checks.append('single_success_with_gpu0_cpu4_mem64G_45m_serial_processes_and_bound_stdio')
r=run_case('timeout_unknown',exception=subprocess.TimeoutExpired('sbatch',60));assert r['error']=='TimeoutExpired' and r['intent']['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY' and r['sbatch_calls']==1
checks.append('unknown_timeout_recorded_no_automatic_second_submit')
r=run_case('malformed_success_stdout',stdout='unexpected response');assert r['error']=='AssertionError' and r['intent']['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
checks.append('uncertain_success_parser_recorded_unknown_not_failed_or_retried')
r=run_case('nonzero_sbatch',returncode=1);assert r['error'] is None and r['intent']['status']=='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY' and r['sbatch_calls']==1
checks.append('nonzero_submission_preserves_stderr_failed_no_retry')
r=run_case('existing_unknown_journal',existing=True);assert r['error']=='FileExistsError' and r['sbatch_calls']==0 and r['intent']['status']=='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
checks.append('existing_exclusive_journal_refuses_duplicate_before_sbatch')
r=run_case('active_duplicate_name',active='3271 tree-context-analysis-3268 RUNNING\n');assert r['error']=='AssertionError' and r['sbatch_calls']==0 and r['intent'] is None
checks.append('active_named_analysis_refuses_duplicate')
r=run_case('over_memory_token_bound',rows=6000);assert r['error']=='AssertionError' and r['sbatch_calls']==0 and r['intent'] is None
checks.append('over20M_potential_response_keys_refused_before_journal_and_submit')
r=run_case('not_completed_contract',terminal='FAILED');assert r['error']=='AssertionError' and r['sbatch_calls']==0
checks.append('failed_parent_terminal_contract_refused')
r=run_case('wrong_host',hostname='other-node');assert r['error']=='AssertionError' and r['sbatch_calls']==0
checks.append('other_node_refused_before_any_submission')
r=run_case('changed_probe_source',bad_hash=True);assert r['error']=='AssertionError' and r['sbatch_calls']==0
checks.append('changed_probe_sha_refused_before_any_submission')

r=run_case('old3270_wrong_named_job_active',active='3270 tree_trajectory_cpu.py RUNNING\n');assert r['error']=='AssertionError' and r['sbatch_calls']==0
checks.append('old3270_jobid_guard_refuses_original_shadowed_job_name')
r=run_case('syntax_invalid_with_matching_sha',bad_syntax=True);assert r['error']=='SyntaxError' and r['sbatch_calls']==0 and r['intent'] is None
checks.append('matching_sha_syntax_invalid_payload_refused_before_journal_and_submit')

print(json.dumps({'status':'CPU_DISPATCH_V2_FIXED_SOURCE_MOCK_PRECHECK_PASS','source_sha256':EXPECTED,'checks':checks,'check_count':len(checks),
    'results':results,'actual_sbatch_calls':0,'remote_calls':False,'model_assets_loaded':False,
    'scope':'Local temporary metadata fixtures and mocked subprocesses only; no actual job or analysis outcome acceptance'},sort_keys=True,allow_nan=False))
