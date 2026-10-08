"""Exact V10 rerun of independently reviewed V9 cases; every scheduler function is mocked."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SUBMIT = ROOT / 'work/tree-fresh200-dispatch-v10/submit_tree_fresh200_once.py'
COMMON = ROOT / 'work/tree-fresh200-dispatch-v10/fresh200_dispatch_common.py'
LAUNCH = ROOT / 'work/tree-fresh200-mu-runtime-v10/launch.sbatch'

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
s = module(SUBMIT, 'independent_submit_v10')
c_source = module(COMMON, 'independent_common_v10')
launch = LAUNCH.read_bytes()


def run_case(label, stdout=b'7001;wuhan-h100\n', stderr=b'', retval=0, timeout=False,
             cluster='wuhan-h100', federation='', final_fail=False, prior=None, mutate=True):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root/'instruct300').mkdir(); (root/'logs').mkdir()
        target = root/'instruct300/candidates/synthetic'; target.mkdir(parents=True)
        path = target/'launch.sbatch'; path.write_bytes(launch)
        c = NS(ROOT=root, STAGE=c_source.STAGE, COMMIT=c_source.COMMIT,
            WRAPPER_HASHES={'launch.sbatch':hashlib.sha256(launch).hexdigest()},
            require=c_source.require, queue_clear=c_source.queue_clear, protected_processes=lambda: None)
        calls=[]; submissions=[]; prior_checked=[]
        if prior:
            old = dict(status=prior, cluster_name='wuhan-h100', job_id=6001)
            (root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-oldmanifest.json').write_text(json.dumps(old))
        def mock(argv, **kw):
            calls.append(argv)
            for key in ('SLURM_CLUSTERS','SLURM_CONF','SLURM_CONF_SERVER','SBATCH_ACCOUNT','SQUEUE_CLUSTERS','SACCT_CLUSTERS','SCONTROL_CLUSTERS'):
                assert key not in kw['env'], key
            assert not any(x == '-M' or x.startswith('--clusters') for x in argv)
            if argv == ['scontrol','--local','show','config']:
                return NS(returncode=0,stdout='ClusterName = '+cluster+'\n',stderr='')
            if argv == ['scontrol','--local','show','federation']:
                return NS(returncode=0,stdout=federation,stderr='')
            if argv[0] == 'squeue':
                assert '--local' in argv
                if mutate: path.write_bytes(b'unreviewed new script after read')
                return NS(returncode=0,stdout='',stderr='')
            if argv[0] == 'sacct':
                assert '--local' in argv; prior_checked.append(argv)
                return NS(returncode=0,stdout='6001|wuhan-h100|COMPLETED|0:0\n',stderr='')
            if argv[0] == 'sbatch':
                submissions.append(kw['input'])
                assert kw['text'] is False and kw['input'] == launch and str(path) not in argv
                jpath = root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-aaaaaaaaaaaa.json'
                j=json.loads(jpath.read_bytes())
                assert j['status']=='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN'
                assert base64.b64decode(j['actual_submit_stdin_base64'])==launch
                assert j['actual_submit_stdin_sha256']==hashlib.sha256(launch).hexdigest()
                assert j['actual_submit_stdin_bytes']==len(launch) and j['actual_submit_argv']==argv
                if timeout: raise subprocess.TimeoutExpired(argv,60,output=b'7001\xff',stderr=b'partial\x80')
                return NS(returncode=retval,stdout=stdout,stderr=stderr)
            raise AssertionError(argv)
        bad_env = dict(SLURM_CLUSTERS='foreign',SLURM_CONF='/foreign',SLURM_CONF_SERVER='foreign',SBATCH_ACCOUNT='foreign',
            SQUEUE_CLUSTERS='foreign',SACCT_CLUSTERS='foreign',SCONTROL_CLUSTERS='foreign')
        accepted=False;exception=None
        try:
            with patch.dict(os.environ,bad_env),contextlib.redirect_stdout(io.StringIO()):
                if final_fail:
                    with patch.object(s,'finish_journal',side_effect=OSError('independent simulated final persistence failure')):
                        s.submit_once(c,target,'a'*64,'b'*64,{},mock)
                else:s.submit_once(c,target,'a'*64,'b'*64,{},mock)
            accepted=True
        except Exception as exc:exception={'type':type(exc).__name__,'message':str(exc)}
        jpath=root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-aaaaaaaaaaaa.json'
        journal=json.loads(jpath.read_bytes()) if jpath.exists() else None
        retry_blocked=None
        if journal is not None and journal['status']!='SUBMITTED':
            path.write_bytes(launch)
            count=len(submissions)
            try:
                with patch.dict(os.environ,bad_env),contextlib.redirect_stdout(io.StringIO()):
                    s.submit_once(c,target,'c'*64,'b'*64,{},mock)
                retry_blocked=False
            except Exception:retry_blocked=len(submissions)==count
        return dict(label=label,accepted=accepted,exception=exception,
            journal=journal,mocked_scheduler_argv=calls,
            submit_count=len(submissions),all_submitted_bytes_match_reviewed=all(v==launch for v in submissions),
            different_manifest_retry_blocked=retry_blocked,prior_terminal_checks=len(prior_checked))

cases=[run_case('same_fd_path_replaced'),run_case('invalid_utf8_stdout',stdout=b'\xff7001'),
       run_case('foreign_returned_cluster',stdout=b'7001;foreign\n'),run_case('nonzero_return_with_job_output',retval=1),
       run_case('timeout_raw_bytes_retained',timeout=True),run_case('final_persistence_failure_leaves_submitting',final_fail=True),
       run_case('foreign_local_configuration_refused',cluster='foreign'),run_case('federation_refused',federation='foreign federation'),
       run_case('previous_unknown_blocks_other_manifest',prior='SUBMISSION_UNKNOWN_DO_NOT_RETRY'),
       run_case('previous_local_terminal_allows_new_intent',prior='SUBMITTED')]
assert cases[0]['accepted'] and cases[0]['submit_count']==1
for case in cases[1:6]:
    assert not case['accepted'] and case['submit_count']==1 and case['different_manifest_retry_blocked'] is True
for case in cases[6:9]:assert not case['accepted'] and case['submit_count']==0
assert cases[-1]['accepted'] and cases[-1]['prior_terminal_checks']==1
assert base64.b64decode(cases[1]['journal']['stdout_raw']['data'])==b'\xff7001'
assert base64.b64decode(cases[4]['journal']['exception_stdout']['data'])==b'7001\xff'
assert base64.b64decode(cases[4]['journal']['exception_stderr']['data'])==b'partial\x80'
assert cases[5]['journal']['status']=='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN'
result=dict(status='INDEPENDENT_V10_CPU_SUBMIT_FUNCTION_PROBES_PASSED',actual_remote_executed=False,
    actual_Slurm_executed=False,actual_gpu_accepted=False,official_assets_loaded=False,
    source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SUBMIT,COMMON,LAUNCH)},cases=cases)
print(json.dumps(result,indent=2,sort_keys=True))
