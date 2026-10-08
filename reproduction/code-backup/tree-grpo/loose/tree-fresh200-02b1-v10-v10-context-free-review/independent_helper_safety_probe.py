"""Only synthetic proc metadata and scheduler functions; never system queries."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
c=load(ROOT/'work/tree-fresh200-dispatch-v10/fresh200_dispatch_common.py','independent_common')
s=load(ROOT/'work/tree-fresh200-dispatch-v10/submit_tree_fresh200_once.py','independent_submit')
contract=load(ROOT/'work/tree-fresh200-mu-runtime-v10/calibration_contract.py','independent_zero_contract')
started=time.process_time();cases=[]
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);proc=root/'synthetic-proc';proc.mkdir()
    for pid,start in c.PROTECTED.items():
        directory=proc/pid;directory.mkdir();fields=['S']+['0']*18+[start]+['0']*3
        (directory/'stat').write_text(pid+' (synthetic process name) '+' '.join(fields))
    c.protected_processes(proc)
    changed=proc/next(iter(c.PROTECTED))/'stat';original=changed.read_text();changed.write_text(original.replace('1502352181','1502352182'))
    try:c.protected_processes(proc)
    except ValueError:cases.append(dict(name='real_protected_pid_start_rejects_synthetic_drift',passed=True))
    else:raise AssertionError('protected PID drift accepted')
    changed.write_text(original)
    (root/'instruct300').mkdir();target=root/'instruct300/candidates/test';target.mkdir(parents=True);(root/'logs').mkdir()
    launch=(ROOT/'work/tree-fresh200-mu-runtime-v10/launch.sbatch').read_bytes();(target/'launch.sbatch').write_bytes(launch)
    mockc=NS(ROOT=root,COMMIT=c.COMMIT,STAGE=c.STAGE,require=c.require,WRAPPER_HASHES=c.WRAPPER_HASHES,
        protected_processes=lambda:c.protected_processes(proc),queue_clear=c.queue_clear)
    fsync_events=[];real_fsync=s.os.fsync
    def sync(fd):fsync_events.append('fsync');return real_fsync(fd)
    def run(argv,**kwargs):
        if argv==['scontrol','--local','show','config']:return NS(returncode=0,stdout='ClusterName = wuhan-h100\n',stderr='')
        if argv==['scontrol','--local','show','federation']:return NS(returncode=0,stdout='',stderr='')
        if argv[0]=='squeue':return NS(returncode=0,stdout='',stderr='')
        assert argv[0]=='sbatch';assert len(fsync_events)>=2
        journal=root/'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-aaaaaaaaaaaa.json'
        intent=json.loads(journal.read_text());assert intent['status']=='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN'
        assert kwargs['input']==launch and kwargs['text'] is False
        fsync_events.append('mock_sbatch');return NS(returncode=0,stdout=b'7001;wuhan-h100\n',stderr=b'')
    with patch.object(s.os,'fsync',sync),contextlib.redirect_stdout(io.StringIO()):
        result=s.submit_once(mockc,target,'a'*64,'b'*64,{},run)
    assert result['code_commit']==c.COMMIT and fsync_events[:3]==['fsync','fsync','mock_sbatch']
    cases.append(dict(name='real_intent_file_and_directory_fsync_precede_mock_submit',passed=True,events=fsync_events))
    with s.task_submit_lock(mockc):
        try:
            with s.task_submit_lock(mockc):raise AssertionError('duplicate lock accepted')
        except ValueError:cases.append(dict(name='real_nonblocking_whole_task_lock_rejects_duplicate',passed=True))
    calls=[]
    def forbidden(argv,**kwargs):calls.append(argv);raise AssertionError('query before cgroup validation')
    for cgroup in ('0::/slurm/job_888/step_batch\n','0::/slurm/job_7770/step_batch\n','0::/slurm/job_777/job_888/step_batch\n',''):
        try:contract.verify_zero_runtime(environment={'SLURM_JOB_ID':'777','SLURM_JOB_NODELIST':'h100-9'},hostname='h100-9',runner=forbidden,cgroup_text=cgroup)
        except ValueError:pass
        else:raise AssertionError('wrong current cgroup accepted')
    assert not calls;cases.append(dict(name='real_zero_runtime_refuses_sibling_before_mock_queries',passed=True,queries=calls))
report=dict(status='INDEPENDENT_HELPER_EXTRA_SAFETY_PROBES_PASSED',source_commit=c.COMMIT,cases=cases,
    process_cpu_seconds=time.process_time()-started,actual_gpu_executed=False,actual_remote_executed=False,
    actual_slurm_executed=False,all_scheduler_calls_mocked=True,actual_assets_loaded=False)
(OUT/'INDEPENDENT_HELPER_SAFETY_PROBE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASSED',cases=len(cases),process_cpu_seconds=report['process_cpu_seconds'])))
