"""Close this observed failed job's private tmpfs on its original node only.

Slurm must already have terminated the job. Preserve original logs/claims:
cleanup uses a shadow ledger and checks all recorded readers are quiescent.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from reproduction.common8.native_allocation import fields
from reproduction.common8.stage_retrieval import assert_quiescent
from reproduction.hpc3_qprobe.control import publish

FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717765-20261014')
SHADOW=FAILED/'failure-cleanup717827'


def main():
    assert os.getuid()==14090 and socket.gethostname().split('.')[0]=='ACD1-67'
    assert os.environ.get('SLURM_JOB_ID') not in (None,'717827')
    proof=json.loads((FAILED/'failed-job-717827/terminal-and-files.json').read_text())
    old=fields(proof['terminal']['stdout'])
    assert old['JobId']=='717827' and old['JobState']=='FAILED' and old['UserId']=='jxu957(14090)' and old['NodeList']=='ACD1-67'
    actual=subprocess.check_output(['sacct','-nP','-j','717827','-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True).strip()
    assert actual=='717827|FAILED|1:0|14090|ACD1-67'
    assert SHADOW.is_dir() and not (SHADOW/'complete.json').exists()
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    assert run['completed_updates']==0 and run['training_world_size']==4 and run['source_commit']=='a75e0c944e2bbae284e52ce9b9f8bf909ab89564'
    from reproduction.common8.temporary_cache import cleanup as clean_corpus
    from reproduction.common8.stage_retrieval import cleanup as clean_stage
    from reproduction.a800.owned_processes import OwnedProcesses
    for name in ('corpus-cache-claim.json','retrieval-staging-claim.json'):
        assert_quiescent(SHADOW,json.loads((SHADOW/'logs'/name).read_text()))
    owned=OwnedProcesses(SHADOW/'logs/process-ledger.json',run['run_id'])
    # No live process is eligible after terminal Slurm teardown. Fail rather
    # than signal any recovered PID when the kernel has no pidfd support.
    assert all(not owned.observe(n) for n in owned.jobs)
    owned.stop_all()
    corpus=clean_corpus(SHADOW);stage=clean_stage(SHADOW)
    value=dict(cleaned=True,run_id=run['run_id'],failed_job='717827',cleanup_job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),actual_terminal=actual,corpus=corpus,stage=stage,original_logs_unchanged=True,checked_unix=time.time())
    publish(SHADOW/'complete.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
