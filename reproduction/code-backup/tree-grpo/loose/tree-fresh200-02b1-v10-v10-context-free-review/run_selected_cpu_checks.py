"""Bounded local CPU reruns; only direct Python synthetic tests are invoked."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'work/tree-behavior-mu-utf8-v2'
checks=[('SOURCE_SELECTED_8',SOURCE,[sys.executable,'-B','-m','unittest','test_tree_raw_semantics_cpu',
    'test_tree_behavior_mu_actor_cpu.Tests.test_resealed_wrong_legacy_semantics_refuse_mu_body_and_actor_before_writes','-v']),
    ('HELPER_INDEPENDENT_10',ROOT,[sys.executable,'-B',str(OUT/'independent_submit_v10_probe.py')])]
records=[]
for name,cwd,argv in checks:
    start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    completed=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,timeout=45)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    (OUT/(name+'.stdout')).write_bytes(completed.stdout);(OUT/(name+'.stderr')).write_bytes(completed.stderr)
    if name=='HELPER_INDEPENDENT_10' and completed.returncode==0:
        (OUT/'INDEPENDENT_SUBMIT_V10_PROBE_RESULT.json').write_bytes(completed.stdout)
    record=dict(name=name,argv=argv,cwd=str(cwd),exit_code=completed.returncode,wall_seconds=time.perf_counter()-start,
        children_process_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(),stderr_sha256=hashlib.sha256(completed.stderr).hexdigest(),
        actual_remote_executed=False,actual_slurm_executed=False,all_scheduler_calls_mocked=True,
        actual_gpu_executed=False,official_assets_loaded=False)
    records.append(record);print(json.dumps(record))
    if completed.returncode:raise RuntimeError(completed.stderr.decode(errors='replace'))
(OUT/'ACTUAL_SELECTED_CPU_RUNS.json').write_text(json.dumps(records,indent=2)+'\n')
