"""Tiny mocked commands only; no scheduler command is ever executed."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HELP = BASE/'work/tree-fresh200-dispatch-v5'
spec=importlib.util.spec_from_file_location('independent_dispatch_fixture',HELP/'test_fresh200_dispatch_cpu.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
case=T.Tests();case.setUp()
try:
    first,error,first_calls=case.mocked_submit(stdout='777;other_cluster')
    assert error is None and first['status']=='SUBMITTED' and first['job_id']==777
    first_journal=case.journal()
    calls=[]
    def synthetic_run(command,**kwargs):
        calls.append(dict(command=command,environment_keys_set=sorted(kwargs.get('env',{}))))
        if command[0]=='squeue':return SimpleNamespace(returncode=0,stdout='',stderr='')
        if command[0]=='sacct':return SimpleNamespace(returncode=0,stdout='777|COMPLETED|0:0\n',stderr='')
        if command[0]=='sbatch':return SimpleNamespace(returncode=0,stdout='888',stderr='')
        raise AssertionError('Unplanned synthetic command')
    with patch.object(T.C,'protected_processes',lambda:None):
        second=T.S.submit_once(T.C,case.root/'instruct300/candidates/new','c'*64,'b'*64,{},run=synthetic_run)
    assert second['status']=='SUBMITTED' and second['job_id']==888
    result=dict(status='CLUSTER_SUFFIX_UNBOUND_TERMINAL_COUNTEREXAMPLE_REPRODUCED',
        source_sha256=hashlib.sha256((HELP/'submit_tree_fresh200_once.py').read_bytes()).hexdigest(),
        first_submitted_stdout='777;other_cluster',first_journal=first_journal,
        first_cluster_structured_field_present='cluster' in first_journal,
        next_manifest_calls=calls,second_intent=second,
        interpretation='The accepted cluster suffix is retained only in raw stdout. The next sacct and squeue requests carry no cluster; a same numeric JobID terminal response is accepted and a different manifest can submit.',
        scope='Synthetic default-query/other-cluster identity ambiguity only; no actual cluster or Slurm result was inspected.',
        actual_slurm_commands_executed=0,actual_remote_executed=False,actual_gpu_executed=False,
        cpu_preparation_affected=False,actual_zero_ready_created=False,production_training_release=False)
    (OUT/'CLUSTER_SUFFIX_COUNTEREXAMPLE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
finally:case.tearDown()
