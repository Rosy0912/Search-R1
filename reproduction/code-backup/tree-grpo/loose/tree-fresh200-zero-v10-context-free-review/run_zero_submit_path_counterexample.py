"""Actual source functions plus tiny local file races; no Slurm client executes."""
import contextlib,hashlib,importlib.util,io,json,os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;HELPERS=WORK/'tree-fresh200-dispatch-v8'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name):
    spec=importlib.util.spec_from_file_location('zero_path_review_'+name.split('.')[0],HELPERS/name);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
frozen=(WORK/'tree-fresh200-mu-runtime-v10/launch.sbatch').read_bytes();expected=hashlib.sha256(frozen).hexdigest()
assert expected=='8a4ae2bbfaa744a4f86dbaa14cbbb6453283ac20d812f0a21cb5ed171843eca3'
changed=b'#!/usr/bin/env bash\n# benign CPU fixture: alternate bytes, never executed\nexit 0\n'
rows=[]
for race in ('overwrite_same_inode','replace_inode'):
    h=module('fresh200_dispatch_common.py');s=module('submit_tree_fresh200_once.py')
    with TemporaryDirectory() as tmp:
        root=Path(tmp);target=root/'instruct300/candidates/cpu-code-only';target.mkdir(parents=True);(root/'logs').mkdir()
        launch=target/'launch.sbatch';launch.write_bytes(frozen);old_inode=launch.stat().st_ino
        h.ROOT=root;h.protected_processes=lambda:None
        manifest={'files':{'launch.sbatch':expected},'external_files':{}}
        h.verify_package(target,manifest) # actual held SHA/regular/fstat algorithm
        checked=dict(sha256=sha(launch),inode=old_inode)
        calls=[];captured=[]
        def runner(argv,**kw):
            calls.append(dict(argv=argv,kwargs=dict(kw,env=dict(kw['env']))))
            if argv==['scontrol','--local','show','config']:return SimpleNamespace(returncode=0,stdout='ClusterName = wuhan-h100\n',stderr='')
            if argv==['scontrol','--local','show','federation']:return SimpleNamespace(returncode=0,stdout='',stderr='')
            if argv[0]=='squeue':
                if race=='overwrite_same_inode':launch.write_bytes(changed)
                else:
                    replacement=target/'replacement-cpu-fixture.sbatch';replacement.write_bytes(changed);replacement.replace(launch)
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            if argv[0]=='sbatch':
                assert argv[-1]==str(launch) and 'input' not in kw
                captured.append(dict(path=argv[-1],sha256=sha(launch),inode=launch.stat().st_ino,bytes_hex=launch.read_bytes().hex(),stdin_supplied=False))
                return SimpleNamespace(returncode=0,stdout='777;wuhan-h100\n',stderr='')
            raise AssertionError('Unexpected scheduler fixture command')
        with patch.dict(os.environ,{'PATH':'/cpu-fixture/bin','SLURM_CONF':'foreign','SBATCH_CLUSTERS':'foreign','KEEP_CPU':'1'},clear=True),contextlib.redirect_stdout(io.StringIO()):
            intent=s._submit_once_locked(h,target,'a'*64,'b'*64,{},runner)
        assert len(captured)==1 and captured[0]['sha256']!=expected and intent['status']=='SUBMITTED'
        if race=='replace_inode':assert captured[0]['inode']!=old_inode
        else:assert captured[0]['inode']==old_inode
        rows.append(dict(race=race,source_submit_sha256=sha(HELPERS/'submit_tree_fresh200_once.py'),source_common_sha256=sha(HELPERS/'fresh200_dispatch_common.py'),actual_source_functions=['fresh200_dispatch_common.verify_package','fresh200_dispatch_common.file_sha','submit_tree_fresh200_once._submit_once_locked'],checked_launch=checked,mock_scheduler_materialized_launch=captured[0],calls=calls,returned_status=intent['status'],returned_job_id_is_mock=True,real_scheduler_commands_executed=0,real_script_executions=0,official_assets_loaded=False,synthetic_minimal_manifest_isolation=True))
proof=dict(status='ZERO_HELPER_V8_P05_MUTABLE_LAUNCH_PATH_COUNTEREXAMPLE_OBSERVED_CPU_ONLY',cases=rows,source_lines={'verify_package':147,'mutable_sbatch_path':181},scope='Actual package-file SHA and submit source functions; synthetic minimal launch-only manifest and mock scheduler. Complete final READY/main/official asset validation was not executed.',actual_slurm_executed=False,actual_gpu_executed=False,actual_remote_executed=False,actual_training_release=False,actual_foreign_process_touched=False)
(HERE/'ZERO_SUBMIT_PATH_COUNTEREXAMPLE.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=proof['status'],cases=len(rows),proof_sha256=sha(HERE/'ZERO_SUBMIT_PATH_COUNTEREXAMPLE.json'))))
