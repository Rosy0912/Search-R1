"""Four changed-entry CPU probes. No scheduler/GPU/assets; subprocesses Python only."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import slurm_identity as identity
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'tree-behavior-mu-utf8-v2'

class V5EntryTests(unittest.TestCase):
    def test_required_sitecustomize_import_or_install_failure_is_actual_fatal_before_payload(self):
        for mode in ('import','install'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'search_r1/baselines').mkdir(parents=True)
                (root/'search_r1/__init__.py').write_text('')
                (root/'search_r1/baselines/__init__.py').write_text('')
                code="raise RuntimeError('synthetic required import failure')\n" if mode=='import' else "import sys,types\ndef gpu_snapshot():pass\nclass Locked(types.ModuleType):\n def __setattr__(self,n,v):\n  if n=='gpu_snapshot':raise RuntimeError('synthetic required install failure')\n  super().__setattr__(n,v)\nsys.modules[__name__].__class__=Locked\n"
                (root/'search_r1/baselines/preflight.py').write_text(code)
                payload=root/'PAYLOAD_RAN';environment=dict(os.environ,PYTHONPATH=str(HERE)+':'+str(root),TREE_INSTRUCT_TOOLS='required-fixture-only',TREE_FIXED_ACTOR_PROFILE='0',CUDA_VISIBLE_DEVICES='',PYTHONOPTIMIZE='0')
                result=subprocess.run([sys.executable,'-B','-c',f"from pathlib import Path;Path({str(payload)!r}).write_text('unsafe payload')"],env=environment,cwd=root,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertFalse(payload.exists());self.assertIn('Required Tree own-cgroup GPU-query guard bootstrap failed',result.stderr)
    def test_explicit_cli_wrong_missing_sibling_current_cgroup_has_zero_gpu_or_scheduler_queries(self):
        for text in ('','0::/user.slice\n','0::/slurm/job_98/step_batch\n'):
            payload="""import json,os,socket
from pathlib import Path
import slurm_identity as identity
from search_r1.baselines import preflight
calls=[]
socket.gethostname=lambda:'h100-9'
Path.read_text=lambda self,*a,**kw:CGROUP
preflight.subprocess.check_output=lambda *a,**kw:calls.append(['GPU',str(a)])
identity.subprocess.run=lambda *a,**kw:calls.append(['scheduler',str(a)])
try:identity.guarded_preflight_main()
except ValueError as error:print(json.dumps(dict(refused=str(error),query_calls=calls)))
else:raise AssertionError('wrong actual cgroup accepted')
""".replace('CGROUP',repr(text))
            environment=dict(os.environ,PYTHONPATH=str(HERE)+':'+str(SOURCE),SLURM_JOB_ID='99',SLURM_JOB_NODELIST='h100-9',SLURM_CLUSTER_NAME='wuhan-h100',CUDA_VISIBLE_DEVICES='0',PYTHONOPTIMIZE='0')
            environment.pop('TREE_INSTRUCT_TOOLS',None);environment.pop('TREE_FIXED_ACTOR_PROFILE',None)
            result=subprocess.run([sys.executable,'-B','-c',payload,'--','childpython','owned.py'],env=environment,cwd=SOURCE,capture_output=True,text=True)
            with self.subTest(text=text):
                self.assertEqual(result.returncode,0,result.stderr);proof=json.loads(result.stdout)
                self.assertEqual(proof['query_calls'],[]);self.assertTrue(proof['refused'])
    def test_legal_own_canonical_main_preserves_cli_and_original_child_exec(self):
        from search_r1.baselines import preflight
        env={'SLURM_JOB_ID':'99','SLURM_JOB_NODELIST':'h100-9','SLURM_CLUSTER_NAME':'wuhan-h100','CUDA_VISIBLE_DEVICES':'0'}
        for flags in ([],['--reuse-inherited']):
            expected=['-c',*flags,'--','childpython','owned.py','--arg=exact'];calls=[]
            with self.subTest(flags=flags),patch.dict(os.environ,env,clear=True),patch.object(identity.socket,'gethostname',return_value='h100-9'),patch.object(identity.Path,'read_text',return_value='0::/slurm/job_99/step_batch\n'),patch.object(sys,'argv',expected.copy()),patch.object(preflight,'main',side_effect=lambda:calls.append(sys.argv.copy())):
                identity.guarded_preflight_main();self.assertEqual(calls,[expected])
        argv=['-c','--reuse-inherited','--','childpython','owned.py','--arg=exact'];execs=[]
        with patch.object(sys,'argv',argv),patch.dict(os.environ,{'SEARCH_R1_GPU_LEASES':'[]'}),patch.object(preflight,'validate_inherited_gpu_leases',return_value=None) as leases,patch.object(preflight.os,'execvp',side_effect=lambda *args:execs.append(args)):
            preflight.main();leases.assert_called_once_with();self.assertEqual(execs,[('childpython',['childpython','owned.py','--arg=exact'])])
    def test_all_actual_formal_preflight_entries_use_canonical_wrapper_no_runpy(self):
        launch=(HERE/'launch.sbatch').read_text();runner=(HERE/'run_tree.sh').read_text()
        self.assertNotIn("'-m','search_r1.baselines.preflight'",launch)
        self.assertNotIn('python -m search_r1.baselines.preflight',runner)
        expression='from slurm_identity import guarded_preflight_main; guarded_preflight_main()'
        self.assertEqual(launch.count(expression),1);self.assertEqual(runner.count(expression),2)
        self.assertIn(expression+"' --reuse-inherited -- python",runner)
        tree=ast.parse((HERE/'slurm_identity.py').read_text());entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='guarded_preflight_main')
        calls=[n for n in ast.walk(entry) if isinstance(n,ast.Call)]
        own=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_cgroup')
        install=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='install_gpu_query_guard')
        main=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='main')
        self.assertLess(own.lineno,install.lineno);self.assertLess(install.lineno,main.lineno)

if __name__=='__main__':unittest.main()
