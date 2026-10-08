import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import calibration_contract as c

ENV={'SLURM_JOB_ID':'777','SLURM_JOB_NODELIST':'h100-9','SLURM_CLUSTER_NAME':'wuhan-h100',
    'SLURM_CONF':'foreign','SLURM_CLUSTERS':'foreign','SBATCH_CLUSTERS':'foreign',
    'SCONTROL_FEDERATION':'1','SCANCEL_NAME':'other','KEEP_OWN_SETTING':'ok'}
ALLOCATION='JobId=777 Cluster=wuhan-h100 JobState=RUNNING NodeList=h100-9 BatchHost=h100-9 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=192 MinMemoryNode=256G TimeLimit=03:00:00 Requeue=0 Restarts=0'
CGROUP='0::/slurm/job_777/step_batch\n'


class ZeroSchedulerTests(unittest.TestCase):
    def fake(self,changes=None):
        outputs=['ClusterName = wuhan-h100\n','',ALLOCATION]
        if changes:
            for index,value in changes.items(): outputs[index]=value
        calls=[]
        def run(argv,**kwargs):
            calls.append((argv,kwargs))
            return SimpleNamespace(returncode=0,stderr='',stdout=outputs[len(calls)-1])
        return run,calls

    def test_runtime_queries_explicit_local_and_sanitized_environment(self):
        run,calls=self.fake()
        value=c.verify_zero_runtime(ENV,'h100-9',run,CGROUP)
        self.assertEqual(value['allocated_cpus'],192)
        self.assertEqual(value['wall_hours'],3)
        self.assertEqual([x[0] for x in calls],[['scontrol','--local','show','config'],['scontrol','--local','show','federation'],['scontrol','--local','show','job','777']])
        for _,kw in calls:
            self.assertEqual(kw['env'],{'SLURM_JOB_ID':'777','SLURM_JOB_NODELIST':'h100-9','KEEP_OWN_SETTING':'ok'})
            self.assertEqual(kw['timeout'],30)

    def test_original_wrong_job_cluster_node_cpu_substring_counterexample_rejected(self):
        bad='JobId=888 Cluster=foreign JobState=RUNNING NodeList=h100-16 BatchHost=h100-16 NumNodes=1 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=320 NumCPUs=320 MinMemoryNode=1G TimeLimit=99:00:00 Requeue=1 Restarts=0'
        with self.assertRaises(ValueError):c.zero_allocation_identity(bad,'777')

    def test_individual_exact_allocation_fields_unknown_and_duplicate_refuse(self):
        for key,new in [('JobId','888'),('Cluster','foreign'),('NodeList','h100-16'),('BatchHost','h100-16'),('NumNodes','2'),('CPUs/Task','320'),('MinMemoryNode','1G'),('TimeLimit','99:00:00'),('Requeue','1'),('Restarts','1'),('OverSubscribe','YES'),('TresPerNode','gres:gpu:4'),('JobState','PENDING')]:
            old=dict(field.split('=',1) for field in ALLOCATION.split())[key]
            with self.subTest(key=key),self.assertRaises(ValueError):c.zero_allocation_identity(ALLOCATION.replace(key+'='+old,key+'='+new),'777')
        for raw in [ALLOCATION+' JobId=777',ALLOCATION.replace('NumCPUs=192',''),ALLOCATION.replace('NumCPUs=192','NumCPUs=nan')]:
            with self.assertRaises(ValueError):c.zero_allocation_identity(raw,'777')

    def test_foreign_runtime_or_host_refuses_before_any_client_call(self):
        for env,host in [(dict(ENV,SLURM_CLUSTER_NAME='foreign'),'h100-9'),(ENV,'h100-16'),(dict(ENV,SLURM_JOB_ID='777;foreign'),'h100-9')]:
            run,calls=self.fake()
            with self.assertRaises(ValueError):c.verify_zero_runtime(env,host,run,CGROUP)
            self.assertEqual(calls,[])

    def test_foreign_default_cluster_federation_or_unknown_query_refuse(self):
        for changes in [{0:'ClusterName = foreign\n'},{0:'ClusterName = wuhan-h100\nClusterName = wuhan-h100\n'},{1:'FederationName = foreign\n'}]:
            run,_=self.fake(changes)
            with self.assertRaises(ValueError):c.verify_zero_runtime(ENV,'h100-9',run,CGROUP)
        for result in [SimpleNamespace(returncode=1,stderr='',stdout=''),SimpleNamespace(returncode=0,stderr='warning',stdout='ClusterName = wuhan-h100\n')]:
            with self.assertRaises(ValueError):c.verify_zero_runtime(ENV,'h100-9',lambda *a,**kw:result,CGROUP)

    def test_current_cgroup_unknown_or_other_job_refuses_before_any_query(self):
        for group in ('0::/slurm/job_888/step_batch\n','0::/slurm/job_7770/step_batch\n',
                      '0::/slurm/job_777/step_batch\n1:cpu:/slurm/job_888/step_batch\n','0::/\n',''):
            run,calls=self.fake()
            with self.assertRaises(ValueError):c.verify_zero_runtime(ENV,'h100-9',run,group)
            self.assertEqual(calls,[])
        value=c.validate_zero_cgroup('777','0::/slurm/job_777/step_batch\n1:cpu:/slurm/job_777/step_batch\n')
        self.assertEqual(value['job_id'],'777')

    def test_bare_original_cancel_is_never_executed_and_refusal_is_durable(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'refusal.json';calls=[]
            guarded=c.controlled_guard_run(lambda *a,**kw:calls.append((a,kw)),p)
            with self.assertRaises(RuntimeError):guarded(['scancel','777'],check=False)
            self.assertEqual(calls,[])
            data=json.loads(p.read_text())
            self.assertFalse(data['scheduler_cancellation_executed'])
            self.assertFalse(data['foreign_processes_touched'])

    def test_original_guard_top_level_argv_and_subprocess_restored(self):
        import subprocess,sys
        old_run,old_argv=subprocess.run,sys.argv
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'observed.json'
            raw=b'import json,sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text(json.dumps(sys.argv))\n'
            c.execute_original_guard(raw,'frozen-original.py',p,Path(tmp)/'refusal.json')
            self.assertEqual(json.loads(p.read_text()),['frozen-original.py','--output',str(p)])
        self.assertIs(subprocess.run,old_run)
        self.assertIs(sys.argv,old_argv)

    def test_actual_original_guard_cancel_call_intercepted(self):
        import subprocess
        original=Path(__file__).resolve().parents[1]/'resource_guard_identity.py'
        self.assertEqual(c.sha(original),'284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88')
        tree=ast.parse(original.read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='run')
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'refusal.json'
            with patch.object(subprocess,'run',c.controlled_guard_run(lambda *a,**kw:self.fail('Actual scancel must not execute'),p)):
                with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[call],type_ignores=[]),str(original),'exec'),{'subprocess':subprocess,'job':'777'})
            self.assertTrue(p.is_file())

    def test_supervisor_uses_verified_allocation_and_only_guard_child(self):
        text=(Path(__file__).parent/'supervisor.py').read_text()
        self.assertIn('scheduler_identity=verify_zero_runtime()',text)
        self.assertNotIn("['scontrol','show','job',job]",text)
        self.assertNotIn("str(ROOT/'tools/resource_guard_identity.py'),'--output'",text)
        self.assertIn("str(TOOLS/'calibration_contract.py'),'--guard-child'",text)


if __name__=='__main__':unittest.main()
