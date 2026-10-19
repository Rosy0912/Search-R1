import copy
import unittest
from reproduction.scaleup_probe.contract import JOB, ROOT, RUNTIME, node_coordinates, validate, validate_gate, validate_params, validate_actual_tasks


class ContractTests(unittest.TestCase):
    def plan(self):
        return dict(job=JOB, root=ROOT, runtime=RUNTIME, nodes=2, gpus_per_node=8,
                    purpose='INFRASTRUCTURE_ONLY_NOT_SCIENTIFIC_REPEAT', timeout_seconds=900,
                    no_auto_retry=True, existing_trainings_unchanged=True,
                    production_checkpoint_read=False, production_speedup_claim=False,
                    files={n: 'a' * 64 for n in ('__init__.py', 'contract.py', 'node.py', 'worker.py', 'gate.py')})

    def test_plan(self):
        validate(self.plan())

    def test_production_or_topology_rejected(self):
        for k, v in [('nodes', 1), ('gpus_per_node', 16), ('production_checkpoint_read', True),
                     ('production_speedup_claim', True), ('root', '/tmp/other')]:
            p = self.plan(); p[k] = v
            with self.assertRaises(AssertionError): validate(p)

    def test_node_env(self):
        e = dict(NODE_RANK='1', NODE_COUNT='2', WORLD_SIZE='2', RANK='1',
                 MASTER_ADDR='master-0.cluster.local', MASTER_PORT='6000')
        self.assertEqual(node_coordinates(e), (1, 'master-0.cluster.local', 6000))
        for k,v in [('NODE_COUNT','1'), ('RANK','0'), ('NODE_RANK','2'),
                    ('MASTER_ADDR','x;bad'), ('MASTER_PORT','0')]:
            x=dict(e);x[k]=v
            with self.assertRaises(AssertionError):node_coordinates(x)

    def test_actual_ssp_environment_without_node_aliases(self):
        for rank in ('0','1'):
            env=dict(RANK=rank,WORLD_SIZE='2',MASTER_ADDR='master.local',MASTER_PORT='23456')
            self.assertEqual(node_coordinates(env),(int(rank),'master.local',23456))
        with self.assertRaises(KeyError):node_coordinates(dict(WORLD_SIZE='2'))

    def test_uid_and_plan_gate(self):
        s = dict(job=JOB, uid='one', plan_sha256='a')
        g = dict(**s, state='RUNNING', nodes=2, total_gpus=16)
        validate_gate(g,s,'a')
        for k,v in [('uid','other'),('state','PENDING'),('nodes',1),('plan_sha256','b')]:
            x=dict(g);x[k]=v
            with self.assertRaises(AssertionError):validate_gate(x,s,'a')

    def test_params(self):
        p = dict(job_name=JOB, num_nodes=2,tasks_per_node=1,gpus_per_task=8,
                 cpus_per_task=16,memory_per_task=64,shm_size_gib=16,priority=1,
                 retry=0,no_env=True,enable_privileged=False,cluster='cluster-p',
                 queue='queue-p-reserved-llmarchitecture',machine_type='n2ls.ie.i60b',
                 rdma_name='ib-cluster-01')
        validate_params(p)
        for k,v in [('include','any'),('retry',1),('gpus_per_task',4),('num_nodes',1)]:
            x=dict(p);x[k]=v
            with self.assertRaises(AssertionError):validate_params(x)

    def test_actual_two_task_allocation(self):
        r = dict(accelerate_device_count=8,cpu_count=16,memory_gib=64,
                 rdma_name='ib-cluster-01',machine_types=['n2ls.ie.i60b'])
        ts=[dict(role=k,replicas=1,resource_spec=dict(r)) for k in ('PYTORCH_MASTER','PYTORCH_WORKER')]
        validate_actual_tasks(ts)
        for k,v in [('accelerate_device_count',4),('memory_gib',600),('machine_types',['other'])]:
            x=copy.deepcopy(ts);x[1]['resource_spec'][k]=v
            with self.assertRaises(AssertionError):validate_actual_tasks(x)


if __name__ == '__main__': unittest.main()
