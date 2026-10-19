import copy
import unittest
from reproduction.scaleup_native_gpu.contract import JOB,node_coordinates,validate_gate,validate_tasks


class Contract(unittest.TestCase):
    def test_actual_ssp_two_node_coordinates(self):
        for r in ['0','1']:
            e=dict(RANK=r,WORLD_SIZE='2',MASTER_ADDR='master.local',MASTER_PORT='23456')
            self.assertEqual(node_coordinates(e),(int(r),'master.local',23456))

    def test_bad_node_or_aliases(self):
        e=dict(RANK='0',WORLD_SIZE='2',MASTER_ADDR='master.local',MASTER_PORT='23456')
        for k,v in [('RANK','2'),('WORLD_SIZE','16'),('NODE_RANK','1'),('NODE_COUNT','1'),('MASTER_PORT','0'),('MASTER_ADDR','bad;command')]:
            x=dict(e);x[k]=v
            with self.assertRaises(AssertionError):node_coordinates(x)

    def test_registered_gate(self):
        s=dict(job=JOB,uid='one',plan_sha256='a')
        g=dict(**s,state='RUNNING',nodes=2,total_gpus=16)
        validate_gate(g,s,'a')
        for k,v in [('uid','other'),('plan_sha256','b'),('nodes',1),('state','PENDING')]:
            x=dict(g);x[k]=v
            with self.assertRaises(AssertionError):validate_gate(x,s,'a')

    def test_task_specs(self):
        resource=dict(accelerate_device_count=8,cpu_count=16,memory_gib=64,machine_types=['n2ls.ie.i60b'],rdma_name='ib-cluster-01')
        tasks=[dict(role=role,replicas=1,resource_spec=dict(resource)) for role in ['PYTORCH_MASTER','PYTORCH_WORKER']]
        validate_tasks(tasks)
        for key,value in [('accelerate_device_count',4),('memory_gib',900),('machine_types',['other'])]:
            t=copy.deepcopy(tasks);t[1]['resource_spec'][key]=value
            with self.assertRaises(AssertionError):validate_tasks(t)


if __name__=='__main__':unittest.main()
