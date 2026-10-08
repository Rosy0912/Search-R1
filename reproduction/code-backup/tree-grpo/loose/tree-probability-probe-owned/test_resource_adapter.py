"""Source-only metadata counterexamples; no /proc, GPU or asset reads."""
import copy
import unittest
import resource_adapter as A


class ResourceTests(unittest.TestCase):
    def good(self):
        return dict(job_id='99999',epoch=100.,gpus=[dict(uuid=str(i),foreign_memory_fraction=.01,foreign_sm_upper=0.) for i in range(8)])

    def test_actual_allocation_not_request_inference(self):
        s='JobId=99999 JobState=RUNNING NodeList=h100-9 OverSubscribe=NO TresPerNode=gres:gpu:8 CPUs/Task=32 NumCPUs=192 '
        self.assertEqual(A.allocation(s,'99999'),192)
        for k,v in [('JobState=RUNNING','JobState=PENDING'),('NodeList=h100-9','NodeList=h100-16'),
            ('OverSubscribe=NO','OverSubscribe=YES'),('CPUs/Task=32','CPUs/Task=4'),
            ('TresPerNode=gres:gpu:8','TresPerNode=gres:gpu:0'),('JobState=RUNNING','JobState=RUNNINGjunk')]:
            with self.assertRaises(ValueError):A.allocation(s.replace(k,v),'99999')

    def test_stale_future_missing_stop(self):
        g=self.good();A.guard_record(g,'99999',map(str,range(8)),105.)
        for field,val in [('epoch',80.),('epoch',110.),('epoch',float('nan')),('event','STOP_OWN_JOB'),('job_id','1')]:
            d=copy.deepcopy(g);d[field]=val
            with self.assertRaises(ValueError):A.guard_record(d,'99999',map(str,range(8)),105.)

    def test_uuid_and_foreign_limits(self):
        for field,val in [('foreign_memory_fraction',.151),('foreign_sm_upper',16),
            ('foreign_memory_fraction',None),('foreign_sm_upper',float('nan')),('foreign_memory_fraction',True)]:
            d=self.good();d['gpus'][0][field]=val
            with self.assertRaises(ValueError):A.guard_record(d,'99999',map(str,range(8)),105.)
        g=self.good();g['gpus'][0]['uuid']='1'
        with self.assertRaises(ValueError):A.guard_record(g,'99999',map(str,range(8)),105.)

    def test_same_node_paths_and_unbound_failure(self):
        with self.assertRaises(ValueError):A.local('/Users/x/weights')
        with self.assertRaises(ValueError):A.local('relative')
        with self.assertRaises(ValueError):A.check('fixture')


if __name__=='__main__':unittest.main(verbosity=2)
