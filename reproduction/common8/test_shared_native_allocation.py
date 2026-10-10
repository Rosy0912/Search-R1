import unittest
from reproduction.common8.native_allocation import validate

class SharedAllocationTests(unittest.TestCase):
    def setUp(self):
        self.uuids=['GPU-'+str(i) for i in range(8)]
        self.args=({'JobState':'RUNNING','NodeList':'h100-8','NumNodes':'1',
                    'TresPerNode':'gres:gpu:8','JobId':'42'},'h100-8',list(map(str,range(8))),['42'],
                   [[str(i),u,'81559'] for i,u in enumerate(self.uuids)])
        self.apps=[[u,str(100+i)] for i,u in enumerate(self.uuids)]
    def check(self,apps,allowed=None):
        return validate(*self.args,apps,list(range(8)),self.uuids,allowed)
    def test_legacy_requires_idle_cards(self):
        with self.assertRaises(AssertionError):self.check(self.apps)
        self.assertTrue(self.check([]))
    def test_bound_cohosts_can_remain_or_exit(self):
        self.assertTrue(self.check(self.apps,self.apps))
        self.assertTrue(self.check(self.apps[:4],self.apps))
    def test_new_pid_or_gpu_mapping_rejected(self):
        with self.assertRaises(AssertionError):self.check(self.apps+[[self.uuids[0],'900']],self.apps)
        with self.assertRaises(AssertionError):self.check([[self.uuids[1],'100']],self.apps)

if __name__=='__main__':unittest.main()
