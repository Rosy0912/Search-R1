import importlib.util
import pathlib
import random
import sys
import unittest
spec=importlib.util.spec_from_file_location("recursive_branch",pathlib.Path(__file__).parents[2]/"search_r1/baselines/recursive_branch.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class RecursiveTests(unittest.TestCase):
    def record(self,id="root",generation=0,reward=0,steps=None):
        return m.Record(id,None,"root",generation,reward,None,[0,1,2],steps or ["a","b","c"],"initial")
    def test_generation_not_attempt_controls_boundary(self):
        p=self.record("p",2,steps=["a","x","c"])
        lo=self.record("lo",1,steps=["a","z","c"])
        same=self.record("same",2,steps=p.steps)
        self.assertEqual(m.legal_positions(p,[lo,same])[0],[1,2])
        self.assertEqual(m.legal_positions(self.record(),[lo,same])[0],[0,1,2])
    def test_all_same_fill_without_duplicate_or_extra_rollout(self):
        calls=[]
        def cont(p,d,a):
            calls.append((d,a));return m.Record(str(a),p.trajectory_id,p.root_id,p.generation+1,0,None,p.snapshots,[*p.steps[:d],str(a)],"same_reward")
        result,n,reason=m.search(self.record(),cont,random.Random(42),lambda *a,**k:None)
        self.assertEqual(n,10);self.assertEqual(len(calls),10);self.assertEqual(len(result),5)
        self.assertEqual(len({r.trajectory_id for r in result}),5);self.assertEqual(reason,"budget_exhausted")
    def test_error_not_fabricated_into_pool(self):
        def cont(*args):raise RuntimeError("failed model call")
        result,n,reason=m.search(self.record(),cont,random.Random(42),lambda *a,**k:None)
        self.assertEqual((len(result),n,reason),(1,1,"execution_error"))
    def test_empty_positions_truthful_shortfall(self):
        r=self.record();r.snapshots=[]
        result,n,reason=m.search(r,None,random.Random(42),lambda *a,**k:None)
        self.assertEqual((len(result),n,reason),(1,0,"no_expandable_record"))
    def test_alternating_rewards_accept_and_recurse(self):
        def cont(p,d,a):return m.Record(str(a),p.trajectory_id,p.root_id,p.generation+1,1-p.reward,None,p.snapshots,[*p.steps[:d],str(a)],"contrast_accept")
        result,n,reason=m.search(self.record(),cont,random.Random(1),lambda *a,**k:None)
        self.assertEqual((len(result),n),(5,4));self.assertTrue(any(r.generation>1 for r in result))
