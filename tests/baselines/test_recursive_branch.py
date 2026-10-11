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
        result,n,reason=m.search(self.record(),cont,random.Random(42),lambda *a,**k:None,variant="uniform")
        self.assertEqual(n,10);self.assertEqual(len(calls),10);self.assertEqual(len(result),5)
        self.assertEqual(len({r.trajectory_id for r in result}),5);self.assertEqual(reason,"budget_exhausted")
    def test_error_not_fabricated_into_pool(self):
        def cont(*args):raise RuntimeError("failed model call")
        result,n,reason=m.search(self.record(),cont,random.Random(42),lambda *a,**k:None,variant="uniform")
        self.assertEqual((len(result),n,reason),(1,1,"execution_error"))
    def test_empty_positions_truthful_shortfall(self):
        r=self.record();r.snapshots=[]
        result,n,reason=m.search(r,None,random.Random(42),lambda *a,**k:None,variant="uniform")
        self.assertEqual((len(result),n,reason),(1,0,"no_expandable_record"))
    def test_alternating_rewards_accept_and_recurse(self):
        def cont(p,d,a):return m.Record(str(a),p.trajectory_id,p.root_id,p.generation+1,1-p.reward,None,p.snapshots,[*p.steps[:d],str(a)],"contrast_accept")
        result,n,reason=m.search(self.record(),cont,random.Random(1),lambda *a,**k:None,variant="uniform")
        self.assertEqual((len(result),n),(5,4));self.assertTrue(any(r.generation>1 for r in result))

    def test_planner_budget_closes_search_before_fill(self):
        events=[]
        planner=m.Planner(self.record(),random.Random(42),lambda e,**k:events.append((e,k)),budget=1)
        parent,index,attempt=planner.next_request()
        child=m.Record("c",parent.trajectory_id,parent.root_id,1,0,None,parent.snapshots,[*parent.steps[:index],"new"],"same_reward")
        planner.accept_result(child)
        self.assertIsNone(planner.next_request());result=planner.finish()
        self.assertEqual(len(result),2)
        self.assertEqual(len([e for e,_ in events if e=="search_end"]),1)
        self.assertEqual(events[-1][0],"question_end")
    def test_planner_execution_error_closes_and_fills(self):
        events=[]
        planner=m.Planner(self.record(),random.Random(42),lambda e,**k:events.append((e,k)))
        planner.next_request();planner.accept_result(error=RuntimeError("executor"))
        self.assertIsNone(planner.next_request());self.assertEqual(len(planner.finish()),1)
        self.assertEqual([e for e,_ in events][-3:],["search_end","fill_end","question_end"])
