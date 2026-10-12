"""CPU scheduling contracts: refill ready questions without crossing dependencies."""
import random
import unittest
from search_r1.baselines.recursive_branch import Planner, Record
from search_r1.baselines.recursive_frontier import run_frontier
from search_r1.baselines.rollout import State


class FakeEngine:
    def __init__(self, fail=False):
        self.cost={'trajectories':0};self.batches=[];self.fail=fail
    def pending(self,state):return not state.done and state.turns<state.duration
    def advance(self,states,meta):
        assert len({s.source for s in states})==len(states)
        self.batches.append([(s.source,s.turns) for s in states])
        if self.fail:raise RuntimeError('native generation failure')
        for state in states:
            state.turns+=1;state.response.append(1);state.model_mask.append(1)
            state.steps.append(1);state.rolling=state.prompt+state.response
            state.done=state.turns==state.duration
    def reward_callback(self,states):return [1 for _ in states]
    def timing(self,*args,**kwargs):pass


def planners():
    result=[]
    for question,duration in enumerate((1,3)):
        state=State(source=question,prompt=[question],rolling=[question]);state.duration=duration
        initial=Record('root'+str(question),None,state.tree,0,0,state,[state],[],'initial')
        result.append(Planner(initial,random.Random(42),lambda *args,**kwargs:None,target=3,budget=10))
    return result


def finish(planner,request,child,suffix,reward,started):
    parent,index,attempt=request
    planner.accept_result(Record('child%d-%d'%(child.source,attempt),parent.trajectory_id,
        parent.root_id,parent.generation+1,1-parent.reward,child,
        parent.snapshots[:index]+suffix,child.steps[:],'initial'))


class Frontier(unittest.TestCase):
    def test_ready_question_refills_before_other_question_whole_suffix_ends(self):
        items=planners();engine=FakeEngine();run_frontier(items,engine,{},finish)
        self.assertEqual(engine.batches[0],[(0,0),(1,0)])
        self.assertEqual(engine.batches[1],[(0,0),(1,1)])
        self.assertTrue(all(len({q for q,_ in batch})==len(batch) for batch in engine.batches))
        self.assertEqual([len(p.finish()) for p in items],[3,3])
        self.assertEqual(engine.cost['trajectories'],4)
        self.assertEqual([p.attempts for p in items],[2,2])
    def test_generation_fault_closes_pending_branches_and_raises(self):
        items=planners();engine=FakeEngine(fail=True)
        with self.assertRaisesRegex(RuntimeError,'native generation failure'):
            run_frontier(items,engine,{},finish)
        self.assertTrue(all(p.pending is None and p.reason=='execution_error' for p in items))
        self.assertEqual(engine.cost['trajectories'],0)


if __name__=='__main__':unittest.main()
