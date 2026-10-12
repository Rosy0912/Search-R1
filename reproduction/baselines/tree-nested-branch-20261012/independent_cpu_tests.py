import copy, random, sys, unittest, tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]/'work/tree-speed-repair-20261008/nested-branch-source'
sys.path.insert(0,str(ROOT))
from search_r1.baselines.rollout import State, SearchRollout, select_rows
from search_r1.baselines.tree_nested_branch import nested_branches, OBJECTIVE
from search_r1.baselines.advantages import tree_grpo, grouped_center
from search_r1.baselines.objectives import attach_policy_weights, weighted_policy_loss
from search_r1.baselines.tree_audit import TreeAudit, fingerprint
from omegaconf import OmegaConf
TREE=OmegaConf.create(dict(roots=2,leaves=3,leaf_quotas=[2,3],action_positions='valid_action_start',exclude_initial_from_backtrack=True,fallback='initial_fresh_once_unconditional',remaining_budget='inherit_parent',reward_comparison='absolute_tolerance',reward_difference_tolerance=0.,sampling_objective=OBJECTIVE,objective=OBJECTIVE))

def emit(s, token, valid=True):
    d=len(s.records);s.response += [token,token+100];s.model_mask += [1,0];s.steps += [d,-1]
    s.rolling=s.prompt+s.response;s.anchors.append(str(d));s.records.append(dict(valid_action=valid,generated_tokens=1,action_tokens=1,observation_tokens=1))
    s.raw_ledger.append(dict(sample=str(token),mu=-.3,observation=token+100));s.observation=[token+100];s.turns+=1;s.valid_actions+=int(valid);s.node=str(token)

def root(source,valid=(True,True,True)):
    s=State(source,[9],[9],tree='AB'[source],trajectory='AB'[source]);path=[]
    for d,v in enumerate(valid):path.append(copy.deepcopy(s));emit(s,10+source*10+d,v)
    s.done=True;return s,path

class Fake(SearchRollout):
    def __init__(self,rewards,valid=(True,True,True),suffix=2):
        self.config=NS(max_turns=10,force_final_answer=True,max_prompt_length=100);self.tokenizer=NS(pad_token_id=0)
        self.manager=NS(tensor_fn=NS(create_position_ids=lambda x: x.cumsum(-1)-1))
        self.token_preservation=None;self.behavior_mu=None;self.output_meta={};self.audit=None
        self.rng=random.Random(7);self.cost=dict(generated_tokens=0,prefill_tokens=0,retrieval_calls=0,trajectories=0)
        self.roots,self.paths=map(list,zip(*(root(i,valid) for i in range(2))))
        self.queue=list(rewards);self.scores={'A':0.,'B':0.};self.calls=[];self.suffix=suffix
        self.reward_callback=lambda ss: [self.scores[s.trajectory] for s in ss]
    def complete(self,ss,meta):
        paths=[]
        for s in ss:
            self.calls.append((s.tree,len(s.records),s.raw_ledger.copy(),s.trajectory));p=[]
            for j in range(self.suffix):p.append(copy.deepcopy(s));emit(s,1000+len(self.calls)*10+j)
            self.cost['generated_tokens']+=self.suffix;self.cost['prefill_tokens']+=len(s.prompt)+len(s.response);self.cost['retrieval_calls']+=1;self.cost['trajectories']+=1
            s.done=True;self.scores[s.trajectory]=self.queue.pop(0);paths.append(p)
        return paths
    def sample(self):return nested_branches(self,self.roots,self.paths,(2,3),TREE,{})

class Independent(unittest.TestCase):
    def test_tail_first_failure_then_success_b2_actual_new_path(self):
        e=Fake([0.,0.,1.,0.]);leaves=e.sample()
        self.assertEqual([s.tree for s in leaves],['A','A','B','B','B'])
        b1,b2=leaves[3:];attempts=e.tree_candidate_attempts
        self.assertEqual([(a['stage'],a['action_depth'],a['accepted']) for a in attempts],[('A1',1,True),('B1',2,False),('B1',1,True),('B2',2,True)])
        self.assertEqual(b2.branch_metadata['parent_trajectory_id'],b1.trajectory)
        self.assertEqual(b2.branch_metadata['parent_reward'],1.)
        self.assertEqual(b2.raw_ledger[:2],b1.raw_ledger[:2]);self.assertNotEqual(b2.raw_ledger[1],e.roots[1].raw_ledger[1])
        self.assertEqual(e.cost['generated_tokens'],8);self.assertEqual(e.cost['candidate_discarded'],1)
        self.assertNotIn(attempts[1]['child_trajectory_id'],[s.trajectory for s in leaves])
        self.assertEqual(b2.branch_metadata['inherited_raw_ledger_sha256'],fingerprint(b1.raw_ledger[:2]))
    def test_same_reward_finite_initial_once_and_no_valid_fallback(self):
        for valid in ((True,True,True),(False,False,False)):
            e=Fake([0.]*30,valid);leaves=e.sample();self.assertEqual(len(leaves),5)
            for stage in ('B1','B2'):
                aa=[a for a in e.tree_candidate_attempts if a['stage']==stage]
                ds=[a['action_depth'] for a in aa];self.assertEqual(ds,sorted(ds,reverse=True));self.assertEqual(len(ds),len(set(ds)))
                self.assertEqual(ds.count(0),1);self.assertTrue(aa[-1]['fallback'])
            if not any(valid):self.assertTrue(leaves[1].branch_metadata['fallback'])
    def test_compose_audit_shuffle_and_original_row_mass(self):
        e=Fake([0.,0.,1.,0.])
        with tempfile.TemporaryDirectory() as d:
            e.audit=TreeAudit(d,d,1,1,['q','q']);leaves=e.sample();out,sources=e.compose(leaves)
            e.audit.finish(leaves,out,sources,e.roots,TREE,0)
            self.assertEqual(e.audit.report['status'],'ROLLOUT_DIAGNOSTIC_COMPLETE')
            shuffled=select_rows(out,[4,1,3,0,2])
            for i,j in enumerate([4,1,3,0,2]):
                self.assertEqual(shuffled.non_tensor_batch['trajectory_uid'][i],leaves[j].trajectory)
                self.assertEqual(shuffled.non_tensor_batch['tree_branch_metadata'][i],leaves[j].branch_metadata)
            attach_policy_weights(shuffled,'tree_grpo');torch.testing.assert_close(shuffled.batch['policy_weights'].sum(-1),torch.ones(5))
    def test_equal_root_means_nonzero_inter_and_original_std(self):
        r=torch.tensor([0.,1.,0.,1.,.5],dtype=torch.float64);uid=['q']*5;trees=['A']*2+['B']*3
        inter=grouped_center(r,uid);self.assertGreater(inter.abs().sum(),0)
        expected=(r-r.mean())/(r.std(correction=1)+1e-6)
        torch.testing.assert_close(inter,expected);torch.testing.assert_close(grouped_center(torch.ones(5),uid),torch.zeros(5))
        a=tree_grpo(r[:,None],torch.ones(5,1),uid,trees)[:,0];torch.testing.assert_close(a,expected+grouped_center(r,list(zip(uid,trees))))
    def test_shared_prefix_length_credit_global_splits_empty_row(self):
        mask=torch.tensor([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1],[0,0,0,0]],dtype=torch.float64)
        weights=mask/mask.sum(-1,keepdim=True).clamp_min(1);a=torch.tensor([1.,-1.,2.,-2.,10.])[:,None]*mask
        current=torch.zeros_like(mask,requires_grad=True)
        loss,_,_=weighted_policy_loss(torch.zeros_like(mask),current,a,mask,weights,.2,5.)
        expected=-(1.-1.+2.-2.)/5;self.assertAlmostEqual(loss.item(),expected)
        loss.backward();self.assertAlmostEqual(current.grad[:,0].sum().item(),-(1.-.5+2/3-2/4)/5)
        split=sum(weighted_policy_loss(torch.zeros_like(mask[i:i+1]),current[i:i+1],a[i:i+1],mask[i:i+1],weights[i:i+1],.2,5.)[0] for i in range(5))
        torch.testing.assert_close(loss,split);self.assertEqual(weights[-1].sum().item(),0)
    def test_missing_validity_rejected_and_b2_earlier_inherited_prefix(self):
        e=Fake([0.,1.,1.,1.,0.]);leaves=e.sample();b1,b2=leaves[3:]
        self.assertEqual(b1.branch_metadata['action_depth'],2)
        self.assertEqual(b2.branch_metadata['action_depth'],1)
        self.assertEqual(b2.branch_metadata['actual_prefix_trajectory_id'],'B')
        self.assertEqual(b2.branch_metadata['parent_trajectory_id'],b1.trajectory)
        self.assertEqual(b2.branch_metadata['b2_prefix_origin'],'inherited_parent_prefix')
        self.assertEqual(b2.raw_ledger[0],e.roots[1].raw_ledger[0])
        e=Fake([0.]*10);del e.roots[1].records[1]['valid_action']
        with self.assertRaisesRegex(ValueError,'validity'):e.sample()
    def test_zero_retained_suffix_still_five_unique_leaves(self):
        class Empty(Fake):
            def complete(self,ss,meta):
                paths=[]
                for state in ss:
                    paths.append([copy.deepcopy(state)])
                    state.records.append(dict(valid_action=False,generated_tokens=0,action_tokens=0,observation_tokens=0));state.anchors.append('empty');state.turns+=1;state.done=True
                    self.scores[state.trajectory]=self.queue.pop(0)
                    self.cost['trajectories']+=1
                return paths
        e=Empty([0.]*10,valid=(False,False,False));leaves=e.sample()
        self.assertEqual(len(set(s.trajectory for s in leaves)),5)
        self.assertTrue(all(not s.response for s in (leaves[1],leaves[3],leaves[4])))
        self.assertTrue(all(a['new_retained_model_tokens']==0 for a in e.tree_candidate_attempts))
    def test_controller_scoring_only_does_not_compose_candidates(self):
        from search_r1.baselines.controller import Controller
        from verl import DataProto
        c=Controller.__new__(Controller)
        scored=DataProto.from_dict(dict(responses=torch.ones(2,1,dtype=torch.long)))
        qa=DataProto.from_dict(dict(responses=torch.ones(2,1,dtype=torch.long)),non_tensors={'index':np.array(['a','b'])})
        c.trainer=NS(reward_fn=lambda data:torch.tensor([[0.],[1.]]))
        engine=NS(reward_batch=lambda states:(scored,np.array([1,0])),compose=lambda states: (_ for _ in ()).throw(AssertionError('candidate compose forbidden')))
        self.assertEqual(c.tree_training_rewards(engine,qa,[None,None],scoring_only=True),[0.,1.])
        self.assertEqual(scored.non_tensor_batch['index'].tolist(),['b','a'])
    def test_selection_probability_not_corrected_by_same_token_mu(self):
        mu=np.log(.2);pi=np.log(.3);ratio=np.exp(pi-mu)
        self.assertAlmostEqual(ratio,1.5);self.assertNotEqual(.2/.4,.2/.8)
        e=Fake([0.,0.,1.,0.]);leaves=e.sample()
        self.assertTrue(all(not s.branch_metadata['selection_unbiased'] for s in leaves))
        self.assertTrue(all(s.branch_metadata.get('sampling_objective',s.branch_metadata.get('objective'))==OBJECTIVE for s in leaves))
if __name__=='__main__':unittest.main(verbosity=2)
