import copy
import unittest
import numpy as np
import torch
from tensordict import TensorDict
from verl.protocol import DataProto
from search_r1.baselines.recursive_transport import actor_slots,inference_call,is_zero_policy_slot,FIELD
from search_r1.baselines.objectives import attach_policy_weights,weighted_policy_loss
from search_r1.baselines.advantages import tree_grpo,grouped_center

class TransportTests(unittest.TestCase):
 def data(self):
  return DataProto(batch=TensorDict(dict(responses=torch.tensor([[7,8],[9,0],[10,11]]),
   attention_mask=torch.tensor([[1,1],[1,0],[1,1]]),info_mask=torch.tensor([[1,1],[1,0],[1,1]]),
   loss_mask=torch.tensor([[1,1],[1,0],[1,1]]),advantages=torch.tensor([[1.,1.],[-1.,0.],[2.,2.]])),batch_size=[3]),
   non_tensor_batch={'uid':np.array(['q1','q1','q2'],dtype=object)})
 def test_forward_slots_are_discarded_and_original_remains_three(self):
  d=self.data();calls=[]
  def worker(p):
   calls.append(len(p));return DataProto(batch=TensorDict({'value':p.batch['responses']+1},batch_size=[len(p)]))
  result=inference_call(worker,d,4)
  self.assertEqual(calls,[4]);self.assertEqual(len(result),3)
  self.assertTrue(torch.equal(result.batch['value'],d.batch['responses']+1))
 def test_one_real_forward_row_handles_world_eight(self):
  d=self.data()[:1];calls=[]
  def worker(p):calls.append(len(p));return p
  out=inference_call(worker,d,8)
  self.assertEqual((calls,len(out)),([8],1))
 def test_variable_group_loss_and_gradient_match_without_padding(self):
  d=attach_policy_weights(self.data(),'tree_grpo');before=copy.deepcopy(d)
  p=actor_slots(d)
  self.assertEqual((len(p),p.meta_info['recursive_real_rows'],p.meta_info['recursive_transport_slots']),(2560,3,2557))
  for key in before.batch.keys():self.assertTrue(torch.equal(before.batch[key],d.batch[key]))
  # Uniform rank averaging equals a global row denominator; accumulation sums micro numerators.
  old=torch.zeros((3,2));current=torch.full((3,2),.03,requires_grad=True)
  loss,_,_=weighted_policy_loss(old,current,d.batch['advantages'],d.batch['loss_mask'],d.batch['policy_weights'],.2,3)
  grad=torch.autograd.grad(loss,current)[0]
  cp=torch.cat([current.detach(),torch.zeros((2557,2))]).requires_grad_()
  lp,_,_=weighted_policy_loss(torch.zeros_like(cp),cp,p.batch['advantages'],p.batch['loss_mask'],p.batch['policy_weights'],.2,2560)
  gp=torch.autograd.grad(lp,cp)[0]
  self.assertAlmostEqual(loss.item(),lp.item(),places=6)
  self.assertTrue(torch.allclose(grad,gp[:3],atol=1e-7));self.assertEqual(torch.count_nonzero(gp[3:]).item(),0)
  self.assertEqual(np.count_nonzero(p.non_tensor_batch[FIELD]),2557)
  self.assertTrue(is_zero_policy_slot(p,3));self.assertFalse(is_zero_policy_slot(p,0))
 def test_nonzero_slot_is_refused(self):
  p=actor_slots(attach_policy_weights(self.data(),'tree_grpo'))
  p.batch['advantages'][3,0]=1
  with self.assertRaises(ValueError):is_zero_policy_slot(p,3)
 def test_single_root_is_twice_group_z_and_observations_zero(self):
  r=torch.tensor([[0.,1.],[0.,0.],[0.,1.]])
  mask=torch.tensor([[1.,0.],[1.,0.],[1.,0.]])
  actual=tree_grpo(r,mask,['q']*3,['root']*3)
  expected=2*grouped_center(r.sum(-1),['q']*3)
  self.assertTrue(torch.equal(actual[:,0],expected));self.assertEqual(torch.count_nonzero(actual[:,1]).item(),0)

if __name__=='__main__':unittest.main()
