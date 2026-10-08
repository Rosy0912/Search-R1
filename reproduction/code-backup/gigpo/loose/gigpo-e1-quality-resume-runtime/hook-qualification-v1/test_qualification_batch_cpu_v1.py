"""Actual DataProto/RecoverableSampler/Controller/row-mask boundary on CPU.

Generation, actor/ref RPC and model probability tensors are explicit fixtures.
No CUDA/Ray/FSDP/Adam or official-model qualification is inferred from this.
The source52 mathematical Controller.compute_advantages and objective loss are
executed with genuine DataProto/TensorDict and the full 512 x 5 row layout.
"""
import copy
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from search_r1.baselines.controller import Controller
from search_r1.baselines.recovery import RecoverableSampler, rng_state
from search_r1.baselines.objectives import weighted_policy_loss
from search_r1.baselines.rollout import select_rows
from calibration_state_v1 import state_digest
from actual_batch_builder_v1 import build_actual_outer
from baseline_A_adapter_v1 import BaselineMetadataOnlyA


class CPUQuestionDataset(Dataset):
    def __len__(self):return 169028
    def __getitem__(self,index):
        return dict(input_ids=torch.tensor([index, 3, 4]),attention_mask=torch.ones(3,dtype=torch.long),
                    position_ids=torch.arange(3),index='q'+str(index),
                    reward_model=dict(ground_truth=dict(target=['fixture answer'])))


class ActualBatchPipelineCPU(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        sampler=RecoverableSampler(169028,shuffle=False)
        sampler.load_state_dict(dict(size=169028,shuffle=False,epoch=0,
            order=list(range(169028)),cursor=12800,generator=sampler.generator.get_state()))
        self.iterator=iter(DataLoader(CPUQuestionDataset(),batch_size=512,sampler=sampler,
                                     num_workers=0,drop_last=True,collate_fn=collate_fn))
        self.calls=[];self.poison=None;self.sources=list(reversed(range(2560)))
        config=OmegaConf.create(dict(actor_rollout_ref=dict(actor=dict(ppo_mini_batch_size=256,
            ppo_micro_batch_size=16,use_kl_loss=True,policy_safety=dict(mode='calibration')),
            rollout=dict(n_agent=5,n=1)),algorithm=dict(no_think_rl=False),
            retriever=dict(url='http://127.0.0.1:18081/retrieve'),force_final_answer=False))
        controller=types.SimpleNamespace(method='gigpo',config=OmegaConf.create(dict(
            gigpo=dict(gamma=.95,omega=1.,normalize=False),loss_reduction='sequence_mean')))
        controller.compute_advantages=lambda data,metrics:Controller.compute_advantages(controller,data,metrics)
        def old(data):
            self.calls.append('CPU_old_RPC_fixture');data.check_consistency()
            return DataProto.from_dict(tensors=dict(old_log_probs=torch.full((2560,4),-.5)))
        def reference(data):
            self.calls.append('CPU_ref_RPC_fixture');data.check_consistency()
            self.assertEqual([len(part) for part in data.chunk(8)],[320]*8)
            return DataProto.from_dict(tensors=dict(ref_log_prob=torch.full((2560,4),-.6)))
        def reward(data):
            scores=torch.zeros((2560,4),dtype=torch.float32)
            # Correct identities determine reward; audit UUIDs are irrelevant.
            scores[:,-1]=torch.tensor([int(row)%5 for row in data.non_tensor_batch['audit_trajectory_row']])/4
            return scores
        def balance(data,metrics):
            selected=select_rows(data,list(reversed(range(2560))))
            data.batch,data.non_tensor_batch=selected.batch,selected.non_tensor_batch
        def mask(data,metrics):
            data.batch['loss_mask']=data.batch['info_mask'][:,-4:]
            return data,metrics
        self.trainer=types.SimpleNamespace(config=config,global_steps=25,recovery_sampler=sampler,
            baseline_controller=controller,tokenizer=types.SimpleNamespace(pad_token_id=0,
                decode=lambda *args,**kwargs:'<answer>fixture answer</answer>'),
            actor_rollout_wg=types.SimpleNamespace(compute_log_prob=old),
            ref_policy_wg=types.SimpleNamespace(compute_ref_log_prob=reference),
            reward_fn=reward,_balance_batch=balance,_create_loss_mask=mask)
        self.binding=dict(status='REVIEWED_FOR_EXTERNAL_HOOK_ONLY')
        self.original_generated=None
        def generate(manager,gen_batch,qa_batch,metrics):
            self.calls.append('CPU_generation_fixture')
            self.assertEqual(qa_batch.non_tensor_batch['audit_trajectory_row'].dtype,object)
            self.assertEqual(manager.rollout_trace.ids,[f'q{i}' for i in range(12800,13312) for _ in range(5)])
            part=select_rows(gen_batch,self.sources)
            prompts=part.batch['input_ids'];responses=torch.tensor([[8,9,3,4]]*2560)
            full=torch.cat((prompts,responses),1)
            info=torch.cat((torch.ones_like(prompts),torch.tensor([[1,1,0,1]]*2560)),1)
            anchors=np.empty(2560,dtype=object)
            records=np.empty(2560,dtype=object)
            for row in range(2560):anchors[row]=['anchor0','anchor1'];records[row]=[{'node':'audit-'+str(row)}]
            generated=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,input_ids=full,
                attention_mask=torch.ones_like(full),position_ids=torch.arange(7).expand_as(full).clone(),
                info_mask=info,step_ids=torch.tensor([[0,0,-1,1]]*2560)),non_tensors=dict(anchors=anchors,
                trajectory_uid=np.asarray(['trajectory-'+str(i) for i in range(2560)],dtype=object),
                tree_uid=np.asarray(['tree-'+str(i) for i in range(2560)],dtype=object),records=records))
            self.original_generated=generated
            if self.poison=='source_order':self.sources=self.sources[:-1]+[self.sources[0]]
            elif self.poison=='missing_action':generated.batch['step_ids'][0,-1]=2
            return generated,self.sources
        self.original_generate=generate
        self.generate=BaselineMetadataOnlyA(generate).generate

    def run_actual(self):
        return build_actual_outer(self.trainer,self.iterator,Path(self.temp.name),'A',self.binding,self.generate)

    def test_full512x5_actual_protocol_sampler_source_assignment_normalizer_and_GiGPO(self):
        batch,receipt=self.run_actual()
        self.assertEqual(self.calls,['CPU_generation_fixture','CPU_old_RPC_fixture','CPU_ref_RPC_fixture'])
        self.assertEqual(self.trainer.recovery_sampler.cursor,13312)
        self.assertEqual(self.trainer.global_steps,26)
        self.assertEqual(batch.non_tensor_batch['audit_trajectory_row'].tolist(),list(range(2560)))
        self.assertTrue(all(type(value) is int for value in batch.non_tensor_batch['audit_trajectory_row']))
        self.assertEqual(receipt['retained_goal_contribution'],0)
        self.assertTrue(receipt['five_response_groups_verified'])
        self.assertTrue(torch.equal(batch.batch['policy_weights'],torch.tensor([[1/3,1/3,0.,1/3]]*2560)))
        self.assertTrue(torch.equal(batch.batch['loss_mask'],torch.tensor([[1,1,0,1]]*2560)))
        raw=torch.load(receipt['physical_witness']['path'],map_location='cpu',weights_only=False)
        self.assertEqual(raw['non_tensors']['audit_trajectory_row'].tolist(),list(range(2560)))
        self.assertTrue(torch.equal(raw['tensors']['advantages'],batch.batch['advantages']))

    def test_audit_UUID_perturbation_cannot_change_actual_GiGPO_weights_loss_or_gradient(self):
        batch,_=self.run_actual();other=copy.deepcopy(batch)
        other.non_tensor_batch['trajectory_uid']=np.asarray(['other-'+str(i) for i in range(2560)],dtype=object)
        other.non_tensor_batch['tree_uid']=np.asarray(['newtree-'+str(i) for i in range(2560)],dtype=object)
        for records in other.non_tensor_batch['records']:records[0]['node']='different random audit UUID'
        before=state_digest(rng_state())
        controller=self.trainer.baseline_controller
        for data in (batch,other):controller.compute_advantages(data,{})
        self.assertEqual(before,state_digest(rng_state()))
        for key in ('advantages','returns','policy_weights','step_ids','loss_mask','old_log_probs','ref_log_prob'):
            self.assertTrue(torch.equal(batch.batch[key],other.batch[key]),key)
        gradients=[];losses=[]
        for data in (batch,other):
            current=(data.batch['old_log_probs']+.01).clone().requires_grad_()
            loss,*_=weighted_policy_loss(data.batch['old_log_probs'],current,data.batch['advantages'],
                data.batch['loss_mask'],data.batch['policy_weights'],.2,2560.,data.batch['step_ids'],dual_clip=3.)
            loss.backward();losses.append(loss.detach());gradients.append(current.grad)
        self.assertTrue(torch.equal(losses[0],losses[1]))
        self.assertTrue(torch.equal(gradients[0],gradients[1]))

    def test_duplicate_source_mapping_still_refuses_before_any_possible_Adam(self):
        self.poison='source_order'
        with self.assertRaisesRegex(ValueError,'provenance'):self.run_actual()

    def test_missing_real_action_anchor_is_not_repaired_by_metadata_adapter(self):
        self.poison='missing_action'
        with self.assertRaisesRegex(ValueError,'action metadata'):self.run_actual()


if __name__=='__main__':unittest.main(verbosity=2)
