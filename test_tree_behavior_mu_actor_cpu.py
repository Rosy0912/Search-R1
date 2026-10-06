"""Affected source AST + tiny real CPU actor/Adam/Gloo, not HF/FSDP/Ray.

Imports only the collector's fixture factory, never runs its independent suite.
TinyBatch/Proto emulate row slicing; native scores are a one-parameter model.
"""
import ast
import copy
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
# Load native optimizer wrappers before collector installs its temporary verl stub.
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp

import test_tree_behavior_mu_data_cpu as collector_fixture
from search_r1.baselines.policy_safety_runtime import (PolicySafetyRuntime, PolicySafetyRefusal,
    behavior_mu_correction_enabled, validate_actor_probability_profile)
from search_r1.baselines.policy_safety import dual_clip_terms
from search_r1.baselines.tree_behavior_mu import PROTOCOL, FIELD, validate_behavior_mu_batch, runtime_provenance

ROOT = Path(__file__).resolve().parent


class Config(dict):
    __getattr__ = dict.__getitem__


class TinyBatch(dict):
    def __len__(self): return len(dict.__getitem__(self, 'input_ids'))
    @property
    def device(self): return self['input_ids'].device
    @property
    def batch_size(self): return [len(self)]
    def __getitem__(self, key):
        if isinstance(key, str): return dict.__getitem__(self, key)
        return TinyBatch({name: value[key] for name, value in self.items()})
    def split(self, rows): return [self[i:i+rows] for i in range(0, len(self), rows)]
    def to(self, device): return TinyBatch({name: value.to(device) for name, value in self.items()})
    def cuda(self): return self


class Proto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = TinyBatch(batch)
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}
    def __len__(self): return len(self.batch)
    def select(self, batch_keys):
        return Proto({key:self.batch[key] for key in batch_keys}, self.non_tensor_batch, self.meta_info)
    def reorder(self, order):
        self.batch = self.batch[order]
        self.non_tensor_batch = {key: values[order] for key, values in self.non_tensor_batch.items()}


def ast_members(path, names, scope, klass=None):
    tree = ast.parse(path.read_bytes())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == klass) if klass else tree.body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return {name: scope[name] for name in names}


Fscope = dict(torch=torch)
F = NS(**ast_members(ROOT/'verl/utils/torch_functional.py', ('masked_sum', 'masked_mean'), Fscope))
objective = ModuleType('search_r1.baselines.objectives')
objective.__package__ = 'search_r1.baselines'
objective.__dict__.update(torch=torch, math=math, F=F)
objective.__dict__.update(ast_members(ROOT/'search_r1/baselines/objectives.py',
    ('row_denominator','weighted_policy_loss','reduce_paper_metrics'), objective.__dict__))
sys.modules[objective.__name__] = objective
count = ModuleType('search_r1.baselines.distillation')
count.__dict__.update(torch=torch)
count.__dict__.update(ast_members(ROOT/'search_r1/baselines/distillation.py', ('global_token_count',), count.__dict__))
sys.modules[count.__name__] = count
rank_audit = ModuleType('search_r1.baselines.tree_rank_audit'); rank_audit.enabled = lambda:False
rank_audit.TreeRankAudit = object
sys.modules[rank_audit.__name__] = rank_audit


def append_to_dict(target, values):
    for key,value in values.items(): target.setdefault(key,[]).append(value)


actor_scope = dict(torch=torch, itertools=itertools, DataProto=Proto,
                   append_to_dict=append_to_dict, verl_F=F, masked_mean=F.masked_mean)
actor_methods = ast_members(ROOT/'verl/workers/actor/dp_actor.py', ('update_policy','_optimizer_step'),
                            actor_scope, 'DataParallelPPOActor')


class Actor:
    update_policy = actor_methods['update_policy']
    _optimizer_step = actor_methods['_optimizer_step']
    def __init__(self, cfg):
        self.config = cfg
        self.actor_module = torch.nn.Linear(1,1,bias=False)
        self.actor_module.weight.data.fill_(-.5)
        self.actor_optimizer = torch.optim.Adam(self.actor_module.parameters(),lr=.01)
        self.use_ulysses_sp = False
        self.calls = []
        self.repeat_drift = False
        self.emulate_rng = False
        self.captured_gradients = []
    def _forward_micro_batch(self, micro_batch, temperature):
        self.calls.append((len(micro_batch),micro_batch['input_ids'].detach().clone(),temperature))
        current = self.actor_module(torch.ones(len(micro_batch),1)).expand_as(micro_batch['responses'])
        if self.emulate_rng: current = current + torch.rand(())*0.
        if self.repeat_drift and len(self.calls) > self.config.policy_safety['monitor_rows_per_rank']//self.config.ppo_micro_batch_size:
            current = current + .1
        return current*0, current


trainer_scope = dict(torch=torch, os=os, DataProto=Proto, core_algos=NS(AdaptiveKLController=object), masked_mean=F.masked_mean)
trainer = ast_members(ROOT/'verl/trainer/ppo/ray_trainer.py',
    ('cast_rollout_structural_fields','prepare_behavior_mu_old_policy','validate_final_behavior_mu_batch','apply_kl_penalty'), trainer_scope)


def config(corrected=True, **overrides):
    safety = Config(enabled=True, dual_clip=3., kl_log_bound=30., monitor_rows_per_rank=4,
        target_old_policy_kl=10., negative_tail_limit=1., initial_old_logprob_max_abs_delta=None,
        sampling_probability_correction=Config(enabled=corrected, protocol=PROTOCOL,
                                              initial_actor_repeat_max_abs_delta=0.))
    safety.update(overrides)
    return Config(ppo_mini_batch_size=6, ppo_micro_batch_size=2, ppo_epochs=1, state_masking=True,
        use_kl_loss=False, use_dynamic_bsz=False, baseline_method='tree_grpo', entropy_coeff=0.,
        clip_ratio=.2, grad_clip=1., policy_safety=safety)


def make_data(factory, cfg, rows=6):
    data = factory.output([factory.state(call='actor-factory-0').branch() for _ in range(rows)])
    data = Proto(data.batch, data.non_tensor_batch, data.meta_info)
    width = data.batch['responses'].shape[1]
    data.batch['old_log_probs'] = torch.full((rows,width),-.5)
    trainer['prepare_behavior_mu_old_policy'](data,cfg)
    data.batch['loss_mask'] = data.batch['info_mask'][:,-width:].clone()
    data.batch['advantages'] = torch.ones((rows,width))
    mask = (data.batch['loss_mask']*data.batch['attention_mask'][:,-width:]).float()
    data.batch['policy_weights'] = mask/mask.sum(-1,keepdim=True).clamp_min(1)
    data.meta_info.update(temperature=1.,baseline_outer_step=111)
    return data


def gloo_worker(rank, rendezvous, output, mode):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    factory = collector_fixture.BehaviorMuDataTests(); factory.setUp()
    try:
        cfg = config(); actor = Actor(cfg); data = make_data(factory,cfg)
        os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'] = str(Path(output)/str(rank))
        if mode == 'missing_mu' and rank == 1: del data.batch['sampled_old_log_probs']
        if mode == 'profile' and rank == 1: cfg.policy_safety.sampling_probability_correction['enabled'] = False
        if mode == 'repeat' and rank == 1: actor.repeat_drift = True
        try:
            actor.update_policy(data)
            raise AssertionError('must refuse both ranks')
        except PolicySafetyRefusal as exc:
            committed = getattr(getattr(actor,'_policy_safety',None),'committed',0)
            result = dict(rank=rank,mode=mode,refused=True,committed=committed,forward_calls=len(actor.calls),error=str(exc))
            assert committed == 0
            (Path(output)/f'{rank}.json').write_text(json.dumps(result,allow_nan=False))
    finally:
        factory.tearDown(); dist.destroy_process_group()


class Tests(unittest.TestCase):
    def setUp(self):
        self.factory = collector_fixture.BehaviorMuDataTests(); self.factory.setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ,{'SEARCH_R1_TREE_RANK_AUDIT_DIR':self.tmp.name});self.env.start()
    def tearDown(self):
        self.env.stop(); self.tmp.cleanup(); self.factory.tearDown()
    def runtime(self,cfg=None,data=None):
        cfg=cfg or config(); actor=Actor(cfg); data=data or make_data(self.factory,cfg)
        runtime=PolicySafetyRuntime(actor,data.batch,data.meta_info,cfg.policy_safety,source_data=data)
        return actor,data,runtime
    def events(self):
        return [json.loads(p.read_text()) for p in sorted(Path(self.tmp.name).rglob('event*.json'))]

    def test_legacy_first_and_final_casts_exact_original_and_corrected_preserves_every_float(self):
        for preserve_old in (False,True):
            proto=Proto(dict(input_ids=torch.ones(2,3),old_log_probs=torch.full((2,2),-.25),
                             sampled_old_log_probs=torch.full((2,2),-.75),other_float=torch.full((2,1),.125)))
            original={k:v.clone() if preserve_old and k=='old_log_probs' else v.long() for k,v in proto.batch.items()}
            trainer['cast_rollout_structural_fields'](proto,config(False),preserve_old=preserve_old)
            self.assertTrue(all(torch.equal(proto.batch[k],v) and proto.batch[k].dtype==v.dtype for k,v in original.items()))
        cfg=config(); data=make_data(self.factory,cfg); data.batch['extra_float']=torch.full((len(data),1),.125)
        before=dict(data.batch)
        trainer['cast_rollout_structural_fields'](data,cfg)
        self.assertTrue(all(data.batch[k] is v for k,v in before.items() if v.is_floating_point()))

    def test_original_beta_is_independent_clone_and_mu_denominator_exact_after_joint_balance(self):
        cfg=config(); data=make_data(self.factory,cfg)
        self.assertTrue(torch.equal(data.batch['actor_old_log_probs'],torch.full_like(data.batch['sampled_old_log_probs'],-.5)))
        self.assertNotEqual(data.batch['old_log_probs'].data_ptr(),data.batch['sampled_old_log_probs'].data_ptr())
        self.assertNotEqual(data.batch['actor_old_log_probs'].data_ptr(),data.batch['old_log_probs'].data_ptr())
        data.reorder(np.array([5,2,0,4,1,3]))
        receipt=trainer['validate_final_behavior_mu_batch'](data,cfg)
        self.assertEqual(receipt['leaf_effective_token_occurrences'],18)
        self.assertFalse(receipt['probability_accepted'])

    def test_real_binding_mutations_refuse_actor_before_forward(self):
        for change in ('missing_mu','missing_beta','missing_binding','profile','old_from_beta','loss_mask','position','ledger'):
            cfg=config(); actor=Actor(cfg); data=make_data(self.factory,cfg)
            if change=='missing_mu':del data.batch['sampled_old_log_probs']
            if change=='missing_beta':del data.batch['actor_old_log_probs']
            if change=='missing_binding':del data.non_tensor_batch[FIELD]
            if change=='profile':data.meta_info['tree_behavior_mu_corrected_profile']=False
            if change=='old_from_beta':data.batch['old_log_probs']=data.batch['actor_old_log_probs'].clone()
            if change=='loss_mask':data.batch['loss_mask'][0,-1]=1
            if change=='position':data.batch['position_ids'][0,1]+=1
            if change=='ledger':data.non_tensor_batch[FIELD][0]['binding_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(PolicySafetyRefusal):actor.update_policy(data)
            self.assertEqual(len(actor.calls),0)
            self.assertEqual(len(actor.actor_optimizer.state),0)

    def test_resealed_wrong_legacy_semantics_refuse_mu_body_and_actor_before_writes(self):
        from search_r1.baselines.tree_raw_tokens import FIELD as RAW_FIELD, fingerprint
        for fields in ('ids','reward','both'):
            cfg=config();actor=Actor(cfg);data=make_data(self.factory,cfg)
            projection=data.non_tensor_batch[RAW_FIELD][0];action=projection['raw_ledger'][0]['action']
            fake='<answer>WRONG</answer>'
            if fields in ('ids','both'):action['canonical_semantic_token_ids']=self.factory.tok.encode(fake)
            if fields in ('reward','both'):action['legacy_reward_action_text']=fake
            action['action_binding_sha256']=fingerprint({k:v for k,v in action.items()
                if not k.startswith('prefix_scan_') and k!='action_binding_sha256'})
            obs=projection['raw_ledger'][0]['observation_token_ids']
            projection['semantic_response_token_ids']=action['canonical_semantic_token_ids']+obs
            projection['semantic_model_mask']=[1]*len(action['canonical_semantic_token_ids'])+[0]*len(obs)
            projection['binding_sha256']=fingerprint({k:v for k,v in projection.items() if k!='binding_sha256'})
            binding=data.non_tensor_batch[FIELD][0]
            binding['raw_projection_sha256']=projection['binding_sha256']
            binding['action_receipts'][0]['action_binding_sha256']=action['action_binding_sha256']
            binding['binding_sha256']=fingerprint({k:v for k,v in binding.items() if k!='binding_sha256'})
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError,'Legacy IDs/reward'):
                    validate_behavior_mu_batch(data,expected_provenance=runtime_provenance(os.environ))
                with self.assertRaisesRegex(PolicySafetyRefusal,'Legacy IDs/reward'):actor.update_policy(data)
                self.assertEqual(len(actor.calls),0);self.assertEqual(len(actor.actor_optimizer.state),0)

    def test_wrong_or_empty_protocol_and_missing_repeat_limit_refuse(self):
        for bad in ('',None,'wrong'):
            cfg=config();cfg.policy_safety.sampling_probability_correction['protocol']=bad
            with self.subTest(protocol=bad),self.assertRaises(ValueError):behavior_mu_correction_enabled(cfg.policy_safety)
        for bad in (None,True,float('nan'),float('inf'),-.1):
            cfg=config();cfg.policy_safety.sampling_probability_correction['initial_actor_repeat_max_abs_delta']=bad
            actor=Actor(cfg);data=make_data(self.factory,cfg)
            with self.subTest(limit=bad),self.assertRaises(PolicySafetyRefusal):actor.update_policy(data)
            self.assertEqual(len(actor.calls),0)

    def test_corrected_startup_without_full_provenance_proto_refuses(self):
        cfg=config();data=make_data(self.factory,cfg);actor=Actor(cfg)
        with self.assertRaisesRegex(PolicySafetyRefusal,'source data'):
            PolicySafetyRuntime(actor,data.batch,data.meta_info,cfg.policy_safety)
        self.assertEqual(len(actor.calls),0)

    def test_initial_mu_kl_is_nonzero_not_subtracted_repeat_zero_fullbeta_vsmonitor_logged(self):
        actor,data,r=self.runtime();decision=r.measure('before_optimizer',publish=False)
        self.assertTrue(decision['allowed']);self.assertGreater(decision['sampled_old_policy_kl'],0.)
        self.assertFalse(decision['initial_mu_kl_baseline_subtracted']);self.assertFalse(decision['original_sampler_actor_consistency_accepted'])
        self.assertEqual(decision['global_monitor_actor_repeat_max_abs_delta'],0.)
        self.assertEqual(decision['global_monitor_actor_full_partition_max_abs_delta'],0.)
        self.assertGreater(decision['global_monitor_max_abs_delta'],0.)
        self.assertEqual(decision['global_monitor_mu_full_partition_beta_max_abs_delta'],.25)
        self.assertEqual(decision['global_monitor_initial_mu_monitor_beta_max_abs_delta'],.25)
        self.assertEqual(r.monitor_native_forward_calls,4);self.assertEqual(r.initial_actor_repeat_native_forward_calls,2)
        self.assertTrue(torch.equal(r.actor_monitor_beta,torch.full_like(r.actor_monitor_beta,-.5)))
        # Frozen full-batch beta differs from the same-partition monitor: diagnostic only.
        data.batch['actor_old_log_probs'][:]=-.9
        _,_,different=self.runtime(data=data)
        report=different.measure('before_optimizer',publish=False)
        self.assertTrue(report['allowed']);self.assertGreater(report['global_monitor_actor_full_partition_max_abs_delta'],.39)

    def test_initial_mu_kl_and_negative_tail_can_refuse_zero_writes(self):
        for gate in ('kl','tail'):
            cfg=config(target_old_policy_kl=1e-8) if gate=='kl' else config(dual_clip=1.21,negative_tail_limit=0.)
            actor=Actor(cfg);data=make_data(self.factory,cfg)
            if gate=='tail':data.batch['advantages'].fill_(-1.)
            before=actor.actor_module.weight.detach().clone()
            with self.subTest(gate=gate),self.assertRaises(PolicySafetyRefusal):actor.update_policy(data)
            self.assertEqual(actor._policy_safety.committed,0)
            self.assertTrue(torch.equal(before,actor.actor_module.weight))

    def test_repeat_drift_or_nonfinite_refuses_zero_writes(self):
        actor,data,r=self.runtime();actor.repeat_drift=True
        with self.assertRaisesRegex(PolicySafetyRefusal,'repeat_inconsistency'):r.measure('before_optimizer')
        self.assertEqual(r.committed,0)
        actor,data,r=self.runtime();data.batch['actor_old_log_probs'][0,0]=float('nan')
        # A new preflight catches nonfinite beta on valid token.
        with self.assertRaises(PolicySafetyRefusal):validate_actor_probability_profile(actor,data,actor.config.policy_safety)

    def test_beta_nan_observation_padding_remains_invalid_not_a_policy_input(self):
        cfg=config();data=make_data(self.factory,cfg);actor=Actor(cfg)
        invalid=~PolicySafetyRuntime.effective_mask(data.batch)
        self.assertTrue(invalid.any());data.batch['actor_old_log_probs'][invalid]=float('nan')
        self.assertTrue(validate_actor_probability_profile(actor,data,cfg.policy_safety))
        r=PolicySafetyRuntime(actor,data.batch,data.meta_info,cfg.policy_safety,source_data=data)
        result=r.measure('before_optimizer',publish=False)
        self.assertTrue(result['allowed']);self.assertTrue(result['kl_computation_valid'])
        self.assertTrue(torch.isnan(data.batch['actor_old_log_probs'][invalid]).all())

    def test_initial_repeat_monitor_fork_restores_rng_and_keeps_input_fields(self):
        actor,data,r=self.runtime();actor.emulate_rng=True
        rng=torch.get_rng_state().clone();before={k:v.clone() for k,v in data.batch.items()}
        r.measure('before_optimizer',publish=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in data.batch.items()))
        self.assertTrue(all(rows==2 and temp==1. for rows,ids,temp in actor.calls))
        self.assertTrue(torch.equal(actor.calls[0][1],actor.calls[2][1]));self.assertTrue(torch.equal(actor.calls[1][1],actor.calls[3][1]))

    def test_mu_dualclip_gradient_preserves_tree_weights_and_global_row_denominator(self):
        cfg=config();actor=Actor(cfg);data=make_data(self.factory,cfg)
        data.batch['advantages'][1::2].fill_(-1.)
        mask=PolicySafetyRuntime.effective_mask(data.batch);weights=data.batch['policy_weights'].clone();adv=data.batch['advantages'].clone()
        w=actor.actor_module.weight.detach().clone().requires_grad_();current=w.expand_as(data.batch['old_log_probs'])
        expected=(dual_clip_terms(data.batch['old_log_probs'],current,adv,mask)*weights).sum()/len(data)
        grad=torch.autograd.grad(expected,w)[0]
        beta_loss=(dual_clip_terms(data.batch['actor_old_log_probs'],current,adv,mask)*weights).sum()/len(data)
        beta_grad=torch.autograd.grad(beta_loss,w)[0]
        self.assertFalse(torch.allclose(grad,beta_grad))
        original=actor._optimizer_step
        def capture():actor.captured_gradients.append(actor.actor_module.weight.grad.detach().clone());return original()
        actor._optimizer_step=capture
        metrics=actor.update_policy(data)
        self.assertEqual(metrics['actor/optimizer_steps'],1);torch.testing.assert_close(actor.captured_gradients[0],grad)
        self.assertTrue(torch.equal(weights,data.batch['policy_weights']));self.assertTrue(torch.equal(adv,data.batch['advantages']))
        self.assertEqual(metrics['actor/behavior_mu_additional_initial_repeat_native_forward_calls'],2)

    def test_beta_reward_kl_source_unchanged_and_missing_beta_never_uses_mu(self):
        cfg=config();data=make_data(self.factory,cfg)
        data.batch['token_level_scores']=torch.ones_like(data.batch['old_log_probs'])
        data.batch['ref_log_prob']=torch.full_like(data.batch['old_log_probs'],-.8)
        def penalty(logprob,ref_logprob,kl_penalty):return logprob-ref_logprob
        trainer_scope['core_algos'].kl_penalty=penalty
        ctrl=NS(value=.2,update=lambda **kw:None)
        result,_=trainer['apply_kl_penalty'](data,ctrl)
        mask=data.batch['info_mask'][:,-4:]
        expected=data.batch['token_level_scores']-.2*(data.batch['actor_old_log_probs']-data.batch['ref_log_prob'])*mask
        torch.testing.assert_close(result.batch['token_level_rewards'],expected)
        del data.batch['actor_old_log_probs']
        with self.assertRaises(KeyError):trainer['apply_kl_penalty'](data,ctrl)

    def test_true_ten_inner_optimizer_calls_without_budget_or_scheduler_change(self):
        cfg=config(monitor_rows_per_rank=2);cfg.ppo_mini_batch_size=2;cfg.ppo_micro_batch_size=1
        actor=Actor(cfg);actor.actor_optimizer=torch.optim.SGD(actor.actor_module.parameters(),lr=1e-4)
        data=make_data(self.factory,cfg,rows=20)
        metrics=actor.update_policy(data)
        self.assertEqual(metrics['actor/optimizer_steps'],10);self.assertEqual(actor._policy_safety.committed,10)
        self.assertEqual(metrics['actor/behavior_mu_additional_initial_repeat_native_forward_calls'],2)
        self.assertTrue(all('actor_old_log_probs' in chunk and 'sampled_old_log_probs' in chunk for chunk in actor._policy_safety.monitor.split(1)))

    def test_finite_postwrite_behavior_kl_refusal_counts_real_write(self):
        cfg=config();actor=Actor(cfg);data=make_data(self.factory,cfg)
        original=actor._optimizer_step
        def change_after():
            result=original();actor.actor_module.weight.data.fill_(10.);return result
        actor._optimizer_step=change_after
        with self.assertRaisesRegex(PolicySafetyRefusal,'sampled_old_policy_kl'):actor.update_policy(data)
        self.assertEqual(actor._policy_safety.committed,1)
        rows=[json.loads(p.read_text()) for p in Path(self.tmp.name).rglob('*.json')]
        result=next(row for row in rows if row['phase']=='after_optimizer')
        self.assertEqual(result['committed_optimizer_calls'],1)
        self.assertFalse(result['decision']['allowed']);self.assertTrue(result['decision']['kl_computation_valid'])
        self.assertFalse(result['outer_completed']);self.assertFalse(result['scheduler_advanced'])

    def test_full_source_data_does_not_allow_beta_as_mu_runtime_denominator(self):
        cfg=config();actor=Actor(cfg);data=make_data(self.factory,cfg)
        data.batch['old_log_probs']=data.batch['actor_old_log_probs'].clone()
        with self.assertRaisesRegex(PolicySafetyRefusal,'actual fixed mu'):
            PolicySafetyRuntime(actor,data.batch,data.meta_info,cfg.policy_safety,source_data=data)
        self.assertEqual(len(actor.calls),0)

    def test_refusal_after_one_real_write_retains_committed_count_and_no_outer_credit(self):
        cfg=config();actor=Actor(cfg);data=make_data(self.factory,cfg);before=actor.actor_module.weight.detach().clone()
        original=actor._optimizer_step
        def corrupt_after():
            result=original();actor.actor_module.weight.data.fill_(float('nan'));return result
        actor._optimizer_step=corrupt_after
        with self.assertRaises(PolicySafetyRefusal):actor.update_policy(data)
        self.assertEqual(actor._policy_safety.committed,1);self.assertFalse(torch.equal(before,actor.actor_module.weight))
        records=[json.loads(p.read_text()) for p in Path(self.tmp.name).rglob('*.json')]
        self.assertTrue(any(row['committed_optimizer_calls']==1 and not row['decision']['allowed'] for row in records))
        self.assertTrue(all(row['outer_completed'] is False and row['scheduler_advanced'] is False for row in records))

    def test_legacy_profile_keeps_old_initial_consistency_refusal(self):
        cfg=config(False,initial_old_logprob_max_abs_delta=0.);actor=Actor(cfg)
        data=make_data(self.factory,config())
        for key in ('sampled_old_log_probs','actor_old_log_probs'):del data.batch[key]
        del data.non_tensor_batch[FIELD];data.meta_info.pop('tree_behavior_mu_corrected_profile')
        for key in ('tree_behavior_mu_protocol','tree_behavior_mu_runtime_provenance'):data.meta_info.pop(key)
        data.batch['old_log_probs'].fill_(-.6)
        with self.assertRaisesRegex(PolicySafetyRefusal,'initial_old_current_probability_inconsistency'):actor.update_policy(data)
        self.assertEqual(actor._policy_safety.committed,0)

    def test_disabled_profile_refuses_orphan_mu_metadata_and_null_config(self):
        cfg=config(False,initial_old_logprob_max_abs_delta=0.);actor=Actor(cfg)
        data=make_data(self.factory,config())
        for key in ('sampled_old_log_probs','actor_old_log_probs'):del data.batch[key]
        del data.non_tensor_batch[FIELD];data.meta_info.pop('tree_behavior_mu_corrected_profile')
        with self.assertRaisesRegex(PolicySafetyRefusal,'uncorrected actor profile'):actor.update_policy(data)
        self.assertEqual(len(actor.calls),0)
        with self.assertRaisesRegex(ValueError,'cannot be null'):
            behavior_mu_correction_enabled({'sampling_probability_correction':None})
        self.assertFalse(behavior_mu_correction_enabled({}))

    def test_legacy_success_keeps_gradient_parameter_rng_and_forward_budget(self):
        cfg=config(False,initial_old_logprob_max_abs_delta=0.);actor=Actor(cfg)
        data=make_data(self.factory,config())
        for key in ('sampled_old_log_probs','actor_old_log_probs'):del data.batch[key]
        del data.non_tensor_batch[FIELD]
        for key in ('tree_behavior_mu_corrected_profile','tree_behavior_mu_protocol','tree_behavior_mu_runtime_provenance'):data.meta_info.pop(key)
        data.batch['old_log_probs'].fill_(-.5)
        actor.actor_optimizer=torch.optim.SGD(actor.actor_module.parameters(),lr=.01)
        actor.emulate_rng=True
        expected=actor.actor_module.weight.detach().clone()+.01
        original=actor._optimizer_step
        gradients=[]
        def capture():gradients.append(actor.actor_module.weight.grad.detach().clone());return original()
        actor._optimizer_step=capture
        rng=torch.get_rng_state().clone()
        for _ in range(3):torch.rand(())
        expected_rng=torch.get_rng_state().clone();torch.set_rng_state(rng)
        metrics=actor.update_policy(data)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assertEqual(metrics['actor/optimizer_steps'],1);torch.testing.assert_close(gradients[0],torch.full_like(gradients[0],-1.))
        torch.testing.assert_close(actor.actor_module.weight,expected,atol=1e-7,rtol=0.)
        self.assertEqual(len(actor.calls),7) # two monitor before + three loss + two monitor after, no repeat
        self.assertEqual(actor._policy_safety.initial_actor_repeat_native_forward_calls,0)
        self.assertNotIn('actor/behavior_mu_additional_initial_repeat_native_forward_calls',metrics)

    def test_source_fit_installs_after_beta_compute_validates_after_balance_and_finalmask(self):
        source=(ROOT/'verl/trainer/ppo/ray_trainer.py').read_text()
        before=source.index('prepare_behavior_mu_old_policy(final_gen_batch_output')
        self.assertGreater(before,source.index('self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)'))
        self.assertLess(before,source.index('self._balance_batch(batch, metrics=metrics)'))
        checks=source.count('validate_final_behavior_mu_batch(batch, self.config.actor_rollout_ref.actor)')
        self.assertEqual(checks,2)
        self.assertLess(source.index('self._create_loss_mask(batch, metrics)',before),source.rindex('validate_final_behavior_mu_batch(batch, self.config.actor_rollout_ref.actor)'))
        self.assertLess(source.rindex('validate_final_behavior_mu_batch(batch, self.config.actor_rollout_ref.actor)'),source.index('self.actor_rollout_wg.update_actor(batch)'))

    def test_real_two_rank_gloo_missing_mu_wrong_profile_and_repeat_drift_all_refuse(self):
        for mode in ('missing_mu','profile','repeat'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                mp.spawn(gloo_worker,args=(str(Path(temp)/'rendezvous'),temp,mode),nprocs=2,join=True)
                rows=[json.loads((Path(temp)/f'{rank}.json').read_text()) for rank in range(2)]
                self.assertTrue(all(row['refused'] and row['committed']==0 for row in rows))
                self.assertTrue(all(row['forward_calls']==(4 if mode=='repeat' else 0) for row in rows))


if __name__=='__main__': unittest.main()
