"""Actual opt-in source gate/native metadata fixtures; no production acceptance."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast
import unittest

class Node(dict):
    def __getattr__(self,key):return self[key]
    def __setattr__(self,key,value):self[key]=value
def node(d):return Node({k:node(v) if isinstance(v,dict) else v for k,v in d.items()})
source=Path(__file__).parent/'search_r1/baselines/controller.py'
parsed=ast.parse(source.read_text());selected=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in {'option','validate_behavior_mu_config'}]
assert len(selected)==2
scope={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
validate=scope['validate_behavior_mu_config']
def candidate():
    return node(dict(baselines=dict(method='tree_grpo',tree=dict(roots=2,leaves=3,
        token_preservation=dict(enabled=True,protocol='tree_sampled_prefix_legacy_semantics_v1'),
        behavior_mu=dict(enabled=True,protocol='tree_behavior_mu_centered_ppo_v1'))),
        actor_rollout_ref=dict(actor=dict(policy_safety=dict(enabled=True,sampling_probability_correction=dict(
        enabled=True,protocol='tree_behavior_mu_centered_ppo_v1',initial_actor_repeat_max_abs_delta=None)),
        ppo_mini_batch_size=384,ppo_micro_batch_size=16,ppo_epochs=1,optim=dict(lr=1e-6)),
        rollout=dict(tree_raw_emission=True,temperature=1.,top_p=1.,top_k=-1)),
        data=dict(train_batch_size=640,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500),
        trainer=dict(n_gpus_per_node=8,nnodes=1),max_turns=4,force_final_answer=False))
class BehaviorMuConfigTests(unittest.TestCase):
    def test_default_off_preserves_legacy_configuration(self):
        c=candidate();c.baselines.tree.behavior_mu.enabled=False
        c.actor_rollout_ref.actor.policy_safety.sampling_probability_correction.enabled=False
        c.baselines.method='grpo';c.data.train_batch_size=512;validate(c)
    def test_source_protocol_does_not_accept_unset_actor_threshold(self):
        c=candidate();validate(c)
        self.assertIsNone(c.actor_rollout_ref.actor.policy_safety.sampling_probability_correction.initial_actor_repeat_max_abs_delta)
    def test_optins_reject_mismatch_and_boolean_coercion(self):
        for who,value in [('collector',False),('collector','true'),('actor',None)]:
            with self.subTest(who=who,value=value):
                c=candidate();target=c.baselines.tree.behavior_mu if who=='collector' else c.actor_rollout_ref.actor.policy_safety.sampling_probability_correction
                target.enabled=value
                with self.assertRaisesRegex(ValueError,'boolean opt-ins'):validate(c)
    def test_raw_guard_tree_and_protocol_are_required(self):
        for change in ['method','guard','capture','raw','protocol']:
            with self.subTest(change=change):
                c=candidate()
                if change=='method':c.baselines.method='gigpo'
                elif change=='guard':c.actor_rollout_ref.actor.policy_safety.enabled=False
                elif change=='capture':c.actor_rollout_ref.rollout.tree_raw_emission=False
                elif change=='raw':c.baselines.tree.token_preservation.enabled=False
                else:c.baselines.tree.behavior_mu.protocol='unreviewed'
                with self.assertRaisesRegex(ValueError,'raw-token/native-emission'):validate(c)
    def test_recovery_protocol_drift_is_refused(self):
        for change in ['batch','micro','lr','length','turns','forced','world']:
            with self.subTest(change=change):
                c=candidate()
                if change=='batch':c.data.train_batch_size=512
                elif change=='micro':c.actor_rollout_ref.actor.ppo_micro_batch_size=64
                elif change=='lr':c.actor_rollout_ref.actor.optim.lr=1e-5
                elif change=='length':c.data.max_prompt_length=8192
                elif change=='turns':c.max_turns=5
                elif change=='forced':c.force_final_answer=True
                else:c.trainer.nnodes=2
                with self.assertRaisesRegex(ValueError,'authorized Tree recovery'):validate(c)
    def test_plain_sampler_support_is_preserved(self):
        for name,value in [('temperature',.8),('top_p',.95),('top_k',5),('min_p',.02),('repetition_penalty',1.1)]:
            with self.subTest(name=name):
                c=candidate();c.actor_rollout_ref.rollout[name]=value
                with self.assertRaisesRegex(ValueError,'temperature-one sampling'):validate(c)
class NativeProcessorCaptureTests(unittest.TestCase):
    def params(self):
        return NS(n=1,best_of=1,_real_n=1,temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,
                  repetition_penalty=1.,min_tokens=0,max_tokens=500,ignore_eos=False,stop=None,stop_token_ids=[],
                  include_stop_str_in_output=False,logprobs=1,prompt_logprobs=None,seed=None,logits_processors=None,
                  truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
    def capture(self,params):
        from search_r1.baselines.tree_raw_tokens import capture_native_emissions
        env={k:'fixture-only' for k in ('SEARCH_R1_BASELINE_COMMIT','SEARCH_R1_BASELINE_CONFIG_SHA256','SEARCH_R1_BASELINE_DATA_IDENTITY','SEARCH_R1_MODEL_IDENTITY','SEARCH_R1_BASELINE_RUN_ID','SLURM_JOB_ID')}
        r=NS(request_id='cpu0',finished=True,prompt_token_ids=[8],outputs=[NS(token_ids=[9],logprobs=[{9:NS(logprob=-.5)}],finish_reason='length',stop_reason=None)])
        with patch.dict('os.environ',env):return capture_native_emissions([r],params,'cpu0',0)[0]
    def test_actual_none_processor_field_is_captured(self):
        r=self.capture(self.params());self.assertEqual(len(r['sampling_params']),24)
        self.assertIsNone(r['sampling_params']['logits_processors'])
        self.assertEqual(r['raw_selected_logprobs'],[-.5])
    def test_missing_or_non_none_processor_does_not_get_default(self):
        p=self.params();del p.logits_processors
        with self.assertRaisesRegex(ValueError,'Explicit actual logits_processors metadata'):self.capture(p)
        for processors in [[],[object()]]:
            p=self.params();p.logits_processors=processors
            with self.subTest(processors=bool(processors)),self.assertRaisesRegex(ValueError,'actual logits_processors=None'):self.capture(p)
if __name__=='__main__':unittest.main()
