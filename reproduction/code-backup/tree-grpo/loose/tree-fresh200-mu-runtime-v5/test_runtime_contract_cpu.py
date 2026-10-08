"""Tiny synthetic wrapper contract tests; no assets/GPU/full DataProto proof."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import calibration_contract as contract
from render_dispatch import render
from tokenizer_raw_preflight import inspect_row


class FreshMetadataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)/'tree-fresh200-9001';(self.run/'output').mkdir(parents=True)
        self.package=dict(code_commit='a'*40,common_data_identity='b'*64,
            official_model_path='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725',
            model_identity={'sha256':'c'*64})
        self.expected=contract.expected_runtime(self.run,self.package,'d'*64)
        workers=[dict(self.expected,rank=rank,world_size=8,
            training_protocol=contract.PROFILE,calibration_profile='fresh_base_zero_v1',
            optimizer_state_entries=0,optimizer_step_values=[],scheduler_step=0,
            optimizer_committed=0,scheduler_advanced=False,checkpoint_recovery_environment_absent=True,
            configured_model_path=self.package['official_model_path'],expected_model_path=self.package['official_model_path'],
            complete_model_parameter_bytes_verified=False) for rank in range(8)]
        self.record=dict(self.expected,status='FRESH_OFFICIAL_INITIAL_STATE_OBSERVED',training_protocol=contract.PROFILE,
            calibration_profile='fresh_base_zero_v1',initialized_global_steps=0,parent_completed_step=0,parent_cursor=0,
            driver_completed_outer=0,checkpoint_loaded=False,workers=workers)

    def validate(self, record=None):
        path=self.run/'output/fresh-initialization.json'
        path.write_text(json.dumps(self.record if record is None else record))
        return contract.validate_fresh_start(self.run,self.package,'d'*64)

    def test_all_eight_direct_fresh_metadata(self):
        self.assertEqual(self.validate()['workers'],8)

    def test_old_model_path_is_not_fresh(self):
        record=copy.deepcopy(self.record);record['workers'][3]['configured_model_path']='/old/HF110'
        with self.assertRaisesRegex(ValueError,'official'):self.validate(record)

    def test_nonempty_adam_refuses_even_with_empty_step_claim(self):
        record=copy.deepcopy(self.record);record['workers'][5]['optimizer_state_entries']=1
        with self.assertRaisesRegex(ValueError,'Adam'):self.validate(record)

    def test_scheduler_and_rank_coverage_refuse(self):
        record=copy.deepcopy(self.record);record['workers'][7]['scheduler_step']=110
        with self.assertRaisesRegex(ValueError,'Adam'):self.validate(record)
        record=copy.deepcopy(self.record);record['workers'].pop()
        with self.assertRaisesRegex(ValueError,'eight'):self.validate(record)

    def test_real_receipt_and_collector_env_names_do_not_alias(self):
        mapped=contract.expected_mu_provenance(self.expected)
        self.assertEqual(set(mapped),{'SEARCH_R1_BASELINE_COMMIT','SEARCH_R1_BASELINE_CONFIG_SHA256',
            'SEARCH_R1_BASELINE_DATA_IDENTITY','SEARCH_R1_MODEL_IDENTITY','SEARCH_R1_BASELINE_RUN_ID','SLURM_JOB_ID'})
        self.assertEqual(mapped['SLURM_JOB_ID'],'9001');self.assertNotIn('job_id',mapped)

    def test_missing_checkpoint_evidence_is_not_default_false(self):
        record=copy.deepcopy(self.record);del record['checkpoint_loaded']
        with self.assertRaises(KeyError):self.validate(record)

    def test_recovery_env_refuses(self):
        contract.forbid_recovery_environment({'TREE_FRESH_INITIALIZATION_ONLY':'1'})
        with self.assertRaises(ValueError):contract.forbid_recovery_environment({'TREE_RECOVERY_MANIFEST':'/old'})

    def test_renderer_has_no_submit_and_rejects_non_task_path(self):
        text=render('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates/fresh-test','a'*64,'b'*64)
        self.assertIn('#SBATCH --gres=gpu:8',text);self.assertIn('h100-9',text)
        self.assertNotIn('sbatch ',text);self.assertNotIn('TREE_RECOVERY_',text)
        with self.assertRaises(ValueError):render('/other-cluster','a'*64,'b'*64)


class ActualReadyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.expected=dict(code_commit='a'*40,config_sha256='b'*64,data_identity='c'*64,
                           model_identity='d'*64,job_id='9001',run_id='tree-fresh200-9001')
        # Exact source producer keys: no invented training_protocol/fresh flag,
        # top-level scheduler, run_id or actual_completed_outer.
        self.ready=dict({k:v for k,v in self.expected.items() if k!='run_id'},
            status='FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES',calibration_profile='fresh_base_zero_v1',
            initialized_global_steps=0,behavior_mu_corrected_profile=True,
            original_sampler_actor_probability_gate_accepted=False,parent_completed_outer=0,
            observed_outer=1,new_completed_outer=0,optimizer_committed=0,scheduler_advanced=False,
            sampler_cursor_before_outer=0,sampler_cursor_after_observation=512)
        self.worker=dict(self.expected,rank=2,world_size=8,calibration_profile='fresh_base_zero_v1',
            parent_completed_outer=0,observed_outer=1,optimizer_state_entries=0,restored_optimizer_steps=[],
            optimizer_committed=0,scheduler_step=0,scheduler_advanced=False,
            fresh_post_monitor_state=dict(calibration_profile='fresh_base_zero_v1',restored_optimizer_steps=[],
                                         optimizer_state_entries=0,scheduler_step=0))

    def test_actual_ready_fields_accept_without_fabricated_fields(self):
        contract.validate_ready_boundary(self.ready,self.expected)
        contract.validate_calibration_rank_state(self.worker,2,self.expected)

    def test_old_or_wrong_profile_ready_does_not_alias_fresh(self):
        ready=copy.deepcopy(self.ready);ready['status']='BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES'
        with self.assertRaisesRegex(ValueError,'status'):contract.validate_ready_boundary(ready,self.expected)
        ready=copy.deepcopy(self.ready);ready['sampler_cursor_after_observation']=640
        with self.assertRaisesRegex(ValueError,'cursor'):contract.validate_ready_boundary(ready,self.expected)

    def test_post_monitor_state_writes_or_wrong_source_refuse(self):
        worker=copy.deepcopy(self.worker);worker['fresh_post_monitor_state']['optimizer_state_entries']=1
        with self.assertRaisesRegex(ValueError,'changed actual'):contract.validate_calibration_rank_state(worker,2,self.expected)
        worker=copy.deepcopy(self.worker);worker['run_id']='old-run'
        with self.assertRaisesRegex(ValueError,'identity'):contract.validate_calibration_rank_state(worker,2,self.expected)


class HistoricalBoundaryOnlyTests(unittest.TestCase):
    def setUp(self):
        source=Path(__file__).resolve().parent.parent/'tree-behavior-mu-ppo/search_r1/baselines/tree_raw_tokens.py'
        spec=importlib.util.spec_from_file_location('original_raw_helper_fixture',source)
        self.helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.helper)
        self.record=dict(token_ids=[1,2],prompt_token_ids=[9],input_row=0,request_id='original-request',
            true_response_length=2,logprobs=[{'1':{'logprob':-.5}},{'2':{'logprob':-.25}}],
            finished=True,completion_index=0,finish_reason='length',stop_reason=None)
        self.params=dict(n=1,temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,presence_penalty=0.0,
            frequency_penalty=0.0,repetition_penalty=1.0,min_tokens=0,max_tokens=500,ignore_eos=False,
            stop=[],stop_token_ids=[],include_stop_str_in_output=False,logprobs=1,prompt_logprobs=None,seed=None)
        self.identity={'pad_token_id':151643,'eos_token_id':151645}

    def test_missing_historical_best_of_and_processors_stay_unknown_without_default(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,self.params)
        self.assertFalse(result['actual_native24_complete'])
        self.assertEqual(result['missing_actual_sampling_fields'],['best_of','_real_n','logits_processors',
            'truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids'])
        self.assertEqual(result['actual_best_of_status'],'UNKNOWN_NOT_RECORDED')
        self.assertTrue(all(value=='UNKNOWN_NOT_RECORDED' for value in result['actual_control_fields_status'].values()))
        self.assertFalse(result['actual_sampling_support_validated'])
        self.assertEqual(result['retained_tokens'],2)

    def test_recorded_historical_best_of_is_observed_without_repairing_distribution(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        for value, supported in ((1,True),(None,False),(2,False),(1.,False),(True,False)):
            actual=dict(self.params,best_of=value,_real_n=1,logits_processors=None,
                truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
            result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,actual)
            self.assertTrue(result['actual_native24_complete'])
            self.assertEqual(result['actual_best_of_status'],'RECORDED')
            self.assertEqual(result['actual_sampling_support_validated'],supported)
            self.assertEqual(result['retained_tokens'],2)

    def test_historical_new_controls_missing_or_nontrivial_stay_observations(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        base=dict(self.params,best_of=1,_real_n=1,logits_processors=None,
            truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
        for key,value in (('_real_n',2),('truncate_prompt_tokens',1),('guided_decoding',{'json':{}}),
                          ('logit_bias',{'151645':1.}),('allowed_token_ids',[151645])):
            for missing in (False,True):
                actual=dict(base)
                if missing:del actual[key]
                else:actual[key]=value
                result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,actual)
                self.assertFalse(result['actual_sampling_support_validated'])
                self.assertEqual(result['actual_control_fields_status'][key],
                                 'UNKNOWN_NOT_RECORDED' if missing else 'RECORDED')
                self.assertEqual(result['retained_tokens'],2)

    def test_unicode_intermediate_refusal_survives_metadata_gap(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return '\ufffd' if len(ids)==1 else '中'
            def encode(self,text,**kwargs):return [1,2]
        with self.assertRaisesRegex(ValueError,'UTF8'):
            inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,self.params)


if __name__=='__main__': unittest.main()
