import copy
import unittest
from search_r1.baselines.fresh200_contract import outer_contract, validate_rank_updates, scheduler_may_step, validate_fresh_config


class Fresh200BoundaryContract(unittest.TestCase):
    def test_boundary_49_50_200(self):
        for outer,mini,writes in [(49,128,20),(50,256,10),(200,256,10)]:
            c=outer_contract(outer)
            self.assertEqual((c['global_mini_batch_size'],c['rank_mini_batch_size'],c['expected_optimizer_steps']), (mini,mini//8,writes))
        self.assertEqual(sum(outer_contract(n)['expected_optimizer_steps'] for n in range(1,201)),2490)
        for invalid in (0,201,True,49.0):
            with self.assertRaises(ValueError): outer_contract(invalid)

    def events(self, outer, status='COMPLETE', writes=None):
        expected=outer_contract(outer)['expected_optimizer_steps']
        return [dict(rank=rank, actual_outer=outer,attempt_id=1,status=status,update_complete=status=='COMPLETE',
                     optimizer_steps=expected if writes is None else writes,expected_optimizer_steps=expected,
                     optimizer_opportunities_total=expected,processed_opportunities=expected if writes is None else writes,
                     skipped_opportunities=0,unprocessed_opportunities=0 if writes is None else expected-writes,
                     attempt_complete=status=='COMPLETE',postabort=status=='ABORTED') for rank in range(8)]

    def test_partial_never_advances_scheduler(self):
        events=self.events(50,'ABORTED',3)
        self.assertFalse(validate_rank_updates(events,50))
        self.assertTrue(all(not scheduler_may_step(e) for e in events))
        events=self.events(49)
        self.assertTrue(validate_rank_updates(events,49))
        self.assertTrue(all(scheduler_may_step(e) for e in events))
        events[7].update(status='ABORTED',postabort=True,attempt_complete=False,update_complete=False,optimizer_steps=19,processed_opportunities=19,unprocessed_opportunities=1)
        with self.assertRaises(ValueError):validate_rank_updates(events,49)

    def test_complete_rejects_missing_writes_and_ranks(self):
        for events in (self.events(50,writes=9),self.events(50)[:-1]):
            with self.assertRaises(ValueError): validate_rank_updates(events,50)

    def test_greedy_evaluation_cannot_bind_training(self):
        from types import SimpleNamespace
        from search_r1.baselines.fresh200_contract import verify_sampling_policy_binding
        with self.assertRaises(ValueError):
            verify_sampling_policy_binding(SimpleNamespace(meta_info={'fresh200_evaluation':True}),{})

    def test_missing_calibration_cannot_admit(self):
        from search_r1.baselines.fresh200_contract import verify_policy_safety_artifact
        with self.assertRaises(ValueError):
            verify_policy_safety_artifact(dict(calibration_manifest='/nonexistent/fresh200-calibration.json',calibration_sha256='0'*64))

    def test_warmup_uses_57_actual_updates(self):
        from search_r1.baselines.fresh200_contract import fresh200_lr_factor
        for completed,factor in [(0,1/57),(1,2/57),(56,1.),(57,1.),(200,1.)]:
            self.assertEqual(fresh200_lr_factor(completed),factor)

    def test_fresh_restore_denied(self):
        actor=dict(fresh200_enabled=True,policy_safety=dict(enabled=True,threshold_origin='user_fixed',target_old_policy_kl=.2,negative_tail_limit=.2,skip_on_policy_guard=True),entropy_coeff=0.,use_kl_loss=False,kl_loss_coef=0.,grad_clip=1.,clip_ratio=.2,
            loss_reduction='sequence_mean',ppo_mini_batch_size=128,ppo_micro_batch_size=16,ppo_epochs=1,use_dynamic_bsz=False,
            optim=dict(num_warmup_steps=57,lr=1e-6,lr_warmup_steps_ratio=0.,warmup_style='constant'))
        config=dict(trainer=dict(fresh200_enabled=True,fresh200_gate_enabled=True,e1_gate_enabled=False,total_training_steps=201,
            acceptance_step=5,save_freq=50,n_gpus_per_node=8,nnodes=1,smoke_only=False,fresh200_gate_receipt='/tmp/new-receipt',fresh200_official_model='Qwen/Qwen2.5-3B-Instruct'),
            actor_rollout_ref=dict(actor=actor,model=dict(path='/verified/official/model',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'),rollout=dict(n_agent=5,tensor_model_parallel_size=1,log_prob_micro_batch_size=32),ref=dict(log_prob_micro_batch_size=32)),
            max_turns=4,force_final_answer=False,
            data=dict(train_batch_size=512,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500),baselines=dict(method='bapo',holdout=dict(manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'),bapo=dict(stage_validation_size=64)),algorithm=dict(adv_estimator='grpo',kl_ctrl=dict(kl_coef=0.)))
        self.assertEqual(validate_fresh_config(config),(200,57))
        for key,value in [('max_turns',5),('force_final_answer',True)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):validate_fresh_config(bad)
        for key in ('max_start_length','max_prompt_length','max_response_length','max_obs_length'):
            bad=copy.deepcopy(config);bad['data'][key]+=1
            with self.assertRaises(ValueError):validate_fresh_config(bad)
        from search_r1.baselines.fresh200_contract import critical_source_hashes
        sources=critical_source_hashes()
        for path in ('verl/trainer/config/ppo_bapo_fresh200.yaml','verl/trainer/config/ppo_baselines.yaml',
                     'verl/trainer/config/ppo_trainer.yaml','verl/utils/dataset/rl_dataset.py',
                     'verl/utils/tokenizer.py','verl/utils/torch_functional.py'):
            self.assertIn(path,sources)
        for key in ('resume_full_state','resume_permission','resume_verify_only','resume_from','resume_path'):
            bad=copy.deepcopy(config);bad['trainer'][key]='old-checkpoint'
            with self.assertRaises(ValueError):validate_fresh_config(bad)
        bad=copy.deepcopy(config);bad['trainer']['e1_gate_enabled']=True
        with self.assertRaises(ValueError):validate_fresh_config(bad)
        bad=copy.deepcopy(config);bad['actor_rollout_ref']['actor']['fresh200_diagnostic_only']=True
        with self.assertRaises(ValueError):validate_fresh_config(bad)
        bad=copy.deepcopy(config);bad['trainer']['fresh200_diagnostic_role']='discovery'
        with self.assertRaises(ValueError):validate_fresh_config(bad)
        bad=copy.deepcopy(config);bad['actor_rollout_ref']['model']['override_config']={'rope_theta':10.}
        with self.assertRaises(ValueError):validate_fresh_config(bad)
