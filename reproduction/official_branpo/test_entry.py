import copy
import os
from pathlib import Path
import unittest
from .entry import actor_patch,iteration_patch,actor_batch_patch,prepare_actor_metadata,ray_runtime_env,ray_init_settings,sanitize_nvidia_visibility,prepare_policy_loss_schema,validate_plan,ROOT,VERSIONS
from .contract import training_overrides,UPSTREAM,DATA_SHA,REVISIONS,job_name

class EntryTest(unittest.TestCase):
    def test_observer_insertion_exact_and_replay_rejected(self):
        s='before\n                self.actor_optimizer.step()\nafter\n';out,extra=actor_patch(s)
        self.assertEqual(out.replace(extra,''),s)
        with self.assertRaises(AssertionError):actor_patch(out)
        self.assertLess(out.index('self.actor_optimizer.step()'),out.index('observe_adam('))
        with self.assertRaises(AssertionError):actor_patch(s+s)
        with self.assertRaises(AssertionError):actor_patch('unrecognized method')
        t='                logger.log(data=metrics, step=self.global_steps)\n';out,extra=iteration_patch(t)
        self.assertEqual(out.replace(extra,''),t)
        self.assertLess(out.index('logger.log('),out.index('observe_iteration('))
    def test_ray_environment_preserves_official_values_and_identity(self):
        original={'env_vars':{'VLLM_USE_V1':'1','NCCL_CUMEM_ENABLE':'0'}};before=copy.deepcopy(original)
        env={k:k+' value' for k in ['PYTHONPATH','BRANPO_AUDIT_DIR','SLURM_JOB_ID','SLURM_JOB_NAME']}
        out=ray_runtime_env(original,env)
        self.assertEqual(original,before)
        self.assertEqual(out['env_vars'],dict(original['env_vars'],**env,ROCR_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES=''))
        with self.assertRaises(AssertionError):ray_runtime_env(original,{})

    def test_nvidia_scheduler_mask_is_preserved_and_amd_conflicts_removed(self):
        env={'CUDA_VISIBLE_DEVICES':'4,3,2,1,0,7,6,5','ROCR_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HIP_VISIBLE_DEVICES':'wrong','OTHER':'keep'}
        removed=sanitize_nvidia_visibility(env,['NVIDIA RTX 3090']*8)
        self.assertEqual(env,{'CUDA_VISIBLE_DEVICES':'4,3,2,1,0,7,6,5','OTHER':'keep'})
        self.assertEqual(removed['HIP_VISIBLE_DEVICES'],'wrong')
        with self.assertRaises(AssertionError):sanitize_nvidia_visibility(env,['AMD GPU']*8)

    def test_only_dynamic_policy_loss_schema_opened_without_value_changes(self):
        from omegaconf import OmegaConf
        cfg=OmegaConf.create({'algorithm':{'rollout_correction':{'bypass_mode':True,'use_policy_gradient':False}},'actor_rollout_ref':{'actor':{'policy_loss':{'loss_mode':'vanilla'}}}});OmegaConf.set_struct(cfg,True)
        before=OmegaConf.to_container(cfg,resolve=True);prepare_policy_loss_schema(cfg)
        self.assertEqual(OmegaConf.to_container(cfg,resolve=True),before)
        cfg.actor_rollout_ref.actor.policy_loss.rollout_correction=cfg.algorithm.rollout_correction
        with self.assertRaises(Exception):cfg.actor_rollout_ref.actor.unexpected=1

    def test_optional_ray_initialization_matches_official_driver(self):
        from types import SimpleNamespace
        self.assertEqual(ray_init_settings(SimpleNamespace()),{})
        self.assertEqual(ray_init_settings(SimpleNamespace(ray_init={'num_cpus':8,'address':None})),{'num_cpus':8})

    def test_actor_metadata_preserves_config_and_existing_identity(self):
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{'temperature':.95}}})
        before=OmegaConf.to_container(cfg,resolve=True)
        batch=SimpleNamespace(meta_info={'batch_id':'frozen','other':[1,2]})
        prepare_actor_metadata(batch,cfg)
        self.assertEqual(batch.meta_info,{'batch_id':'frozen','other':[1,2],'temperature':.95})
        self.assertEqual(OmegaConf.to_container(cfg,resolve=True),before)
        prepare_actor_metadata(batch,cfg)
        batch.meta_info['temperature']=1.0
        with self.assertRaises(AssertionError):prepare_actor_metadata(batch,cfg)
        for temperature in [0,-1,float('nan'),float('inf'),'0.95']:
            cfg.actor_rollout_ref.rollout.temperature=temperature
            with self.assertRaises(AssertionError):prepare_actor_metadata(SimpleNamespace(meta_info={}),cfg)

    def test_only_actor_metadata_bridge_added_to_branch_trainer(self):
        anchor='                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()\n'
        original='before\n'+anchor+'after\n';patched,extra=actor_batch_patch(original)
        self.assertEqual(patched.replace(extra,''),original)
        with self.assertRaises(AssertionError):actor_batch_patch(patched)
        with self.assertRaises(AssertionError):actor_batch_patch(anchor+anchor)

    def test_frozen_science_allocation_and_size_identity(self):
        up=Path(os.environ.get('BRANPO_UPSTREAM','../official-branpo'))
        script=(up/'agent/search/start_branpo_trainer.sh').read_text()
        for size in ['3b','7b']:
            plan=dict(protocol='official_branpo_common_initialization_v1',size=size,
                job=job_name(size),root=ROOT+'/runs/'+size+('-v6' if size=='3b' else '-v5'),upstream_commit=UPSTREAM,
                model_revision=REVISIONS[size],versions=VERSIONS,data_manifest_sha256=DATA_SHA,
                expected_actual_iterations=200,upstream_labels=[101,300],gpus=8,no_auto_retry=True,
                no_additional_sft=True,model='/same/init',allocation=dict(gpus=8,cpus=64,memory_gib=600),
                minimum_gpu_memory_mib=44000,
                partition='ADA6000,L40S',node=None)
            plan['prior_attempt']=dict(job_id='139131' if size=='3b' else '139132',scientific_updates=0,raw_policy_tokens=None,status='FAILED_AFTER_FIRST_ROLLOUT_MISSING_ACTOR_TEMPERATURE_METADATA_BEFORE_ADAM')
            if size=='3b':plan['prior_attempt']['raw_policy_tokens']=None
            plan['overrides']=training_overrides(script,size,plan['model'],plan['root'])
            validate_plan(plan,script)
            for key,value in [('gpus',4),('model_revision','wrong'),('no_additional_sft',False)]:
                bad=copy.deepcopy(plan);bad[key]=value
                with self.assertRaises(AssertionError):validate_plan(bad,script)
            bad=copy.deepcopy(plan);bad['overrides']=[v.replace('rollout.n=4','rollout.n=2') for v in plan['overrides']]
            with self.assertRaises(AssertionError):validate_plan(bad,script)

if __name__=='__main__':unittest.main()
