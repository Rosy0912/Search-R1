import copy
import os
from pathlib import Path
import unittest
from .entry import actor_patch,iteration_patch,ray_runtime_env,ray_init_settings,sanitize_nvidia_visibility,validate_plan,ROOT,VERSIONS
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

    def test_optional_ray_initialization_matches_official_driver(self):
        from types import SimpleNamespace
        self.assertEqual(ray_init_settings(SimpleNamespace()),{})
        self.assertEqual(ray_init_settings(SimpleNamespace(ray_init={'num_cpus':8,'address':None})),{'num_cpus':8})

    def test_frozen_science_allocation_and_size_identity(self):
        up=Path(os.environ.get('BRANPO_UPSTREAM','../official-branpo'))
        script=(up/'agent/search/start_branpo_trainer.sh').read_text()
        for size in ['3b','7b']:
            plan=dict(protocol='official_branpo_common_initialization_v1',size=size,
                job=job_name(size),root=ROOT+'/runs/'+size+'-v3',upstream_commit=UPSTREAM,
                model_revision=REVISIONS[size],versions=VERSIONS,data_manifest_sha256=DATA_SHA,
                expected_actual_iterations=200,upstream_labels=[101,300],gpus=8,no_auto_retry=True,
                no_additional_sft=True,model='/same/init',allocation=dict(gpus=8,cpus=64,memory_gib=600),
                minimum_gpu_memory_mib=24000 if size=='3b' else 44000,
                partition='RTX3090' if size=='3b' else 'ADA6000',node='node2' if size=='3b' else None)
            plan['prior_attempt']=dict(job_id='139123' if size=='3b' else '139124',scientific_updates=0,raw_policy_tokens=0,status='FAILED_WORKER_INIT_NVIDIA_AMD_VISIBILITY_CONFLICT_BEFORE_ADAM' if size=='3b' else 'CANCELLED_NEVER_EXECUTED_KNOWN_GPU_VISIBILITY_CONFLICT')
            plan['overrides']=training_overrides(script,size,plan['model'],plan['root'])
            validate_plan(plan,script)
            for key,value in [('gpus',4),('model_revision','wrong'),('no_additional_sft',False)]:
                bad=copy.deepcopy(plan);bad[key]=value
                with self.assertRaises(AssertionError):validate_plan(bad,script)
            bad=copy.deepcopy(plan);bad['overrides']=[v.replace('rollout.n=4','rollout.n=2') for v in plan['overrides']]
            with self.assertRaises(AssertionError):validate_plan(bad,script)

if __name__=='__main__':unittest.main()
