"""Native trainer phases within one owned allocation; no Slurm allocation here."""
import argparse,json,os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['short5','continue200','final_eval']);args=parser.parse_args()
    import ray
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    import main_co_search_ppo as native
    root=Path(os.environ['COSEARCH_RUN_ROOT'])
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Owned Slurm allocation required')
    overrides=json.loads((root/'author-cli.json').read_text())
    resume=5 if args.phase=='continue200' else 200 if args.phase=='final_eval' else None
    if resume is not None:
        overrides.extend(['trainer.resume_mode=resume_path','++trainer.resume_from_path='+str(root/'checkpoints'/('global_step_'+str(resume)))])
    if args.phase=='final_eval':
        overrides.extend(['data.val_files=['+str(root/'data/final.parquet')+']','++trainer.val_only=True'])
    with initialize_config_dir(config_dir=str(root/'repo/config'),version_base=None):
        config=compose(config_name='co_search_trainer',overrides=overrides)
    if config.trainer.n_gpus_per_node!=4 or not config.reranker_actor_rollout_ref.trainable:raise RuntimeError('Invalid joint protocol')
    class Runner(native.DualAgentTaskRunner):
        def run(self,c):
            original=native.CoSearchRayTrainer
            class Trainer(original):
                def fit(self):
                    assert len(self.train_dataset)==169028
                    assert len(self.val_dataset)==(51713 if args.phase=='final_eval' else 512)
                    assert self.total_training_steps==200
                    from runtime_fixes import install_controller
                    install_controller(self,original)
                    # Workers keep the200 horizon. Only this phase's stopping boundary changes.
                    if args.phase=='short5':self.total_training_steps=5
                    return super().fit()
            native.CoSearchRayTrainer=Trainer
            try:return super().run(c)
            finally:native.CoSearchRayTrainer=original
    env={'worker_process_setup_hook':'runtime_audit.install','env_vars':{
        'COSEARCH_AUDIT_ROOT':os.environ['COSEARCH_AUDIT_ROOT'],
        'SLURM_JOB_ID':os.environ['SLURM_JOB_ID'],
        'PYTHONPATH':os.environ['PYTHONPATH'],
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'COSEARCH_LEASED_UUIDS':os.environ['COSEARCH_LEASED_UUIDS'],
        'COSEARCH_LEASED_UUID_TO_INDEX':os.environ['COSEARCH_LEASED_UUID_TO_INDEX']}}
    config.ray_kwargs.ray_init.runtime_env=OmegaConf.create(env)
    (root/('resolved-'+args.phase+'.yaml')).write_text(OmegaConf.to_yaml(config,resolve=True))
    native.run_dual_agent_ppo(config,task_runner_class=ray.remote(num_cpus=1)(Runner))
if __name__=='__main__':main()
