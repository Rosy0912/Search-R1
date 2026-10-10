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
        overrides.extend(['data.val_files=['+str(root/'data/final.parquet')+']','++trainer.val_only=True','trainer.validation_data_dir='+str(root/'final-evaluation')])
    with initialize_config_dir(config_dir=str(root/'repo/config'),version_base=None):
        config=compose(config_name='co_search_trainer',overrides=overrides)
    if config.trainer.n_gpus_per_node!=4 or not config.reranker_actor_rollout_ref.trainable:raise RuntimeError('Invalid joint protocol')
    class Runner(native.DualAgentTaskRunner):
        def run(self,c):
            original=native.CoSearchRayTrainer
            class Trainer(original):
                def _save_checkpoint(self):
                    result=super()._save_checkpoint()
                    from verify_checkpoint import verify
                    verify(root,self.global_steps)
                    return result
                def _validate(self):
                    metrics=super()._validate()
                    from verify_evaluation import verify
                    folder=root/'final-evaluation' if args.phase=='final_eval' else root/'validation'
                    rows=51713 if args.phase=='final_eval' else 512
                    receipt=verify(folder/(str(self.global_steps)+'.jsonl'),rows,self.global_steps)
                    (root/('evaluation-'+args.phase+'-'+str(self.global_steps)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                    return metrics
                def _load_checkpoint(self):
                    result=super()._load_checkpoint()
                    if resume is not None:
                        import torch
                        from runtime_audit import digest
                        expected=torch.load(root/'checkpoints'/('global_step_'+str(resume))/'data.pt',weights_only=False)
                        actual=self.train_dataloader.state_dict()
                        if digest(expected)!=digest(actual):raise RuntimeError('Restored data progress differs')
                        if self.global_steps!=resume:raise RuntimeError('Restored outer step differs')
                        (root/('data-recovery-'+str(resume)+'.json')).write_text(json.dumps({'step':resume,'data_state_sha256':digest(actual),'equal':True})+'\n')
                    return result
                def fit(self):
                    assert len(self.train_dataset)==169028
                    assert len(self.val_dataset)==(51713 if args.phase=='final_eval' else 512)
                    assert self.total_training_steps==200
                    from runtime_fixes import install_controller
                    install_controller(self,original)
                    if args.phase=='final_eval':
                        # Never enter the PPO fit loop; its val_only flag depends on val_before_train.
                        self.global_steps=0
                        self._load_checkpoint()
                        assert self.global_steps==200
                        import phase_sampling
                        metrics=phase_sampling.json_metadata(self._validate())
                        from verify_updates import finite
                        if not metrics or not finite(metrics):raise RuntimeError('Invalid final evaluation metrics')
                        (root/'final-evaluation-metrics.json').write_text(json.dumps(metrics,allow_nan=False,indent=2)+'\n')
                        return
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
    OmegaConf.update(config,'ray_kwargs.ray_init.address','local',force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init.num_gpus',4,force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init.num_cpus',48,force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init._temp_dir','/data/user/jxu957/r200-'+os.environ['SLURM_JOB_ID']+'-'+args.phase,force_add=True)
    (root/('resolved-'+args.phase+'.yaml')).write_text(OmegaConf.to_yaml(config,resolve=True))
    try:
        native.run_dual_agent_ppo(config,task_runner_class=ray.remote(num_cpus=1)(Runner))
    finally:
        ray.shutdown()
if __name__=='__main__':main()
