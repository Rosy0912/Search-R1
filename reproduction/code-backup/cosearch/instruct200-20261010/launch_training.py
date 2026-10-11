"""Native trainer phases within one owned allocation; no Slurm allocation here."""
import argparse,json,os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['short5','continue200','final_eval','validation5']);args=parser.parse_args()
    import ray
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    import main_co_search_ppo as native
    root=Path(os.environ['COSEARCH_RUN_ROOT'])
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Owned Slurm allocation required')
    overrides=json.loads((root/'author-cli.json').read_text())
    resume=5 if args.phase in ('continue200','validation5') else 200 if args.phase=='final_eval' else None
    recovery_plan=json.loads((root/'recovery-plan.json').read_text()) if (root/'recovery-plan.json').exists() else None
    if recovery_plan:
        if args.phase=='continue200':resume=int(recovery_plan['step'])
        # Extra checkpoint10 remains historical; no extra scheduled validation.
    if resume is not None:
        overrides.extend(['trainer.resume_mode=resume_path','++trainer.resume_from_path='+str(root/'checkpoints'/('global_step_'+str(resume)))])
    if args.phase in ('final_eval','validation5'):
        if args.phase=='final_eval':overrides.append('data.val_files=['+str(root/'data/final.parquet')+']')
        overrides.append('++trainer.val_only=True')
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
                    common_config=root/"common-validation"/(str(self.global_steps)+".config.json")
                    if common_config.exists():
                        import hashlib
                        metadata=json.loads(common_config.read_text())
                        metadata["checkpoint_receipt_sha256"]=hashlib.sha256((root/"checkpoints"/("global_step_"+str(self.global_steps))/"full-sha-receipt.json").read_bytes()).hexdigest()
                        common_config.write_text(json.dumps(metadata,indent=2)+"\n")
                    from archive_checkpoint import request,allow_cleanup
                    request(root,self.global_steps)
                    if recovery_plan and self.global_steps==10:
                        from verify_updates import verify
                        from run_allocation import recovery
                        recovery(5)
                        receipt=verify(root/'evidence',10,os.environ['SLURM_JOB_ID'],prior_job=recovery_plan['source_job'],prior_through=5)
                        samples=[json.loads(line) for line in (root/'gpu-memory.jsonl').read_text().splitlines() if json.loads(line)['phase']=='continue200']
                        if not samples:raise RuntimeError('Missing recovery GPU memory evidence')
                        headroom_pass=not any(d['used_mib']>0.95*d['total_mib'] for sample in samples for d in sample['devices'])
                        events=[json.loads(line) for p in (root/'evidence').glob('main-*.jsonl') for line in p.read_text().splitlines()]
                        updates={(e['outer'],e['rank']):e for e in events if e['kind']=='update' and str(e['job'])==os.environ['SLURM_JOB_ID']}
                        identities={e['rank']:e for e in events if e['kind']=='initialized' and str(e['job'])==os.environ['SLURM_JOB_ID']}
                        for rank in range(4):
                            growth=updates[(10,rank)]['metrics']['perf/max_memory_reserved_gb']-updates[(9,rank)]['metrics']['perf/max_memory_reserved_gb']
                            if growth>0.05*identities[rank]['total_memory_bytes']/(1024**3):raise RuntimeError('Recovery short-run GPU memory growth')
                        receipt['gpu_memory_stability_still_requires_review']=not headroom_pass
                        receipt['actual_checkpoint5_recovery_pass']=True
                        receipt['checkpoint_resume_still_requires_verification']=False
                        receipt['gpu_memory_headroom_pass']=headroom_pass
                        (root/'short-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    if self.global_steps not in (5,200):allow_cleanup(root,self.global_steps)
                    return result
                def _validate(self):
                    from common_em_runtime import begin_validation,verify_validation
                    common_folder="common-final-evaluation" if args.phase=="final_eval" else "common-historical-validation" if args.phase=="validation5" else "common-validation"
                    begin_validation(self.global_steps,common_folder,self.config)
                    metrics=super()._validate()
                    common_receipt=verify_validation(self.global_steps,common_folder,51713 if args.phase=="final_eval" else 512)
                    metrics.update({"common/em":common_receipt["em"],"common/f1":common_receipt["f1"],"common/correct":common_receipt["correct"]})
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
                        if resume==5 and args.phase!='validation5':
                            from archive_checkpoint import allow_cleanup
                            allow_cleanup(root,5)
                    return result
                def fit(self):
                    assert len(self.train_dataset)==169028
                    assert len(self.val_dataset)==(51713 if args.phase=='final_eval' else 512)
                    assert self.total_training_steps==200
                    from runtime_fixes import install_controller
                    install_controller(self,original)
                    if args.phase in ('final_eval','validation5'):
                        # Never enter the PPO fit loop; its val_only flag depends on val_before_train.
                        self.global_steps=0
                        self._load_checkpoint()
                        assert self.global_steps==resume
                        import phase_sampling
                        metrics=phase_sampling.json_metadata(self._validate())
                        from verify_updates import finite
                        if not metrics or not finite(metrics):raise RuntimeError('Invalid final evaluation metrics')
                        (root/(args.phase+'-metrics.json')).write_text(json.dumps(metrics,allow_nan=False,indent=2)+'\n')
                        return
                    # Workers keep the200 horizon. Only this phase's stopping boundary changes.
                    if args.phase=='short5':self.total_training_steps=5
                    return super().fit()
            native.CoSearchRayTrainer=Trainer
            try:return super().run(c)
            finally:native.CoSearchRayTrainer=original
    env={'worker_process_setup_hook':'runtime_audit.install','env_vars':{
        'COSEARCH_AUDIT_ROOT':os.environ['COSEARCH_AUDIT_ROOT'],
        'COSEARCH_RUN_ROOT':str(root),
        'SLURM_JOB_ID':os.environ['SLURM_JOB_ID'],
        'PYTHONPATH':os.environ['PYTHONPATH'],
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        # All four ranks run on one node; avoid unaddressed sysfs entries.
        'GLOO_SOCKET_IFNAME':'lo','NCCL_SOCKET_IFNAME':'lo',
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES':'1',
        'VLLM_RPC_BASE_PATH':os.environ['VLLM_RPC_BASE_PATH'],
        'COSEARCH_LEASED_UUIDS':os.environ['COSEARCH_LEASED_UUIDS'],
        'COSEARCH_LEASED_UUID_TO_INDEX':os.environ['COSEARCH_LEASED_UUID_TO_INDEX']}}
    OmegaConf.update(config,'ray_kwargs.ray_init.runtime_env',env,force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init.address','local',force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init.num_gpus',4,force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init.num_cpus',48,force_add=True)
    OmegaConf.update(config,'ray_kwargs.ray_init._temp_dir','/tmp/r200-'+os.environ['SLURM_JOB_ID']+'-'+args.phase,force_add=True)
    (root/('resolved-'+args.phase+'.yaml')).write_text(OmegaConf.to_yaml(config,resolve=True))
    try:
        native.run_dual_agent_ppo(config,task_runner_class=ray.remote(num_cpus=1)(Runner))
    finally:
        ray.shutdown()
if __name__=='__main__':main()
