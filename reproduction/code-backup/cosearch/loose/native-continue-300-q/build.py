from pathlib import Path
import hashlib
base=Path('work/author-native-phase-aware-m');out=Path('work/native-continue-300-q')
s=(base/'launch_observed.py').read_text();assert hashlib.sha256(s.encode()).hexdigest()=='956f763d018d05fefedab1407b04bf04d47337d7e663853c3dd63c831d05015f'
s=s.replace("    if reload_only:\n        from omegaconf import open_dict", "    from continuation import target, source_checkpoint\n    continue_to=target()\n    if reload_only or continue_to:\n        from omegaconf import open_dict",1)
s=s.replace("checkpoint=ROOT/'runs'/source_id/'checkpoints/global_step_5'", "checkpoint=ROOT/'runs'/source_id/'checkpoints/global_step_5' if reload_only else source_checkpoint()[2]",1)
s=s.replace("            config.trainer.del_local_ckpt_after_load=False", "            config.trainer.del_local_ckpt_after_load=False\n            if continue_to:\n                config.trainer.test_freq=50\n                config.trainer.save_freq=50",1)
s=s.replace("    config,reload_only=compose_launch_config()", "    config,reload_only=compose_launch_config()\n    from continuation import target, after_original_load\n    continue_to=target()",1)
s=s.replace('fresh_five=not reload_only,reload_only=reload_only', 'fresh_five=not reload_only and not continue_to,continue_to=continue_to,reload_only=reload_only',1)
s=s.replace('            class FiveTrainer(original_class):\n                def fit(self):', '            class FiveTrainer(original_class):\n                def _load_checkpoint(self):\n                    result=super()._load_checkpoint()\n                    if continue_to:after_original_load(self,write)\n                    return result\n                def fit(self):',1)
s=s.replace('                    self.total_training_steps=5','                    assert self.total_training_steps >= (continue_to or 5)\n                    self.total_training_steps=continue_to or 5',1)
(out/'launch_observed.py').write_text(s)
s=(base/'native_job.py').read_text();assert hashlib.sha256(s.encode()).hexdigest()=='2848ac1b2d20ef0fdeec8854f4210a1696fba5f6848e68746f71daea240653e6'
s=s.replace("    if os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')=='1':", "    continue_to=os.environ.get('COSEARCH_NATIVE_CONTINUE_TO')\n    if continue_to is not None:\n        assert continue_to=='300' and os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')!='1'\n    if os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')=='1' or continue_to:",1)
s=s.replace("        assert (source/'checkpoints/global_step_5').is_dir()", "        step=int(os.environ.get('COSEARCH_NATIVE_RESUME_STEP','5')) if continue_to else 5\n        assert 0<step<300\n        checkpoint=source/'checkpoints'/f'global_step_{step}'\n        assert checkpoint.is_dir()\n        if continue_to:assert (checkpoint/'data.pt').is_file()",1)
(out/'native_job.py').write_text(s)
