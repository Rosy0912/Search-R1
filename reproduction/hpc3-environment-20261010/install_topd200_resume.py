"""Apply audited recovery changes only while no TOP-D job is active."""
import ast,json,pathlib,subprocess
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');ROOT=BASE/'runtime'
rt=ROOT/'tools/topd-instruct300-launch-v6/runtime'
assert 'topd' not in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower()
def change(p,old,new):
 s=p.read_text();assert s.count(old)==1,(p,old,s.count(old));s=s.replace(old,new);ast.parse(s);p.write_text(s)
change(rt/'train_entry.py','install(workers.ActorRolloutRefWorker)','install(workers.ActorRolloutRefWorker)\n from topd_resume_state import install as install_resume\n install_resume(workers.ActorRolloutRefWorker)')
p=ROOT/'tools/topd-instruct300-source-v1/verl/trainer/ppo/ray_trainer.py'
change(p,'        # start training loop\n','        from topd_resume_state import prepare_data\n        prepare_data(self)\n        # start training loop\n')
change(p,'            for batch_dict in self.train_dataloader:\n','            for batch_index, batch_dict in enumerate(self.train_dataloader):\n                if batch_index < self._topd_resume_batch: continue\n                self._topd_batch_index = batch_index\n')
change(p,'        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)','        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)\n        from topd_resume_state import save_driver\n        save_driver(self)')
change(rt/'r1_observation.py'," old['export_step']=step;old['checkpoint_directory']=str(pathlib.Path(checkpoint_root)/'actor'/('global_step_'+str(step)))", " checkpoint=pathlib.Path(checkpoint_root)/'actor'/('global_step_'+str(step))\n if (checkpoint/'resume-complete.json').is_file():\n  old['export_step']=step;old['checkpoint_directory']=str(checkpoint)\n else:\n  old.setdefault('export_step',None);old.setdefault('checkpoint_directory',None)")
print('Resume state integration applied; bindings must be refreshed before submission')
