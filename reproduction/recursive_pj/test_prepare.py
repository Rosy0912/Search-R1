import copy
import unittest
from reproduction.recursive_lumia.test_prepare_ablation import MatchedAblationConfig
from .prepare import derive

class PJConfig(unittest.TestCase):
    def test_only_paths_schedule_and_condition_change(self):
        t=MatchedAblationConfig().baseline()
        t['data']=dict(train_files='/old/train.parquet',val_files='/old/validation.parquet',final_val_files='/old/final.parquet',train_batch_size=512)
        t['baselines']['holdout']=dict(manifest_path='/old/manifest.json',data_root='/old')
        t['baselines']['bapo']=dict(stage_validation_files='/old/stage.parquet')
        t['actor_rollout_ref']['model']=dict(path='/old/model')
        t['actor_rollout_ref']['rollout']=dict(model_path='/old/model',temperature=1.)
        t['critic']['model']=dict(tokenizer_path='/old/model')
        t['reward_model']=dict(model=dict(input_tokenizer='/old/model'))
        original=copy.deepcopy(t)
        for group in ('position-early','position-late','generation-2'):
            c=derive(t,group,'/new/run','pod-host','/new/data','/new/model')
            self.assertEqual(c['actor_rollout_ref']['actor'],t['actor_rollout_ref']['actor'])
            self.assertEqual(c['data']['train_batch_size'],512)
            self.assertEqual(c['actor_rollout_ref']['rollout']['temperature'],1.)
            self.assertEqual(c['baselines']['local_runtime_host'],'pod-host')
            self.assertEqual(c['trainer']['evaluation_steps'],[5,50,100,150])
            self.assertEqual(c['data']['train_files'],'/new/data/train.parquet')
        self.assertEqual(t,original)
if __name__=='__main__':unittest.main()
