"""Synthetic eight-record fresh driver boundary; original source entrypoints."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch
from search_r1.baselines import sampling_consistency as source
from search_r1.baselines import policy_safety_runtime as runtime_module
from test_calibration_actor_mode_cpu import FakeRuntime, worker_fixture
from test_fresh_calibration_worker_cpu import C, ENV, PROVENANCE

class D(dict):
    def __getattr__(self,key):
        try:return self[key]
        except KeyError:raise AttributeError(key)

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.env=patch.dict(os.environ,dict(ENV,SEARCH_R1_BASELINE_OUTPUT=self.tmp.name));self.env.start()
        for key in list(os.environ):
            if key.startswith('TREE_RECOVERY_'):os.environ.pop(key)
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def fixture(self):
        identity=dict(code_commit=ENV['SEARCH_R1_BASELINE_COMMIT'],config_sha256=ENV['SEARCH_R1_BASELINE_CONFIG_SHA256'],
            data_identity=ENV['SEARCH_R1_BASELINE_DATA_IDENTITY'],model_identity=ENV['SEARCH_R1_MODEL_IDENTITY'],
            job_id=ENV['SLURM_JOB_ID'],run_id=ENV['SEARCH_R1_BASELINE_RUN_ID'])
        proof=dict(PROVENANCE,workers=[dict(identity,rank=r,world_size=8,optimizer_state_entries=0,optimizer_step_values=[],
            scheduler_step=0,optimizer_committed=0,scheduler_advanced=False,calibration_profile='fresh_base_zero_v1',
            configured_model_path='/synthetic/official-model',expected_model_path='/synthetic/official-model',
            checkpoint_recovery_environment_absent=True) for r in range(8)],**identity)
        safety=D(enabled=True,sampling_probability_correction=dict(enabled=False,protocol='tree_behavior_mu_centered_ppo_v1'))
        trainer=N(global_steps=1,actual_outer_updates=0,checkpoint_sampler=N(cursor=512),fresh_initialization_provenance=proof,
            config=C(actor_rollout_ref=C(actor=C(fresh200_protocol='tree_fresh200_five_leaf_v1',policy_safety=safety),
                model=C(path='/synthetic/official-model'),rollout=C(temperature=1.,sampling_consistency=C(enabled=True,enforce=False)))))
        def observe(batch):
            root=Path(self.tmp.name)/'sampling-consistency'
            for rank in range(8):
                snapshot=root/f'rank{rank}.pt';snapshotsha=source.atomic_save({'synthetic':torch.zeros(1)},snapshot)
                rawpath=root/f'rank{rank}.json';raw=dict(identity,rank=rank,scheduler_step=0,snapshot_path=str(snapshot),snapshot_sha256=snapshotsha)
                rawsha=source.atomic_save(raw,rawpath,json_format=True)
                worker,_=worker_fixture();worker.config.actor=C(policy_safety=safety,fresh200_protocol='tree_fresh200_five_leaf_v1')
                worker.actor_optimizer.state={};worker.actor_lr_scheduler.last_epoch=0
                worker._sampling_consistency_records=[str(rawpath)]
                worker._sampling_consistency_entries=[dict(path=str(rawpath),sha256=rawsha,snapshot_path=str(snapshot),snapshot_sha256=snapshotsha)]
                with patch.object(source.dist,'get_rank',return_value=rank),patch.object(source.dist,'get_world_size',return_value=8),patch.object(source.dist,'all_reduce',lambda *a,**k:None),patch.object(runtime_module,'PolicySafetyRuntime',FakeRuntime):
                    source.calibrate_actor(worker,batch)
        trainer.actor_rollout_wg=N(calibrate_policy_safety=observe)
        batch=N(meta_info={},batch={'token_level_rewards':torch.ones(1,2)})
        return trainer,batch

    def test_original_finalize_fresh8_mode_sha_boundary_and_preserved_no_recovery(self):
        trainer,batch=self.fixture();ready=source.finalize_calibration(trainer,batch)
        self.assertEqual(ready['initialized_global_steps'],0);self.assertEqual(ready['observed_outer'],1)
        self.assertEqual(ready['new_completed_outer'],0);self.assertEqual(ready['sampler_cursor_before_outer'],0)
        self.assertEqual(ready['sampler_cursor_after_observation'],512);self.assertEqual(len(ready['workers']),8)
        self.assertTrue(all(r['actor_calibration_mode']['original_module_modes_restored'] for r in ready['workers']))
        self.assertFalse(hasattr(trainer,'recovery_provenance'))
        # Finalize must reject an internally changed mode record; no recalc or overwrite.
        rankpath=Path(self.tmp.name)/'probability-calibration/rank-00.json'
        changed=json.loads(rankpath.read_text());changed['actor_calibration_mode']['original_module_modes_restored']=False
        rankpath.write_text(json.dumps(changed,allow_nan=False));trainer.actor_rollout_wg.calibrate_policy_safety=lambda batch:None
        with self.assertRaisesRegex(RuntimeError,'train-mode/restoration'):source.finalize_calibration(trainer,batch)

    def test_driver_observed1_actual0_cursor512_and_each_actual_initializer_required(self):
        trainer,_=self.fixture();self.assertIs(source.fresh_calibration_driver_provenance(trainer),trainer.fresh_initialization_provenance)
        for mutate in (lambda t:setattr(t,'global_steps',0),lambda t:setattr(t,'actual_outer_updates',1),
                       lambda t:setattr(t.checkpoint_sampler,'cursor',511),
                       lambda t:t.fresh_initialization_provenance['workers'][7].update(optimizer_state_entries=1),
                       lambda t:t.fresh_initialization_provenance.update(initialized_global_steps=1)):
            broken=copy.deepcopy(trainer);mutate(broken)
            with self.assertRaises(ValueError):source.fresh_calibration_driver_provenance(broken)

if __name__=='__main__':unittest.main(verbosity=2)
