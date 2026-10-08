"""Affected real CPU module-mode/RNG boundaries; no HF/FSDP/Ray/GPU assets."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from search_r1.baselines import sampling_consistency as source
from search_r1.baselines import policy_safety_runtime as runtime_module


ENV=dict(TREE_PPO_CALIBRATION_ONLY='1',TREE_RECOVERY_PARENT_OPTIMIZER_STEPS='2500',
 SEARCH_R1_BASELINE_COMMIT='a'*40,SEARCH_R1_BASELINE_CONFIG_SHA256='b'*64,
 SEARCH_R1_BASELINE_DATA_IDENTITY='c'*64,SEARCH_R1_MODEL_IDENTITY='d'*64,
 SEARCH_R1_BASELINE_RUN_ID='synthetic-mode-cpu',SLURM_JOB_ID='0')
IDENTITY=dict(rank=0,world_size=1,code_commit='a'*40,config_sha256='b'*64,
 data_identity='c'*64,model_identity='d'*64,job_id='0',run_id='synthetic-mode-cpu')


def flags(model):return [(name,module.training) for name,module in model.named_modules()]


def model_fixture():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Sequential(torch.nn.Linear(2,2)))
    model.eval();model[0].training=True
    return model


class TrainRaises(torch.nn.Sequential):
    def train(self,mode=True):
        super().train(mode)
        torch.rand(())
        raise RuntimeError('synthetic train install failure')


class TrainIgnoresChildren(torch.nn.Sequential):
    def train(self,mode=True):
        self.training=mode
        return self


class FakeRuntime:
    fail=False
    calls=0
    def __init__(self,actor,batch,meta,config,source_data):
        self.actor=actor;self.device=torch.device('cpu');self.indices=torch.tensor([0])
        self.monitor={'old_log_probs':torch.zeros(1,2)}
        self.actor_monitor_beta=torch.zeros(1,2)
        self.initial_actor_repeat_native_forward_calls=0;self.monitor_native_forward_calls=1
    @staticmethod
    def effective_mask(data):return torch.ones_like(data['old_log_probs'],dtype=torch.bool)
    def measure(self,*args,**kwargs):
        FakeRuntime.calls+=1
        if not all(mode for name,mode in flags(self.actor.actor_module)):
            raise AssertionError('Actual calibrate native monitor was not train')
        self.actor.actor_module(torch.ones(1,2))
        if torch.is_grad_enabled():raise AssertionError('Calibration enabled autograd')
        torch.rand(())
        if FakeRuntime.fail:raise RuntimeError('synthetic native monitor failure')
        return dict(allowed=False,reasons=['synthetic diagnostic only'])


def worker_fixture(model=None,corrected=False):
    model=model or model_fixture()
    safety={'sampling_probability_correction':{'enabled':corrected,'protocol':'tree_behavior_mu_centered_ppo_v1'}}
    worker=N(_is_actor=True,actor=N(actor_module=model,use_ulysses_sp=False),
        actor_optimizer=N(state={0:{'step':torch.tensor(2500)}}),actor_lr_scheduler=N(last_epoch=110),
        config=N(actor=N(policy_safety=safety)),_sampling_consistency_records=['synthetic-raw'],
        _sampling_consistency_entries=[dict(path='synthetic-raw')])
    batch=N(batch={'input_ids':torch.ones(1,3,dtype=torch.long)},
        meta_info=dict(calibration_parent_outer=110,baseline_outer_step=111),non_tensor_batch={})
    if corrected:
        import numpy as np
        batch.non_tensor_batch['tree_behavior_mu_binding']=np.array([{'synthetic_only':True}],dtype=object)
    return worker,batch


def gloo_mode_worker(rank,rendezvous,output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        model=(TrainRaises(torch.nn.Linear(2,2)) if rank==1 else model_fixture())
        if rank==1:
            # Set the initial modes directly; calling overridden train() is the subject.
            for module in model.modules():module.training=False
        before=flags(model);worker=N(actor=N(actor_module=model));entered=False
        env=dict(ENV,SEARCH_R1_BASELINE_OUTPUT=str(Path(output)/str(rank)))
        with patch.dict(os.environ,env):
            try:
                with source.calibration_actor_training_mode(worker,torch.device('cpu'),dict(IDENTITY,rank=rank,world_size=2)):
                    entered=True
                raise AssertionError('Both ranks must refuse')
            except RuntimeError as error:
                record=dict(rank=rank,entered=entered,restored=flags(model)==before,error=str(error))
                assert not entered and record['restored']
                (Path(output)/f'{rank}.json').write_text(json.dumps(record,allow_nan=False))
    finally:dist.destroy_process_group()


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.env=patch.dict(os.environ,dict(ENV,SEARCH_R1_BASELINE_OUTPUT=self.tmp.name));self.env.start()
        FakeRuntime.calls=0;FakeRuntime.fail=False
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def mode_record(self):return json.loads((Path(self.tmp.name)/'probability-calibration/rank-00-mode.json').read_text())

    def test_actual_train_all_modules_then_restore_original_mixed_child_modes(self):
        model=model_fixture();before=flags(model);worker=N(actor=N(actor_module=model))
        rng=torch.get_rng_state().clone()
        with source.calibration_actor_training_mode(worker,torch.device('cpu'),IDENTITY) as mode:
            self.assertTrue(all(flag is True for name,flag in flags(model)))
            torch.rand(())
        self.assertEqual(flags(model),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(mode['original_module_modes_restored']);self.assertTrue(mode['body_returned'])
        self.assertTrue(self.mode_record()['all_monitor_modules_training'])

    def test_native_exception_restores_flags_rng_and_persists_restoration_without_success(self):
        model=model_fixture();before=flags(model);worker=N(actor=N(actor_module=model));rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'native body'):
            with source.calibration_actor_training_mode(worker,torch.device('cpu'),IDENTITY):
                torch.rand(());raise RuntimeError('native body')
        self.assertEqual(flags(model),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        record=self.mode_record();self.assertTrue(record['original_module_modes_restored']);self.assertFalse(record['body_returned'])

    def test_partial_train_install_exception_refuses_before_body_restores_every_flag(self):
        model=TrainRaises(torch.nn.Linear(2,2));
        for module in model.modules():module.training=False
        worker=N(actor=N(actor_module=model));before=flags(model);rng=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'preparation refusal'):
            with source.calibration_actor_training_mode(worker,torch.device('cpu'),IDENTITY):self.fail('body entered')
        self.assertEqual(flags(model),before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(self.mode_record()['original_module_modes_restored'])

    def test_true_root_but_eval_child_refuses_before_monitor_and_restores(self):
        model=TrainIgnoresChildren(torch.nn.Linear(2,2));
        for module in model.modules():module.training=False
        before=flags(model);worker=N(actor=N(actor_module=model))
        with self.assertRaisesRegex(RuntimeError,'every module'):
            with source.calibration_actor_training_mode(worker,torch.device('cpu'),IDENTITY):self.fail('body entered')
        self.assertEqual(flags(model),before);self.assertFalse(self.mode_record()['all_monitor_modules_training'])

    def test_evidence_io_failure_keeps_restored_modes_and_is_not_success(self):
        model=model_fixture();before=flags(model);worker=N(actor=N(actor_module=model))
        with patch.object(source,'atomic_save',side_effect=OSError('synthetic mode IO')):
            with self.assertRaisesRegex(RuntimeError,'synthetic mode IO'):
                with source.calibration_actor_training_mode(worker,torch.device('cpu'),IDENTITY):pass
        self.assertEqual(flags(model),before);self.assertTrue(worker._last_calibration_mode_evidence['original_module_modes_restored'])
        self.assertFalse((Path(self.tmp.name)/'probability-calibration/rank-00.json').exists())

    def test_actual_calibrate_actor_success_is_train_restored_and_zero_optimizer_scheduler(self):
        worker,data=worker_fixture();before=flags(worker.actor.actor_module)
        steps=copy.deepcopy(worker.actor_optimizer.state);rng=torch.get_rng_state().clone()
        with patch.object(runtime_module,'PolicySafetyRuntime',FakeRuntime),patch.object(source.dist,'get_rank',return_value=0),patch.object(source.dist,'get_world_size',return_value=1),patch.object(source.dist,'all_reduce',lambda *a,**k:None):
            record=source.calibrate_actor(worker,data)
        self.assertEqual(FakeRuntime.calls,1);self.assertEqual(flags(worker.actor.actor_module),before)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(worker.actor_lr_scheduler.last_epoch,110)
        self.assertTrue(torch.equal(worker.actor_optimizer.state[0]['step'],steps[0]['step']))
        self.assertEqual(record['optimizer_committed'],0);self.assertFalse(record['scheduler_advanced'])
        self.assertTrue(record['actor_calibration_mode']['all_monitor_modules_training']);self.assertTrue(record['actor_calibration_mode']['original_module_modes_restored'])
        self.assertEqual(source.file_sha(record['actor_calibration_mode']['mode_evidence_path']),record['actor_calibration_mode']['mode_evidence_sha256'])

    def test_actual_calibrate_actor_monitor_exception_restores_without_final_rank_success(self):
        worker,data=worker_fixture();before=flags(worker.actor.actor_module);FakeRuntime.fail=True
        with patch.object(runtime_module,'PolicySafetyRuntime',FakeRuntime),patch.object(source.dist,'get_rank',return_value=0),patch.object(source.dist,'get_world_size',return_value=1),patch.object(source.dist,'all_reduce',lambda *a,**k:None):
            with self.assertRaisesRegex(RuntimeError,'native monitor failure'):source.calibrate_actor(worker,data)
        self.assertEqual(flags(worker.actor.actor_module),before);self.assertTrue(self.mode_record()['original_module_modes_restored'])
        self.assertFalse(self.mode_record()['body_returned']);self.assertEqual(worker.actor_lr_scheduler.last_epoch,110)
        self.assertFalse((Path(self.tmp.name)/'probability-calibration/rank-00.json').exists())

    def test_real_two_rank_gloo_one_rank_train_install_refuses_before_both_monitors(self):
        with tempfile.TemporaryDirectory() as output:
            mp.spawn(gloo_mode_worker,args=(str(Path(output)/'rendezvous'),output),nprocs=2,join=True)
            records=[json.loads((Path(output)/f'{rank}.json').read_text()) for rank in range(2)]
            self.assertTrue(all(not r['entered'] and r['restored'] for r in records))


if __name__=='__main__':unittest.main()
