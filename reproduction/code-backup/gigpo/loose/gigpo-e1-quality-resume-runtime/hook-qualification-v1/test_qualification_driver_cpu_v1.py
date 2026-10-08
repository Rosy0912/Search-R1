"""Real CPU DataLoader/sampler/RNG saved-state replay, no Ray/CUDA/Adam."""
import copy
import types
import unittest
import torch
from torch.utils.data import DataLoader,TensorDataset
from omegaconf import OmegaConf
from search_r1.baselines.recovery import RecoverableSampler
from calibration_state_v1 import state_digest
from qualification_driver_state_v1 import capture_driver_state,reload_driver_state


class RealDriverReplayCPU(unittest.TestCase):
    def trainer(self):
        sampler=RecoverableSampler(100,shuffle=True,seed=17)
        stage=types.SimpleNamespace(values=dict(stage_counter=5))
        stage.state_dict=lambda:copy.deepcopy(stage.values)
        stage.load_state_dict=lambda value:setattr(stage,'values',copy.deepcopy(value))
        return types.SimpleNamespace(config=OmegaConf.create(dict(micro=16,mini=256,lr=1e-6)),
            recovery_sampler=sampler,train_dataloader=DataLoader(TensorDataset(torch.arange(100)),
                batch_size=5,sampler=sampler,num_workers=0),
            instruct_data_identity={'explicit_CPU_dataset':100},global_steps=26,actual_outer_updates=1,
            kl_ctrl=types.SimpleNamespace(value=.001),quality_health={'explicit_CPU_health_fixture':True},
            validation_trace_serial=0,baseline_controller=types.SimpleNamespace(method='gigpo',stage=stage,
                config=OmegaConf.create(dict(gigpo=dict(gamma=.95,omega=1.,normalize=False),
                    loss_reduction='sequence_mean',base_commit='CPU_base',algorithm_source_commit='CPU_gigpo'))))

    def test_actual_saved_driver_rng_cursor_iterator_and_next_batch_replay(self):
        trainer=self.trainer();iterator=iter(trainer.train_dataloader);next(iterator)
        before=capture_driver_state(trainer,iterator)
        expected=next(iterator)[0].clone()
        trainer.kl_ctrl.value=.2;trainer.global_steps=99
        trainer.baseline_controller.stage.values={'stage_counter':999};torch.rand(9)
        replay=reload_driver_state(trainer,before)
        self.assertEqual(state_digest(capture_driver_state(trainer,replay)),state_digest(before))
        actual=next(replay)[0]
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(trainer.recovery_sampler.cursor,10)
        self.assertEqual(replay._num_yielded,2)
        self.assertEqual(trainer.global_steps,26);self.assertEqual(trainer.kl_ctrl.value,.001)

    def test_changed_math_config_or_driver_identity_refuses_without_rng_or_data_reset(self):
        for kind in ('config','pid','data','algorithm','iterator'):
            with self.subTest(kind=kind):
                trainer=self.trainer();iterator=iter(trainer.train_dataloader);next(iterator)
                saved=capture_driver_state(trainer,iterator)
                if kind=='config':saved['config']['lr']=.1
                elif kind=='pid':saved['pid']+=1
                elif kind=='data':saved['data_identity']={'wrong_dataset':True}
                elif kind=='algorithm':saved['algorithm']['parameters']['gamma']=.9
                else:saved['iterator']['num_yielded']=-1
                before=capture_driver_state(trainer,iterator)
                with self.assertRaises(ValueError):reload_driver_state(trainer,saved)
                self.assertEqual(state_digest(capture_driver_state(trainer,iterator)),state_digest(before))


if __name__=='__main__':unittest.main(verbosity=2)
