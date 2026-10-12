"""Actual recipe argv with a scalar manifest fixture, without Hydra or training."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import configuration as config
from reproduction.common8.formal_contract import formal_mathematics
from tests.reproduction.test_actor_probe_recovery import documents


class ActorProbeConfigurationTests(unittest.TestCase):
    def test_generated_hydra_files_stay_outside_exact_source_map(self):
        for actor in (True,False):
            with self.subTest(actor=actor),tempfile.TemporaryDirectory() as root:
                base=Path(root).resolve();(base/'logs').mkdir()
                run=copy.deepcopy(documents()[0]);run['completed_updates']=0
                run['budget']=dict(json.loads((config.REPO/'reproduction/common8/budget.json').read_text()),
                    **formal_mathematics(run))
                run['source_sha256']={name:config.sha(config.REPO/name) for name in
                    ('reproduction/common8/recipe.json','verl/trainer/config/ppo_trainer.yaml')}
                if not actor:
                    for key in ('probe_representation_source','probe_refresh_interval','probe_parameter_ownership'):
                        run.pop(key)
                    run['reward_schedule']='g1_through50_then_binary';run['reward_schedule_boundary']=50
                (base/'logs/run-manifest.json').write_text(json.dumps(run))
                # The deliberately small fixture is not a native run admission.
                with patch.object(config,'assert_formal_run'):
                    _,argv=config.arguments(base)
                by_key=dict(value.split('=',1) for value in argv)
                self.assertEqual(len(by_key),len(argv))
                self.assertEqual(by_key['actor_rollout_ref.actor.ppo_micro_batch_size'],'8')
                self.assertEqual(by_key['trainer.n_gpus_per_node'],'4')
                if actor:
                    directory=Path(json.loads(by_key['hydra.run.dir']))
                    self.assertEqual(directory,base/'logs/trainer-hydra')
                    self.assertFalse(directory.is_relative_to(config.REPO))
                    self.assertEqual(by_key['hydra.job.chdir'],'false')
                else:
                    self.assertNotIn('hydra.run.dir',by_key)
                    self.assertNotIn('hydra.job.chdir',by_key)


if __name__=='__main__':unittest.main()
