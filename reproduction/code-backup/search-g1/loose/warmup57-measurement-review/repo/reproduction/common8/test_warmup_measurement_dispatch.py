"""CPU dispatch exercises the actual driver branch; no native training claims."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from reproduction.common8 import configuration,validation_identity
from reproduction.common8.formal_contract import verify_formal_config,verify_scientific_config
from reproduction.common8.run_pipeline import Pipeline,adoption_inputs
from reproduction.common8.test_warmup_measurement_plan import fixture
from reproduction.common8.warmup_measurement_plan import ROLE


class WarmupMeasurementDispatchTests(unittest.TestCase):
    def test_actual_resolved_recipe_keeps_horizon201_and_formal_math_for_new_role(self):
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as root:
            base,_,run,_=fixture(root);(base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,config,_=configuration.resolved(base);document=OmegaConf.to_container(config,resolve=True)
            proof=verify_scientific_config(document,run,base)
            self.assertEqual(proof['execution_role'],ROLE)
            self.assertEqual(document['trainer']['total_training_steps'],201)
            self.assertEqual(document['trainer']['acceptance_step'],5)
            self.assertEqual(document['actor_rollout_ref']['actor']['optim']['lr_warmup_steps'],57)
            with self.assertRaises(AssertionError):verify_formal_config(document,run,base)
            for key,value in (('total_training_steps',3),('acceptance_step',2)):
                bad=copy.deepcopy(document);bad['trainer'][key]=value
                with self.subTest(key=key),self.assertRaises(AssertionError):verify_scientific_config(bad,run,base)

    def test_dispatch_never_reads_old301_plan_or_old_failed_probe_source(self):
        with tempfile.TemporaryDirectory() as root:
            base,_,run,_=fixture(root);(base/'logs/run-manifest.json').write_text(json.dumps(run))
            marker=(base,run,{'role':ROLE},'CPU-only')
            with patch.dict('os.environ',SEARCH_G1_PROFILE='common8',SEARCH_G1_BASE=str(base)),patch(
                    'reproduction.common8.calibration_runtime.active_plan',side_effect=AssertionError('Old schema must stay isolated')),patch(
                    'reproduction.common8.warmup_measurement_runtime.active_plan',return_value=marker) as new:
                self.assertEqual(validation_identity.active_phase(),(marker,ROLE));new.assert_called_once_with()
            self.assertEqual(adoption_inputs(run,base),{})
            pipeline=Pipeline.__new__(Pipeline);pipeline.base=base;pipeline.run=run;pipeline.python='CPU-only-python'
            pipeline.start=Mock(return_value=object());pipeline.wait=Mock();pipeline.stage=Mock()
            pipeline.initial_probe_calibration()
            command=pipeline.start.call_args.args[1]
            self.assertIn('reproduction.common8.refit',command);self.assertEqual(command[command.index('--step')+1],'0')
            self.assertFalse(any('adopt' in str(value) for value in command))

    def test_actual_driver_first_outer_continues_and_second_stops_before_formal_five(self):
        # Execute the real isolated phase branch in a small CPU loop, retaining
        # its continue/return semantics. No rewritten mirror of that branch.
        repo=Path(__file__).resolve().parents[2]
        tree=ast.parse((repo/'verl/trainer/ppo/ray_trainer.py').read_text())
        trainer=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RayPPOTrainer')
        fit=next(node for node in trainer.body if isinstance(node,ast.FunctionDef) and node.name=='fit')
        branches=[node for node in ast.walk(fit) if isinstance(node,ast.If) and
                  isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and
                  node.test.left.id=='calibration_phase' and any(isinstance(child,ast.Name) and
                  child.id=='finish_after_update' for child in ast.walk(node))]
        self.assertEqual(len(branches),1)
        branch=copy.deepcopy(branches[0])
        function=ast.parse('def exercise(self,calibration_phase,calibration_role):\n    for _ in range(3):\n        raise AssertionError("unexpected formal path")\n').body[0]
        function.body[0].body.insert(0,branch)
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));namespace={}
        exec(compile(module,'actual-driver-phase-CPU-fixture','exec'),namespace)
        trainer=SimpleNamespace(global_steps=1);phase=('CPU-only-phase',)
        with patch('reproduction.common8.warmup_measurement_runtime.finish_after_update') as finish,patch(
                'reproduction.common8.calibration_runtime.finish_after_update',side_effect=AssertionError('Old one-outer finisher forbidden')):
            namespace['exercise'](trainer,phase,ROLE)
            finish.assert_called_once_with(trainer,phase);self.assertEqual(trainer.global_steps,2)


if __name__=='__main__':unittest.main()
