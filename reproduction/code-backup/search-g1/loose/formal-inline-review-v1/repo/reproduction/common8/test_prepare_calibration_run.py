"""Filesystem/identity fixtures, not actual native preparation admission."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8 import prepare_calibration_run as module
from reproduction.common8.policy_calibration_plan import PlanError
from reproduction.common8 import test_policy_calibration_plan as plans


class PrepareCalibrationTests(unittest.TestCase):
    def setUp(self):
        fixture = plans.CalibrationPlanTests(); fixture.setUp()
        self.template = copy.deepcopy(fixture.parent)

    def fake_prepare(self, base, model, official, commit):
        run = copy.deepcopy(self.template)
        run.update(run_id=base.name, initial_model=str(model), reference_model=str(model),
                   train_log=str(base / 'logs/train.log'), checkpoint_dir=str(base / 'checkpoints'))
        target = base / 'logs/run-manifest.json'; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as f: json.dump(run, f)
        return run

    def setup_paths(self, root):
        parent, child = Path(root) / 'parent', Path(root) / 'discovery'
        for b in (parent, child): (b / 'preparation').mkdir(parents=True)
        return parent, child

    def test_separate_fresh_parent_child_and_original_bytes_no_ready_marker(self):
        with tempfile.TemporaryDirectory() as root:
            parent, child = self.setup_paths(root)
            with patch.object(module, 'native_preparation', return_value='123'), \
                 patch.object(module, 'prepare', side_effect=self.fake_prepare):
                run = module.prepare_child(child, parent, module.DISCOVERY_ROLE, '/model', '/official', 'a'*40)
            original = (parent / 'logs/run-manifest.json').read_bytes()
            self.assertEqual((child / 'preparation/parent-run-manifest.json').read_bytes(), original)
            self.assertNotEqual(run['run_id'], run['parent_production_run_id'])
            self.assertEqual(run['completed_updates'], 0)
            self.assertEqual(run['policy_calibration_parent_manifest_sha256'], module.digest(child / 'preparation/parent-run-manifest.json'))
            self.assertEqual(run['policy_calibration_discovery_base'], str(child.resolve()))
            self.assertFalse(list((child / 'logs').glob('*accepted*')))
            self.assertFalse((child / 'logs/native-allocation-admission.json').exists())

    def test_existing_child_is_never_reprepared(self):
        with tempfile.TemporaryDirectory() as root:
            parent, child = self.setup_paths(root); self.fake_prepare(child, '/model', '', '')
            original = (child / 'logs/run-manifest.json').read_bytes()
            with patch.object(module, 'native_preparation', return_value='123'), \
                 patch.object(module, 'prepare') as prepare, self.assertRaises(AssertionError):
                module.prepare_child(child, parent, module.DISCOVERY_ROLE, '/model', '/official', 'a'*40)
            prepare.assert_not_called()
            self.assertEqual((child / 'logs/run-manifest.json').read_bytes(), original)

    def test_confirmation_requires_original_parent_and_distinct_discovery(self):
        with tempfile.TemporaryDirectory() as root:
            parent, child = self.setup_paths(root)
            with patch.object(module, 'native_preparation', return_value='123'), \
                 patch.object(module, 'prepare') as prepare, self.assertRaises(AssertionError):
                module.prepare_child(child, parent, module.CONFIRMATION_ROLE, '/model', '/official', 'a'*40, Path(root)/'other')
            prepare.assert_not_called()
            with self.assertRaises(AssertionError):
                module.prepare_child(child, parent, module.CONFIRMATION_ROLE, '/model', '/official', 'a'*40, child)
            with self.assertRaisesRegex(AssertionError, 'explicit discovery'):
                module.prepare_child(child, parent, module.CONFIRMATION_ROLE, '/model', '/official', 'a'*40)

    def test_confirmation_binds_exact_parent_descriptor_before_archiving(self):
        with tempfile.TemporaryDirectory() as root:
            parent, child = self.setup_paths(root)
            discovery = (Path(root) / 'known-discovery').resolve()
            self.fake_prepare(parent, '/model', '', '')
            with patch.object(module, 'native_preparation', return_value='123'), \
                 patch.object(module, 'prepare', side_effect=self.fake_prepare):
                run = module.prepare_child(child, parent, module.CONFIRMATION_ROLE,
                                           '/model', '/official', 'a'*40, discovery)
            original = json.loads((child / 'preparation/parent-run-manifest.json').read_text())
            self.assertEqual(original['policy_calibration_confirmation_base'], str(child.resolve()))
            self.assertEqual(original['policy_calibration_discovery_base'], str(discovery))
            self.assertEqual(run['parent_production_run_id'], original['run_id'])

    def test_confirmation_cannot_replace_previously_bound_child(self):
        with tempfile.TemporaryDirectory() as root:
            parent, child = self.setup_paths(root)
            run = self.fake_prepare(parent, '/model', '', '')
            run['policy_calibration_confirmation_base'] = str((Path(root) / 'other').resolve())
            (parent / 'logs/run-manifest.json').write_text(json.dumps(run))
            with patch.object(module, 'native_preparation', return_value='123'), \
                 patch.object(module, 'prepare', side_effect=self.fake_prepare), self.assertRaisesRegex(AssertionError, 'another confirmation'):
                module.prepare_child(child, parent, module.CONFIRMATION_ROLE, '/model', '/official', 'a'*40, Path(root)/'source')
            self.assertFalse((child / 'preparation/parent-run-manifest.json').exists())

    def test_dirty_or_wrong_source_parent_cannot_silently_change_identity(self):
        for wrong in ('claim', 'updates', 'checkpoint', 'trainlog', 'source', 'segment', 'partial'):
            with self.subTest(wrong=wrong), tempfile.TemporaryDirectory() as root:
                parent, child = self.setup_paths(root); original = self.fake_prepare(parent, '/model', '', '')
                if wrong in ('claim', 'updates'):
                    (parent / 'logs' / ('training-launch-claim.json' if wrong == 'claim' else 'actual-updates.jsonl')).write_text('preserve')
                elif wrong == 'checkpoint':
                    (parent / 'checkpoints').mkdir(); (parent / 'checkpoints/preserve').write_text('old')
                elif wrong == 'trainlog':
                    Path(original['train_log']).write_text('preserve')
                elif wrong == 'segment':
                    (parent / 'logs/policy-calibration-segment.json').write_text('preserve')
                elif wrong == 'partial':
                    (parent / 'logs/policy-calibration-optimizer-rank-0.json').write_text('preserve')
                else:
                    original['source_commit'] = 'b'*40
                    (parent / 'logs/run-manifest.json').write_text(json.dumps(original))
                with patch.object(module, 'native_preparation', return_value='123'), \
                     patch.object(module, 'prepare', side_effect=self.fake_prepare), self.assertRaises((AssertionError, PlanError)):
                    module.prepare_child(child, parent, module.DISCOVERY_ROLE, '/model', '/official', 'a'*40)
                self.assertFalse((child / 'preparation/parent-run-manifest.json').exists())


if __name__ == '__main__': unittest.main()
