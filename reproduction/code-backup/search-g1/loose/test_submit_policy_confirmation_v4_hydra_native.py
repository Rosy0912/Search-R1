"""CPU filesystem/OmegaConf protocol tests; no Slurm/GPU/health assertions.

Synthetic sources and descriptors below test the reader, not actual discovery
completion. The old actual metadata observation is checked only for parser
semantics, never adopted as the new source's healthy evidence.
"""
import copy
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'repo'))
spec = importlib.util.spec_from_file_location('confirmation_v4_hydra', HERE / 'submit_policy_confirmation_v4_hydra_native.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import submit_policy_discovery_v4_native as terminal
import test_submit_policy_confirmation_v4_native as prior


def snapshot_fixture(root):
    parent, child = root / 'parent', root / 'child'
    sources = {f'cpu_fixture_{i}.py': b'# CPU structural fixture\n' for i in range(350)}
    sources.update({'verl/version/version': b'0.0.cpu\n', 'verl/single_controller/version/version': b'0.0.cpu\n',
                    'reproduction/common8/recipe.json': b'["CPU_fixture=true"]\n'})
    mapping = {name: m.sha(content) for name, content in sources.items()}
    for base in (parent, child):
        for name, content in sources.items():
            path = base / 'repo' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    logs = child / 'logs'; logs.mkdir()
    run_id, job = '1' * 32, '98765'
    write_json = lambda path, document: path.write_bytes(m.canonical(document))
    run = dict(run_id=run_id, source_sha256=mapping, policy_calibration_slurm_job_id=job)
    write_json(logs / 'run-manifest.json', run)
    claim = dict(run_id=run_id, slurm_job_id=job, shell_pid=12345, shell_start_ticks=987654,
                 claimed_unix=1000.0)
    write_json(logs / 'training-launch-claim.json', claim)
    config = dict(run_id=run_id, slurm_job_id=job, checked_unix=1001.0,
                  resolved_config={'trainer': {'total_training_steps': 301}, 'optimizer': {'lr': 1e-6}},
                  actual_argv_overrides=['trainer.total_training_steps=301', 'optimizer.lr=1e-6'],
                  recipe_sha256=mapping['reproduction/common8/recipe.json'])
    config['resolved_config_sha256'] = m.sha(m.canonical(config['resolved_config']))
    write_json(logs / 'resolved-training-config.json', config)
    ledger = dict(run_id=run_id, jobs={'trainer': {'leader': {'pid': 12345, 'start_ticks': 987654, 'uid': os.getuid()}}},
                  CPU_fixture_padding='x' * 1040323)
    write_json(logs / 'process-ledger.json', ledger)
    folder = child / 'repo/outputs/2026-10-06/08-27-31/.hydra'; folder.mkdir(parents=True)
    documents = {'config.yaml': config['resolved_config'], 'overrides.yaml': config['actual_argv_overrides'],
                 'hydra.yaml': {'hydra': {'runtime': {'cwd': str(child / 'repo'), 'output_dir': str(folder.parent)},
                                         'job': {'name': 'main_ppo_eco', 'config_name': 'ppo_trainer', 'chdir': None},
                                         'overrides': {'task': config['actual_argv_overrides']}}}}
    for name, doc in documents.items():
        (folder / name).write_text(OmegaConf.to_yaml(OmegaConf.create(doc)))
    descriptor = dict(schema_version=1, run_id=run_id, slurm_job_id=job, source_map_sha256=m.sha(m.canonical(mapping)),
                      training_launch_claim_sha256=m.sha((logs / 'training-launch-claim.json').read_bytes()),
                      resolved_training_config_file_sha256=m.sha((logs / 'resolved-training-config.json').read_bytes()),
                      resolved_config_sha256=config['resolved_config_sha256'],
                      files={str(path.relative_to(child / 'repo')): dict(size_bytes=path.stat().st_size, sha256=m.sha(path.read_bytes()))
                             for path in sorted(folder.iterdir())})
    selection = dict(discovery_run_id=run_id, discovery_slurm_job_id=job, discovery_generated_hydra_snapshots=descriptor)
    launch = dict(job_id=job, source_sha256=mapping)
    return S(parent=parent, child=child, selection=selection, launch=launch, descriptor=descriptor,
             folder=folder, logs=logs, config=config, claim=claim, ledger=ledger, documents=documents)


class HydraSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = snapshot_fixture(Path(self.temp.name).resolve())

    def verify(self):
        f = self.f
        return m.verify_source_snapshot_sets(f.parent, f.child, f.launch, f.selection)

    def reseal_yaml(self, name, document):
        f = self.f; path = f.folder / name
        path.write_text(OmegaConf.to_yaml(OmegaConf.create(document)))
        f.descriptor['files'][str(path.relative_to(f.child / 'repo'))] = dict(size_bytes=path.stat().st_size, sha256=m.sha(path.read_bytes()))

    def test_exact_three_and_real_omegaconf_semantics(self):
        result = self.verify()
        self.assertEqual(result['source_snapshot_kind'], 'current_disk_not_loaded_module_proof')
        for key in ('production_admission', 'healthy_checkpoint_accepted', 'physical_execution_authenticated'):
            self.assertIs(result[key], False)
        self.assertGreater((self.f.logs / 'process-ledger.json').stat().st_size, 1040323)

    def test_original_source_changed(self):
        (self.f.child / 'repo/cpu_fixture_0.py').write_text('# changed')
        with self.assertRaises(Exception): self.verify()

    def test_unknown_extra_all_runtime_suffixes(self):
        for suffix in ('.py', '.sh', '.json', '.yaml', '.yml'):
            with self.subTest(suffix=suffix):
                path = self.f.child / ('repo/unknown' + suffix); path.write_text('CPU')
                with self.assertRaises(Exception): self.verify()
                path.unlink()

    def test_parent_gets_no_generated_exception(self):
        (self.f.parent / 'repo/unknown.yaml').write_text('CPU: true')
        with self.assertRaises(Exception): self.verify()

    def test_missing_snapshot_and_changed_bytes_rejected(self):
        path = self.f.folder / 'config.yaml'; raw = path.read_bytes(); path.unlink()
        with self.assertRaises(Exception): self.verify()
        path.write_bytes(raw + b'\n')
        with self.assertRaises(Exception): self.verify()

    def test_second_empty_hydra_directory_rejected(self):
        (self.f.child / 'repo/outputs/2026-10-06/08-28-31/.hydra').mkdir(parents=True)
        with self.assertRaises(Exception): self.verify()

    def test_descriptor_missing_unknown_bool_size_and_source_drift(self):
        f = self.f
        for kind in ('missing', 'unknown', 'bool', 'source'):
            original = copy.deepcopy(f.descriptor)
            if kind == 'missing': del f.descriptor['files'][next(iter(f.descriptor['files']))]
            elif kind == 'unknown': f.descriptor['unknown'] = True
            elif kind == 'bool': next(iter(f.descriptor['files'].values()))['size_bytes'] = True
            else: f.descriptor['source_map_sha256'] = 'a' * 64
            with self.subTest(kind=kind), self.assertRaises(Exception): self.verify()
            f.descriptor.clear(); f.descriptor.update(original)

    def test_sealed_yaml_still_must_match_actual_config_and_overrides(self):
        for name, wrong in (('config.yaml', {'trainer': {'total_training_steps': 1}}),
                            ('overrides.yaml', ['CPU_wrong=true'])):
            with self.subTest(name=name):
                self.reseal_yaml(name, wrong)
                with self.assertRaises(Exception): self.verify()
                self.reseal_yaml(name, self.f.documents[name])

    def test_sealed_hydra_metadata_must_match_actual_namespace(self):
        for section, key, wrong in (('runtime', 'cwd', '/outside'), ('runtime', 'output_dir', '/outside'),
                                    ('job', 'name', 'other'), ('job', 'config_name', 'other'), ('job', 'chdir', True)):
            document = copy.deepcopy(self.f.documents['hydra.yaml']); document['hydra'][section][key] = wrong
            self.reseal_yaml('hydra.yaml', document)
            with self.subTest(key=key), self.assertRaises(Exception): self.verify()
        self.reseal_yaml('hydra.yaml', self.f.documents['hydra.yaml'])
        self.verify()

    def test_claim_and_resolved_raw_bytes_bound(self):
        for name in ('training-launch-claim.json', 'resolved-training-config.json'):
            path = self.f.logs / name; raw = path.read_bytes(); path.write_bytes(raw + b'\n')
            with self.subTest(name=name), self.assertRaises(Exception): self.verify()
            path.write_bytes(raw)

    def test_claim_birth_matches_existing_ledger(self):
        f = self.f; ledger = copy.deepcopy(f.ledger); ledger['jobs']['trainer']['leader']['start_ticks'] += 1
        (f.logs / 'process-ledger.json').write_bytes(m.canonical(ledger))
        with self.assertRaises(Exception): self.verify()

    def test_symlink_and_fd_alias_rejected(self):
        path = self.f.folder / 'config.yaml'; raw = path.read_bytes(); path.unlink()
        target = Path(self.temp.name) / 'outside.yaml'; target.write_bytes(raw); path.symlink_to(target)
        with self.assertRaises(Exception): self.verify()

    def test_dangling_unlisted_source_symlink_rejected(self):
        (self.f.child / 'repo/unknown.py').symlink_to('/missing_cpu_fixture')
        with self.assertRaises(Exception): self.verify()

    def test_small_limit_not_invented_for_large_ledger(self):
        path = self.f.logs / 'process-ledger.json'
        with self.assertRaises(Exception): m.owned_small_bytes(path)
        self.assertEqual(m.owned_small_bytes(path, maximum=None), path.read_bytes())

    def test_path_closed_directory_and_invalid_calendar(self):
        original = copy.deepcopy(self.f.descriptor['files'])
        for replacement in ('outputs/2026-02-30/08-27-31/.hydra/config.yaml',
                            'outputs/2026-10-06/08-28-31/.hydra/config.yaml',
                            '../outside/config.yaml'):
            files = copy.deepcopy(original); key = next(key for key in files if key.endswith('/config.yaml'))
            files[replacement] = files.pop(key); self.f.descriptor['files'] = files
            with self.subTest(path=replacement), self.assertRaises(Exception): self.verify()
        self.f.descriptor['files'] = original

    def test_terminal_helper_frozen_exact_and_embedding_compile(self):
        for name in ('read_terminal_observation', 'verify_terminal_current', '_snapshot_binding', '_job_fields', '_scheduler_time'):
            self.assertEqual(inspect.getsource(getattr(m, name)), inspect.getsource(getattr(terminal, name)))
        compile(m.verification_code(), '<light reader>', 'exec')
        self.assertIn('verify_source_snapshot_sets(parent_base, discovery, launch, selection)', inspect.getsource(m.source_precheck))
        self.assertIn('verify_source_snapshot_sets(parent_base, discovery, launch, selection)', inspect.getsource(m.target_evidence))

    def test_missing_future_descriptor_fails_before_any_intent(self):
        f = prior.fixture()
        with patch.object(m, 'FROZEN_METHOD_SHA256', m.sha(f.method_raw)), self.assertRaises(Exception):
            m.selection_schema(f.selection, f.launch)

    def test_actual_old_metadata_is_only_parser_relationship_evidence(self):
        content = (HERE / 'policy-discovery-v3-hydra-schema-readonly-observation.json').read_bytes()
        self.assertEqual(m.sha(content), 'f61636633a8f5876f3882b512b2f1929e921b87d928b2464cf49a565c9647326')
        observation = m.strict_json(content)
        self.assertTrue(observation['config_matches_resolved'])
        self.assertTrue(observation['overrides_match_actual_argv'])
        self.assertEqual(len(observation['files']), 5)


if __name__ == '__main__':
    unittest.main()
