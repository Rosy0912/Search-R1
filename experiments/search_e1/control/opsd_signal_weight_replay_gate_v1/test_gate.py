"""Small synthetic evidence fixtures; original replay GPU execution is never used."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    from . import gate
    from . import common as c
except ImportError:
    import gate
    import common as c


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve(); self.n = 0
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', E1_RUN_ID='synthetic-replay'); self.env.start()
        self.uid = patch.object(os, 'getuid', return_value=1001); self.uid.start()

    def tearDown(self):
        self.uid.stop(); self.env.stop(); self.tmp.cleanup()

    def write(self, value, name=None):
        self.n += 1; path = self.root / (name or ('ref-%d.json' % self.n)); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

    def source(self, root):
        files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in root.rglob('*') if p.is_file() and p.name != 'source-manifest.json'}
        manifest = self.write({'files': files})
        publication = self.write(dict(accepted=True, branch='search-e1', source_commit='a' * 40, remote_readback_sha='a' * 40,
                                      source_manifest_sha256=manifest['sha256']))
        return dict(helper_root=str(root), source_manifest=manifest, publication=publication)

    def packet(self, output, source):
        stdout = self.write(output)
        execution = self.write(dict(accepted=True, actual_wait_returncode=0, torch_imported=False, stdout=stdout,
                                    source_manifest=source['source_manifest'], publication=source['publication']))
        return dict(source, actual_stdout=stdout, output=stdout, execution=execution)

    def fixture(self):
        original = self.root / 'original'; original.mkdir(); (original / '__init__.py').write_text('# ' + self.root.name + '\n')
        (original / 'replay.py').write_text('''import hashlib,json,os\nfrom pathlib import Path\nCALLS=[]\ndef run(plan,output):\n CALLS.append(os.getpid())\n root=Path(output);root.mkdir(parents=True)\n p=root/'complete.json'\n p.write_text(json.dumps(dict(accepted=True,plan=plan,posthoc_responses=4608,new_samples_or_training=0,actual_workers_exit_and_release=True,final_release={'accepted':True})))\n return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)\n''')
        old_source = self.source(original); own_source = self.source(Path(gate.__file__).resolve().parent)
        validation = self.write({'synthetic_validation': True}); configs = {}; finals = {}; native_groups = []; strong_groups = []
        for variant in c.VARIANTS:
            config = self.write(dict(variant=variant, study_id='study', run_id=variant,
                                     helper_source_manifest={'sha256': c.ORIGINAL_SHA}, validation_identity=validation))
            configs[variant] = config
            final = self.write(dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=variant,
                config_sha256=config['sha256'], cursor={'outer_index': 24, 'next_pair_cursor': 7, 'usable_pair_count': 7},
                complete_rank_count=4, all_rank_adapter_sha256=['d' * 64] * 4))
            finals[variant] = final; ledger = self.write({'synthetic_ledger': True}); actual = self.write({'actual_wait_returncode': 0})
            native_groups.append(dict(variant=variant, actual_wait_returncode=0, actual_terminal=actual, summary=ledger, release={'accepted': True}))
            strong_groups.append(dict(variant=variant, config=config, train_ledger=ledger, actual_terminal=actual, actual_controller_wait_returncode=0))
        study = self.write(dict(study_base=str(self.root), study_id='study', groups=[dict(variant=v, config=configs[v]) for v in c.VARIANTS]))
        native = self.write(dict(accepted=True, completed_groups=4, completed_outers=100, full_training_em_rows=100, study_manifest=study, groups=native_groups))
        strong = self.write(dict(schema='opsd-signal-weight-driver-terminal-v1', accepted=True, actual_supervisor_wait_returncode=0,
                driver_exit_evidence=dict(source='same-retained-tmux-pane', pane_dead=1, pane_dead_status=0, no_exit_code_inferred_from_absence=True),
                owned_gpu_release={'accepted': True}, study_manifest=study, native_study_completion=native, groups=strong_groups))
        endpoints = []
        for version, t in sorted(gate.ENDPOINTS):
            variant = 'G_POS' if t == 0 else version; cp = None
            if t: cp = finals[variant] if t == 25 else self.write(dict(cursor={'outer_index': 14, 'next_pair_cursor': 9, 'usable_pair_count': 9}, all_rank_adapter_sha256=['d' * 64] * 4))
            endpoints.append(self.write(dict(schema='opsd-signal-native512-complete-v1', accepted=True, version=version, completed_outer=t,
                     primary_slots=512, actual_child_exit_and_release=True, run_id=variant, config=configs[variant],
                     raw_checkpoint=None if t == 0 else {'kind': 'raw_end_of_outer', 'manifest': cp},
                     input_identity={'expected_adapter_tensor_sha256': 'd' * 64})))
        zero = next(ref for ref in endpoints if c.read(ref)[0]['completed_outer'] == 0)
        bank = self.write(dict(schema='opsd-signal-fixed-common-zero-bank-v1', accepted=True, study_id='study',
               frozen_before_all_group_15_25_outputs=True, configs=configs, validation_identity=validation, common_zero_evaluation=zero))
        plan = self.write(dict(schema='opsd-signal-heldout-replay-plan-v1', analysis_run_id='synthetic-replay',
                     helper_root=old_source['helper_root'], source_manifest=old_source['source_manifest'], publication=old_source['publication'],
                     four_group_terminal=native, configs=list(configs.values()), bank=bank, endpoints=endpoints))
        coverage_request = self.write(dict(study_manifest=study, study_terminal=strong, native_study_completion=native,
                         groups=[dict(variant=v, config=configs[v]) for v in c.VARIANTS]))
        coverage = self.write(dict(schema='opsd-signal-final-coverage-v1', accepted=True, coverage_accepted=True,
                     complete_groups=4, complete_outers=100, train_trajectory_slots=256000, formal_native512_slots=4608,
                     final_checkpoints=finals, study_manifest=study, study_terminal=strong, native_study_completion=native,
                     torch_imported=False, final_accepted=False, request=coverage_request,
                     evaluations=[{'child': e} for e in endpoints], outer_rows=[{'variant': v, 'outer_index': i} for v in c.VARIANTS for i in range(25)]))
        coverage_packet = self.packet(dict(accepted=True, final_accepted=False, output=coverage), own_source)
        coverage_packet['output'] = coverage
        request = dict(schema='opsd-signal-posthoc-replay-gate-request-v1', gate_source=own_source,
               old_plan=plan, study_manifest=study, strong_terminal=strong, native_study_completion=native,
               configs=configs, final_checkpoints=finals, bank=bank, endpoints=endpoints,
               initial_coverage=coverage_packet, replay_output=str(self.root / 'analysis/replay'), preflight_output=str(self.root / 'analysis/gate/preflight.json'))
        entry = str(original / 'replay.py'); argv = ['gate.py', '--original-replay-entry', entry]
        return request, entry, argv

    def context(self):
        return patch.object(gate, 'STUDY_ROOT', str(self.root))

    def test_final_false_coverage_allows_preflight_without_final_dependency_deadlock(self):
        request, entry, argv = self.fixture()
        with self.context():
            result = gate.preflight(request, entry, argv)
        self.assertTrue(result['accepted']); self.assertFalse(result['initial_coverage']['final_accepted'])
        self.assertFalse(Path(request['replay_output']).exists())

    def test_same_process_run_calls_original_source_without_subprocess_or_patch(self):
        request, entry, argv = self.fixture(); ref = self.write(request)
        with self.context(), patch('subprocess.Popen', side_effect=AssertionError('wrapper subprocess forbidden')):
            result = gate.run(ref, entry, argv)
        self.assertEqual(result['same_process_pid'], os.getpid()); self.assertFalse(result['scientific_functions_patched'])
        self.assertTrue(Path(request['preflight_output']).is_file())

    def test_native_or_proc_absence_cannot_replace_strong_wait0(self):
        request, entry, argv = self.fixture(); strong, _ = c.read(request['strong_terminal'])
        strong['driver_exit_evidence']['pane_dead_status'] = 'UNKNOWN'; request['strong_terminal'] = self.write(strong)
        with self.context(), self.assertRaisesRegex(ValueError, 'Strong supervisor'):
            gate.preflight(request, entry, argv)

    def test_actual_argv_original_entry_is_required(self):
        request, entry, argv = self.fixture()
        with self.context(), self.assertRaisesRegex(ValueError, 'actual CLI|Actual CLI'):
            gate.preflight(request, entry, ['gate.py'])

    def test_coverage_report_not_falsely_labeled_as_actual_stdout(self):
        request, entry, argv = self.fixture(); request['initial_coverage']['actual_stdout'] = request['initial_coverage']['output']
        with self.context(), self.assertRaises(ValueError): gate.preflight(request, entry, argv)

    def test_partial_coverage_or_missing_endpoint_rejected_before_run(self):
        request, entry, argv = self.fixture(); coverage, _ = c.read(request['initial_coverage']['output']); coverage['complete_outers'] = 99
        new = self.write(coverage); request['initial_coverage']['output'] = new
        cli = self.packet(dict(accepted=True, final_accepted=False, output=new), request['gate_source']); cli['output'] = new
        request['initial_coverage'] = cli
        with self.context(), self.assertRaisesRegex(ValueError, 'partial'): gate.preflight(request, entry, argv)

    def test_cache_or_other_final_cp_is_not_replay_raw(self):
        request, entry, argv = self.fixture(); ref = next(r for r in request['endpoints'] if c.read(r)[0]['completed_outer'] == 25)
        endpoint, _ = c.read(ref); endpoint['raw_checkpoint']['kind'] = 'merged_cache'; changed = self.write(endpoint)
        request['endpoints'] = [changed if r == ref else r for r in request['endpoints']]
        plan, _ = c.read(request['old_plan']); plan['endpoints'] = request['endpoints']; request['old_plan'] = self.write(plan)
        with self.context(), self.assertRaisesRegex(ValueError, 'cache/best'): gate.preflight(request, entry, argv)

    def test_no_precreation_of_original_replay_output(self):
        request, entry, argv = self.fixture(); request['preflight_output'] = str(Path(request['replay_output']) / 'preflight.json')
        with self.context(), self.assertRaisesRegex(ValueError, 'independent new remote'):
            gate.preflight(request, entry, argv)


if __name__ == '__main__':
    unittest.main()
