"""Stdlib scalar protocol fixtures only; no native/health/selection is produced."""
import copy
import importlib.util
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('local_v4_selection', HERE / 'prepare_policy_confirmation_v4_hydra_selection.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def readers():
    launch = m.strict_json((HERE / 'policy-discovery-v4-native-launch.json').read_bytes())
    return m.pure_readers(launch['source_sha256'])


def evaluation_fixture():
    ids = ['CPU-scalar-question-' + str(index) for index in range(512)]
    sources = ['nq'] * 256 + ['hotpotqa'] * 256
    rows = [dict(question_id=identity, data_source=source, em=1, f1=1., actual_calls=1,
        valid_action=2, turns=2, answer_present=True, answer_sha256=m.sha(b'CPU scalar answer hash'),
        response_token_count=4, model_token_count=2) for identity, source in zip(ids, sources)]
    return rows, ids, sources


class LocalSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = readers()

    def test_actual_fixed_method_cpu_and_birth_contract_but_RUNNING_is_rejected(self):
        raw = b'{}'
        birth = (HERE / 'policy-discovery-v4-zero-write-observation.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'terminal state'):
            m.prepare_selection(raw, evidence_sha256=m.sha(raw),
                method_bytes=(HERE / 'policy-calibration-v4-selection-method.json').read_bytes(),
                launch_bytes=(HERE / 'policy-discovery-v4-native-launch.json').read_bytes(),
                birth_bytes=birth, terminal_bytes=birth, terminal_sha256=m.sha(birth),
                preregistration_bytes=(HERE / 'policy-calibration-v4-method-preregistration.json').read_bytes(),
                cpu_bytes=(HERE / 'behavior-consistency-target-cpu-v2-result.json').read_bytes())

    def test_missing_terminal_and_explicit_files_reject_without_writer(self):
        result = subprocess.run([sys.executable, str(HERE / 'prepare_policy_confirmation_v4_hydra_selection.py')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--terminal', result.stderr)
        self.assertNotIn('--output', inspect.getsource(m.main))
        self.assertNotIn('.write_', inspect.getsource(m))
        self.assertNotIn('subprocess', inspect.getsource(m))

    def test_wrong_original_method_and_evidence_SHA_rejected_first(self):
        kwargs = dict(evidence_sha256=m.sha(b'{}'), method_bytes=b'{}', launch_bytes=b'',
            birth_bytes=b'', terminal_bytes=b'', terminal_sha256='0' * 64,
            preregistration_bytes=b'{}', cpu_bytes=b'')
        with self.assertRaisesRegex(ValueError, 'root61eeb'):
            m.prepare_selection(b'{}', **kwargs)
        kwargs['method_bytes'] = (HERE / 'policy-calibration-v4-selection-method.json').read_bytes()
        kwargs['preregistration_bytes'] = (HERE / 'policy-calibration-v4-method-preregistration.json').read_bytes()
        kwargs['evidence_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'scalar-input SHA'):
            m.prepare_selection(b'{}', **kwargs)

    def test_exact_frozen_aggregate_without_Torch_import(self):
        rows, ids, sources = evaluation_fixture()
        value = self.r.aggregate(rows, ids, sources)
        self.assertEqual(value['question_count'], 512)
        self.assertEqual(value['weighted_em'], 1.)
        self.assertEqual(value['weighted_f1'], 1.)
        self.assertEqual(value['valid_action_rate'], 1.)
        self.assertEqual(value['executed_search_calls_per_question'], 1.)
        self.assertNotIn('torch', sys.modules)
        bad = copy.deepcopy(rows); bad[511]['question_id'] = ids[510]
        with self.assertRaisesRegex(ValueError, 'ordered question/source'):
            self.r.aggregate(bad, ids, sources)
        bad = copy.deepcopy(rows); bad[511]['raw_answer'] = 'Forbidden fixture text'
        with self.assertRaisesRegex(ValueError, 'unexpected measurement row fields'):
            self.r.aggregate(bad, ids, sources)

    def test_true_8rank_ledger_reconstruction_once_and_rank_drift_rejected(self):
        # Fabricated scalar ledgers test parser arithmetic only. No full-native
        # bundle, successful selection, physical receipt or tensor is invented.
        sys.path.insert(0, str(m.REPO))
        from reproduction.common8.test_safety_calibration_evidence import fabricated_segment
        logs, binding = fabricated_segment(role='policy_safety_calibration_discovery')
        binding['run_id'] = '1' * 32
        gate = m.sha(b'CPU scalar public receipt')
        for rows in logs.values():
            for row in rows:
                row['run_id'] = binding['run_id']
                if row['kind'] in ('POLICY_MONITOR', 'OPTIMIZER_WRITE'):
                    row['context']['behavior_probability_gate_sha256'] = gate
                if row['kind'] == 'POLICY_MONITOR':
                    row.update(sampled_old_policy_kl=.0005, weighted_k3_numerator=.008,
                        tail_effective_weight_mass=0., negative_tail_count=0,
                        negative_tail_count_fraction=0., negative_tail_weight_fraction=0.,
                        dual_clip_saturated_objective_weight_fraction=0.,
                        negative_log_ratio_p95=.01, negative_log_ratio_p99=.01, negative_log_ratio_max=.01)
        structural = self.r.ledger.verify_segment(logs, binding)
        raw = {'discovery/logs/optimizer-rank-' + str(rank) + '.jsonl':
            b'\n'.join(m.canonical(row) for row in rows) + b'\n' for rank, rows in logs.items()}
        segment = dict(role=binding['role'] + '_evidence', run_id=binding['run_id'],
            plan_sha256=binding['calibration_sha256'], structural=structural,
            worker_log_sha256={str(rank): m.sha(raw['discovery/logs/optimizer-rank-' + str(rank) + '.jsonl']) for rank in range(8)})
        rebuilt = m.check_ledgers(raw, segment, self.r)
        self.assertEqual(len(rebuilt['global_monitors']), 40)
        self.assertEqual(rebuilt['optimizer_writes_per_rank'], 20)
        self.assertIs(rebuilt['production_admission'], False)
        changed = copy.deepcopy(logs); changed[7][0]['negative_tail_weight_fraction'] = .1
        name = 'discovery/logs/optimizer-rank-7.jsonl'
        changed_raw = dict(raw); changed_raw[name] = b'\n'.join(m.canonical(row) for row in changed[7]) + b'\n'
        changed_segment = copy.deepcopy(segment); changed_segment['worker_log_sha256']['7'] = m.sha(changed_raw[name])
        with self.assertRaises(ValueError): m.check_ledgers(changed_raw, changed_segment, self.r)
        changed = copy.deepcopy(logs); changed[7][0]['input_ids'] = [1, 2, 3]
        changed_raw[name] = b'\n'.join(m.canonical(row) for row in changed[7]) + b'\n'
        changed_segment['worker_log_sha256']['7'] = m.sha(changed_raw[name])
        with self.assertRaisesRegex(ValueError, 'Unknown/raw token'):
            m.check_ledgers(changed_raw, changed_segment, self.r)

    def test_native_hydra_descriptor_only_from_actual_supplied_bytes(self):
        job, run_id = '999999', '1' * 32
        launch = dict(job_id=job, source_sha256={'fixture.py': m.sha(b'fixture')})
        raw = {'discovery/logs/training-launch-claim.json': m.canonical(dict(run_id=run_id, slurm_job_id=job, claimed_unix=1.)),
            'discovery/logs/resolved-training-config.json': m.canonical(dict(run_id=run_id, slurm_job_id=job,
                checked_unix=2., resolved_config={}, resolved_config_sha256=m.sha(m.canonical({}))))}
        extras = {'discovery/repo/outputs/2026-10-06/10-00-00/.hydra/' + name for name in ('config.yaml', 'hydra.yaml', 'overrides.yaml')}
        raw.update({name: b'# CPU scalar metadata only\n' for name in extras})
        descriptor = m.hydra_descriptor(raw, extras, {'run_id': run_id}, launch)
        self.assertEqual(set(descriptor['files']), {name.removeprefix('discovery/repo/') for name in extras})
        receipt = dict(role='exact_generated_hydra_file_snapshot_verification', run_id=run_id, slurm_job_id=job,
            source_map_sha256=descriptor['source_map_sha256'], descriptor_sha256=m.sha(m.canonical(descriptor)),
            source_snapshot_kind='current_disk_not_loaded_module_proof', production_admission=False,
            healthy_checkpoint_accepted=False, physical_execution_authenticated=False)
        m.check_snapshot_receipt(receipt, descriptor)
        bad = dict(receipt, physical_execution_authenticated=True)
        with self.assertRaisesRegex(ValueError, 'not physical proof'): m.check_snapshot_receipt(bad, descriptor)
        bad = copy.deepcopy(descriptor); bad['files'][next(iter(bad['files']))]['sha256'] = '0' * 64
        with self.assertRaises(ValueError): m.check_snapshot_receipt(receipt, bad)

    def test_actual_paired_scalar_arithmetic_and_zero_drop_not_caller_boolean(self):
        rows, ids, sources = evaluation_fixture()
        run = dict(run_id='1' * 32, parent_production_run_id='2' * 32,
            policy_calibration_slurm_job_id='999999', input_sha256={'validation.parquet': m.sha(b'CPU dataset bytes')})
        canonical = dict(validation_sha256=run['input_sha256']['validation.parquet'], question_ids=ids, data_sources=sources)
        segment = dict(role='policy_safety_calibration_discovery_evidence', plan_sha256=m.sha(b'CPU plan'), evaluation_sha256={})
        raw = {}
        for label, step in (('before', 0), ('after', 1)):
            value = self.r.aggregate(rows, ids, sources)
            value.update(run_id=run['run_id'], parent_production_run_id=run['parent_production_run_id'],
                slurm_job_id=run['policy_calibration_slurm_job_id'], plan_sha256=segment['plan_sha256'],
                step=step, phase_role=segment['role'].removesuffix('_evidence'),
                validation_sha256=canonical['validation_sha256'], checked_unix=100.)
            name = 'discovery/logs/policy-calibration-evaluation-' + label + '.json'
            raw[name] = m.canonical(value); segment['evaluation_sha256'][label] = m.sha(raw[name])
            segment['evaluation_' + label] = value
        m.check_evaluations(raw, segment, run, canonical, self.r.aggregate)
        bad = copy.deepcopy(segment)
        bad['evaluation_after']['rows'][511]['em'] = 0
        bad['evaluation_after']['rows'][511]['f1'] = .5
        changed = self.r.aggregate(bad['evaluation_after']['rows'], ids, sources)
        bad['evaluation_after'].update(changed)
        name = 'discovery/logs/policy-calibration-evaluation-after.json'
        changed_raw = dict(raw, **{name: m.canonical(bad['evaluation_after'])})
        bad['evaluation_sha256']['after'] = m.sha(changed_raw[name])
        with self.assertRaisesRegex(ValueError, 'zero-allowed-drop'):
            m.check_evaluations(changed_raw, bad, run, canonical, self.r.aggregate)
        bad = copy.deepcopy(segment); bad['evaluation_after']['weighted_em'] = .1
        changed_raw[name] = m.canonical(bad['evaluation_after'])
        bad['evaluation_sha256']['after'] = m.sha(changed_raw[name])
        with self.assertRaisesRegex(ValueError, 'arithmetic differs'):
            m.check_evaluations(changed_raw, bad, run, canonical, self.r.aggregate)

    def test_fresh_checkpoint_only_scalar_receipt_and_canonical435(self):
        run = dict(run_id='1' * 32, checkpoint_dir='/mnt/data/shaoshitong/g1d-cpufixture/checkpoints')
        complete = dict(run_id=run['run_id'], checkpoint=run['checkpoint_dir'] + '/actor/global_step_1',
            step=1, validated_tensor_count=435, finite_all_saved_tensors=True,
            files_sha256={'model.safetensors.index.json': m.sha(b'CPU checkpoint index pointer')})
        segment = dict(checkpoint=complete, changed_parameters=['CPU.parameter.name'])
        raw = {'discovery/checkpoint-complete.json': m.canonical(complete)}
        result = m.check_checkpoint(raw, segment, run, '/mnt/data/shaoshitong/g1d-cpufixture')
        self.assertIs(result['independent_tensor_verification_pending'], True)
        self.assertNotIn('accepted', result)
        for value in (434, True):
            bad = copy.deepcopy(segment); bad['checkpoint']['validated_tensor_count'] = value
            changed_raw = {'discovery/checkpoint-complete.json': m.canonical(bad['checkpoint'])}
            with self.assertRaisesRegex(ValueError, 'canonical435'):
                m.check_checkpoint(changed_raw, bad, run, '/mnt/data/shaoshitong/g1d-cpufixture')

    def test_public_receipt_update_stat_and_context_links_without_raw_journal(self):
        run_id, job, trace_sha = '1' * 32, '999999', m.sha(b'CPU private trace pointer only')
        ids = ['CPU-trajectory-' + str(index) for index in range(2560)]
        sources = {'fixture.py': m.sha(b'CPU source bytes')}
        sampling = dict(n=1, do_sample=True, temperature=1., top_p=1., top_k=-1, min_p=0.,
            presence_penalty=0., frequency_penalty=0., repetition_penalty=1., ignore_eos=False,
            logprobs=1, max_tokens=500, min_tokens=0, logits_processors=[], best_of=1,
            logit_bias=None, allowed_token_ids=None, guided_decoding=None, truncate_prompt_tokens=None,
            logprob_semantics='selected_token_logprob_at_actual_sampling_call')
        gate = dict(schema_version=1, role='common8_behavior_probability_gate', accepted=True,
            run_id=run_id, source_sha256=sources, slurm_job_id=job, logical_update=1,
            policy_snapshot_id='CPU scalar snapshot', sampling=sampling, trajectory_count=2560,
            sampling_distribution='full_softmax; actual neutral filters and penalties verified',
            trajectory_ids=ids, actor_token_counts=[2] * 2560, included_actor_token_count=5120,
            max_abs_discrepancy=.0005, tolerance=.05, trace_sha256=trace_sha,
            actor_rows_sha256=m.sha(b'CPU actor row pointer'), audit_sha256=m.sha(b'CPU audit pointer'),
            tensor_sha256={name: m.sha(name.encode()) for name in ('hf_old_log_probs', 'sampling_log_probs', 'actor_mask', 'attention_mask')},
            hf_dtype='torch.float32', sampling_dtype='torch.float32', actual_frozen_old_all_finite=True,
            scope='CPU fabricated public-receipt protocol only', production_admission=False,
            healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
            row_tensor_sha256={identity: m.sha(identity.encode()) for identity in ids},
            upstream_group_sha256={identity: m.sha(b'CPU group pointer') for identity in ids},
            private_trace_sha256=trace_sha,
            denominator='actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate')
        gate_raw = m.canonical(gate); gate_sha = m.sha(gate_raw)
        summary = dict(proof_sha256=gate_sha, private_trace_sha256=trace_sha,
            included_actor_token_count=5120, max_abs_discrepancy=.0005)
        update = dict(run_id=run_id, step=1, probe_version='CPU probe version',
            metrics={'actor/optimizer_steps_this_update': 20.},
            behavior_probability_gate_sha256=gate_sha, behavior_trace_sha256=trace_sha)
        update_raw = m.canonical(update) + b'\n'
        raw = {'discovery/logs/behavior-probability-outer-1.json': gate_raw,
               'discovery/logs/actual-updates.jsonl': update_raw}
        run = dict(run_id=run_id, source_sha256=sources, policy_calibration_slurm_job_id=job,
                   initial_probe_version='CPU probe version')
        segment = dict(behavior_probability_gate=summary, actual_updates=[update],
            structural={'global_monitors': [dict(context={'behavior_probability_gate_sha256': gate_sha})] * 40})
        stat = dict(size_bytes=len(update_raw), sha256=m.sha(update_raw), mtime_unix=2.)
        result = m.check_gate_and_update(raw, segment, run, {'frozen_unix': 1.}, stat, self.r.verify_sampling)
        self.assertEqual(result, summary)
        self.assertNotIn('accepted', result)
        for change in ('source', 'probability', 'top_p', 'raw_payload'):
            bad = copy.deepcopy(gate)
            if change == 'source': bad['source_sha256'] = {}
            if change == 'probability': bad['max_abs_discrepancy'] = .051
            if change == 'top_p': bad['sampling']['top_p'] = .95
            if change == 'raw_payload': bad['raw_generated_token_ids'] = [1]
            changed = dict(raw, **{'discovery/logs/behavior-probability-outer-1.json': m.canonical(bad)})
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.check_gate_and_update(changed, segment, run, {'frozen_unix': 1.}, stat, self.r.verify_sampling)
        with self.assertRaisesRegex(ValueError, 'prospective method'):
            m.check_gate_and_update(raw, segment, run, {'frozen_unix': 1.}, dict(stat, mtime_unix=.5), self.r.verify_sampling)
        bad = copy.deepcopy(segment)
        bad['structural']['global_monitors'][39] = dict(context={'behavior_probability_gate_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'context changed'):
            m.check_gate_and_update(raw, bad, run, {'frozen_unix': 1.}, stat, self.r.verify_sampling)

    def test_empirical_max_has_no_margin_floor_and_keeps_context4(self):
        reader = m.confirmation_reader()
        run_id, plan_sha, gate = '1' * 32, m.sha(b'CPU plan'), m.sha(b'CPU gate')
        observations = []
        for ordinal in range(40):
            mini, post = ordinal // 2, ordinal % 2
            observations.append(dict(context=dict(logical_update=1, minibatch_index=mini,
                prior_optimizer_writes=mini, behavior_probability_gate_sha256=gate),
                stage=('before_minibatch', 'after_minibatch')[post], run_id=run_id,
                calibration_sha256=plan_sha, optimizer_writes=mini + post,
                execution_role=reader.DISCOVERY_ROLE, enforcement={'negative_tail': 'observe_only'},
                allowed=True, reasons=[], negative_tail_weight_fraction=.003 if ordinal == 39 else 0.))
        segment = dict(structural=dict(execution_role=reader.DISCOVERY_ROLE, run_id=run_id,
            calibration_sha256=plan_sha, outer_count=1, optimizer_writes_per_rank=20,
            enforcement={'negative_tail': 'observe_only'}, global_monitors=observations),
            behavior_probability_gate={'proof_sha256': gate})
        selection = dict(discovery_run_id=run_id, discovery_plan_sha256=plan_sha, negative_tail_limit=.003)
        self.assertEqual(reader.measured_selection(selection, segment), .003)
        for value in (.003000000001, .01, 0.):
            with self.assertRaisesRegex(ValueError, 'no margin/floor'):
                reader.measured_selection(dict(selection, negative_tail_limit=value), segment)
        bad = copy.deepcopy(segment); bad['structural']['global_monitors'][39]['context'].pop('behavior_probability_gate_sha256')
        with self.assertRaisesRegex(ValueError, 'global observation'):
            reader.measured_selection(selection, bad)

    def test_no_unknown_future_fields_or_optimizer_raw_transport(self):
        import base64
        packet = {'files': {name: dict(base64=base64.b64encode(b'{}').decode(), sha256=m.sha(b'{}'))
                    for name in m.METADATA_FILES}}
        with self.assertRaisesRegex(ValueError, 'exactly three'): m.original_files(packet)
        for name in ('config.yaml', 'hydra.yaml', 'overrides.yaml'):
            packet['files']['discovery/repo/outputs/2026-10-06/10-00-00/.hydra/' + name] = dict(
                base64=base64.b64encode(b'fixture').decode(), sha256=m.sha(b'fixture'))
        m.original_files(packet)
        packet['files']['discovery/logs/policy-calibration-optimizer-rank-0.json'] = dict(
            base64=base64.b64encode(b'{}').decode(), sha256=m.sha(b'{}'))
        with self.assertRaisesRegex(ValueError, 'Closed metadata set'): m.original_files(packet)
        self.assertNotIn('actual_optimizer_states', m.SCALAR_SEGMENT_FIELDS)
        self.assertNotIn('actual_training_inputs', m.SCALAR_SEGMENT_FIELDS)
        self.assertNotIn('actual_training_groups', m.SCALAR_SEGMENT_FIELDS)


if __name__ == '__main__':
    unittest.main()
