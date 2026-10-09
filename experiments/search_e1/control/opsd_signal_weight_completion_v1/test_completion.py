"""Synthetic stdlib fixtures only: no real questions, weights, remote or models."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
try:
    from . import assemble as a
    from . import common as c
except ImportError:
    import assemble as a
    import common as c


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve(); self.serial = 0
        self.env = patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''); self.env.start()
        self.config = dict(variant='G_POS', run_id='synthetic-run', owner=dict(uid=1001), _config_sha256='c' * 64)

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def write(self, value, name=None):
        self.serial += 1; path = self.root / (name or ('f%d.json' % self.serial))
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

    def raw(self, text, name):
        path = self.root / name; path.write_text(text)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

    def jsonl(self, records, name):
        return self.raw(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records), name)

    def native(self):
        root = Path(os.environ.get('OPSD_COMPLETION_TEST_ORIGINAL_HELPER_ROOT',
                         str(Path(__file__).resolve().parent.parent / 'opsd_signal_weight_v1'))).resolve()
        manifest = root / 'source-manifest.json'
        return c.bind_helper(root, {'path': str(manifest), 'sha256': c.ORIGINAL_SHA}, ('checkpoint', 'training_metrics'))

    def exits(self, count, phase='training-running'):
        identities = [dict(identity=dict(pid=10 + i, uid=1001, starttick=100 + i, run_id='synthetic-run'), actual_wait_returncode=0) for i in range(count)]
        terminal = self.write(dict(run_id='synthetic-run', config_sha256='c' * 64, phase=phase, child_exit_evidence=identities))
        return dict(accepted=True, child_exit_evidence=identities, actual_terminal=terminal,
                    gpu_release=dict(all_owned_release=dict(accepted=True)))

    def packet(self, output, torch=False):
        payload = self.raw('"synthetic published helper"\n', 'source%d.py' % self.serial)
        name = Path(payload['path']).name
        manifest = self.write(dict(files={name: dict(sha256=payload['sha256'], bytes=payload['bytes'])}))
        publication = self.write(dict(accepted=True, source_commit='a' * 40, remote_readback_sha='a' * 40,
                                      source_manifest_sha256=manifest['sha256']))
        stdout = self.write(output)
        execution = self.write(dict(accepted=True, actual_wait_returncode=0, stdout=stdout,
                                    source_manifest=manifest, publication=publication, torch_imported=torch))
        return dict(output=stdout, execution=execution, source_manifest=manifest,
                    publication=publication, helper_root=str(self.root))

    def test_exact_sha_and_exclusive_output(self):
        ref = self.write(dict(v=1)); c.read(ref)
        Path(ref['path']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            c.read(ref)
        out = self.root / 'new.json'; c.exclusive(out, dict(accepted=True))
        with self.assertRaises(FileExistsError):
            c.exclusive(out, dict(accepted=True))

    def test_parsed_bytes_must_match_pin_after_checked_file_return(self):
        original = c.checked_file
        for reader, suffix in ((c.read, '.json'), (c.rows, '.jsonl')):
            ref = self.raw('{"original":true}\n', 'read-race' + suffix)
            def replace_after_hash(bound):
                proof = original(bound)
                Path(proof['path']).write_text('{"replacement":true}\n')
                return proof
            with patch.object(c, 'checked_file', side_effect=replace_after_hash):
                with self.assertRaisesRegex(ValueError, 'pinned SHA'):
                    reader(ref)

    def test_unknown_wait_wrong_phase_and_duplicate_pid_rejected(self):
        proof = self.exits(2); a.wait_release(proof, 2, self.config, 'training-running')
        bad = copy.deepcopy(proof); bad['child_exit_evidence'][0]['actual_wait_returncode'] = 'UNKNOWN'
        with self.assertRaises(ValueError): a.wait_release(bad, 2, self.config, 'training-running')
        with self.assertRaises(ValueError): a.wait_release(proof, 2, self.config, 'sampling-running')
        bad = copy.deepcopy(proof); bad['child_exit_evidence'][1]['identity'] = bad['child_exit_evidence'][0]['identity']
        with self.assertRaises(ValueError): a.wait_release(bad, 2, self.config, 'training-running')

    def test_bridge_preserves_native_body_and_source_receipt(self):
        cfg = self.write(self.config); config = dict(self.config, _config_sha256=cfg['sha256'])
        exits = self.exits(8, 'sampling-running')
        terminal, _ = c.read(exits['actual_terminal']); terminal['config_sha256'] = cfg['sha256']; exits['actual_terminal'] = self.write(terminal)
        body = dict(accepted=True, variant='G_POS', outer_index=3, overall=dict(denominator=2560, strict_correct=1280))
        sample = self.write(dict(run_id='synthetic-run', train_em=body, exit_proof=exits))
        before = Path(sample['path']).read_bytes(); result = a.bridge(cfg, sample, self.root / 'bridge.json')
        value, _ = c.read(result)
        self.assertEqual({key: value[key] for key in body}, body); self.assertEqual(value['config']['sha256'], cfg['sha256'])
        self.assertEqual(Path(sample['path']).read_bytes(), before)
        with self.assertRaises(ValueError): a.bridge(cfg, sample, sample['path'])

    def test_actual_native_final_ledger_has_only_row_config_sha(self):
        ledger = dict(accepted=True, run_id=self.config['run_id'], variant='G_POS', outer_iterations=25,
             rows=[dict(outer_index=i, run_id=self.config['run_id'], variant='G_POS', config_sha256='c' * 64) for i in range(25)])
        self.assertNotIn('config_sha256', ledger)
        a.ledger_identity(ledger, self.config, {'sha256': 'c' * 64})
        broken = copy.deepcopy(ledger); broken['rows'][24]['config_sha256'] = 'd' * 64
        with self.assertRaises(ValueError): a.ledger_identity(broken, self.config, {'sha256': 'c' * 64})
        broken = copy.deepcopy(ledger); broken['rows'][24]['outer_index'] = 23
        with self.assertRaises(ValueError): a.ledger_identity(broken, self.config, {'sha256': 'c' * 64})

    def test_cached_token_inventory_fresh_sha_counts_and_committed_only(self):
        token = self.raw('{"synthetic":true}\n', 'tokens.jsonl'); score = self.write(dict(synthetic=True)); batch = self.write(dict(synthetic=True))
        update = self.write(dict(score_batch=batch, global_actual_pairs=1,
                     all_rank_actual_token_metrics=[dict(rank=0, pairs=[dict(signal_score=score, token_log=token, valid_tokens=7)])]))
        cp = dict(outer_update_receipts=[update]); files = [c.checked_file(score), c.checked_file(token), c.checked_file(update), c.checked_file(batch)]
        output = dict(referenced_record_inventory_sha256=c.canonical(files), referenced_record_file_count=4,
                      actual_pair_exposures=1, valid_token_rows=7)
        self.raw('{"failed_uncommitted":true}\n', 'failed-attempt.jsonl')
        self.assertEqual(a.verify_token_inventory(cp, output), 7)
        bad = dict(output, actual_pair_exposures=2)
        with self.assertRaises(ValueError): a.verify_token_inventory(cp, bad)
        Path(token['path']).write_text('{"changed":true}\n')
        with self.assertRaises(ValueError): a.verify_token_inventory(cp, output)

    def test_k0_metadata_na_without_token_audit_and_no_invented_update(self):
        cp = dict(cursor=dict(usable_pair_count=0, next_pair_cursor=0), optimizer_updates=21, outer_update_receipts=[])
        self.assertEqual(a.token_audit(dict(token_audit=None), self.config, {}, cp, {}, 21)['state'], 'K0_METADATA_NA')
        for broken in (dict(cp, optimizer_updates=22), dict(cp, outer_update_receipts=[{}])):
            with self.assertRaises(ValueError): a.token_audit(dict(token_audit=None), self.config, {}, broken, {}, 21)
        with self.assertRaises(ValueError): a.token_audit(dict(token_audit={'accepted': False}), self.config, {}, cp, {}, 21)

    def test_external_missing_is_hold_and_source_wait_not_borrowed(self):
        configs = dict(G_POS=dict(study_id='synthetic-study')); finals = dict(G_POS=dict(path='cp', sha256='c' * 64))
        result = a.final_requirements({}, configs, finals, set(), [], [])
        self.assertFalse(any(row['accepted'] for row in result.values()))
        packet = self.packet(dict(accepted=True))
        self.assertTrue(c.execution_packet(packet)[0]['accepted'])
        execution, _ = c.read(packet['execution']); execution['actual_wait_returncode'] = 'UNKNOWN'; packet['execution'] = self.write(execution)
        with self.assertRaises(ValueError): c.execution_packet(packet)

    def test_external_numeric_may_import_torch_but_coverage_does_not(self):
        packet = self.packet(dict(accepted=True), torch=True)
        self.assertTrue(c.execution_packet(packet, require_no_torch=False)[0]['accepted'])
        with self.assertRaises(ValueError): c.execution_packet(packet)

    def test_native_numeric_final_schema_parsed_not_generic_envelope_claim(self):
        configs = {}; finals = {}; packets = {}
        for variant in c.VARIANTS:
            initial = self.write(dict(synthetic=True)); cp = self.write(dict(optimizer_updates=251,
                         all_rank_optimizer_tensor_sha256=['d' * 64] * 4, all_rank_adapter_sha256=['e' * 64] * 4))
            original_config = dict(path='config-' + variant, sha256='c' * 64)
            configs[variant] = dict(variant=variant, run_id=variant, _config_sha256='c' * 64,
                                   _original_config_ref=original_config, common_initial_adapter_manifest=initial)
            finals[variant] = cp
            output = dict(schema='opsd-ablation-independent-cpu-state-readback-v1', accepted=True,
                  variant=variant, run_id=variant, config=original_config, checkpoint=dict(manifest=cp),
                  outer_index=24, optimizer_updates=251, all_four_exact_full_adam=True,
                  ranks=[dict(rank=r, actual_adam_step=251, full_rng_validated=True, optimizer_tensor_sha256='d' * 64) for r in range(4)],
                  adapter_tensors=504, adapter_finite_fp32=True, adapter_tensor_sha256='e' * 64,
                  common_initial=initial, changed_tensors_vs_common_initial=504, cuda_initialized=False)
            packets[variant] = self.packet(output, torch=True)
        self.assertTrue(a.numeric_final(packets, configs, finals)['accepted'])
        bad = copy.deepcopy(packets); value, _ = c.read(bad['G_POS']['output']); value['ranks'][3]['actual_adam_step'] = 250
        bad['G_POS'] = self.packet(value, torch=True)
        with self.assertRaises(ValueError): a.numeric_final(bad, configs, finals)
        bad = dict(packets, G_POS=self.packet(dict(accepted=True, final_checkpoints=finals), torch=True))
        with self.assertRaises((ValueError, KeyError)): a.numeric_final(bad, configs, finals)

    def test_vocab_cap_is_shared_and_arithmetic_not_required_for_every_npy(self):
        vocab = self.root / 'vocabulary'; vocab.mkdir(); output = self.root / 'group-G_POS' / 'runs'; output.mkdir(parents=True)
        fake_array = self.raw('SYNTHETIC BYTE FILE ONLY\n', 'group-G_POS/runs/token.npy')
        ledger = self.write(dict(bytes=fake_array['bytes'], files=[fake_array], reservations=[{'status': 'complete'}]), 'vocabulary/ledger.json')
        cfg = dict(logging=dict(full_vocab_study_root=str(vocab)), paths=dict(output_root=str(output)))
        result = a.vocabulary_scope(dict(ledger=ledger), {v: cfg for v in c.VARIANTS})
        self.assertTrue(result['accepted']); self.assertFalse(result['all_positions_or_all_npy_arithmetic_required'])
        over = self.write(dict(bytes=24 * (1 << 30) + 1, files=[], reservations=[]), 'vocabulary/too-much.json')
        with self.assertRaises(ValueError): a.vocabulary_scope(dict(ledger=over), {v: cfg for v in c.VARIANTS})

    def test_source_only_compile_rejects_extra_python_and_modified_source(self):
        root = self.root / 'helper'; root.mkdir()
        (root / '__init__.py').write_text(''); (root / 'leaf.py').write_text('VALUE = 7\n')
        files = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in root.iterdir()}
        manifest = self.write(dict(files=files)); module = c.bind_helper(root, manifest, ('leaf',))['leaf']
        self.assertEqual(module.VALUE, 7)
        (root / 'extra.py').write_text('VALUE=0\n')
        with self.assertRaises(ValueError): c.bind_helper(root, manifest, ('leaf',))
        (root / 'extra.py').unlink(); (root / 'leaf.py').write_text('VALUE = 8\n')
        with self.assertRaises(ValueError): c.bind_helper(root, manifest, ('leaf',))

    def outer_zero(self):
        native = self.native(); cpmod = native['checkpoint']; initial = self.write({'synthetic_initial': True})
        config = dict(self.config, source_model_manifest={'sha256': 'm' * 64}, runtime_manifest={'sha256': 'r' * 64},
             common_initial_adapter_manifest=initial, common_initial_adapter_tensor_sha256='a' * 64,
             common_cohort_manifest={'sha256': 'o' * 64}, precision={'mode': 'bf16'}, scientific={'global_batch': 32})
        config_ref = {'path': str(self.root / 'config-synthetic.json'), 'sha256': 'c' * 64}
        questions = [dict(question_id='q%d' % i, question='synthetic %d' % i, answers=['gold'], data_source='nq' if i < 256 else 'hotpotqa') for i in range(512)]
        response = '<answer>wrong</answer>'
        rollouts = [dict(q, sibling=s, response=response, policy_spans=[[0, len(response)]], reward=0, search_calls=0, termination='answer') for q in questions for s in range(5)]
        qp = self.jsonl(questions, 'questions.jsonl'); rp = self.jsonl(rollouts, 'rollouts.jsonl'); pairs = self.jsonl([], 'pairs.jsonl')
        selection = dict(before={'cycle': 0, 'offset': 0}, after={'cycle': 0, 'offset': 512})
        identity = dict(checkpoint=None, optimizer_updates=0, source_model_identity='m' * 64, adapter_tensor_sha256='a' * 64, sha256='i' * 64)
        stats = {'P': 0}; contract = cpmod.science_contract(config)
        manifest = self.write(dict(outer_index=0, run_id=config['run_id'], config_sha256='c' * 64,
             runtime_identity='r' * 64, source_model_identity='m' * 64, contract=contract,
             common_initial_adapter_manifest=initial, common_cohort_manifest=config['common_cohort_manifest'],
             previous_complete_checkpoint=None, outer_old_adapter=initial, cumulative_optimizer_updates_before=0,
             questions=qp, rollouts=rp, pairs=pairs, train_cursor=selection, P=0, sampling_stats=stats,
             sampling_model_identity=identity, explicit_fresh=True))
        core = types.SimpleNamespace(reward=lambda response, answers, spans: int(response == '<answer>gold</answer>'))
        train = native['training_metrics'].calculate(rollouts, questions, 'G_POS', 0, identity, core)
        sampling = self.write(dict(schema='opsd-online-sampling-complete-v1', accepted=True, run_id=config['run_id'],
             train_em=train, rollouts=rp, pairs=pairs, stats=stats, sampling_model_identity=identity, exit_proof=self.exits(8, 'sampling-running')))
        order = self.write({'synthetic_order': True}); encoding = self.write({'synthetic_encoding': True})
        rank_files = [self.raw('SYNTHETIC DECLARED STATE ONLY %d' % r, 'rank-%d-state.pt' % r) for r in range(4)]
        adapter = self.root / 'adapter'; adapter.mkdir(); am = self.raw('SYNTHETIC NO TENSOR BYTES', 'adapter/adapter_model.safetensors'); ac = self.write({}, 'adapter/adapter_config.json')
        optimizer = self.raw('SYNTHETIC EMPTY DECLARED OPTIMIZER', 'optimizer-shared.pt')
        cpvalue = dict(schema='opsd-online-checkpoint-v1', accepted=True, run_id=config['run_id'], config_sha256='c' * 64,
             runtime_identity='r' * 64, source_model_identity='m' * 64, contract=contract, common_initial_adapter=initial,
             outer_old_adapter=initial, all_rank_scaler_state=[{}] * 4, optimizer_updates=0, cumulative_pair_exposure=0,
             complete_rank_count=4, equal_optimizer_steps=True, optimizer_storage='shared_exact_four_rank_optimizer_v1', shared_optimizer=optimizer,
             ranks=[dict(rank=r, file=rank_files[r]) for r in range(4)], all_rank_optimizer_tensor_sha256=['d' * 64] * 4,
             adapter=dict(path=str(adapter), files=[am, ac]), all_rank_adapter_sha256=['a' * 64] * 4,
             all_rng=['python', 'numpy', 'torch_cpu', 'cuda_all'], scheduler={'kind': 'none', 'warmup_updates': 0},
             cursor=dict(outer_index=0, next_pair_cursor=0, usable_pair_count=0),
             training_input=dict(outer_manifest=manifest, pairs=pairs, order=order, encoding=encoding),
             sampling_model_identity=identity, outer_update_receipts=[])
        cp = self.write(cpvalue)
        completion = self.write(dict(schema='opsd-online-outer-complete-v1', accepted=True, run_id=config['run_id'],
             outer_index=0, P=0, K=0, optimizer_updates_delta=0, optimizer_updates=0, cumulative_pair_exposure=0,
             scheduler_advanced_updates=0, checkpoint=cp))
        summary = native['training_metrics'].complete_outer_summary(train, c.read(completion)[0])
        summary.update(run_id=config['run_id'], config_sha256='c' * 64, raw_checkpoint=cp, sampling_model_sha256='i' * 64)
        entry = dict(outer_index=0, outer_manifest=manifest, sampling_complete=sampling, outer_complete=completion,
                     end_checkpoint=cp, trainer_exit=self.write(self.exits(1)), summary=self.write(summary), token_audit=None)
        native['sampling'] = types.SimpleNamespace(cohort=lambda config, cursor: (questions, selection),
                     verify_rollouts=lambda *args: ([], stats))
        return entry, config, config_ref, native, core

    def test_actual_native_zero_k_outer_and_no_exposure_fabrication(self):
        entry, config, cfg, native, core = self.outer_zero()
        result, summary, cursor = a.outer(entry, config, cfg, native, core, None, 0, 0, {'cycle': 0, 'offset': 0})
        self.assertEqual((result['K'], result['delta'], result['train_denominator'], result['token']['valid_token_rows']), (0, 0, 2560, 0))
        self.assertEqual(cursor['offset'], 512)
        complete, _ = c.read(entry['outer_complete']); complete['cumulative_pair_exposure'] = 1
        broken = dict(entry, outer_complete=self.write(complete))
        with self.assertRaisesRegex(ValueError, 'exposure'): a.outer(broken, config, cfg, native, core, None, 0, 0, {'cycle': 0, 'offset': 0})

    def test_actual_native_rejects_wrong_prior_chain_and_train_em(self):
        entry, config, cfg, native, core = self.outer_zero()
        manifest, _ = c.read(entry['outer_manifest']); manifest['cumulative_optimizer_updates_before'] = 1
        with self.assertRaisesRegex(ValueError, 'Cross-outer'):
            a.outer(dict(entry, outer_manifest=self.write(manifest)), config, cfg, native, core, None, 0, 0, {'cycle': 0, 'offset': 0})
        sampling, _ = c.read(entry['sampling_complete']); sampling['train_em']['overall']['strict_correct'] = 1
        with self.assertRaisesRegex(ValueError, 'TRAIN EM'):
            a.outer(dict(entry, sampling_complete=self.write(sampling)), config, cfg, native, core, None, 0, 0, {'cycle': 0, 'offset': 0})

    def endpoint(self):
        native = self.native(); root = Path(os.environ.get('OPSD_COMPLETION_TEST_ORIGINAL_HELPER_ROOT',
                         str(Path(__file__).resolve().parent.parent / 'opsd_signal_weight_v1'))).resolve()
        manifest = {'path': str(root / 'source-manifest.json'), 'sha256': c.ORIGINAL_SHA}
        original_eval = c.bind_helper(root, manifest, ('evaluation',))['evaluation']
        native['evaluation'] = types.SimpleNamespace(PROTOCOL=original_eval.PROTOCOL,
                    aggregate=lambda records, data, config, request, verify_traces: dict(accepted=True, overall={'hits': 239}, primary_slots=512))
        config = dict(self.config, source_model_manifest={'sha256': 'm' * 64}, runtime_manifest={'sha256': 'r' * 64},
                      common_initial_adapter_manifest=self.write({'synthetic': True}), study_id='s')
        cfg = {'path': str(self.root / 'cfg.json'), 'sha256': 'c' * 64}
        data = [{'synthetic_question': i} for i in range(512)]; validation = self.jsonl(data, 'validation.jsonl'); config['validation_identity'] = validation
        checkpoint = self.write(dict(cursor=dict(outer_index=14, next_pair_cursor=7, usable_pair_count=7), all_rank_adapter_sha256=['d' * 64] * 4))
        raw = dict(kind='raw_end_of_outer', manifest=checkpoint)
        identity = dict(protocol=original_eval.PROTOCOL, source_model_manifest=config['source_model_manifest'],
                      runtime={'manifest': config['runtime_manifest']}, validation=validation,
                      common_initial=config['common_initial_adapter_manifest'], expected_adapter_tensor_sha256='d' * 64, raw_checkpoint=raw)
        request = self.write(dict(input_identity=identity, completed_outer=15, version='G_POS', evaluation_id='ev', config=cfg))
        records = [dict(validation_index=i, synthetic=True) for i in range(512)]; results = self.jsonl(records, 'results.jsonl')
        proofs = []; exits = []
        loaded = dict(accepted=True, adapter_name='G_POS', layers=252, layer_checks=[{}] * 252, scaling=2., rank=16,
                      alpha=32, dropout=0., A_B_dtype='torch.float32')
        for rank in range(8):
            process = dict(pid=100 + rank, starttick=1000 + rank, run_id=config['run_id'], uid=1001)
            exits.append(dict(identity=process, actual_wait_returncode=0)); rr = self.jsonl(records[rank::8], 'worker-%d.jsonl' % rank)
            complete = self.write(dict(schema='opsd-signal-native512-worker-v1', accepted=True, status='COMPLETE',
                    evaluation_id='ev', run_id=config['run_id'], config=cfg, request=request, rank=rank, primary_rows=64,
                    results=rr, actual_loaded_adapter={'tensor_sha256': 'd' * 64}, actual_loaded_forward=loaded, process=process))
            proofs.extend([complete, rr])
        child = self.write(dict(schema='opsd-signal-native512-complete-v1', accepted=True, run_id=config['run_id'], config=cfg,
                variant='G_POS', study_id='s', completed_outer=15, primary_slots=512, actual_child_exit_and_release=True,
                no_training=True, checkpoint_metadata_unchanged=True, child_exit_evidence=exits, gpu_release={'released': True},
                input_identity=identity, raw_checkpoint=raw, results=results, request=request,
                summary=self.write(dict(accepted=True, overall={'hits': 239}, primary_slots=512)), worker_proofs=proofs,
                evaluation_id='ev', version='G_POS'))
        parent = self.write(dict(schema='opsd-signal-evaluation-parent-v1', accepted=True, run_id=config['run_id'], config_sha256='c' * 64,
                completed_outer=15, raw_checkpoint=checkpoint, child=child, isolated_child_no_training_rng_consumed=True,
                raw_state_unchanged=True, parent_torch_imported=False, exit_proof=self.exits(1, 'evaluation-running')))
        return parent, child, config, cfg, checkpoint, native, data

    def test_formal_raw_endpoint_512_eight_workers_not_partial_or_cache(self):
        parent, child, config, cfg, checkpoint, native, data = self.endpoint()
        result = a.evaluation(parent, child, config, cfg, 15, checkpoint, native, data)
        self.assertEqual((result['slots'], result['hits']), (512, 239))
        changed, _ = c.read(child); changed['raw_checkpoint']['kind'] = 'merged_model_cache'
        bad_child = self.write(changed); pp, _ = c.read(parent); pp['child'] = bad_child
        with self.assertRaisesRegex(ValueError, 'raw outer'):
            a.evaluation(self.write(pp), bad_child, config, cfg, 15, checkpoint, native, data)

    def test_formal_duplicate_worker_wait_or_nonzero_is_rejected(self):
        parent, child, config, cfg, checkpoint, native, data = self.endpoint()
        changed, _ = c.read(child); changed['child_exit_evidence'][7]['identity'] = changed['child_exit_evidence'][0]['identity']
        bad_child = self.write(changed); pp, _ = c.read(parent); pp['child'] = bad_child
        with self.assertRaisesRegex(ValueError, 'worker waits'):
            a.evaluation(self.write(pp), bad_child, config, cfg, 15, checkpoint, native, data)

    def test_backup_original_member_inventory_and_target_marker_sha(self):
        files = [self.raw('SYNTHETIC %d' % i, 'state-%d.txt' % i) for i in range(7)]
        cp = self.write(dict(ranks=[{'file': f} for f in files[:4]], shared_optimizer=files[4], adapter={'files': files[5:]}))
        refs = [cp] + files; members = {Path(r['path']).name: c.checked_file(r) for r in refs}
        cfg = {'_config_sha256': 'c' * 64, 'run_id': 'r', 'study_id': 's'}
        inventory = dict(variant='G_POS', checkpoint=cp, config={'sha256': 'c' * 64}, run_id='r', study_id='s',
                         member_bytes=sum(r['bytes'] for r in members.values()), member_count=len(members), members=members)
        marker = self.write(dict(schema='opsd-signal-pinned-backup-complete-v1', accepted=True,
                actual_tar_and_transfer_returncodes_zero=True, large_outputs_remote_only=True,
                inventory=inventory, inventory_sha256=c.canonical(inventory), member_readback=members))
        value = dict(accepted=True, already_complete=True, remote_backup=dict(accepted=True, state='COMPLETE_FRESH_SHA_READBACK',
                 receipt=marker, member_bytes=inventory['member_bytes'], member_count=len(members)), inventory_sha256=c.canonical(inventory),
                 inventory_actual_wait={'actual_ssh_wait_returncode': 0}, target_probe_actual_wait={'actual_ssh_wait_returncode': 0},
                 local_checkpoint_payload_files_created=0)
        packet = dict(self.packet(value), target_marker_copy=marker); expected = {('G_POS', cp['path'], cp['sha256'])}
        self.assertTrue(a.backups([packet], {'G_POS': cfg}, expected)['accepted'])
        wrong = self.write(dict(c.read(marker)[0], member_readback={}))
        bad = dict(packet, target_marker_copy=wrong)
        with self.assertRaises(ValueError): a.backups([bad], {'G_POS': cfg}, expected)
        with self.assertRaises(ValueError): a.backups([packet, packet], {'G_POS': cfg}, expected)

    def test_actual_analysis_cli_stdout_artifact_chain_not_forged_report_stdout(self):
        request = self.write(dict(synthetic_request=True)); report = self.write(dict(accepted=True, schema='synthetic-report', request=request))
        cli = self.packet(dict(accepted=True, output=report))
        packet = dict(cli, mode='native_cli_artifact', actual_stdout=cli['output'], output=report)
        value, req, proof = a.analysis_artifact(packet, 'effects')
        self.assertEqual(value['schema'], 'synthetic-report'); self.assertTrue(req['synthetic_request'])
        self.assertEqual(proof['execution']['output']['sha256'], cli['output']['sha256'])
        false_stdout = dict(packet, actual_stdout=report)
        with self.assertRaises(ValueError): a.analysis_artifact(false_stdout, 'effects')
        wrong_report = self.write(dict(accepted=True, schema='foreign-report', request=request))
        with self.assertRaises(ValueError): a.analysis_artifact(dict(packet, output=wrong_report), 'effects')
        complete = self.write(dict(accepted=True, complete_100_outers=True, outputs={'summary': report}, request=request))
        train_cli = self.packet(complete)
        train_packet = dict(train_cli, mode='native_cli_artifact', actual_stdout=train_cli['output'], output=report)
        value, req, proof = a.analysis_artifact(train_packet, 'train_relationship')
        self.assertTrue(value['accepted']); self.assertEqual(proof['native_complete']['sha256'], complete['sha256'])


if __name__ == '__main__':
    unittest.main()
