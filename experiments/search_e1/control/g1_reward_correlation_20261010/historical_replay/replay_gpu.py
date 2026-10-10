"""Independent historical G1 probe reward reconstruction on approved auxiliary5.

No generation, retrieval, HTTP, production service reload, training update or
production file write. The production online per-row reward was not saved;
all output reward values are explicitly R_reconstructed. Raw per-row artifacts
remain in a new directory on the original Linux node.
"""
import argparse
import datetime
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bound_json(path, expected, core):
    p = Path(path)
    core.require(p.is_file() and not p.is_symlink() and p.resolve() == p,
                 'missing_or_symlink_bound_metadata')
    core.require(core.sha_file(p) == expected, 'bound_metadata_sha_changed')
    return json.loads(p.read_bytes())


def publish(path, document, core, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('xb') as f:
            f.write(core.canonical(document)); f.flush(); os.fsync(f.fileno())
    else:
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            f.write(core.canonical(document)); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)


def resource_gate(base, run, config, pf, self_identity, core, initial=False):
    observed = pf.resource_snapshot(base, run, config['gpu_index'])
    gpu = observed['candidate_gpu']
    core.require(gpu['uuid'] == config['gpu_uuid'], 'candidate_GPU_UUID_changed')
    foreign = []
    for p in observed['compute_processes']:
        if p['matched_recorded_pid_tick_sid_uid']:
            core.require(p.get('recorded_command_sha256') == p['command_sha256'], 'original_retriever_command_changed')
            continue
        if p['pid'] == self_identity['pid'] and pf.same_process(p, self_identity):
            core.require(p['memory_mib'] <= config['max_replay_memory_mib'],
                         'replay_GPU_memory_cap_exceeded')
        else:
            foreign.append(p['pid'])
    core.require(not foreign, 'foreign_or_unbound_candidate_GPU_process')
    floor = config['initial_min_free_mib'] if initial else config['running_min_free_mib']
    core.require(gpu['actual_free_mib'] >= floor, 'candidate_direct_free_below_floor')
    return dict(observed_utc=utc(), candidate=gpu, processes=observed['compute_processes'],
                only_original_retriever_and_exact_replay=True)


def validate_parameter_proof(proof, config, core):
    core.require(proof['readonly'] is True and proof['rc'] == 0, 'invalid_parameter_identity_proof')
    live = json.loads(proof['stdout'])
    core.require(live['pid'] == 1338978 and live['start_ticks'] == 552409911 and live['uid'] == 0,
                 'original_training_process_identity_mismatch')
    expected = dict(ECO_TRUST_VARIANT='balanced', ECO_TRUST_SIGNAL='probe',
        ECO_PROBE_GATE='1', ECO_BAL_GROUND='0.6', ECO_BAL_NOSEARCH_BONUS='0.5',
        ECO_W_COST='0.02', ECO_WRONG_GROUND_SCALE='0', ECO_WRONG_FLOOR=None,
        ECO_PROBE_BASE='0', ECO_PROBE_ALPHA_COST='0', ECO_USE_CF='0', ECO_NO_SELF='1', ECO_ALPHA=None)
    core.require(live['ECO_whitelist'] == expected, 'native_reward_environment_unsupported')
    return live


def native_scalars(run, end, core):
    values = {}
    with Path(run['train_log']).open('rb') as f:
        for raw in f:
            text = raw.decode(errors='replace')
            match = re.search(r'\bstep:(\d+)\s*-', text)
            if not match or not 1 <= int(match[1]) <= end:
                continue
            metrics = {}
            for key in ['critic/score/mean', 'critic/score/min', 'critic/score/max', 'train/perf/em']:
                m = re.search(r'\b' + re.escape(key) + r':(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', text)
                if m:
                    metrics[key] = float(m[1])
            if len(metrics) == 4:
                step = int(match[1]); core.require(step not in values, 'duplicate_native_scalar_step')
                values[step] = dict(metrics=metrics, line_sha256=hashlib.sha256(raw).hexdigest())
    return values


def summarize_native(rows, native, update, torch, core):
    tensor = torch.tensor([r['R_reconstructed'] for r in rows], dtype=torch.float32)
    observed = {'critic/score/mean': tensor.mean().item(), 'critic/score/min': tensor.min().item(),
                'critic/score/max': tensor.max().item(), 'train/perf/em': sum(r['C_raw'] for r in rows) / len(rows)}
    difference = {k: abs(v - native['metrics'][k]) for k, v in observed.items()}
    precise = update['metrics']['critic/rewards/mean']
    core.require(isinstance(precise, (int, float)) and math.isfinite(precise), 'missing_native_full_precision_R_mean')
    precise_difference = abs(observed['critic/score/mean'] - precise)
    core.require(precise_difference <= 2e-6, 'reconstructed_R_precise_native_mean_mismatch')
    core.require(all(v <= 0.0005001 for v in difference.values()), 'reconstructed_R_native_scalar_mismatch')
    return dict(reconstructed_metrics=observed, native_printed=native['metrics'],
                native_scalar_line_sha256=native['line_sha256'], rounding_tolerance=0.0005001,
                absolute_differences=difference, precise_native_update_R_mean=precise,
                precise_mean_absolute_difference=precise_difference, precise_mean_tolerance=2e-6, passed=True,
                native_scalar_agreement_does_not_prove_online_per_row_bitwise_identity=True)


def run(args):
    directory = Path(__file__).parent
    core = load_module('g1_R_replay_core', directory / 'replay_core.py')
    manifest = bound_json(args.source_manifest, args.source_manifest_sha256, core)
    for name, expected in manifest['source_sha256'].items():
        core.require(core.sha_file(directory / name) == expected, 'replay_source_not_published')
    config = bound_json(args.config, args.config_sha256, core)
    core.require(sys.platform.startswith('linux') and socket.gethostname() == config['expected_node'] and
                 os.environ.get('CUDA_VISIBLE_DEVICES') == '5' and 'torch' not in sys.modules,
                 'wrong_remote_node_or_isolated_GPU_environment')
    core.require(config['authorization'] == 'user_existing_run_auxiliary_full_reward_diagnostic' and
                 config['gpu_index'] == 5 and config['initial_min_free_mib'] >= 32768 and
                 config['running_min_free_mib'] >= 8192 and 0 < config['max_replay_memory_mib'] <= 24576,
                 'invalid_auxiliary_replay_resource_contract')
    core.require(1 <= config['start_step'] <= config['end_step'] <= 150, 'invalid_scope')
    base = Path(config['base']); out = Path(config['output']); c_base = Path(config['C_base'])
    core.require(base.resolve() == base and out.resolve() == out and not out.exists() and
                 out.parent.is_dir() and not out.is_relative_to(base) and out.is_relative_to(Path('/data/shaoshitong')), 'output_not_fresh_remote_directory')
    run_record = bound_json(base / 'logs/run-manifest.json', config['run_manifest_sha256'], core)
    core.require(run_record['run_id'] == config['expected_run_id'] == 'db474b6abf654ad9892b02ae67ab6f86' and
                 run_record['execution_host'] == socket.gethostname() and
                 (base / 'logs/run-manifest.json').stat().st_uid == os.getuid(), 'run_owner_identity_mismatch')
    core.require(run_record['training_world_size'] == 4 and run_record['training_physical_indices'] == [0, 1, 2, 3] and
                 run_record['probe_physical_index'] == 6, 'native_resource_profile_changed')
    pf = load_module('g1_R_replay_preflight', directory / 'preflight_cpu.py')
    c_path = Path(config['C_helper_path'])
    core.require(core.sha_file(c_path) == config['C_helper_sha256'], 'unpublished_C_dependency')
    ch = load_module('g1_R_C_dependency', c_path)
    preflight = bound_json(config['encoder_preflight_path'], config['encoder_preflight_sha256'], core)
    core.require(preflight['run_id'] == run_record['run_id'] and preflight['no_model_inference'] is True and
                 all(v['all_manifest_files_fresh_equal'] for v in preflight['versions']), 'encoder_preflight_not_accepted')
    parameter = bound_json(config['reward_parameter_proof_path'], config['reward_parameter_proof_sha256'], core)
    parameter_identity = validate_parameter_proof(parameter, config, core)
    resolved_proof = bound_json(base / 'logs/resolved-training-config.json', config['resolved_config_sha256'], core)
    resolved = resolved_proof['resolved_config']
    core.require(resolved_proof['run_id'] == run_record['run_id'] and
        hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == resolved_proof['resolved_config_sha256'], 'resolved_config_wrapper_mismatch')
    actor_uses_KL_loss = resolved['actor_rollout_ref']['actor'].get('use_kl_loss', False)
    outcome_KL_coefficient = resolved['algorithm']['kl_ctrl']['kl_coef']
    core.require(actor_uses_KL_loss or outcome_KL_coefficient == 0, 'outcome_rewards_not_original_scores')
    repo = base / 'repo'; sources = run_record['source_sha256']
    for name in ['probe_server.py', 'search_g1/signals.py', 'verl/trainer/main_ppo_eco.py',
                 'verl/trainer/ppo/ray_trainer.py', 'verl/utils/seqlen_balancing.py',
                 'reproduction/common8/run_pipeline.py', 'reproduction/a800/owned_processes.py', 'verl/utils/tokenizer.py']:
        core.require(core.sha_file(repo / name) == sources[name], 'original_frozen_source_changed')
    balance_method_sha = core.verify_balance_method(repo / 'verl/trainer/ppo/ray_trainer.py', sources['verl/trainer/ppo/ray_trainer.py'])
    balance = core.load_balance(repo / 'verl/utils/seqlen_balancing.py', sources['verl/utils/seqlen_balancing.py'])
    score = core.load_reward(repo / 'verl/trainer/main_ppo_eco.py', sources['verl/trainer/main_ppo_eco.py'], config['reward_constants'])
    versions = {r['version']: r for r in preflight['versions']}
    # Every manifest encoder member is SHA-streamed again immediately before replay.
    for version in versions.values():
        m = bound_json(version['manifest_path'], version['manifest_fresh_sha256'], core)
        for name, expected in m['encoder_sha256'].items():
            core.require(core.sha_file(Path(m['encoder']) / name) == expected, 'historical_encoder_fresh_SHA_changed')
        for role in ['commit', 'boundary']:
            core.require(core.sha_file(m[role]) == m['head_sha256'][role + '_probe.pkl'], 'historical_head_fresh_SHA_changed')
        version['document'] = m
    sys.path.insert(0, str(repo))
    signals = ch.load_pure_module('g1_replay_signals', repo / 'search_g1/signals.py')
    hf_utils = ch.load_pure_module('g1_replay_original_tokenizer_helper', repo / 'verl/utils/tokenizer.py')
    core.require('torch' not in sys.modules, 'early_Torch_import')
    lock = (out.parent / '.historical-R-replay-gpu5.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = pf.proc_identity(os.getpid())
    core.require(own['sid'] == own['pid'], 'replay_not_independent_new_session')
    initial_resource = resource_gate(base, run_record, config, pf, own, core, initial=True)
    out.mkdir(mode=0o700)
    (out / 'cache').mkdir(); (out / 'tmp').mkdir()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HOME=str(out / 'cache' / 'hf'), XDG_CACHE_HOME=str(out / 'cache'), TMPDIR=str(out / 'tmp'))
    os.environ.pop('USE_TORCH', None)
    import numpy as np
    import torch
    import joblib
    import transformers
    import tokenizers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    core.require(torch.cuda.device_count() == 1, 'replay_CUDA_visibility_not_one')
    torch.cuda.set_per_process_memory_fraction(config['max_replay_memory_mib'] / initial_resource['candidate']['total_mib'], 0)
    initial = hf_utils.hf_tokenizer(run_record['initial_model'], local_files_only=True)
    # Same exact initial decode assets as separately accepted C reconstruction.
    for name, expected in run_record['initial_model_sha256'].items():
        if not name.endswith(('.bin', '.pt', '.safetensors')):
            core.require(core.sha_file(Path(run_record['initial_model']) / name) == expected, 'initial_decode_tokenizer_changed')
    ns = dict(torch=torch, np=np, joblib=joblib, AutoTokenizer=AutoTokenizer,
              AutoModelForCausalLM=AutoModelForCausalLM, DTYPE=torch.bfloat16, contexts=signals.contexts)
    probe = core.extract_functions(repo / 'probe_server.py', ['_load', '_last_hidden', '_commit_ctx'], ns)
    stop, errors = threading.Event(), []
    def monitor():
        with (out / 'memory.jsonl').open('x') as f:
            while not stop.is_set():
                try:
                    row = resource_gate(base, run_record, config, pf, own, core)
                    f.write(core.canonical(row).decode() + '\n'); f.flush()
                except Exception as e:
                    errors.append(type(e).__name__ + ':' + str(e)); stop.set()
                    break
                stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    publish(out / 'inputs.json', dict(config_sha256=args.config_sha256, source_manifest_sha256=args.source_manifest_sha256,
        original_run_id=run_record['run_id'], own_identity=own, initial_resource=initial_resource,
        original_probe_environment='BF16/HF-SDPA/left-padding/left-truncation/4096/micro4; original AST functions',
        native_balance_method_AST_sha256=balance_method_sha, reward_parameter_identity=parameter_identity,
        execution=dict(python_executable=sys.executable, torch=torch.__version__, transformers=transformers.__version__,
            tokenizers=tokenizers.__version__, numpy=np.__version__, joblib=joblib.__version__,
            CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], default_dtype=str(torch.get_default_dtype()),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'), OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
            MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'), historical_runtime_flags_not_individually_persisted=True),
        native_outcome_KL_route=dict(actor_use_kl_loss=actor_uses_KL_loss, outcome_KL_coefficient=outcome_KL_coefficient),
        raw_online_R_saved=False, result_semantics='R_reconstructed; not original online recorded reward'), core, True)
    updates_raw = (base / 'logs/actual-updates.jsonl').read_bytes()
    core.require(updates_raw.endswith(b'\n'), 'partial_committed_ledger')
    updates = [json.loads(x) for x in updates_raw.splitlines()]
    selected = [x for x in updates if config['start_step'] <= x['step'] <= config['end_step']]
    core.require([x['step'] for x in selected] == list(range(config['start_step'], config['end_step'] + 1)), 'uncompleted_requested_scope')
    native = native_scalars(run_record, config['end_step'], core)
    current_version, model = None, None; summaries = []
    try:
        for update in selected:
            started = time.monotonic(); step = update['step']; attempt = update['attempt_id']
            core.require(update['run_id'] == run_record['run_id'] and re.fullmatch('[0-9a-f]{32}', attempt), 'invalid_committed_update_identity')
            # C can finish its independent loop ahead of this replay; no invented partial C.
            cp = c_base / ('TRAIN-' + str(step) + '-summary.json')
            while not cp.is_file():
                core.require(not errors, 'resource_observer_HOLD')
                publish(out / 'status.json', dict(status='WAIT_C_STEP', step=step, observed_utc=utc()), core)
                time.sleep(5)
            c_summary = json.loads(cp.read_bytes()); c_summary_sha = core.sha_file(cp)
            core.require(c_summary['step'] == step and c_summary['attempt_id'] == attempt and
                c_summary['native_EM_check_pass'] is True and c_summary['trajectory_count'] == 2560 and
                c_summary['probe_version_for_future_R_replay'] == update['probe_version'] and
                c_summary['complete_trace_fresh_sha256'] == update['behavior_trace_sha256'], 'independent_C_step_not_accepted')
            crows_path = Path(c_summary['raw_C_remote_path'])
            core.require(crows_path.parent == c_base and core.sha_file(crows_path) == c_summary['raw_C_fresh_sha256'], 'C_rows_sha_changed')
            crows = [json.loads(line) for line in crows_path.read_bytes().splitlines()]
            gate = ch.read_bound(base / 'logs' / ('behavior-probability-attempt-' + attempt + '.json'), update['behavior_probability_gate_sha256'])
            ch.verify_native_gate(gate, update, run_record)
            trace = base / 'artifacts/behavior-traces' / ('attempt-' + attempt + '.json')
            trace_identity = ch.stream_trace_sha(trace, update['behavior_trace_sha256'])
            items, lengths, actor_sha = core.request_items(ch.stream_actor_rows(trace), crows, gate, ch, initial, signals)
            order = core.balance_order(lengths, balance, world=4); balanced = [items[i] for i in order]
            version = update['probe_version']; core.require(version in versions, 'historical_probe_version_not_verified')
            core.require(versions[version]['round'] == ((step - 1) // 50) * 50, 'historical_probe_refit_boundary_mismatch')
            if version != current_version:
                if model is not None:
                    del model, tok, commit, boundary; gc.collect(); torch.cuda.empty_cache()
                m = versions[version]['document']
                tok, model, commit, boundary = probe['_load'](m['encoder'], m['commit'], m['boundary'])
                core.require(all(list(h['clf'].classes_) == [0, 1] and np.isfinite(h['auc']) for h in [commit, boundary]), 'historical_head_classes_or_auc_invalid')
                current_version = version
                publish(out / ('probe-loaded-' + str(m['round']) + '.json'), dict(probe_version=version,
                    encoder=m['encoder'], actual_dtype=str(next(model.parameters()).dtype),
                    attention_implementation=model.config._attn_implementation, device_map=model.hf_device_map,
                    tokenizer_class=type(tok).__name__, padding_side=tok.padding_side, truncation_side=tok.truncation_side,
                    commit_layer=commit['layer'], boundary_layer=boundary['layer'], commit_position=commit['position'],
                    model_eval=not model.training, observed_utc=utc()), core, True)
            core.require(not errors, 'resource_observer_HOLD')
            publish(out / 'status.json', dict(status='REPLAY_COMMIT', step=step, probe_version=version, observed_utc=utc()), core)
            cc = [probe['_commit_ctx'](i['prompt'], i['response'], i['model_ranges'], commit['position']) for i in balanced]
            bc = [i['prompt'] for i in balanced]
            hc = probe['_last_hidden'](cc, commit['layer'], model, tok)
            core.require(not errors, 'resource_observer_HOLD')
            publish(out / 'status.json', dict(status='REPLAY_BOUNDARY', step=step, observed_utc=utc()), core)
            hb = probe['_last_hidden'](bc, boundary['layer'], model, tok)
            d = commit['clf'].predict_proba(hc)[:, 1]; b = boundary['clf'].predict_proba(hb)[:, 1]
            core.require(len(d) == len(b) == 2560 and np.isfinite(d).all() and np.isfinite(b).all() and
                         ((0 <= d) & (d <= 1)).all() and ((0 <= b) & (b <= 1)).all(), 'invalid_probe_output')
            rows = []
            for j, item in enumerate(balanced):
                c = item['C_record']; rr = core.scalar_reward(score, c['final_action_valid'], c['C_raw'], c['model_authored_search_count'], float(b[j]), float(d[j]))
                rows.append(dict(trajectory_id=c['trajectory_id'], run_id=run_record['run_id'], step=step,
                    attempt_id=attempt, original_sample_index=order[j], balanced_request_index=j,
                    upstream_group_sha256=c['upstream_group_sha256'], C_raw=c['C_raw'],
                    final_action_valid=c['final_action_valid'], model_authored_search_count=c['model_authored_search_count'],
                    b_reconstructed=float(b[j]), d_reconstructed=float(d[j]), probe_version=version,
                    C_record_sha256=hashlib.sha256(core.canonical(c)).hexdigest(), terminal_response_token_index=item['valid_rlen'] - 1,
                    original_online_R_available=False, **rr))
            after = trace.stat()
            core.require((after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) ==
                (trace_identity['bytes'], trace_identity['mtime_ns'], trace_identity['inode'], trace_identity['device']), 'original_trace_changed_during_replay')
            core.require(core.sha_file(crows_path) == c_summary['raw_C_fresh_sha256'] and core.sha_file(cp) == c_summary_sha, 'C_artifact_changed_during_R_replay')
            comparison = summarize_native(rows, native[step], update, torch, core)
            core.require(not errors, 'resource_observer_HOLD')
            target = out / ('TRAIN-' + str(step) + '-pairs-R-reconstructed.jsonl')
            with target.open('xb') as f:
                for row in rows:
                    f.write(core.canonical(row) + b'\n')
                f.flush(); os.fsync(f.fileno())
            elapsed = time.monotonic() - started
            summary = dict(status='ACCEPTED_RECONSTRUCTED_R_STEP', step=step, trajectory_count=2560,
                observed_utc=utc(), elapsed_seconds=elapsed, actual_probe_rows=2560,
                commit_micro4_batches=640, boundary_micro4_batches=640, probe_version=version,
                independent_C_summary_sha256=c_summary_sha, independent_C_rows_sha256=c_summary['raw_C_fresh_sha256'],
                original_trace_sha256=update['behavior_trace_sha256'], actor_rows_sha256=actor_sha,
                post_balance_permutation_sha256=hashlib.sha256(core.canonical(order)).hexdigest(),
                output_path=str(target), output_sha256=core.sha_file(target), native_scalar_comparison=comparison,
                actual_memory=resource_gate(base, run_record, config, pf, own, core),
                **core.compact_sufficient_statistics(rows))
            publish(out / ('TRAIN-' + str(step) + '-summary.json'), summary, core, True)
            summaries.append(summary)
            publish(out / 'status.json', dict(status='STEP_COMPLETE', step=step, completed_iterations=len(summaries),
                completed_individual_pairs=len(summaries) * 2560, elapsed_seconds=elapsed, observed_utc=utc()), core)
            print(json.dumps({k: summary[k] for k in ['status', 'step', 'trajectory_count', 'elapsed_seconds', 'observed_utc']}), flush=True)
            del items, balanced, cc, bc, hc, hb, d, b, rows, crows, gate
        publish(out / 'accepted-summary.json', dict(status='ACCEPTED_COMPLETE_RECONSTRUCTED_R_SCOPE',
            original_online_R_available=False, reconstructed_individual_pairs=len(summaries) * 2560,
            completed_iterations=len(summaries), scope=[config['start_step'], config['end_step']],
            iterations=summaries, observed_utc=utc()), core, True)
    finally:
        stop.set(); thread.join(30)
        # No signals/cleanup of any production process or cache; normal exit releases only own CUDA.
    return dict(status='COMPLETE_RECONSTRUCTED_R_ONLY', completed_iterations=len(summaries), output=str(out), observed_utc=utc())


def main():
    p = argparse.ArgumentParser()
    for name in ['config', 'config-sha256', 'source-manifest', 'source-manifest-sha256']:
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    try:
        result = run(args)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)  # Private remote diagnostic log, never raw payload stdout.
        result = dict(status='HOLD_OR_OWN_REPLAY_FAILURE', exception_type=type(e).__name__,
            reason=str(e) if type(e).__name__ == 'Hold' else 'private_exception_in_remote_log',
            original_training_unmodified=True, original_probe_unmodified=True,
            original_online_R_available=False, reconstructed_scope_not_complete=True, observed_utc=utc())
        print(json.dumps(result), flush=True); return 2
    print(json.dumps(result), flush=True); return 0


if __name__ == '__main__':
    raise SystemExit(main())
