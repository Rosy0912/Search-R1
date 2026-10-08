"""Local read-only actual-config supplement; never execute a native inspector."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import stat
import types

H = Path(__file__).resolve().parent
CAPSULE = H / 'job3304-first5-complete-originals-v3'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
RUN = ROOT + '/runs/formal-3304'
REPO = ROOT + '/repo-fcecbea5d939'
MODEL = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725'
DATA = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/common-data'
CFG_SHA = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
MANIFEST_SHA = '30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
INVENTORY_SHA = '59e4b1daa4d1d5647420c012a0ca4d853768cf360c2e373a4e374e92ba10ba58'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable_raw(path, expected=None):
    before = path.lstat()
    assert path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
    assert before.st_uid == os.getuid() and before.st_nlink == 1
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        with os.fdopen(fd, 'rb', closefd=False) as f:
            raw = f.read()
        after = os.fstat(fd)
        final = path.lstat()
    finally:
        os.close(fd)
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    assert all(tuple(getattr(s, k) for k in fields) == tuple(getattr(before, k) for k in fields)
               for s in (opened, after, final))
    assert len(raw) == before.st_size and (expected is None or sha(raw) == expected)
    return raw


def unique(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            assert key not in out
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def main(output):
    inventory = unique(stable_raw(CAPSULE / 'inventory.json', INVENTORY_SHA))
    assert inventory['job_id'] == '3304' and inventory['source_commit'] == SOURCE
    assert inventory['collector_source_sha256'] == 'c5636c9c6a12b40a397ff58fd09d6b8ac5afc8bfa058b6691f2a51c36847ff97'
    verified = {}

    def member(name):
        record = inventory['files'][name]
        raw = stable_raw(CAPSULE / 'raw' / name, record['sha256'])
        assert len(raw) == record['bytes'] and record['uid'] == 0
        verified[name] = {'sha256': sha(raw), 'bytes': len(raw), 'physical': record['physical'],
                          'reading_scope': record['reading_scope']}
        return raw

    def document(name):
        return unique(member(name))

    config_raw = member('full_state/global_step_5/resolved-config.json')
    config = unique(config_raw)
    assert sha(config_raw) == CFG_SHA
    assert json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode() == config_raw
    manifest = document('full_state/global_step_5/manifest.json')
    assert verified['full_state/global_step_5/manifest.json']['sha256'] == MANIFEST_SHA
    assert manifest['config_sha256'] == CFG_SHA and manifest['source_commit'] == SOURCE
    assert manifest['job_id'] == '3304' and manifest['driver_pid'] == 2548319
    assert manifest['actual_outer_updates'] == manifest['global_step'] == 5
    assert manifest['horizon'] == 200 and manifest['warmup_steps'] == 57
    assert manifest['next_global_step'] == 6 and len(manifest['files']) == 141

    def get(key):
        value = config
        for part in key.split('.'):
            value = value[part]
        return value

    expected = {
        'baselines.method': 'opsd', 'baselines.opsd_e1.profile': 'multitraj200',
        'baselines.local_runtime_host': 'h100-10', 'baselines.local_runtime_only': True,
        'baselines.holdout.input_mode': 'prebuilt_manifest', 'baselines.holdout.manifest_sha256': COMMON_SHA,
        'actor_rollout_ref.model.path': MODEL, 'actor_rollout_ref.actor.baseline_method': 'opsd',
        'actor_rollout_ref.actor.ppo_mini_batch_size': 256, 'actor_rollout_ref.actor.ppo_micro_batch_size': 32,
        'actor_rollout_ref.actor.ppo_epochs': 1, 'actor_rollout_ref.actor.grad_clip': 1,
        'actor_rollout_ref.actor.entropy_coeff': 0, 'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.kl_loss_coef': 0, 'algorithm.kl_ctrl.kl_coef': 0,
        'actor_rollout_ref.actor.optim.lr': 1e-6, 'actor_rollout_ref.actor.optim.lr_warmup_steps': 57,
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio': 0,
        'actor_rollout_ref.actor.optim.total_training_steps': 200,
        'actor_rollout_ref.actor.optim.warmup_style': 'constant',
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size': 1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size': 64,
        'actor_rollout_ref.ref.log_prob_micro_batch_size': 64,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.n': 1, 'actor_rollout_ref.rollout.n_agent': 5,
        'actor_rollout_ref.rollout.temperature': 1, 'actor_rollout_ref.rollout.top_p': 1,
        'actor_rollout_ref.rollout.top_k': -1,
        'data.train_batch_size': 512, 'data.val_batch_size': 512,
        'data.shuffle_train_dataloader': True,
        'data.max_prompt_length': 4096, 'data.max_start_length': 2048,
        'data.max_response_length': 500, 'data.max_obs_length': 500,
        'max_turns': 4, 'force_final_answer': False,
        'retriever.topk': 3, 'retriever.url': 'http://127.0.0.1:18084/retrieve',
        'trainer.n_gpus_per_node': 8, 'trainer.nnodes': 1,
        'trainer.total_training_steps': 200, 'trainer.acceptance_step': 5,
        'trainer.save_freq': 50, 'trainer.test_freq': 50, 'trainer.val_before_train': True,
        'baselines.opsd_e1.stage_num': 64, 'baselines.opsd_e1.stage_freq': 50,
        'baselines.opsd_e1.binding_manifest_path': RUN + '/bindings/manifest.json',
        'baselines.opsd_e1.run_root': RUN,
        'data.train_files': DATA + '/train.parquet', 'data.val_files': DATA + '/validation.parquet',
        'data.final_val_files': DATA + '/final.parquet', 'baselines.opsd_e1.stage_files': DATA + '/stage.parquet',
    }
    for key, value in expected.items():
        actual = get(key)
        assert type(actual) is type(value) and actual == value, (key, actual, value)
    expected_distillation = {'beta': 0, 'temperature': 1, 'component_clip': None,
        'teacher_path': MODEL, 'teacher_device': 'cuda', 'privileged_answers': True,
        'teacher_prompt_mode': 'user_context', 'teacher_forward_micro_batch': 4,
        'response_only_logits': False, 'teacher_logits_recompute': True,
        'loss_checkpointing': True, 'chunk_size': 16, 'max_student_logits_gib': 8.0}
    assert config['baselines']['distillation'] == config['actor_rollout_ref']['actor']['distillation'] == expected_distillation
    bound = document('bindings/manifest.json')
    assert bound['source_commit'] == SOURCE and bound['run_root'] == RUN
    assert get('baselines.opsd_e1.binding_manifest_sha256') == verified['bindings/manifest.json']['sha256']
    for role, record in bound['files'].items():
        document('bindings/' + role + '.json')
        assert record == {'path': RUN + '/bindings/' + role + '.json',
                          'sha256': verified['bindings/' + role + '.json']['sha256']}
    source = document('bindings/source.json')
    assert source['source_commit'] == SOURCE and source['clean'] is True and source['repo'] == REPO
    local_bound_raw = stable_raw(H / 'runtime-tools-bound-fcecbea5d939-v5.json',
                               '8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228')
    local_bound = unique(local_bound_raw)
    assert source['source_files'] == local_bound['source_files'] and len(source['source_files']) == 19
    assert source['formal_manifest_sha256'] == local_bound['formal']['manifest_sha256']
    for relative, digest in source['source_files'].items():
        assert sha(member('source/' + relative)) == digest
    closure = document('bindings/import_closure.json')
    assert len(closure['modules']) == 211
    entry_names = ('search_r1.baselines.main', 'search_r1.baselines.actor',
                   'search_r1.baselines.distillation', 'verl.trainer.main_ppo',
                   'verl.trainer.ppo.ray_trainer', 'verl.workers.fsdp_workers',
                   'verl.workers.actor.dp_actor', 'reproduction.baselines.e1_import_bootstrap')
    entry = {name: closure['modules'][name] for name in entry_names}
    for name, record in entry.items():
        relative = record['path'].removeprefix(REPO + '/')
        assert record['path'].startswith(REPO + '/')
        if relative in source['source_files']:
            assert record['sha256'] == source['source_files'][relative]
    checker = get('actor_rollout_ref.actor.e1_import_checker')
    assert checker['checker_path'] == entry['reproduction.baselines.e1_import_bootstrap']['path']
    assert checker['checker_sha256'] == entry['reproduction.baselines.e1_import_bootstrap']['sha256']
    assert checker['closure_path'] == RUN + '/bindings/import_closure.json'
    assert checker['closure_sha256'] == verified['bindings/import_closure.json']['sha256']
    models = document('bindings/model_manifest.json')
    assert verified['bindings/model_manifest.json']['sha256'] == MODEL_SHA
    assert models['repo'] == 'Qwen/Qwen2.5-3B-Instruct'
    assert models['revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert len(models['files']) == 11 and len({r['file'] for r in models['files']}) == 11
    common = document('bindings/data_manifest.json')
    assert verified['bindings/data_manifest.json']['sha256'] == COMMON_SHA
    counts = {'train': 169028, 'validation': 512, 'stage': 64, 'final': 51713}
    for split, count in counts.items():
        record = common['splits'][split]
        assert type(record['rows']) is int and record['rows'] == count
        assert sum(record['source_counts'].values()) == count
    assert common['splits']['validation']['source_counts'] == {'nq': 256, 'hotpotqa': 256}
    assert common['splits']['stage']['source_counts'] == {'nq': 32, 'hotpotqa': 32}
    assert common['splits']['final']['source_counts'] == {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267,
        'hotpotqa': 7405, '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}
    native = document('evidence/first5-cpu-inspection-once-v1/native.json')
    hf = document('evidence/first5-cpu-inspection-once-v1/hf.json')
    for report in (native, hf):
        assert report['passed'] is True and report['source_commit'] == SOURCE
        assert report['job_id'] == '3304' and report['checkpoint_manifest_sha256'] == MANIFEST_SHA
        assert report['cuda_is_initialized'] is False and report['run_root'] == RUN
    assert native['scope'] == 'NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE'
    assert hf['scope'] == 'ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA'
    assert native['inspector_source_sha256'] == local_bound['native']['files']['inspect_payloads.py']
    assert hf['inspector_source_sha256'] == local_bound['first5']['files']['hf_cpu_reload.py']
    assets = {MODEL + '/' + r['file']: r['sha256'] for r in models['files']}
    assets[MODEL + '/VERIFIED-MODEL-MANIFEST.json'] = MODEL_SHA
    assert hf['base_all_files'] == assets
    for path, digest in assets.items():
        assert native['all_files'][path] == digest
    ranks = []
    for rank, audited in enumerate(native['ranks']):
        raw_rank = document('checkpoints/actor/global_step_5/training_state/rank_' + str(rank) + '.json')
        assert audited['rank'] == raw_rank['rank'] == rank and raw_rank['step'] == 5
        assert raw_rank['algorithm'] == 'pure_opsd' and raw_rank['world_size'] == 8
        assert raw_rank['status'] == 'EXACT_FULL_STATE_RELOAD_OK'
        assert audited['pid'] == raw_rank['pid'] and audited['state_sha256'] == raw_rank['state_sha256']
        assert audited['optimizer_step_values'] == [50] and audited['scheduler_last_epoch'] == 5
        assert len(audited['scheduler_last_lr']) == 1 and math.isclose(audited['scheduler_last_lr'][0], 1e-6 * (5 / 57), rel_tol=1e-14, abs_tol=1e-20)
        teacher = raw_rank['teacher']
        assert teacher['asset_directory'] == MODEL and teacher['asset_files'] == {Path(p).name: s for p, s in assets.items()}
        assert teacher['model_content_digest'] == audited['teacher_content_digest'] == native['base_named_parameter_digest']
        assert teacher['execution_device'] == 'cuda' and teacher['storage_device'] == 'cpu'
        assert audited['teacher_cuda_update_entries'] == teacher['counters']['cuda_update_entries'] == 5
        ranks.append({'rank': rank, 'pid': audited['pid'], 'actor_config_digest': audited['actor_config_digest'],
                      'adam_steps': 50, 'scheduler_step': 5, 'post_lr': audited['scheduler_last_lr'][0],
                      'teacher_official_parameter_digest_equal': True})
    assert len(ranks) == 8 and len({r['pid'] for r in ranks}) == 8
    algorithm = document('checkpoints/algorithm/global_step_5/state.json')
    assert algorithm['method'] == 'pure_opsd' and algorithm['global_step'] == 5
    assert algorithm['distillation'] == expected_distillation and algorithm['outcome_rewards_train_policy'] is False
    assert native['driver']['consumed_cursor'] == native['driver']['sampler_cursor'] == 2560
    assert native['driver']['size'] == 169028 and native['driver']['epoch'] == 0
    assert native['driver']['pid'] == 2548319
    gate_raw = member('source/search_r1/baselines/checkpoint_gate.py')
    gate = types.ModuleType('same_bytes_original_gate')
    gate.__file__ = REPO + '/search_r1/baselines/checkpoint_gate.py'
    exec(compile(gate_raw, gate.__file__, 'exec', dont_inherit=True), gate.__dict__)
    outer_rows = []
    workers = [r['pid'] for r in ranks]
    for name in ('train.log', 'driver-current-prefix.out'):
        values = []
        for line in member(name).decode().splitlines():
            if 'OPSD_E1_OUTER_JSON=' in line:
                row = gate.parse_log_marker(line, 'OPSD_E1_OUTER_JSON=', driver_pid=2548319,
                    job_id='3304', worker_pids=workers,
                    checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')
                assert row['source_commit'] == SOURCE and row['config_sha256'] == CFG_SHA
                gate.validate_outer(row['metrics'], profile='multitraj200')
                values.append(row)
        assert [r['actual_outer_updates'] for r in values] == list(range(1, 6))
        if outer_rows:
            assert values == outer_rows
        outer_rows = values
    metrics = []
    for outer, row in enumerate(outer_rows, 1):
        m = row['metrics']
        assert all(math.isclose(m['actor/optimizer_lr_used_mean/rank_' + str(r)],
            1e-6 * (outer - 1) / 57, rel_tol=1e-14, abs_tol=1e-20) for r in range(8))
        assert m['rollout/trajectories'] == 2560
        assert all(m[k] == 0 for k in ('critic/advantages/min', 'critic/advantages/mean', 'critic/advantages/max',
                                      'critic/returns/min', 'critic/returns/mean', 'critic/returns/max'))
        metrics.append({'outer': outer, 'reward_random_train_batch': m['critic/rewards/mean'],
                        'rank0_loss_mean': m['actor/opsd_loss'],
                        'global_mean_of_mini_losses': sum(m['actor/opsd_loss_mean/rank_' + str(r)] for r in range(8)) / 8,
                        'used_lr': m['actor/optimizer_lr_used_mean/rank_0'], 'post_lr': m['actor/lr'],
                        'optimizer_calls_per_rank': 10, 'samples_per_rank': 320,
                        'finish_ratio': m['env/finish_ratio'], 'valid_action_ratio': m['env/ratio_of_valid_action']})
    report = {'version': 2, 'status': 'PASS_INDEPENDENT_ACTUAL_FIRST5_CONFIG_RUNTIME_MATH_SUBREVIEW_NOT_TOTAL_FIRST5_ACCEPTANCE',
        'source_commit': SOURCE, 'job_id': '3304', 'checkpoint_manifest_sha256': MANIFEST_SHA,
        'config_sha256': CFG_SHA, 'capsule_inventory_sha256': INVENTORY_SHA,
        'parent_math_subreview_sha256': '574ebd63a32a784aeb0278c561621a6a69a5a18f7608a253c45fcfea2d964193',
        'scope': 'Read actual preserved full resolved config, 19 source originals, model/data/source/import bindings and CPU/native/HF report budget joins; not rerun inspectors, complete archive semantics or resource acceptance',
        'actual_config_equalities': expected, 'distillation': expected_distillation,
        'effective_budget': {'questions': 512, 'responses_per_question': 5, 'trajectories': 2560,
            'global_mini': 256, 'local_mini': 32, 'global_micro': 32, 'local_micro': 4,
            'global_old_log_prob_micro': 64, 'local_old_log_prob_micro': 8,
            'reference_log_prob_executed': False, 'optimizer_calls_per_rank_per_outer': 10,
            'saved_adam_calls_per_rank': 50, 'scheduler_outer_ticks': 5},
        'pure_opsd_scope': 'Forward KL(q teacher || p student), beta0/T1/no component cap. PPO clip_ratio .2 and grpo enum are inactive driver/template fields: pinned main removes reference role, pinned actor never calls PG, algorithm state explicitly rewards do not train policy.',
        'warmup_scope': 'Outer1 usedLR0 and executes10Adam calls. Parameters change across later warmup updates, not claimed changed in outer1. Actual five used/post learning rates match57 outer warmup.',
        'entrypoint_records': entry, 'closure_modules': 211,
        'models': {'official_repo': models['repo'], 'official_revision': models['revision'],
                   'manifest_sha256': MODEL_SHA, 'asset_sha_map': assets,
                   'scope': 'Read actual manifest and join existing native/HF full asset SHA records and all8 teacher official content digests; no fresh local full-weight loading or hashing'},
        'data': {'manifest_sha256': COMMON_SHA, 'splits': common['splits'],
                 'scope': 'Actual common immutable manifest full bytes and exact configured shared paths/counts plus stored sampler169028/cursor2560. This subreview does not freshly rehash/download shared parquet files; existing preGPU/runtime loader is separately reviewed.'},
        'rank_budget_report_joins': ranks, 'actual_outer_metrics': metrics,
        'verified_originals': verified,
        'remaining_for_root_total_acceptance': ['Independent full100 archive/tokenizer semantics', 'Phase-qualified resource and retrieval acceptance', 'Root full first5 verdict and signed release'],
        'author_disclosure': 'This agent authored archive/collector tooling; separate peer independently reviews full archive semantics and first5 acceptance. This supplement covers original config/mathematics/entrypoint bindings only.',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    assert output.parent == H and not output.exists()
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'report_sha256': sha(output.read_bytes()),
                      'verified_small_originals': len(verified), 'actual_outer_updates': 5}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
