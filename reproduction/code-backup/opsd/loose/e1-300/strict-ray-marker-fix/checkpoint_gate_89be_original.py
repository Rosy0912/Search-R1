"""Opt-in pure-OPSD E1 state boundary and independent signed release consumer.

There is deliberately no receipt producer or signing operation in this module.
The training process waits while retaining its objects, leases and data iterator.
"""
import hashlib
import hmac
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import types


def enabled(config):
    baselines = config.get('baselines') if hasattr(config, 'get') else getattr(config, 'baselines', None)
    settings = (baselines.get('opsd_e1') if hasattr(baselines, 'get')
                else getattr(baselines, 'opsd_e1', None))
    return bool(settings and (settings.get('enabled', False) if hasattr(settings, 'get')
                             else getattr(settings, 'enabled', False)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def owned_file(path, root):
    path, root = Path(path), Path(root)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('Evidence must be an absolute physical file')
    path.relative_to(root)
    if not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError('Evidence is not an owned regular file')
    return path


def verify_file(record, root):
    if set(record) != {'path', 'sha256'}:
        raise ValueError('Invalid file binding schema')
    path = owned_file(record['path'], root)
    if sha256(path) != record['sha256']:
        raise ValueError('Evidence bytes changed: ' + str(path))
    return path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + secrets.token_hex(8))
    with open(temporary, 'xb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


IMPORT_OBJECTS = {
    'search_r1.baselines.checkpoint_gate': ('CheckpointGate.verify_receipt',),
    'search_r1.baselines.full_state': ('rng_state', 'StatefulSampler.state_dict'),
    'search_r1.baselines.controller': ('Controller.compute_advantages',),
    'search_r1.baselines.distillation': ('opsd_loss', 'teacher_inputs'),
    'search_r1.baselines.actor': ('DistillationActor._update_policy',),
    'verl.trainer.ppo.ray_trainer': ('RayPPOTrainer.fit',),
    'verl.workers.fsdp_workers': ('ActorRolloutRefWorker._training_state_payload',
                                'ActorRolloutRefWorker._load_training_state_payload'),
}


def verify_import_closure(record, root, repo, required=None):
    """Check actual module paths and live implementation code, not pyc labels."""
    closure = load_json(verify_file(record, root))
    modules = closure.get('modules', {})
    required = IMPORT_OBJECTS if required is None else required
    for name, objects in required.items():
        if name not in modules or set(modules[name]) != {'path', 'sha256'}:
            raise ValueError('Import closure is missing a required module')
        expected_path = Path(repo) / (name.replace('.', '/') + '.py')
        actual_module = importlib.import_module(name)
        if (Path(actual_module.__file__).resolve() != expected_path or
                modules[name]['path'] != str(expected_path) or
                modules[name]['sha256'] != sha256(expected_path)):
            raise ValueError('Actual module does not match the reviewed source: ' + name)
        compiled = compile(expected_path.read_bytes(), str(expected_path), 'exec', optimize=sys.flags.optimize)
        for dotted in objects:
            actual, expected = actual_module, compiled
            for component in dotted.split('.'):
                actual = getattr(actual, component)
                expected = next((code for code in expected.co_consts
                    if inspect.iscode(code) and code.co_name == component), None)
                if expected is None:
                    raise ValueError('Reviewed import object not found in source')
            live = inspect.unwrap(actual).__code__
            if (live != expected or live.co_filename != expected.co_filename or
                    live.co_firstlineno != expected.co_firstlineno):
                raise ValueError('Live imported implementation differs from source: ' + name + '.' + dotted)


def full_import_checker(closure_record, root, repo):
    """Execute only externally pinned checker bytes, then audit this process."""
    closure_path = verify_file(closure_record, root)
    closure = load_json(closure_path)
    name = 'reproduction.baselines.e1_import_bootstrap'
    record = closure.get('modules', {}).get(name)
    expected = Path(repo) / 'reproduction/baselines/e1_import_bootstrap.py'
    if (not isinstance(record, dict) or set(record) != {'path', 'sha256'} or
            record['path'] != str(expected) or expected.resolve(strict=True) != expected):
        raise ValueError('Pinned import checker path/schema differs')
    source = expected.read_bytes()
    if hashlib.sha256(source).hexdigest() != record['sha256']:
        raise ValueError('Pinned import checker source changed')
    module = types.ModuleType(name)
    module.__file__ = str(expected)
    # Execute the very bytes verified above; never import a possibly stale pyc.
    exec(compile(source, str(expected), 'exec', dont_inherit=True,
                 optimize=sys.flags.optimize), module.__dict__)
    sys.modules[name] = module
    module.verify_imports(Path(repo), closure, required=())
    return {'checker_path': str(expected), 'checker_sha256': record['sha256'],
            'closure_path': str(closure_path), 'closure_sha256': closure_record['sha256']}


def worker_import_checker_config(config):
    """Fail before creating Ray workers; propagate identical external pins."""
    settings = config.baselines.opsd_e1
    root = Path(settings.run_root)
    binding = load_json(verify_file({'path': str(settings.binding_manifest_path),
                                   'sha256': str(settings.binding_manifest_sha256)}, root))
    repo = Path(__file__).resolve().parents[2]
    if binding.get('run_root') != str(root):
        raise ValueError('Worker import binding belongs to another run')
    return full_import_checker(binding['files']['import_closure'], root, repo)


def should_save(step):
    return step == 5 or (1 <= step <= 300 and step % 25 == 0)


def validate_evaluation(metrics, count):
    if metrics.get('val/qa_count/all') != count:
        raise RuntimeError('E1 evaluation lost or duplicated fixed questions')
    evaluated = sum(value for key, value in metrics.items() if key.startswith('val/evaluated_count/'))
    if evaluated != count:
        raise RuntimeError('E1 reward/QA evaluation counts differ')


def validate_outer(metrics, world=8):
    """A skipped empty mini is a failure, not a fabricated Adam clock tick."""
    for rank in range(world):
        expected = {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0,
                    'actor/samples': 320, 'actor/loss_mask_valid': 1,
                    'actor/opsd_loss_finite': 1, 'actor/grad_norm_finite': 1,
                    'teacher/trainable_parameters_per_rank': 0,
                    'teacher/gradient_tensors_per_rank': 0,
                    'teacher/student_optimizer_membership_per_rank': 0,
                    'teacher/storage_is_cpu_per_rank': 1,
                    'teacher/all_parameter_dtype_is_fp32_per_rank': 1,
                    'teacher/attention_is_eager_per_rank': 1,
                    'actor/allocator_is_native': 1,
                    'actor/allocator_configuration_is_default': 1}
        for key, value in expected.items():
            actual = metrics.get(key + '/rank_' + str(rank))
            if isinstance(actual, bool) or actual != value or not math.isfinite(float(actual)):
                raise RuntimeError('E1 rank update audit failed: ' + key + '/rank_' + str(rank))


def verify_rsa_signature(payload, signature_hex, public_key):
    """Strict RSASSA-PKCS1-v1_5/SHA256, public verification only (RFC 8017)."""
    if set(public_key) != {'algorithm', 'n_hex', 'e'} or public_key['algorithm'] != 'RSA-SHA256':
        raise ValueError('Unsupported review public key')
    if public_key['e'] != 65537 or not isinstance(public_key['n_hex'], str):
        raise ValueError('Invalid review key parameters')
    modulus = int(public_key['n_hex'], 16)
    if modulus.bit_length() < 2048 or modulus.bit_length() > 8192 or modulus % 2 == 0:
        raise ValueError('Review RSA modulus must be 2048..8192 bits and odd')
    size = (modulus.bit_length() + 7) // 8
    signature = bytes.fromhex(signature_hex)
    if len(signature) != size or int.from_bytes(signature, 'big') >= modulus:
        raise ValueError('Invalid review signature size/value')
    actual = pow(int.from_bytes(signature, 'big'), 65537, modulus).to_bytes(size, 'big')
    # DER DigestInfo for SHA256, including the mandatory NULL parameter.
    tail = bytes.fromhex('3031300d060960864801650304020105000420') + hashlib.sha256(canonical(payload)).digest()
    expected = b'\x00\x01' + b'\xff' * (size - len(tail) - 3) + b'\x00' + tail
    if not hmac.compare_digest(actual, expected):
        raise ValueError('Independent review signature does not verify')


class CheckpointGate:
    def __init__(self, trainer):
        from omegaconf import OmegaConf
        from .full_state import resolve_warmup_steps
        self.trainer = trainer
        self.settings = trainer.config.baselines.opsd_e1
        self.root = Path(self.settings.run_root)
        if not self.root.is_absolute() or self.root.resolve(strict=True) != self.root:
            raise ValueError('E1 run root must be an existing physical directory')
        self.pid, self.uid = os.getpid(), os.getuid()
        self.job = os.environ.get('SLURM_JOB_ID')
        if not self.job or not self.settings.gate_required:
            raise ValueError('E1 requires a real job and independent gate')
        if (trainer.config.baselines.method != 'opsd' or
                trainer.config.trainer.n_gpus_per_node != 8 or trainer.config.trainer.nnodes != 1 or
                resolve_warmup_steps(trainer.config.actor_rollout_ref.actor.optim) != 57 or
                trainer.config.trainer.total_training_steps != 300):
            raise ValueError('E1 requires pure OPSD, eight ranks, 57 warmup and 300 actual outers')
        if (not trainer.config.trainer.val_before_train or trainer.config.trainer.get('val_only', False) or
                trainer.config.trainer.test_freq != 50 or trainer.config.trainer.get('smoke_only', False) or
                self.settings.acceptance_step != 5 or self.settings.stage_freq != 50):
            raise ValueError('E1 requires initial/dev50, stage5/50, and live continuation')
        actor = trainer.config.actor_rollout_ref.actor
        rollout = trainer.config.actor_rollout_ref.rollout
        distillation = trainer.config.baselines.distillation
        if (actor.use_kl_loss or actor.entropy_coeff or trainer.config.algorithm.kl_ctrl.kl_coef or
                distillation.component_clip is not None or distillation.beta != 0 or
                distillation.teacher_prompt_mode != 'user_context' or distillation.teacher_device != 'cuda'):
            raise ValueError('E1 gate supports the reviewed pure/unclipped privileged OPSD only')
        if (trainer.config.data.train_batch_size != 512 or actor.ppo_mini_batch_size != 128 or
                actor.ppo_micro_batch_size != 16 or actor.ppo_epochs != 1 or
                rollout.n != 1 or rollout.n_agent != 5 or rollout.tensor_model_parallel_size != 1 or
                rollout.temperature != 1 or distillation.temperature != 1):
            raise ValueError('E1 requires 512x5, global mini128/micro16, one epoch and TP/T1')
        self.repo = Path(__file__).resolve().parents[2]
        self.source = self.current_source()
        self.config = OmegaConf.to_container(trainer.config, resolve=True)
        self.config_sha = hashlib.sha256(canonical(self.config)).hexdigest()
        self.binding_record = {'path': str(self.settings.binding_manifest_path),
                               'sha256': str(self.settings.binding_manifest_sha256)}
        binding = load_json(verify_file(self.binding_record, self.root))
        if binding.get('source_commit') != self.source or binding.get('run_root') != str(self.root):
            raise ValueError('E1 binding belongs to another source/run')
        required = {'allocation', 'source', 'import_closure', 'data_manifest', 'model_manifest',
                    'tool_manifest', 'environment'}
        if not required <= set(binding.get('files', {})):
            raise ValueError('E1 launch bindings are incomplete')
        self.bindings = binding['files']
        for record in self.bindings.values():
            verify_file(record, self.root)
        full_import_checker(self.bindings['import_closure'], self.root, self.repo)
        verify_import_closure(self.bindings['import_closure'], self.root, self.repo)
        if len({record['path'] for record in self.bindings.values()}) != len(self.bindings):
            raise ValueError('E1 launch roles must bind distinct evidence files')
        self.key_record = {'path': str(self.settings.public_key_path),
                           'sha256': str(self.settings.public_key_sha256)}
        self.public_key = load_json(verify_file(self.key_record, self.root))
        self.receipt = Path(self.settings.receipt_path)
        if not self.receipt.is_absolute() or self.receipt.parent.resolve(strict=True) != self.receipt.parent:
            raise ValueError('Receipt path must have a physical parent')
        self.receipt.relative_to(self.root)
        if self.receipt.exists() or self.receipt.is_symlink():
            raise ValueError('A release must not pre-exist training')

    def current_source(self):
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=self.repo, check=True)
        return source

    def save_driver(self, step, epoch, roundtrip):
        import torch
        from .full_state import rng_state, restore_rng, state_digest, atomic_save
        t = self.trainer
        directory = self.root / 'full_state' / ('global_step_' + str(step))
        directory.mkdir(parents=True, exist_ok=False)
        payload = {'version': 1, 'method': 'pure_opsd', 'global_step': step, 'next_global_step': step + 1,
                   'actual_outer_updates': step, 'epoch': epoch, 'sampler': t.e1_sampler.state_dict(),
                   'rng': rng_state(), 'algorithm': t.baseline_controller.opsd_state_dict(step),
                   'pid': self.pid, 'uid': self.uid, 'job_id': self.job,
                   'source_commit': self.source, 'config_sha256': self.config_sha}
        before = state_digest(payload)
        atomic_save(payload, directory / 'driver.pt')
        if roundtrip:
            loaded = torch.load(directory / 'driver.pt', map_location='cpu', weights_only=False)
            if state_digest(loaded) != before:
                raise RuntimeError('Saved driver state differs')
            t.e1_sampler.load_state_dict(loaded['sampler'])
            t.baseline_controller.load_opsd_state_dict(loaded['algorithm'], step)
            restore_rng(loaded['rng'])
            observed = dict(payload, sampler=t.e1_sampler.state_dict(), rng=rng_state(),
                            algorithm=t.baseline_controller.opsd_state_dict(step))
            if state_digest(observed) != before:
                raise RuntimeError('Actual driver roundtrip differs')
        report = {'status': 'EXACT_DRIVER_STATE_RELOAD_OK' if roundtrip else 'DRIVER_STATE_SERIALIZED_OK',
                  'step': step, 'pid': self.pid, 'uid': self.uid, 'job_id': self.job,
                  'digest': before, 'state_sha256': sha256(directory / 'driver.pt')}
        atomic_json(directory / 'driver.json', report)
        return directory

    def restore_driver(self, state_record, authorization_record):
        """Explicit signed recovery primitive; never called by the first-five gate.

        Worker recovery remains a separate RPC requiring its own rank bindings.
        This function does not auto-resume fit or reuse a same-PID step-five receipt.
        """
        import torch
        from .full_state import restore_rng, rng_state, state_digest
        envelope = load_json(verify_file(authorization_record, self.root))
        if set(envelope) != {'payload', 'signature_hex'}:
            raise ValueError('Invalid driver recovery authorization envelope')
        verify_file(self.key_record, self.root)
        auth = envelope['payload']
        verify_rsa_signature(auth, envelope['signature_hex'], self.public_key)
        expected = {'version': 1, 'authorization': 'EXPLICIT_OPSD_E1_DRIVER_RECOVERY',
                    'checkpoint': state_record, 'destination_pid': self.pid,
                    'destination_uid': self.uid, 'destination_job_id': self.job,
                    'source_commit': self.source, 'config_sha256': self.config_sha}
        if set(auth) != set(expected) | {'source_pid', 'source_job_id', 'state_digest', 'step'}:
            raise ValueError('Invalid driver recovery authorization schema')
        if any(auth[k] != v for k, v in expected.items()):
            raise ValueError('Driver recovery destination or identity differs')
        if self.current_source() != self.source or os.getpid() != self.pid or os.environ.get('SLURM_JOB_ID') != self.job:
            raise ValueError('Driver recovery live process changed')
        payload = torch.load(verify_file(state_record, self.root), map_location='cpu', weights_only=False)
        if (state_digest(payload) != auth['state_digest'] or payload['pid'] != auth['source_pid'] or
                payload['job_id'] != auth['source_job_id'] or payload['global_step'] != auth['step'] or
                payload['actual_outer_updates'] != auth['step'] or payload['next_global_step'] != auth['step'] + 1 or
                payload['uid'] != self.uid or payload['source_commit'] != self.source or
                payload['config_sha256'] != self.config_sha or payload['method'] != 'pure_opsd' or
                payload['version'] != 1 or not 1 <= auth['step'] < 300):
            raise ValueError('Driver recovery checkpoint differs from authorization')
        t = self.trainer
        t.e1_sampler.load_state_dict(payload['sampler'])
        t.baseline_controller.load_opsd_state_dict(payload['algorithm'], auth['step'])
        restore_rng(payload['rng'])
        if (state_digest(t.e1_sampler.state_dict()) != state_digest(payload['sampler']) or
                state_digest(rng_state()) != state_digest(payload['rng']) or
                t.baseline_controller.opsd_state_dict(auth['step']) != payload['algorithm']):
            raise RuntimeError('Actual recovered driver state differs')
        t.actual_outer_updates = auth['step']
        t.global_steps = payload['next_global_step']
        t.e1_epoch = payload['epoch']
        return {'status': 'EXACT_AUTHORIZED_DRIVER_STATE_RELOAD_OK',
                'actual_outer_updates': t.actual_outer_updates, 'next_global_step': t.global_steps,
                'pid': self.pid, 'job_id': self.job, 'worker_recovery_verified': False}

    def manifest(self, step, driver_directory, actor_directory, metrics):
        if self.current_source() != self.source:
            raise RuntimeError('Source changed during E1 training')
        files = dict(self.bindings)
        files['launch_binding'] = self.binding_record
        files['review_public_key'] = self.key_record
        actor_directory = Path(actor_directory)
        hf_index = load_json(owned_file(actor_directory / 'model.safetensors.index.json', self.root))
        shards = set(hf_index.get('weight_map', {}).values())
        if not shards or any(Path(name).name != name for name in shards):
            raise RuntimeError('HF checkpoint index has no complete local shards')
        for name in shards | {'config.json', 'tokenizer_config.json', 'tokenizer.json'}:
            path = owned_file(actor_directory / name, self.root)
            if path.stat().st_size < 1:
                raise RuntimeError('HF checkpoint file is empty')
        for path in sorted(actor_directory.rglob('*')):
            if path.is_file():
                owned_file(path, self.root)
                role = 'actor/' + str(path.relative_to(actor_directory))
                files[role] = {'path': str(path), 'sha256': sha256(path)}
        for path in sorted(driver_directory.iterdir()):
            if path.is_file():
                files['driver/' + path.name] = {'path': str(path), 'sha256': sha256(path)}
        algorithm_path = Path(self.trainer.config.trainer.default_local_dir) / 'algorithm' / ('global_step_' + str(step)) / 'state.json'
        owned_file(algorithm_path, self.root)
        files['algorithm'] = {'path': str(algorithm_path), 'sha256': sha256(algorithm_path)}
        for rank in range(8):
            report_path = actor_directory / 'training_state' / ('rank_' + str(rank) + '.json')
            report = load_json(owned_file(report_path, self.root))
            if (report['status'] != 'EXACT_FULL_STATE_RELOAD_OK' or report['rank'] != rank or
                    report['world_size'] != 8 or report['step'] != step or
                    report['algorithm'] != 'pure_opsd' or report['version'] != 3 or
                    report['model_state_dict_type'] != 'SHARDED_STATE_DICT' or
                    report['model_dtensor_entries'] < 1 or report['optimizer_state_entries'] < 1 or
                    report['scheduler_last_epoch'] != step or
                    report['path'] != str(actor_directory / 'training_state' / ('rank_' + str(rank) + '.pt')) or
                    report['state_sha256'] != sha256(actor_directory / 'training_state' / ('rank_' + str(rank) + '.pt'))):
                raise RuntimeError('Rank full-state roundtrip is incomplete')
        expected_native = {'rank_' + str(rank) + extension for rank in range(8) for extension in ('.pt', '.json')}
        if {path.name for path in (actor_directory / 'training_state').iterdir()} != expected_native:
            raise RuntimeError('Native state directory mixes unexpected rank files')
        ranks = [load_json(actor_directory / 'training_state' / ('rank_' + str(rank) + '.json')) for rank in range(8)]
        if len({record['pid'] for record in ranks}) != 8:
            raise RuntimeError('Native rank reports alias worker processes')
        driver_report = load_json(driver_directory / 'driver.json')
        if (driver_report['status'] != 'EXACT_DRIVER_STATE_RELOAD_OK' or
                driver_report['step'] != step or driver_report['pid'] != self.pid or
                driver_report['state_sha256'] != sha256(driver_directory / 'driver.pt')):
            raise RuntimeError('Driver roundtrip is incomplete')
        config_path = driver_directory / 'resolved-config.json'
        atomic_json(config_path, self.config)
        files['resolved_config'] = {'path': str(config_path), 'sha256': sha256(config_path)}
        record = {'version': 1, 'method': 'pure_opsd', 'run_root': str(self.root),
                  'source_commit': self.source, 'config_sha256': self.config_sha,
                  'global_step': step, 'actual_outer_updates': step, 'next_global_step': step + 1,
                  'horizon': 300, 'warmup_steps': 57, 'world_size': 8,
                  'driver_pid': self.pid, 'uid': self.uid, 'job_id': self.job,
                  'pause_nonce': secrets.token_hex(32), 'files': files, 'outer_metrics': metrics}
        path = driver_directory / 'manifest.json'
        atomic_json(path, record)
        return path, record

    def verify_receipt(self, path, manifest_path, manifest):
        envelope = load_json(owned_file(path, self.root))
        if set(envelope) != {'payload', 'signature_hex'}:
            raise ValueError('Invalid signed review envelope')
        verify_file(self.key_record, self.root)
        payload = envelope['payload']
        verify_rsa_signature(payload, envelope['signature_hex'], self.public_key)
        expected = {key: manifest[key] for key in ('method', 'run_root', 'source_commit', 'config_sha256',
                    'global_step', 'actual_outer_updates', 'next_global_step', 'horizon', 'warmup_steps',
                    'world_size', 'driver_pid', 'uid', 'job_id', 'pause_nonce')}
        expected['status'] = 'PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
        expected['checkpoint_manifest_sha256'] = sha256(manifest_path)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError('Independent receipt identity mismatch: ' + key)
        if self.current_source() != self.source or self.pid != os.getpid() or self.job != os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Training process/source changed while paused')
        for record in manifest['files'].values():
            verify_file(record, self.root)
        if payload.get('checkpoint_files') != manifest['files']:
            raise ValueError('Receipt does not bind every checkpoint/launch file')
        verify_file(payload['independent_review'], self.root)
        evidence = payload['evidence']
        if not isinstance(evidence, dict) or not {'resources', 'retrieval', 'hf_reload', 'rank_audit'} <= set(evidence):
            raise ValueError('Independent receipt evidence is incomplete')
        for record in evidence.values():
            verify_file(record, self.root)
        prefix = payload['train_log_prefix']
        live = owned_file(prefix['live_path'], self.root)
        if live != Path(self.settings.train_log_path):
            raise ValueError('Receipt log is not this training log')
        snapshot = verify_file({'path': prefix['snapshot_path'], 'sha256': prefix['sha256']}, self.root)
        if live == snapshot or type(prefix['bytes']) is not int or prefix['bytes'] <= 0:
            raise ValueError('Invalid immutable log prefix')
        if snapshot.stat().st_size != prefix['bytes']:
            raise ValueError('Log snapshot size differs')
        with open(live, 'rb') as handle:
            if hashlib.sha256(handle.read(prefix['bytes'])).hexdigest() != prefix['sha256']:
                raise ValueError('Live training prefix changed')
        text = snapshot.read_text()
        rows = [json.loads(line.split('OPSD_E1_OUTER_JSON=', 1)[1]) for line in text.splitlines()
                if 'OPSD_E1_OUTER_JSON=' in line]
        if [row['actual_outer_updates'] for row in rows] != list(range(1, 6)):
            raise ValueError('Independent snapshot lacks exactly five completed outers')
        for row in rows:
            if (row['global_step'] != row['actual_outer_updates'] or row['driver_pid'] != self.pid or
                    row['job_id'] != self.job or row['source_commit'] != self.source or
                    row['config_sha256'] != self.config_sha):
                raise ValueError('Outer log row belongs to another process/configuration')
            validate_outer(row['metrics'])
        return payload

    def wait(self, manifest_path, manifest):
        print('OPSD_E1_STEP5_PAUSED_JSON=' + json.dumps({'manifest_path': str(manifest_path),
              'manifest_sha256': sha256(manifest_path), 'driver_pid': self.pid,
              'job_id': self.job, 'pause_nonce': manifest['pause_nonce']}, sort_keys=True), flush=True)
        while not self.receipt.exists():
            if self.receipt.is_symlink():
                raise ValueError('Receipt is a symlink')
            time.sleep(2)
        self.verify_receipt(self.receipt, manifest_path, manifest)
        print('OPSD_E1_STEP5_RELEASED_JSON=' + json.dumps({'driver_pid': self.pid,
              'actual_outer_updates': 5, 'next_global_step': 6, 'receipt_sha256': sha256(self.receipt)},
              sort_keys=True), flush=True)
