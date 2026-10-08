"""Owned same-node live200 executor; launch requires new exact live goal authority."""
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from owned_runtime_resume_v1 import ROOT, require, sha, cgroup_scope, OwnedProcesses, gpu_sample, available
from calibration_contract_v1 import SCOPE, LIVE_SCOPE, LIVE_GOAL_UPDATED_AT, DATA, validate_binding, validate_guard_binding


def query_gpu_coexistence_snapshot(timeout_seconds, *, command_runner=subprocess.check_output,
                                  monotonic=time.monotonic):
    """Same source parser/checker; exact raw reads are bounded by one shared budget."""
    from search_r1.baselines.preflight import parse_gpu_inventory
    require(0 < timeout_seconds <= 60., 'Invalid GPU cooldown command budget')
    deadline = monotonic() + timeout_seconds
    query = ['nvidia-smi', '--format=csv,noheader,nounits']
    raw = dict(gpu_inventory_csv=None, compute_apps_csv=None)
    try:
        raw['gpu_inventory_csv'] = command_runner(
            query + ['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'],
            text=True, timeout=timeout_seconds)
        remaining = deadline - monotonic()
        require(remaining > 0, 'GPU inventory query exceeded cooldown command budget')
        raw['compute_apps_csv'] = command_runner(query + ['--query-compute-apps=gpu_uuid,pid'],
                                                text=True, timeout=remaining)
        require(monotonic() < deadline, 'GPU process query exceeded cooldown command budget')
        inventory = parse_gpu_inventory(raw['gpu_inventory_csv'])
        processes = {line.split(',')[0].strip() for line in raw['compute_apps_csv'].splitlines() if ',' in line}
        return inventory, processes, raw
    except BaseException as error:
        # Preserve malformed/partial originals even when parsing or a later read
        # fails. This is evidence only and cannot substitute missing gate data.
        partial = getattr(error, 'output', None)
        if partial is not None:
            raw['failed_command_partial_output'] = partial.decode(errors='replace') if isinstance(partial, bytes) else partial
        error.gpu_coexistence_raw_snapshot = raw
        raise


def wait_retriever_gpu_coexistence(run, selected, guard, retriever, *, snapshot, checker,
                                  monotonic=time.monotonic, sleep=time.sleep):
    """Keep both original total15% gates; allow own HTTP-probe activity to cool.

    The accepted guard-v2 continues enforcing foreign use throughout. No actor,
    Ray head, sampler or optimizer may start until this separate launch gate passes.
    Dependencies, unknown snapshots and the fixed60s deadline fail closed.
    """
    journal = Path(run) / 'gpu-retriever-coexistence-before-ray.jsonl'
    receipt = Path(run) / 'gpu-retriever-coexistence-before-ray.json'
    started = monotonic()
    deadline = started + 60.
    attempt = 0

    def append(row):
        # Preserve unknown/nonfinite readings without writing invalid JSON.
        import math
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return {'nonfinite_float_repr': repr(value)}
            if isinstance(value, dict):
                return {str(k): finite_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(v) for v in value]
            return value
        with journal.open('a') as stream:
            stream.write(json.dumps(finite_json(row), allow_nan=False, sort_keys=True) + '\n')
            stream.flush()
        if row['decision'] != 'WAIT_RETRY':
            receipt.write_text(json.dumps(finite_json(row), indent=2, allow_nan=False) + '\n')

    while True:
        attempt += 1
        row = dict(scope='POST_HTTP_LOCAL_RETRIEVER_TOTAL15_COOLDOWN', attempt=attempt,
                   selected_devices=list(selected), timeout_seconds=60., retry_seconds=3.,
                   elapsed_seconds=monotonic() - started, total_sm_limit_percent=15.,
                   total_memory_limit_percent=15., foreign_guard_v2_unchanged=True,
                   guard_pid=guard.pid, retriever_pid=retriever.pid,
                   dependency_exit_codes=dict(guard=guard.poll(), retriever=retriever.poll()),
                   gate_snapshot=None, raw_gpu_sample=None)
        try:
            require(all(v is None for v in row['dependency_exit_codes'].values()),
                    'Owned guard/retriever exited during GPU cooldown')
            require(monotonic() < deadline, 'Local GPU retriever exceeds existing launch eligibility after60s')
            # Exact raw CSV and its parsed gate inventory come from the same
            # bounded reads, not a later potentially quieter substitute snapshot.
            row['sample_started_elapsed_seconds'] = monotonic() - started
            inventory, processes, raw = snapshot(deadline - monotonic())
            row['raw_gpu_sample'] = raw
            row['gate_sample_elapsed_seconds'] = monotonic() - started
            row['gate_snapshot'] = dict(inventory=inventory, compute_gpu_uuids=sorted(processes))
            row['dependency_exit_codes'] = dict(guard=guard.poll(), retriever=retriever.poll())
            require(all(v is None for v in row['dependency_exit_codes'].values()),
                    'Owned guard/retriever exited while reading GPU cooldown snapshot')
            require(monotonic() < deadline, 'GPU cooldown snapshot exceeded60s deadline')
            try:
                row['selected_gpu_records'] = checker(selected, inventory, processes)
            except RuntimeError as error:
                # Retry ONLY the frozen checker's explicit total occupancy case.
                # NaN/unknown/invalid readings must not become a later admission.
                retry_messages = {f'Allocated GPU {device} exceeds the 15% idle thresholds'
                                  for device in selected}
                if str(error) not in retry_messages:
                    raise
                # The frozen global checker stops at its first overbusy GPU.
                # Validate EVERY selected GPU before treating that short-circuit
                # as a retry: later unknown data or duplicate UUIDs are fatal.
                resolved_uuids = []
                for device in selected:
                    matches = [v for index, v in inventory.items()
                               if device == index or device == v['uuid']]
                    require(len(matches) == 1, 'Cannot resolve every allocated GPU during cooldown retry')
                    resolved_uuids.append(matches[0]['uuid'])
                    try:
                        checker([device], inventory, processes)
                    except RuntimeError as device_error:
                        if str(device_error) != f'Allocated GPU {device} exceeds the 15% idle thresholds':
                            raise
                require(len(set(resolved_uuids)) == len(selected),
                        'Duplicate allocated GPU UUIDs during cooldown retry')
                row.update(decision='WAIT_RETRY', eligibility_error_type=type(error).__name__,
                           eligibility_error=str(error))
            else:
                row['dependency_exit_codes'] = dict(guard=guard.poll(), retriever=retriever.poll())
                require(all(v is None for v in row['dependency_exit_codes'].values()),
                        'Owned guard/retriever exited before GPU coexistence admission')
                require(monotonic() < deadline, 'GPU coexistence admission exceeded60s deadline')
                row.update(decision='PASS_ORIGINAL_TOTAL15', elapsed_seconds=monotonic() - started)
                append(row)
                return row
        except BaseException as error:
            if hasattr(error, 'gpu_coexistence_raw_snapshot'):
                row['raw_gpu_sample'] = error.gpu_coexistence_raw_snapshot
            row.update(decision='FAILED_CLOSED', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=monotonic() - started)
            append(row)
            raise
        append(row)
        sleep(min(3., max(0., deadline - monotonic())))


def main():
    require(len(sys.argv) == 3, 'Need reviewed binding and exact SHA')
    binding_path = Path(sys.argv[1])
    require(sha(binding_path) == sys.argv[2], 'Calibration binding changed')
    binding = json.loads(binding_path.read_text())
    require(binding.get('human_nonblocking_diagnostics') is True
            and binding.get('external_diagnostic_review_blocks_training') is False
            and binding.get('diagnostic_authorization_scope') == 'HUMAN_NONBLOCKING_DIAGNOSTICS_V1',
            'This isolated runtime requires explicit human nonblocking diagnostic authorization')
    require(binding['allow_bf16_reduced_precision_reduction'] is False,
            'This immutable BF16 candidate requires a new binding declaring the disabled reduction flag')
    require(binding['initial_baseline_policy']['status'] == 'REUSED_USER_AUTHORIZED_BASELINE'
            and binding['initial_baseline_policy']['parent_job'] == '3332'
            and binding['initial_baseline_policy']['new_run_evaluation_forward_calls'] == 0,
            'This immutable runtime requires explicit user-authorized initial baseline reuse')
    helpers = Path(__file__).resolve().parent
    for name in ('calibration_run_v1.py', 'calibration_driver_v1.py',
                 'calibration_worker_v1.py', 'calibration_state_v1.py'):
        require(sum(Path(item['path']).resolve() == helpers / name for item in binding['files']) == 1,
                'New binding must cover the actual BF16 candidate helper directory: ' + name)
    repo = Path(os.environ['GIGPO_RESUME_REPO'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == binding['code_commit'] and not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
            'Dirty or wrong frozen source')
    validate_binding(binding, commit)
    require(repo.resolve() == Path(binding['repo']).resolve(), 'Different source repo path')
    require(Path(sys.executable).resolve() == (Path(binding['environment_root']) / 'bin/python').resolve(),
            'Unbound actual interpreter/environment')
    for item in binding['files'] + binding['evidence']:
        require(Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256'],
                'Bound source/helper/physical/evidence bytes changed')
    guard_identity = validate_guard_binding(binding)
    for item in guard_identity.values():
        path = Path(item['path'])
        require(path.resolve() == path and path.is_file() and sha(path) == item['sha256'],
                'Actual guard/module path or physical bytes differ')
    # The historical env-variable name is a source API, not a native result.
    # Its authorization content is the new exact first5-retained live200 scope.
    os.environ.update(GIGPO_POLICY_NATIVE_BINDING=str(binding_path), GIGPO_POLICY_NATIVE_BINDING_SHA256=sys.argv[2],
        SEARCH_R1_BASELINE_COMMIT=commit, GIGPO_E1_COMMON_ROOT=binding['common_root'])
    require(not any(key.startswith('GIGPO_QUALITY_') for key in os.environ), 'Inherited historical quality-resume authorization')
    job = os.environ['SLURM_JOB_ID']
    scheduler = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=15)
    actual_node = subprocess.check_output(['scontrol', 'show', 'node', 'h100-8', '-o', '-d'], text=True, timeout=15)
    scheduler_config = subprocess.check_output(['scontrol', 'show', 'config'], text=True, timeout=15)
    actual_node_jobs = subprocess.check_output(['squeue', '--noheader', '--all',
        '--states=RUNNING,SUSPENDED,COMPLETING', '--nodelist=h100-8', '--format=%A|%N'], text=True, timeout=15)
    from slurm_allocation_v2 import validate_exclusive_allocation
    allocation = validate_exclusive_allocation(scheduler, actual_node, actual_node_jobs, os.environ,
        socket.gethostname(), scheduler_config_raw=scheduler_config)
    require(socket.gethostname().split('.')[0] == 'h100-8' and os.environ['SLURM_JOB_NODELIST'] == 'h100-8', 'Wrong node')
    require(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 8 and os.environ.get('SLURM_JOB_GPUS'), 'Not full8 GPU allocation')
    require(int(os.environ['SLURM_CPUS_PER_TASK']) >= 128 and int(os.environ['SLURM_MEM_PER_NODE']) >= 256 * 1024, 'Allocation too small')
    lock = (ROOT / 'gpu-job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from search_r1.baselines.preflight import validate_inherited_gpu_leases, gpu_snapshot, check_selected_gpus
    validate_inherited_gpu_leases(8)
    scope, relative = cgroup_scope(job)
    owned = OwnedProcesses(scope, relative)
    for port in (18081, 16381):
        available(port)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    os.environ.update(NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1')
    headers = '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include'
    header_cpath = ':'.join((headers, headers + '/python3.10', headers + '/x86_64-linux-gnu/python3.10'))
    os.environ.update(CPATH=header_cpath, SEARCH_R1_PYTHON_HEADER_CPATH=header_cpath)
    run_id = f'gigpo-e1-fresh200-live-job{job}-{time.time_ns()}'
    run = ROOT / 'runs' / run_id
    run.mkdir()
    (run / 'actual-exclusive-slurm-allocation.json').write_text(json.dumps(allocation, indent=2, allow_nan=False))
    for name in ('audit', 'tmp', 'ray', 'cache'):
        (run / name).mkdir()
    alias = Path('/tmp') / f'gigpo-c{job}'
    require(not alias.exists() and not alias.is_symlink(), 'Existing owned Ray alias')
    alias.symlink_to(run, target_is_directory=True)
    os.environ.update(GIGPO_E1_ROOT=str(ROOT), GIGPO_E1_RUN=str(run), AGENT_ROOT=str(ROOT),
        SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root', SEARCH_R1_BASELINE_OUTPUT=str(run),
        SEARCH_R1_BASELINE_RUN_ROOT=str(alias), SEARCH_R1_BASELINE_RUN_ID=run_id,
        SEARCH_R1_BASELINE_AUDIT_DIR=str(run / 'audit'), SEARCH_R1_BASELINE_DATA_IDENTITY=DATA,
        RAY_ADDRESS='127.0.0.1:16381', TMPDIR=str(alias / 'tmp'), RAY_TMPDIR=str(alias / 'ray'),
        XDG_CACHE_HOME=str(alias / 'cache'), HF_HOME=str(alias / 'cache/hf'),
        HF_DATASETS_CACHE='/mnt/data/shaoshitong/search-r1-rl/gigpo/cache/retriever-datasets',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WANDB_MODE='offline', VLLM_NO_USAGE_STATS='1', NCCL_NVLS_ENABLE='0', VLLM_ATTENTION_BACKEND='FLASH_ATTN')
    result = dict(status='FAILED', scope=LIVE_SCOPE, job_id=job, run_id=run_id, source_commit=commit,
                  binding_sha256=sha(binding_path), retained_training_updates_contributed=None,
                  restore_checkpoint=None, restore_manifest_sha256=None,
                  calibration_phase=binding['calibration_phase'], cgroup=str(scope), human_old_policy_kl_selected=True, token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908, negative_tail_threshold_not_selected=True,
                  guard_identity=guard_identity, actual_slurm_allocation=allocation, source_native_acceptance=None,
                  actual_gpu_acceptance='PENDING_THIS_LIVE_JOB')
    started = time.monotonic()
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from search_r1.baselines.holdout import prepare_common_holdout
        from search_r1.baselines.controller import validate_config
        from search_r1.baselines.policy_safety import validate_config as validate_safety
        with initialize_config_dir(config_dir=str(repo / 'verl/trainer/config'), version_base=None):
            config = compose(config_name='ppo_gigpo_instruct200_safe')
        config.actor_rollout_ref.actor.policy_safety.mode = 'calibration'
        config.actor_rollout_ref.actor.policy_safety.calibration_scope = SCOPE
        config.actor_rollout_ref.actor.policy_safety.candidate_old_policy_kl = .2
        config.actor_rollout_ref.actor.policy_safety.target_old_policy_kl = .2
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.token_kl_threshold_source', 'human', force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.token_kl_human_goal_updated_at', 1791282908, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.fresh200_live_protocol', True, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.human_nonblocking_diagnostics', True, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.human_nonblocking_diagnostics', True, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.live_training_protocol', True, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.first5_known_token_kl_enforced', True, force_add=True)
        OmegaConf.update(config, 'actor_rollout_ref.actor.policy_safety.first5_tail_observation_only', True, force_add=True)
        OmegaConf.update(config, 'trainer.fresh200.live_goal_updated_at', LIVE_GOAL_UPDATED_AT, force_add=True)
        config.trainer.experiment_name = 'gigpo-e1-fresh200-live'
        config.trainer.default_local_dir = str(run / 'checkpoints')
        # Persist the same deterministic constructor metadata injection before
        # binding C0; the GiGPO path has no critic and keeps its mathematics.
        config.critic.optim.total_training_steps = config.trainer.total_training_steps
        OmegaConf.resolve(config)
        from search_r1.baselines.fresh200 import validate_protocol
        validate_protocol(config)
        # Latest human instruction skips only the initial512 evaluation. Preserve
        # every other frozen protocol check and save the actual False config.
        config.trainer.val_before_train = False
        require(config.actor_rollout_ref.model.path == config.actor_rollout_ref.ref.model_path == binding['official_model_path'],
                'Actual actor/reference official source differs from binding')
        split = prepare_common_holdout(config)
        validate_config(config)
        require(split['content_identity']['sha256'] == DATA and split['train_rows'] == 169028
                and split['holdout_rows'] == 512, 'Actual common immutable inputs differ')
        OmegaConf.save(config, run / 'resolved-config.yaml')
        from live_continuation_v10 import immutable_json
        c0 = immutable_json(run / 'resolved-config-C0.json', OmegaConf.to_container(config, resolve=True))
        os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = c0['sha256']
        validate_safety(OmegaConf.to_container(config.actor_rollout_ref.actor.policy_safety, resolve=True))
        validate_inherited_gpu_leases(8)  # Actual inherited OFD proof immediately before guard/services.
        inherited_leases = json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
        inventory, processes = gpu_snapshot()
        selected = check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','), inventory, processes)
        from guard_uuid_input_v5 import validate_actual_guard_uuid_input
        uuid_proof = validate_actual_guard_uuid_input(binding, selected, inherited_leases)
        (run / 'guard-actual-uuid-input.raw.json').write_bytes(uuid_proof['raw_utf8'].encode('utf-8'))
        (run / 'actual-guard-full8-uuid-proof.json').write_text(json.dumps(uuid_proof, indent=2, allow_nan=False))
        (run / 'actual-selected-and-inherited-ofd-uuids.json').write_text(json.dumps(
            dict(selected=selected, inherited_leases=inherited_leases, inventory=inventory), indent=2, allow_nan=False))
        result['actual_guard_uuid_input'] = uuid_proof
        result['initial_gpu'] = gpu_sample()
        # Recheck at spawn, not just when reading the binding. Never silently run
        # the old ROOT guard while listing v2 only as documentary evidence.
        for item in guard_identity.values():
            require(sha(item['path']) == item['sha256']
                and ('bytes' not in item or Path(item['path']).stat().st_size == item['bytes']),
                'Bound actual guard/module/UUID input changed before spawn')
        guard = owned.spawn([sys.executable, guard_identity['resource_guard']['path'],
                             '--output', str(run / 'guard-resources.jsonl')], run / 'guard.log')
        deadline = time.monotonic() + 30
        while not (run / 'guard-resources.ready').exists():
            require(guard.poll() is None and time.monotonic() < deadline, 'Owned dual15% guard failed')
            time.sleep(1)
        retriever = owned.spawn([sys.executable, 'reproduction/h20/retriever_local.py',
            '--index_path', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/e5_Flat.index',
            '--corpus_path', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/wiki-18.jsonl',
            '--retriever_model', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/e5-base-v2',
            '--retriever_name', 'e5', '--topk', '3', '--faiss_gpu', '--device', 'cuda', '--threads', '8',
            '--host', '127.0.0.1', '--port', '18081'], run / 'retriever.log',
            env=dict(os.environ, PYTHONPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/retriever-gpu-lib' + os.pathsep + str(repo),
                     OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8'))
        import requests
        session = requests.Session()
        session.trust_env = False
        deadline = time.monotonic() + 900
        while True:
            require(retriever.poll() is None and guard.poll() is None and time.monotonic() < deadline, 'Owned GPU retriever failed')
            try:
                if session.get('http://127.0.0.1:18081/openapi.json', timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        subprocess.run([sys.executable, binding['retrieval_verifier_path']], check=True,
                       env=dict(os.environ, SEARCH_R1_HTTP_RECEIPT=str(run / 'local-http-retrieval-verification.json')))
        result['retriever_coexistence_evidence'] = dict(
            journal_path=str(run / 'gpu-retriever-coexistence-before-ray.jsonl'),
            receipt_path=str(run / 'gpu-retriever-coexistence-before-ray.json'))
        result['retriever_coexistence_admission'] = wait_retriever_gpu_coexistence(
            run, os.environ['CUDA_VISIBLE_DEVICES'].split(','), guard, retriever,
            snapshot=query_gpu_coexistence_snapshot, checker=check_selected_gpus)
        helpers = Path(__file__).resolve().parent
        head = owned.spawn([str(Path(binding['environment_root']) / 'bin/ray'), 'start', '--head', '--node-ip-address=127.0.0.1',
            '--port=16381', '--num-cpus=24', '--num-gpus=8', '--include-dashboard=false', '--object-store-memory=8589934592',
            '--temp-dir=' + str(alias / 'ray'), '--min-worker-port=21081', '--max-worker-port=22080',
            '--disable-usage-stats', '--block'], run / 'ray.log')
        deadline = time.monotonic() + 120
        while True:
            require(head.poll() is None and guard.poll() is None and time.monotonic() < deadline, 'Owned Ray failed')
            try:
                with socket.create_connection(('127.0.0.1', 16381), timeout=1):
                    break
            except OSError:
                time.sleep(2)
        subprocess.run([sys.executable, str(helpers / 'ray_audit_resume_v1.py'), str(run / 'ray-audit-environment-verification.json')], check=True)
        driver = owned.spawn([sys.executable, str(helpers / 'calibration_driver_v1.py'), str(run / 'resolved-config.yaml'), str(binding_path)],
                            run / 'calibration.log', pass_fds=tuple(int(v['fd']) for v in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])))
        while driver.poll() is None:
            require(all(p.poll() is None for p in (guard, retriever, head)), 'Owned calibration dependency died')
            time.sleep(5)
        require(driver.returncode == 0, 'Actual calibration/probability identity failed; preserve evidence')
        observed = json.loads((run / 'driver-result.json').read_text())
        require(observed['retained_training_updates_contributed'] == 200
                and observed['scope'] == LIVE_SCOPE and observed['live_goal_updated_at'] == LIVE_GOAL_UPDATED_AT
                and observed['official_training_outer_target'] == observed['scheduler_after'] == 200
                and observed['calibration_actual_outer_updates'] == 200
                and observed['human_old_policy_kl_selected'] is True
                and observed['candidate_old_policy_kl'] == observed['target_old_policy_kl'] == .2
                and observed['token_kl_threshold_source'] == 'human'
                and observed['token_kl_human_goal_updated_at'] == 1791282908
                and observed['negative_tail_limit'] is None
                and observed['negative_tail_threshold_status'] == 'PENDING_BACKGROUND'
                and observed['independent_review_status'] == 'PENDING_BACKGROUND'
                and observed['actual_probability_accepted'] is False
                and observed['human_nonblocking_diagnostics'] is True
                and observed['optimal_threshold_accepted'] is False
                and observed['output_quality_accepted'] is False
                and observed['production_admission_accepted'] is False and observed['no_discarded_training_updates'] is True,
                'Live result did not retain actual200 under the bound human nonblocking scope')
        result.update(status=observed['status'], driver_exit=driver.returncode, observations=observed,
            retained_training_updates_contributed=200, negative_tail_threshold_not_selected=True,
            negative_tail_threshold_status='PENDING_BACKGROUND', independent_review_status='PENDING_BACKGROUND')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            owned.cleanup()
        except BaseException as error:
            result.update(status='FAILED', cleanup_error=repr(error))
            raise
        finally:
            alias.unlink(missing_ok=True)
            if (run / 'live-progress.json').is_file():
                try:
                    result['last_actual_driver_progress'] = json.loads((run / 'live-progress.json').read_text())
                    result['retained_training_updates_contributed'] = result['last_actual_driver_progress']['completed_outer_updates']
                except BaseException as progress_error:
                    result['progress_read_error'] = repr(progress_error)
            result.update(cleanup_residual_states=getattr(owned, 'cleanup_residual', {}), wall_seconds=time.monotonic() - started)
            try:
                result['final_gpu'] = gpu_sample()
                require(not result['final_gpu']['processes'], 'GPU compute remains after owned cleanup')
            except BaseException as error:
                result.update(status='FAILED', terminal_gpu_error=repr(error))
            (run / 'run-result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    require(result['status'] != 'FAILED', 'Calibration/cleanup failed')
    print(json.dumps(result, allow_nan=False), flush=True)


def interrupted(sig, frame):
    raise RuntimeError('Owned calibration executor interrupted by signal ' + str(sig))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
