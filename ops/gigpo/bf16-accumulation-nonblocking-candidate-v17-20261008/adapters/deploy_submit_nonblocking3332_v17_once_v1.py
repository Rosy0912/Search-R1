"""Single-use v17: six bound runtime files and isolated source delta; one official200 submission."""
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import zlib
from zero_adam_stop3332_nonblocking_v17_once_v1 import (BASE, OUT, ROOT, RUN, STATE, OLD, NEW, PARENT_BINDING,
    PARENT_SHA, PARENT_SOURCE, SOURCE, OLD_REPO, NEW_REPO, ENV, require, sha, packed, digest, checked, record, once, gates, transport, query, no_updates)

OPTIONS = ['--parsable', '--partition=gpu', '--nodelist=h100-8', '--nodes=1', '--ntasks=1',
           '--gres=gpu:h100:8', '--cpus-per-task=128', '--mem=512G', '--exclusive',
           '--no-requeue', '--time=1-00:00:00', '--export=ALL']
PATCH_KEYS = {'helpers', 'runtime_files', 'runtime_version', 'runtime_variant',
    'runtime_relative_files_digest', 'runtime_canonical_files_digest', 'declared_primary_files_digest',
    'owned_submit_namespace', 'allow_bf16_reduced_precision_reduction', 'actual_runtime_static_review',
    'actual_runtime_static_manifest', 'binding_derivation', 'bf16accum_repair_parent_job',
    'bf16accum_repair_parent_zero_adam_required', 'bf16accum_repair_stop_fact_path', 'initial_baseline_policy',
    'repo', 'code_commit', 'source_commit', 'source_tree', 'source_version', 'source_files',
    'source_file_count', 'source_relative_files_digest', 'source_canonical_files_digest',
    'human_nonblocking_diagnostics', 'external_diagnostic_review_blocks_training', 'diagnostic_authorization_scope',
    'capture_actual_runner_inputs', 'capture_actual_normalizer_inputs', 'calibration_phase',
    'initial_calibration_counted_outer_limit', 'production_tail_required_before_outer6', 'allowed_safety_modes',
    'accepted_actual_source_CPU_case_count', 'accepted_actual_source_CPU_result',
    'independently_accepted_current_source_runtime_CPU_only', 'source_runtime_cpu_pending',
    'source_runtime_controls_cpu_pending'}


def actual_stop(row):
    require(Path(row['path']) == STATE / 'STOP-FACTS.json', 'Wrong parent stop facts')
    facts = json.loads(checked(row['path'], row))
    require(facts['status'] == 'ACTUAL_OWN3332_GRACEFUL_ZERO_ADAM_STOPPED'
            and facts['parent_job'] == '3332' and facts['parent_binding_sha256'] == PARENT_SHA
            and facts['source_commit'] == PARENT_SOURCE and facts['completed_outer'] == facts['actual_adam_writes_per_rank'] == 0
            and facts['target_source_commit'] == SOURCE
            and facts['scheduler_step'] == 0 and facts['data_cursor'] == 512
            and facts['cgroup_gone'] is True and facts['original_pids_gone'] is True
            and facts['actual_residual_pids'] == [] and facts['no_probability_approval'] is True,
            'No actual terminal zero-Adam parent facts')
    pids = [r['pid'] for r in facts['original_processes']] + [facts['executor']]
    require(not any((Path('/proc') / str(pid)).exists() for pid in pids)
            and not Path(facts['cgroup']).exists(), 'Original PID/cgroup remains or PID reused')
    terminal = json.loads(checked(facts['original_run_result']['path'], facts['original_run_result']))
    require(terminal['status'] == 'FAILED' and terminal['cleanup_residual_states'] == {}
            and terminal['error'] == 'Owned calibration executor interrupted by signal 15', 'Wrong parent terminal')
    scheduler = query(['scontrol', 'show', 'job', '3332', '-o'])
    require(re.search(r'\bJobId=3332\b', scheduler['stdout'])
            and re.search(r'\bJobState=FAILED\b', scheduler['stdout']), 'Missing actual Slurm terminal; pending is insufficient')
    gpu = query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    require(not gpu['stdout'].strip(), 'GPU compute remains; do not affect foreign jobs')
    no_updates()
    return facts, scheduler, gpu


def remote(payload):
    require(socket.gethostname().split('.')[0] == 'h100-8' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Wrong CPU-only submission receiver')
    require(STATE.resolve() == STATE and STATE.is_dir() and not STATE.is_symlink(), 'Missing exact parent stop namespace')
    # Exclusive before staging: uncertainty or any partial attempt consumes this invocation.
    once(STATE / 'SUBMIT-INTENT.json', packed(dict(status='CLAIMED_NEVER_REPLAY',
        binding=payload['binding'], parent_stop_facts=payload['parent_stop_facts'],
        adapter_review=payload['adapter_review'], adapter_publication=payload['adapter_publication'],
        source_publication=payload['source_publication'],
        queued_submission_is_not_gpu_admission=True, extra_validation_job=False)))
    facts, scheduler, gpu = actual_stop(payload['parent_stop_facts'])
    queue = query(['squeue', '-h', '--all', '-o', '%i|%j|%T|%R'])
    for line in queue['stdout'].splitlines():
        fields = line.split('|')
        require(len(fields) == 4, 'Unparseable scheduler queue')
        require(not fields[1].startswith('gigpo-e1-fresh200-live'), 'Own old/new job remains; NEVER REPLAY')
    parent = json.loads(checked(PARENT_BINDING, payload['parent_binding']))
    require(len(parent['files']) == 39961 and len(parent['runtime_files']) == 44
            and parent['helpers'] == str(OLD) and parent['code_commit'] == PARENT_SOURCE, 'Wrong complete parent binding')
    require(set(payload['patch']) == PATCH_KEYS, 'Unexpected binding mutation')
    child = dict(parent)
    child.update(payload['patch'])
    new_rows = {r['path']: r for r in child['runtime_files']}
    source_rows = {r['path']: r for r in child['source_files']}
    require(len(source_rows) == 494 and child['repo'] == str(NEW_REPO)
            and child['code_commit'] == child['source_commit'] == SOURCE
            and child['human_nonblocking_diagnostics'] is True
            and child['external_diagnostic_review_blocks_training'] is False, 'Wrong isolated source/nonblocking scope')
    require(len(new_rows) == 44 and child['helpers'] == str(NEW) and child['runtime_version'] == 17
            and child['allow_bf16_reduced_precision_reduction'] is False
            and child['owned_submit_namespace'] == str(STATE)
            and child['initial_baseline_policy']['status'] == 'REUSED_USER_AUTHORIZED_BASELINE'
            and child['initial_baseline_policy']['parent_job'] == '3332'
            and child['initial_baseline_policy']['new_run_evaluation_forward_calls'] == 0,
            'Wrong exact numerical candidate and historical baseline policy')
    replaced = source_replaced = 0
    child_files = []
    for row in parent['files']:
        if row['path'].startswith(str(OLD) + '/'):
            require(Path(row['path']).parent == OLD, 'Unexpected runtime subdirectory')
            child_files.append(new_rows[str(NEW / Path(row['path']).name)])
            replaced += 1
        elif row['path'].startswith(str(OLD_REPO) + '/'):
            child_files.append(source_rows[str(NEW_REPO / Path(row['path']).relative_to(OLD_REPO))])
            source_replaced += 1
        else:
            child_files.append(row)
    require(replaced == 44 and source_replaced == 494, 'All44 runtime and494 isolated source substitutions required')
    child['files'] = child_files
    child['evidence'] = parent['evidence'] + payload['additional_evidence']
    require(len(child['files']) == 39961 and child['evidence'][:len(parent['evidence'])] == parent['evidence'],
            'Original source/assets/evidence lost')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        require(child[key] is parent[key] is None, 'Four unknown tolerances must remain null')
    require(child['official_training_outer_target'] == parent['official_training_outer_target'] == 200
            and child['negative_tail_limit'] is parent['negative_tail_limit'] is None,
            'Scientific target or unselected safety tail changed')
    raw = packed(child)
    require(len(raw) == payload['binding']['bytes'] and sha(raw) == payload['binding']['sha256'],
            'Remote delta differs from independently reviewed exact child binding')
    require(not NEW.exists() and not NEW.is_symlink() and NEW.parent.resolve() == NEW.parent,
            'Frozen v17 namespace already exists; no overwrite/replay')
    NEW.mkdir()
    require(len(payload['runtime']) == 44 and set(r['name'] for r in payload['runtime']) == {Path(k).name for k in new_rows},
            'Incomplete runtime staging')
    for row in payload['runtime']:
        name = row['name']
        require(Path(name).name == name and name in {Path(k).name for k in new_rows}, 'Runtime path escaped')
        if 'b64' in row:
            require(name in payload['changed_files'], 'Unexpected changed runtime payload')
            data = zlib.decompress(base64.b64decode(row['b64']))
        else:
            require(name not in payload['changed_files'], 'Changed runtime cannot clone old bytes')
            data = checked(OLD / name, row)
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Staged bytes differ')
        once(NEW / name, data)
    require(sorted(payload['changed_files']) == ['calibration_contract_v1.py', 'calibration_driver_v1.py', 'calibration_run_v1.py',
            'calibration_state_v1.py', 'calibration_worker_v1.py', 'live_continuation_v10.py'], 'Beyond six bound runtime changes')
    for row in payload['evidence_staging']:
        path = Path(row['path'])
        require(path.parent == STATE and path.name != 'BINDING.json', 'Metadata escaped private namespace')
        data = zlib.decompress(base64.b64decode(row['b64']))
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Evidence frame changed')
        once(path, data)
    source_bundle = payload['source_bundle']
    bundle_path = STATE / 'SOURCE-DELTA.bundle'
    bundle_bytes = zlib.decompress(base64.b64decode(source_bundle['b64']))
    require(len(bundle_bytes) == source_bundle['bytes'] and sha(bundle_bytes) == source_bundle['sha256'], 'Source bundle frame changed')
    once(bundle_path, bundle_bytes)
    require(not NEW_REPO.exists() and not NEW_REPO.is_symlink() and NEW_REPO.parent.resolve() == NEW_REPO.parent,
            'Isolated source namespace already exists; no overwrite')
    require(subprocess.check_output(['git', '-C', str(OLD_REPO), 'rev-parse', 'HEAD'], text=True).strip() == PARENT_SOURCE,
            'Original frozen source differs')
    for command in (['git', 'clone', '--shared', '--no-checkout', str(OLD_REPO), str(NEW_REPO)],
                    ['git', '-C', str(NEW_REPO), 'fetch', str(bundle_path), 'HEAD'],
                    ['git', '-C', str(NEW_REPO), 'checkout', '--detach', SOURCE]):
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=ENV)
    for row in child['runtime_files'] + child['source_files'] + child['evidence']:
        checked(row['path'], row)
    repo = Path(child['repo'])
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
            and not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip(),
            'Frozen source changed')
    # The retained18 NFS4 read is a shared visibility proof, not a durable independent copy.
    storage = payload['shared_storage']
    require(storage['client_node'] == 'h100-10' and storage['mount_type'] == 'nfs4'
            and storage['source'] == '10.32.0.8:/mnt/data' and storage['binding_sha256'] == PARENT_SHA,
            'Output storage lacks the verified same shared server/client binding')
    once(STATE / 'BINDING.json', raw)
    actual_stop(payload['parent_stop_facts'])
    command = ['sbatch'] + OPTIONS + [str(NEW / 'calibration_v1.sbatch'), str(STATE / 'BINDING.json'),
        payload['binding']['sha256'], child['repo'], str(NEW), child['environment_root']]
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=ENV, timeout=60)
        job = result.stdout.strip()
        receipt = dict(status='SUBMITTED_EXACTLY_ONCE' if result.returncode == 0 and job.isdecimal()
                       else 'UNCERTAIN_NEVER_REPLAY', command=command, returncode=result.returncode,
                       stdout=result.stdout, stderr=result.stderr, job_id=job if result.returncode == 0 and job.isdecimal() else None,
                       binding=payload['binding'], parent_stop_facts=payload['parent_stop_facts'],
                       parent_completed_outer=0, parent_adam_writes=0, training_success_not_yet_observed=True,
                       extra_validation_job=False, queued_submission_is_not_gpu_admission=True,
                       shared_storage=storage, server8_release_date='2026-10-20',
                       retained_independent_checkpoint_copy='NOT_YET_CREATED_MIGRATE_BEFORE_SERVER8_RELEASE')
    except BaseException as error:
        receipt = dict(status='UNCERTAIN_NEVER_REPLAY', error_type=type(error).__name__, error=str(error),
                       binding=payload['binding'], parent_stop_facts=payload['parent_stop_facts'])
    once(STATE / 'SUBMIT-RECEIPT.json', packed(receipt))
    print(json.dumps(receipt))
    require(receipt['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Read durable intent/queue; NEVER REPLAY')


def main():
    require(len(sys.argv) == 2, 'Supply the exact reviewed/publication/actual-stop gate file')
    gate, derivation = gates(sys.argv[1], 'submit')
    stop_row = gate['parent_stop_facts']
    local_facts = json.loads(checked(stop_row['local_path'], stop_row))
    require(local_facts['status'] == 'ACTUAL_OWN3332_GRACEFUL_ZERO_ADAM_STOPPED', 'Actual stop receipt required')
    runtime = []
    for row in derivation['runtime_manifest']['files']:
        item = dict(name=row['path'], **row['after'])
        data = checked(Path(derivation['runtime_manifest']['candidate_runtime']) / row['path'], item)
        if row['path'] in derivation['changed_files']:
            item['b64'] = base64.b64encode(zlib.compress(data)).decode()
        runtime.append(item)
    staging = []
    for row in derivation['additional_evidence_staging']:
        data = checked(row['local_path'], row)
        staging.append(dict(path=row['path'], bytes=len(data), sha256=sha(data),
                            b64=base64.b64encode(zlib.compress(data)).decode()))
    bundle_row = derivation['source_bundle']
    bundle_data = checked(bundle_row['local_path'], bundle_row)
    payload = dict(source_bundle=dict(bytes=len(bundle_data), sha256=sha(bundle_data),
        b64=base64.b64encode(zlib.compress(bundle_data)).decode()), parent_binding=derivation['parent_binding'], binding=derivation['child_binding'],
        patch=derivation['binding_patch'], additional_evidence=derivation['additional_evidence'],
        runtime=runtime, changed_files=derivation['changed_files'], evidence_staging=staging,
        parent_stop_facts={k: v for k, v in stop_row.items() if k != 'local_path'},
        adapter_review=gate['adapter_review'], adapter_publication=gate['adapter_publication'],
        shared_storage=derivation['shared_storage'], source_publication=gate['source_publication'])
    transport(payload, 'submit', (BASE / 'work/zero_adam_stop3332_nonblocking_v17_once_v1.py').read_bytes(), Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
