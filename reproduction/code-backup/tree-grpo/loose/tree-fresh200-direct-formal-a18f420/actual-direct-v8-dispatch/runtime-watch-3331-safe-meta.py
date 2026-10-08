import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess

BASE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-current200-continue-3331'
BLOCKED = ('question', 'context', 'token_ids', 'tokenids', 'logprob', 'log_prob', 'logps', 'retrieval', 'prompt', 'response_text', 'weights', 'index')
STRING_KEYS = {'state', 'status', 'stage', 'decision', 'continuation_decision', 'reason', 'job_id', 'run_id', 'attempt_id', 'phase', 'device', 'rank', 'scheduler_type', 'optimizer_type', 'event', 'kind', 'outcome', 'error_type', 'exception_type', 'code_commit', 'source_commit', 'manifest_sha256', 'created_at', 'completed_at', 'timestamp', 'hostname', 'node', 'initialization', 'initialization_mode', 'warm_start', 'checkpoint_loaded', 'cold_start', 'optimizer_state_loaded', 'scope', 'schema', 'schema_version', 'version', 'method', 'generation_id', 'data_id', 'config_id', 'source_id', 'protocol', 'parent_job_id', 'parent_run_id', 'parent_code_commit', 'parent_manifest_sha256', 'child_code_commit', 'child_job_id', 'child_run_id', 'scope'}

def allowed_string(key):
    return key in STRING_KEYS or key.lower().endswith(('_sha256', '_sha')) or key.lower() == 'sha256'

SAFE_NUMERIC_ARRAY_KEYS = {'actual_optimizer_calls_per_rank', 'actual_completed_optimizer_calls_per_rank', 'total_optimizer_calls_per_rank', 'actual_scheduler_steps_per_rank', 'optimizer_step_values', 'cleanup_residual', 'optimizer_lrs', 'learning_rates_before', 'learning_rates_after', 'adam_steps'}

def allowed_list(value, key):
    return all(isinstance(x, dict) for x in value) or (key in SAFE_NUMERIC_ARRAY_KEYS and all(isinstance(x, (bool, int, float)) or x is None for x in value))

def scalar_metadata(value, key=''):
    if any(term in key.lower() for term in BLOCKED):
        return None
    if isinstance(value, dict):
        return {str(k): scalar_metadata(v, str(k)) for k, v in value.items()
                if not any(term in str(k).lower() for term in BLOCKED)
                and (isinstance(v, (dict, bool, int, float)) or v is None or (isinstance(v, str) and allowed_string(str(k))) or (isinstance(v, list) and allowed_list(v, str(k))))}
    if isinstance(value, list):
        if key in SAFE_NUMERIC_ARRAY_KEYS and all(isinstance(x, (bool, int, float)) or x is None for x in value):
            return value
        return [scalar_metadata(v) for v in value if isinstance(v, dict)]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str) and allowed_string(key):
        return value[:200]
    return None

def numeric_tree(value):
    if isinstance(value, dict):
        return {str(k): numeric_tree(v) for k, v in value.items() if isinstance(v, (dict, bool, int, float)) or v is None}
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return None

result = {'source_commit_expected': '9693f0c846fab483a03e1d70ce1109aeddc5e118', 'manifest_sha256_expected': '336a0a60df561fd3fcf50e7fdc03cee344a0b33e9d916193c7b4a4716b1b6a12', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': 3331, 'parent_job_id_expected': 3328, 'migration_boundary_expected': {'completed_outer': 2, 'optimizer_calls_per_rank': 40, 'scheduler_step': 2, 'sampler_cursor': 1024}, 'observer_hostname': os.uname().nodename, 'run_exists': os.path.isdir(BASE), 'files': {}}
policy_paths = sorted(glob.glob(os.path.join(BASE, 'output/tree-rank-audit/policy-safety/*.json')))
policy_metadata = {'statistics_scope': 'latest_event_per_rank_only', 'historical_event_file_count': len(policy_paths), 'selected_event_file_count': 0, 'valid_selected_json_file_count': 0, 'invalid_selected_json_file_count': 0, 'latest_by_rank': {}}
policy_latest_order = {}
selected_policy_paths = {}
for candidate_path in policy_paths:
    rank_match = re.search(r'rank-(\d+)-event-(\d+)', os.path.basename(candidate_path))
    if not rank_match: continue
    rank_key = rank_match.group(1)
    candidate_order = (os.path.getmtime(candidate_path), os.path.basename(candidate_path))
    prior = selected_policy_paths.get(rank_key)
    if prior is None or candidate_order > prior[0]: selected_policy_paths[rank_key] = (candidate_order, candidate_path)
policy_metadata['selected_event_file_count'] = len(selected_policy_paths)
for _, policy_path in sorted(selected_policy_paths.values()):
    try:
        with open(policy_path) as f:
            event = json.load(f)
    except (OSError, json.JSONDecodeError):
        policy_metadata['invalid_selected_json_file_count'] += 1
        continue
    policy_metadata['valid_selected_json_file_count'] += 1
    identity = event.get('identity', {}) if isinstance(event.get('identity'), dict) else {}
    names = {'job_id', 'run_id', 'code_commit', 'source_commit', 'rank', 'phase', 'event_index', 'event', 'global_step', 'attempt_id', 'committed_optimizer_calls', 'monitor_native_forward_calls', 'micro'}
    selected = {k:v for k,v in event.items() if k in names and (isinstance(v, (str, bool, int, float)) or v is None)}
    selected['identity'] = {k:v for k,v in identity.items() if k in names and (isinstance(v, (str, bool, int, float)) or v is None)}
    decision = event.get('decision', {})
    decision_numeric_names = {'allowed', 'kl_computation_valid', 'nonfinite_global_reductions', 'sampled_old_policy_kl', 'negative_tail_fraction', 'negative_tail_count_fraction', 'dual_clip_saturated_weight_fraction', 'mass', 'kl_sum', 'negative_mass', 'negative_tail_mass', 'negative_tail_count', 'negative_count', 'saturation_mass', 'invalid_count', 'overflow_count', 'global_monitor_max_abs_delta', 'global_monitor_mu_actor_max_abs_delta', 'global_monitor_mu_full_partition_beta_max_abs_delta', 'global_monitor_initial_mu_monitor_beta_max_abs_delta', 'original_sampler_actor_consistency_accepted', 'initial_mu_kl_baseline_subtracted', 'global_monitor_actor_full_partition_max_abs_delta', 'global_monitor_actor_frozen_monitor_max_abs_delta', 'global_monitor_actor_repeat_max_abs_delta', 'initial_actor_repeat_checked', 'additional_initial_actor_repeat_native_forward_calls'}
    if isinstance(decision, dict):
        selected['decision'] = {k:v for k,v in decision.items() if k in decision_numeric_names and (isinstance(v,(bool,int,float)) or v is None)}
        safe_reasons = {'empty_effective_mass','nonfinite_or_invalid_input_or_reduction','delta_exceeds_kl_log_bound_before_exp','sampled_old_policy_kl','negative_tail_fraction','nonfinite_actor_beta_or_monitor_difference','initial_actor_same_partition_repeat_inconsistency','initial_old_current_probability_inconsistency'}
        reasons=decision.get('reasons',[])
        selected['decision']['reasons']=[reason for reason in reasons if isinstance(reason,str) and reason in safe_reasons] if isinstance(reasons,list) else []
        selected['decision_scope']='Fixed length-stratified local-outer subset; not all rollout tokens; not unbiased full-policy KL'
    for numeric_key in ['outer_step','outer_completed','scheduler_advanced','automatic_rollback','monitor_rows','local_outer_rows','monitor_forward_wall_seconds','additional_initial_actor_repeat_native_forward_calls','measured_gradient_preclip','measured_gradient_postclip']:
        if isinstance(event.get(numeric_key),(bool,int,float)) or event.get(numeric_key) is None:
            selected[numeric_key]=event.get(numeric_key)

    config = event.get('config', {}) if isinstance(event.get('config'), dict) else {}
    selected['config'] = {}
    for config_key in ['monitor_rows_per_rank', 'target_old_policy_kl', 'negative_tail_limit', 'dual_clip', 'kl_log_bound']:
        if isinstance(config.get(config_key), (bool, int, float)):
            selected['config'][config_key] = config[config_key]
    correction = config.get('sampling_probability_correction', {})
    if isinstance(correction, dict) and isinstance(correction.get('enabled'), (bool, int, float)):
        selected['config']['sampling_probability_correction'] = {'enabled': correction['enabled']}
    rank = event.get('rank', identity.get('rank'))
    filename_rank = re.search(r'rank-(\d+)-event-(\d+)', os.path.basename(policy_path))
    if rank is None and filename_rank:
        rank = int(filename_rank.group(1))
        selected['rank_from_filename'] = rank
    order = (os.path.getmtime(policy_path), os.path.basename(policy_path))
    if rank is not None and (str(rank) not in policy_latest_order or order > policy_latest_order[str(rank)]):
        selected['name'] = os.path.relpath(policy_path, BASE)
        selected['mtime'] = os.path.getmtime(policy_path)
        policy_latest_order[str(rank)] = order
        policy_metadata['latest_by_rank'][str(rank)] = selected
result['policy_safety_events'] = policy_metadata
protected_pids = {1680782, 1680783, 1680784, 1680785, 1696443, 1696444, 1696445, 1696446}
own_pids = set()
for job_dir in glob.glob('/sys/fs/cgroup/freezer/**/job_3331', recursive=True):
    for directory, _, filenames in os.walk(job_dir):
        if 'cgroup.procs' in filenames:
            try:
                with open(os.path.join(directory, 'cgroup.procs')) as f:
                    own_pids.update(int(p) for p in f.read().split())
            except (OSError, ValueError):
                pass
own_processes = []
for pid in sorted(own_pids - protected_pids):
    try:
        with open('/proc/%s/cgroup' % pid) as f:
            membership = f.read()
        if '/job_3331/' not in membership:
            continue
        with open('/proc/%s/stat' % pid) as f:
            proc_stat = f.read()
        split = proc_stat.rfind(')')
        comm = proc_stat[proc_stat.find('(') + 1:split]
        fields = proc_stat[split + 2:].split()
        own_processes.append({'pid': pid, 'comm': comm, 'state': fields[0], 'utime_jiffies': int(fields[11]), 'stime_jiffies': int(fields[12]), 'starttime_jiffies': int(fields[19])})
    except (OSError, ValueError, IndexError):
        pass
result['original_eight_pid_identities'] = []
for pid in sorted(protected_pids):
    expected = 1502352181 if pid < 1690000 else 1502398762
    record = {'pid': pid, 'expected_starttime_jiffies': expected, 'exists': False}
    try:
        with open('/proc/%s/stat' % pid) as stream:
            raw = stream.read()
        values = raw[raw.rfind(')') + 2:].split()
        actual = int(values[19])
        record.update({'exists': True, 'starttime_jiffies': actual, 'starttime_preserved': actual == expected})
    except (OSError, ValueError, IndexError) as error:
        record['stat_read_error_type']=type(error).__name__
        record['stat_read_errno']=getattr(error,'errno',None)
    result['original_eight_pid_identities'].append(record)
result['own_job_processes'] = {'count': len(own_processes), 'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'), 'processes': own_processes}
result['py_spy_installed_path'] = shutil.which('py-spy')
result['own_measured_task_wait'] = []
for proc in own_processes:
    if proc['comm'] == 'ray::measured_t':
        try:
            with open('/proc/%s/wchan' % proc['pid']) as f:
                wchan = f.read().strip()
            result['own_measured_task_wait'].append({'pid': proc['pid'], 'starttime_jiffies': proc['starttime_jiffies'], 'state': proc['state'], 'wchan': wchan})
        except OSError:
            pass
anchors = ['Loading checkpoint shards', 'Loading safetensors checkpoint shards', 'Processed prompts', 'Training Progress', 'rollout', 'generate', 'Traceback', 'RuntimeError', 'ValueError', 'Error executing job']
result['anchored_log_metadata'] = {}
for name in ['stage.log', 'retriever.log', 'ray.log', 'guard.log']:
    path = os.path.join(BASE, name)
    metadata = {'exists': os.path.isfile(path)}
    if metadata['exists']:
        stat = os.stat(path)
        metadata.update({'native_log_numeric_metrics': {}, 'size_bytes': stat.st_size, 'mtime': stat.st_mtime, 'anchor_counts': {a: 0 for a in anchors}, 'epoch_step_count': 0, 'recent_progress_numeric': {}, 'traceback_files': [], 'exception_classes': []})
        with open(path, errors='replace') as f:
            for line in f:
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                # Existing published training scalars only; never emit log lines.
                metric_keys = ['critic/score/mean', 'critic/rewards/mean', 'critic/rewards/max', 'critic/rewards/min', 'actor/lr', 'actor/ppo_loss', 'actor/grad_norm', 'actor/entropy', 'training/actual_outer_updates', 'train/reward_mean', 'reward_mean']
                numeric_value = r'(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
                for metric_key in metric_keys:
                    metric_match = re.search(r'(?:[\"\']?)' + re.escape(metric_key) + r'(?:[\"\']?)\s*[:=]\s*' + numeric_value, clean_line)
                    if metric_match:
                        metric_value = float(metric_match.group(1))
                        if math.isfinite(metric_value): metadata['native_log_numeric_metrics'][metric_key] = metric_value
                for metric_match in re.finditer(r'(?:[\"\']?)((?:timing_s|timing_per_token_ms)/[A-Za-z0-9_]+)(?:[\"\']?)\s*[:=]\s*' + numeric_value, clean_line):
                    metric_value = float(metric_match.group(2))
                    if math.isfinite(metric_value): metadata['native_log_numeric_metrics'][metric_match.group(1)] = metric_value
                epoch_step = re.search(r'\bepoch\s+(\d+),\s+step\s+(\d+)\b', clean_line)
                if epoch_step:
                    metadata['epoch_step_count'] += 1
                    metadata['recent_progress_numeric']['epoch_step'] = {'epoch': int(epoch_step.group(1)), 'step': int(epoch_step.group(2))}
                for anchor in anchors:
                    count = clean_line.lower().count(anchor.lower())
                    if count:
                        metadata['anchor_counts'][anchor] += count
                        if anchor in ['Loading checkpoint shards', 'Loading safetensors checkpoint shards', 'Processed prompts', 'Training Progress']:
                            segment = clean_line[clean_line.lower().find(anchor.lower()) + len(anchor):]
                            progress = {}
                            percentage = re.search(r'(?<!\d)(\d+(?:\.\d+)?)%', segment)
                            ratio = re.search(r'(?<![\w/])(\d+)\s*/\s*(\d+)(?![\w/])', segment)
                            duration = re.search(r'\[(\d+:\d+(?::\d+)?)(?:<([^,\]]+))?', segment)
                            if percentage:
                                progress['percent'] = float(percentage.group(1))
                            if ratio:
                                progress['completed'] = int(ratio.group(1))
                                progress['total'] = int(ratio.group(2))
                            if duration:
                                progress['elapsed_time_numeric'] = duration.group(1)
                                if duration.group(2) and re.fullmatch(r'\d+:\d+(?::\d+)?', duration.group(2)):
                                    progress['remaining_time_numeric'] = duration.group(2)
                            if progress:
                                metadata['recent_progress_numeric'][anchor] = progress
                file_match = re.search(r'File "(/[^"\n]+)", line (\d+)', clean_line)
                if file_match:
                    metadata['traceback_files'].append({'file': file_match.group(1), 'line': int(file_match.group(2))})
                    metadata['traceback_files'] = metadata['traceback_files'][-10:]
                for exception_class in re.findall(r'\b(RuntimeError|ValueError|MemoryError|AssertionError|TimeoutError|OSError|TypeError|KeyError|SystemExit)\b', clean_line):
                    if exception_class not in metadata['exception_classes']:
                        metadata['exception_classes'].append(exception_class)
    result['anchored_log_metadata'][name] = metadata
for label, argv in [('slurm_queue', ['squeue', '-j', '3331', '-h', '-o', '%i|%T|%M|%D|%C|%N|%R']), ('slurm_identity', ['scontrol', 'show', 'job', '3331', '-o'])]:
    p = subprocess.run(argv, text=True, capture_output=True, timeout=20)
    if label == 'slurm_identity':
        fields = {}
        allowed = {'JobId','JobState','Reason','ExitCode','SubmitTime','StartTime','EndTime','RunTime','TimeLimit','NodeList','NumNodes','NumCPUs','NumTasks','CPUs/Task','ReqTRES','AllocTRES','MinMemoryNode','Requeue','BatchHost'}
        for token in p.stdout.split():
            if '=' in token:
                k, v = token.split('=', 1)
                if k in allowed:
                    fields[k] = v
        result[label] = {'exit_code': p.returncode, 'fields': fields, 'stderr': p.stderr.strip()[:200]}
        terminal_states = {'FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
        if p.returncode == 0 and fields.get('JobId') == '3331' and fields.get('JobState') in terminal_states:
            result['positive_terminal_slurm_raw_stdout'] = p.stdout
    else:
        result[label] = {'exit_code': p.returncode, 'stdout': p.stdout.strip()[:1600], 'stderr': p.stderr.strip()[:200]}

patterns = ['output/current-chain-migrated.json', 'output/fresh-initialization.json', 'checkpoints/training_state/attempt-results/attempt-*.json', 'checkpoints/training_state/partial_attempt_*/manifest.json', 'output/fresh-state/outer-*-attempt-*-rank??.json', 'output/trajectories/**/manifest.json', 'output/training-rollout-cost/attempt-*.json', 'output/tree-audit/step_1_rollout_1.json', 'output/tree-audit/step_1_rollout_1_policy_batch.json', 'run-result.json', 'cost.json', 'termination-cost.json', 'output/validation-cost/validation-*.json', 'checkpoints/training_state/global_step_5/manifest.json']
for pattern in patterns:
    paths = sorted(glob.glob(os.path.join(BASE, pattern), recursive=True))
    result['files'][pattern] = {'count': len(paths), 'recent': []}
    result['files'][pattern]['metadata_read_scope'] = 'latest_record_only_except_latest_eight_rank_records'
    for path in paths[-8:] if pattern == 'output/fresh-state/outer-*-attempt-*-rank??.json' else paths[-1:]:
        try:
            with open(path) as f:
                obj = json.load(f)
            original_obj = obj
            if pattern in {'output/tree-audit/step_1_rollout_1.json', 'output/tree-audit/step_1_rollout_1_policy_batch.json'}:
                obj = {k: v for k, v in obj.items() if not isinstance(v, (dict, list)) or k == 'identity'}
            if pattern == 'checkpoints/training_state/partial_attempt_*/manifest.json':
                allowed_partial_keys = {'status', 'identity', 'source', 'config', 'code_commit', 'source_commit', 'config_sha256', 'actual_optimizer_calls_per_rank', 'actual_scheduler_steps_per_rank', 'worker_states_verified', 'continuation_decision', 'credited_outer_candidate', 'opportunity_processing_completed', 'attempt_id', 'attempted_outer', 'completed_outer', 'run_id', 'job_id'}
                obj = {k: v for k, v in obj.items() if k in allowed_partial_keys}
            if pattern == 'run-result.json':
                allowed_result_keys = {'status', 'stage', 'job_id', 'code_commit', 'error_type', 'optimizer_committed', 'scheduler_advanced', 'actual_progress_evidence', 'cleanup_residual', 'original_eight_pid_start_times_preserved', 'final_gpu'}
                obj = {k: v for k, v in obj.items() if k in allowed_result_keys}
            if pattern == 'output/training-rollout-cost/attempt-*.json':
                obj = {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}
            metadata = scalar_metadata(obj)
            if pattern in {'cost.json', 'termination-cost.json'}:
                metadata['native_numeric_fields'] = numeric_tree(original_obj)
            if pattern == 'run-result.json':
                residual = original_obj.get('cleanup_residual')
                if isinstance(residual, dict):
                    metadata['cleanup_residual_shape'] = {k: {'type': type(v).__name__, 'count': len(v)} if isinstance(v, (dict, list)) else {'type': type(v).__name__, 'value': v if isinstance(v, (bool, int, float)) or v is None else None} for k, v in residual.items()}
                else:
                    metadata['cleanup_residual_shape'] = {'type': type(residual).__name__, 'count': len(residual) if isinstance(residual, list) else None}
            if pattern in {'output/tree-audit/step_1_rollout_1.json', 'output/tree-audit/step_1_rollout_1_policy_batch.json'}:
                metadata['top_numeric_counts'] = {k: v for k, v in original_obj.items() if isinstance(v, (bool, int, float)) and not any(term in k.lower() for term in ['token_ids', 'tokenids', 'logps', 'logprob', 'action', 'sample'])}
                if pattern == 'output/tree-audit/step_1_rollout_1_policy_batch.json':
                    costs = original_obj.get('diagnostic_check_cost', {})
                    metadata['diagnostic_check_cost'] = {k: costs.get(k) for k in ['policy_wall_seconds', 'policy_process_cpu_seconds'] if k in costs and (isinstance(costs[k], (int, float)) or costs[k] is None)}
            if pattern == 'output/training-rollout-cost/attempt-*.json':
                if isinstance(original_obj.get('costs'), dict):
                    metadata['costs'] = numeric_tree(original_obj['costs'])
                if isinstance(original_obj.get('question_rows'), (int, float)):
                    metadata['question_rows'] = original_obj['question_rows']
            result['files'][pattern]['recent'].append({'name': os.path.relpath(path, BASE), 'mtime': os.path.getmtime(path), 'metadata': metadata})
        except Exception as e:
            result['files'][pattern]['recent'].append({'name': os.path.relpath(path, BASE), 'read_error_type': type(e).__name__})

for rel in ['resource-samples.jsonl', 'output/resource-samples.jsonl']:
    path = os.path.join(BASE, rel)
    if os.path.isfile(path):
        with open(path) as f:
            lines = f.readlines()
        samples = []
        for line in lines[-2:]:
            try:
                sample = json.loads(line)
                cards = sample.get('gpus', [])
                samples.append({
                    'epoch': sample.get('epoch'),
                    'job_id': sample.get('job_id'),
                    'gpu_count': len(cards),
                    'max_foreign_memory_fraction': max((c.get('foreign_memory_fraction', 0) for c in cards), default=None),
                    'max_foreign_sm_upper': max((c.get('foreign_sm_upper', 0) for c in cards), default=None),
                    'gpu_util_min': min((c.get('total_gpu_util', 0) for c in cards), default=None),
                    'gpu_util_max': max((c.get('total_gpu_util', 0) for c in cards), default=None),
                    'total_used_memory_bytes': sum(c.get('total_used_memory_bytes', 0) for c in cards),
                    'used_memory_bytes_per_card': [c.get('total_used_memory_bytes') for c in cards],
                    'gpu_util_per_card': [c.get('total_gpu_util') for c in cards],
                    'foreign_process_count': sum(len(c.get('foreign', [])) for c in cards),
                })
            except Exception as e:
                samples.append({'read_error_type': type(e).__name__})
        result['files'][rel] = {'line_count': len(lines), 'recent': samples}

# Only these two native JSON names are opened; PT/state bytes are never read.
rolling_root = os.path.join(BASE, 'checkpoints/training_state/rolling-completed')
rolling = {'exists': os.path.isdir(rolling_root), 'directories': [], 'json_metadata': [], 'files': []}
known_hashes = {}
for name in sorted(glob.glob(os.path.join(rolling_root, 'global_step_*/manifest.json'))) + [os.path.join(rolling_root, 'latest.json')]:
    if not os.path.isfile(name) or os.path.islink(name):
        continue
    record = {'name': os.path.relpath(name, BASE), 'bytes': os.path.getsize(name), 'mtime': os.path.getmtime(name)}
    try:
        with open(name, 'rb') as stream:
            contents = stream.read()
        record['sha256'] = hashlib.sha256(contents).hexdigest()
        obj = json.loads(contents)
        record['metadata'] = scalar_metadata(obj)
        for state in obj.get('worker_states', []) + ([obj['driver']] if isinstance(obj.get('driver'), dict) else []):
            if not isinstance(state, dict):
                continue
            path, digest_value = state.get('path'), state.get('sha256')
            if isinstance(path, str) and os.path.abspath(path).startswith(os.path.abspath(rolling_root) + os.sep) and isinstance(digest_value, str) and re.fullmatch('[a-f0-9]{64}', digest_value):
                known_hashes[os.path.abspath(path)] = digest_value
    except (OSError, ValueError, TypeError, KeyError) as error:
        record['read_error_type'] = type(error).__name__
    rolling['json_metadata'].append(record)
for directory, children, names in os.walk(rolling_root):
    children[:] = sorted(n for n in children if not os.path.islink(os.path.join(directory, n)))
    rolling['directories'].append(os.path.relpath(directory, BASE))
    for name in sorted(names):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        stat = os.stat(path)
        rolling['files'].append({'name': os.path.relpath(path, BASE), 'bytes': stat.st_size, 'mtime': stat.st_mtime, 'sha256_from_native_manifest': known_hashes.get(os.path.abspath(path)), 'pt_bytes_read': False})
result['rolling_completed_native_metadata'] = rolling
result['permanent_step5_metadata'] = {'manifest_present': False, 'files': [], 'hf_files': []}
step5_root = os.path.join(BASE, 'checkpoints/training_state/global_step_5')
step5_manifest = os.path.join(step5_root, 'manifest.json')
step5_hashes = {}
hf5_hashes = {}
if os.path.isfile(step5_manifest) and not os.path.islink(step5_manifest):
    try:
        with open(step5_manifest, 'rb') as stream: step5_contents = stream.read()
        step5_obj = json.loads(step5_contents)
        result['permanent_step5_metadata'].update({'manifest_present': True, 'manifest_sha256': hashlib.sha256(step5_contents).hexdigest(), 'manifest_bytes': len(step5_contents), 'manifest_metadata': scalar_metadata(step5_obj)})
        hf5_root = os.path.abspath(os.path.join(BASE,'checkpoints/actor/global_step_5'))
        hf5_identity = step5_obj.get('hf_identity',{})
        hf5_files = hf5_identity.get('files',{}) if isinstance(hf5_identity,dict) else {}
        for hf_name,hf_digest in hf5_files.items() if isinstance(hf5_files,dict) else []:
            if isinstance(hf_name,str) and isinstance(hf_digest,str) and re.fullmatch('[a-f0-9]{64}',hf_digest):
                hf_path = os.path.abspath(os.path.join(hf5_root,hf_name))
                if hf_path.startswith(hf5_root + os.sep): hf5_hashes[hf_path] = hf_digest
        for state in step5_obj.get('worker_states', []) + ([step5_obj['driver']] if isinstance(step5_obj.get('driver'), dict) else []):
            if isinstance(state, dict) and isinstance(state.get('path'), str) and isinstance(state.get('sha256'), str):
                step5_hashes[os.path.abspath(state['path'])] = state['sha256']
    except (OSError, ValueError, TypeError, KeyError) as error:
        result['permanent_step5_metadata']['read_error_type'] = type(error).__name__
for directory, children, names in os.walk(step5_root):
    children[:] = sorted(n for n in children if not os.path.islink(os.path.join(directory,n)))
    for name in sorted(names):
        path = os.path.join(directory,name)
        if os.path.isfile(path) and not os.path.islink(path):
            stat = os.stat(path)
            result['permanent_step5_metadata']['files'].append({'name': os.path.relpath(path,BASE), 'bytes': stat.st_size, 'mtime': stat.st_mtime, 'sha256_from_native_manifest': step5_hashes.get(os.path.abspath(path)), 'asset_bytes_read': False})
for directory, children, names in os.walk(os.path.join(BASE,'checkpoints/actor/global_step_5')):
    children[:] = sorted(n for n in children if not os.path.islink(os.path.join(directory,n)))
    for name in sorted(names):
        path = os.path.join(directory,name)
        if os.path.isfile(path) and not os.path.islink(path):
            stat = os.stat(path)
            result['permanent_step5_metadata']['hf_files'].append({'name': os.path.relpath(path,BASE), 'bytes': stat.st_size, 'mtime': stat.st_mtime, 'sha256_from_native_manifest': hf5_hashes.get(os.path.abspath(path)), 'asset_bytes_read': False})


print(json.dumps(result, sort_keys=True))
