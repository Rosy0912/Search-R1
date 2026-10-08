import datetime
import glob
import json
import os
import re
import shutil
import subprocess

BASE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fresh200-train-3322'
BLOCKED = ('question', 'context', 'token_ids', 'tokenids', 'logprob', 'log_prob', 'logps', 'retrieval', 'prompt', 'response_text', 'weights', 'index')
STRING_KEYS = {'state', 'status', 'stage', 'decision', 'continuation_decision', 'reason', 'job_id', 'run_id', 'attempt_id', 'phase', 'device', 'rank', 'scheduler_type', 'optimizer_type', 'event', 'kind', 'outcome', 'error_type', 'exception_type', 'code_commit', 'source_commit', 'manifest_sha256', 'created_at', 'completed_at', 'timestamp', 'hostname', 'node', 'initialization', 'initialization_mode', 'warm_start', 'checkpoint_loaded', 'cold_start', 'optimizer_state_loaded', 'scope', 'schema', 'schema_version', 'version', 'method', 'generation_id', 'data_id', 'config_id', 'source_id'}

def allowed_string(key):
    return key in STRING_KEYS or key.lower().endswith(('_sha256', '_sha')) or key.lower() == 'sha256'

SAFE_NUMERIC_ARRAY_KEYS = {'actual_optimizer_calls_per_rank', 'actual_completed_optimizer_calls_per_rank', 'total_optimizer_calls_per_rank', 'actual_scheduler_steps_per_rank', 'optimizer_step_values', 'cleanup_residual'}

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

result = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': 3322, 'run_exists': os.path.isdir(BASE), 'files': {}}
protected_pids = {1680782, 1680783, 1680784, 1680785, 1696443, 1696444, 1696445, 1696446}
own_pids = set()
for job_dir in glob.glob('/sys/fs/cgroup/freezer/**/job_3322', recursive=True):
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
        if '/job_3322/' not in membership:
            continue
        with open('/proc/%s/stat' % pid) as f:
            proc_stat = f.read()
        split = proc_stat.rfind(')')
        comm = proc_stat[proc_stat.find('(') + 1:split]
        fields = proc_stat[split + 2:].split()
        own_processes.append({'pid': pid, 'comm': comm, 'state': fields[0], 'utime_jiffies': int(fields[11]), 'stime_jiffies': int(fields[12]), 'starttime_jiffies': int(fields[19])})
    except (OSError, ValueError, IndexError):
        pass
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
        metadata.update({'size_bytes': stat.st_size, 'mtime': stat.st_mtime, 'anchor_counts': {a: 0 for a in anchors}, 'epoch_step_count': 0, 'recent_progress_numeric': {}, 'traceback_files': [], 'exception_classes': []})
        with open(path, errors='replace') as f:
            for line in f:
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
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
for label, argv in [('slurm_queue', ['squeue', '-j', '3322', '-h', '-o', '%i|%T|%M|%D|%C|%N|%R']), ('slurm_identity', ['scontrol', 'show', 'job', '3322', '-o'])]:
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
    else:
        result[label] = {'exit_code': p.returncode, 'stdout': p.stdout.strip()[:1600], 'stderr': p.stderr.strip()[:200]}

patterns = ['output/fresh-initialization.json', 'checkpoints/training_state/attempt-results/attempt-*.json', 'checkpoints/training_state/partial_attempt_*/manifest.json', 'output/fresh-state/outer-*-attempt-*-rank??.json', 'output/trajectories/**/manifest.json', 'output/training-rollout-cost/attempt-*.json', 'output/tree-audit/step_1_rollout_1.json', 'output/tree-audit/step_1_rollout_1_policy_batch.json', 'run-result.json', 'cost.json', 'termination-cost.json']
for pattern in patterns:
    paths = sorted(glob.glob(os.path.join(BASE, pattern), recursive=True))
    result['files'][pattern] = {'count': len(paths), 'recent': []}
    for path in paths[-8:]:
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
print(json.dumps(result, sort_keys=True))
