"""Local saved metadata/log review; no remote calls, Torch, PT/HF or scheduler IO."""
import ast
import base64
import copy
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/tree-instruct-300'
HERE = Path(__file__).resolve().parent
PKG = ROOT / 'work/probability-probe-49a6b35-29822f9-r2'
checks = []

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def read(path):
    return json.loads(path.read_bytes())

raw_path = OUT / 'PPO_PROBE_GPU3280_PROGRESS4_RAW.json'
meta_path = OUT / 'PPO_PROBE_GPU3280_FAILURE_METADATA_RAW.json'
summary_path = OUT / 'PPO_PROBE_GPU3280_FAILURE_SUMMARY.json'
check('exact progress4 capture', sha(raw_path.read_bytes()) ==
      '72d06a5afe53f3f859c4cc4b3ee9a377722cc1043090a950677e58de26499c99')
check('exact failure metadata', sha(meta_path.read_bytes()) ==
      '33b4f9dd366e571720db00752518503bb297999a49c535dc2867a8186348419a')
raw, meta, summary = read(raw_path), read(meta_path), read(summary_path)
saved_files = {}
for remote, entry in raw['files'].items():
    data = gzip.decompress(base64.b64decode(entry['gzip_base64']))
    local = OUT / 'gpu3280' / Path(remote).name
    check('raw length ' + local.name, len(data) == entry['bytes'])
    check('raw SHA ' + local.name, sha(data) == entry['sha256'])
    check('local bytes ' + local.name, local.read_bytes() == data)
    saved_files[local.name] = {'sha256': sha(data), 'bytes': len(data)}
check('all11 metadata files and no tensor assets', len(saved_files) == 11 and
      all(not name.endswith(('.pt', '.safetensors')) for name in saved_files))
r = read(OUT / 'gpu3280/result.json')
launch = read(OUT / 'gpu3280/launch.json')
initial = read(OUT / 'gpu3280/initial-allocation.json')
termination = read(OUT / 'gpu3280/termination-cost.json')
bundle = read(OUT / 'gpu3280/bundle.json')
manifest = read(PKG / 'package-manifest.json')
package_sha = sha((PKG / 'package-manifest.json').read_bytes())
release_path = ROOT / 'work/tree-probability-probe-owned/pre-dispatch-r2/RELEASE.json'
release_sha = sha(release_path.read_bytes())
check('exact released R2 manifest', package_sha ==
      '0a1a39ec5d9f86498d8e66a9f9291aefffba02f88776e2fa1aaeb200085e7dc8')
check('exact R2 source-only release', release_sha ==
      'bb4558fcc00ec7d0a3c3e9dc1d5c8db4ef70e8a3df998ae9bf685f47423cc66d')
for record in (r, launch, initial, raw['journal']['identity']):
    check('package identity ' + str(len(checks)), record['package_manifest_sha256'] == package_sha)
    check('release identity ' + str(len(checks)), record['release_sha256'] == release_sha)
check('launch exact741 source hashes', launch['source_manifest_files'] == manifest['files'] and
      len(manifest['files']) == 741)
check('launch actual bundle SHA', launch['bundle_sha256'] == saved_files['bundle.json']['sha256'])
check('bundle envelope release', bundle['release']['sha256'] == release_sha)
tree = ast.parse((PKG / 'probe_package.py').read_bytes())
names = {'require', 'canonical', 'digest', 'bundle_payload', 'runtime_output', 'guard_record'}
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                    type_ignores=[])
check('pure exact six package functions selected', len(module.body) == 6)
namespace = {'json': json, 'hashlib': hashlib, 'copy': copy, 'math': math, 're': re,
             'ROOT': Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')}
token_assign = next(n for n in tree.body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'OUTPUT_TOKEN' for t in n.targets))
namespace['OUTPUT_TOKEN'] = ast.literal_eval(token_assign.value)
exec(compile(ast.fix_missing_locations(module), str(PKG / 'probe_package.py'), 'exec'), namespace)
check('actual dynamic bundle normalized identity', namespace['digest'](
      namespace['bundle_payload'](bundle, '3280')) == manifest['bundle_payload_sha256'] ==
      launch['bundle_payload_sha256'] == raw['journal']['identity']['bundle_payload_sha256'])
check('rank plan ready count0', len(raw['rank_summaries']) == 8 and
      {x['rank'] for x in raw['rank_summaries']} == set(range(8)) and
      all(x['ready'] is False and x['json_files'] == 0 and x['comparisons'] == []
          for x in raw['rank_summaries']))
check('failure not a numeric receipt', r['status'] == 'FAILED_OR_UNKNOWN_ZERO_UPDATE_GPU_PROBE' and
      r['actual_optimizer_writes'] is None and r['planned_optimizer_writes'] == 0 and
      all(r[k] is False for k in ('actual_training_accepted', 'probability_consistency_accepted',
                                  'gpu_training_capacity_proven')))
check('once intent is used no automatic retry', raw['journal']['job_id'] == '3280' and
      raw['journal']['automatic_retry_allowed'] is False and
      raw['journal']['actual_gpu_probe_accepted'] is False)
check('poll did not load tensors', raw['tensor_files_loaded'] is False and
      raw['optimizer_writes_by_this_poll'] == 0 and meta['gpu_or_tensor_loaded_by_reader'] is False)
for cmd, x in raw['commands'].items():
    check('purged current scheduler ' + cmd, x['rc'] == 1 and x['stdout'] == '' and
          x['stderr'] == 'slurm_load_jobs error: Invalid job id specified\n')
check('accounting unavailable', next(v for k, v in meta['commands'].items() if k.startswith('sacct ')) ==
      {'rc': 1, 'stdout': '', 'stderr': 'Slurm accounting storage is disabled\n'})
check('body snapshot is running not terminal', 'JobState=RUNNING ' in r['slurm_at_supervisor_end'] and
      'Requeue=0 Restarts=0 ' in r['slurm_at_supervisor_end'])
for text in (initial['allocation_text'], launch['allocation_text'], r['slurm_at_supervisor_end']):
    check('same-node exclusive original allocation ' + str(len(checks)),
          all(value in text for value in ('NodeList=h100-9', 'NumNodes=1 NumCPUs=192 NumTasks=1',
              'CPUs/Task=32', 'mem=256G', 'OverSubscribe=NO', 'gres:gpu:8', 'TimeLimit=00:30:00')))
protected = {str(x): 1502352181 for x in range(1680782, 1680786)}
protected.update({str(x): 1502398762 for x in range(1696443, 1696447)})
check('original8 saved start ticks', launch['background_pid_start_ticks'] ==
      initial['background_pid_start_ticks'] == protected)
check('own scope and loopback port', launch['cgroup_relative'] ==
      '/slurm_h100-9/uid_0/job_3280/step_0' and launch['torchrun_loopback_port'] == manifest['torchrun_port'])
check('eight distinct inherited lease FDs', len(launch['leases']) == 8 and
      {x['uuid'] for x in launch['leases']} == set(launch['gpu_uuids']) and
      len({x['fd'] for x in launch['leases']}) == 8)
check('cleanup result and original identity', r['cleanup_residual'] == {} and
      r['original_background_pid_start_ticks_preserved'] is True)
check('final GPU processes equal initial background', r['final_gpu']['processes'] ==
      initial['initial_gpu']['processes'] and len(r['final_gpu']['processes']) == 8 and
      {line.split(', ')[1] for line in r['final_gpu']['processes']} == set(protected))
check('final card snapshot restored exactly', r['final_gpu']['gpus'] == initial['initial_gpu']['gpus'])
resources = [json.loads(x) for x in (OUT / 'gpu3280/resources.jsonl').read_text().splitlines()]
processes = [json.loads(x) for x in (OUT / 'gpu3280/process-resources.jsonl').read_text().splitlines()]
for row in resources:
    namespace['guard_record'](row, '3280', launch['gpu_uuids'], row['epoch'])
check('all34 saved historical guard rows exact dual15', len(resources) == 34)
check('guard times monotonically recorded', all(a['epoch'] < b['epoch'] for a, b in zip(resources, resources[1:])))
for row in processes:
    a = row['adapter']
    check('saved adapter row ' + str(len(checks)), a['job_id'] == '3280' and
          a['phase'] == 'supervisor_live' and a['original_background_processes_preserved'] is True and
          0 <= row['epoch'] - a['guard_epoch'] <= 15)
check('153 saved process sample rows', len(processes) == 153)
check('original requested32 actual192 and8GPU',
      r['requested_cpus'] == initial['requested_cpus'] == 32 and
      r['allocated_cpus'] == initial['allocated_cpus'] == 192 and
      r['allocated_gpus'] == initial['allocated_gpus'] == 8)
duration = r['end_epoch'] - r['start_epoch']
check('supervisor interval172.370s', 0 < duration < 180 and abs(duration - r['wall_seconds']) < 1e-5)
check('GPU8 arithmetic only supervisor scope', abs(r['gpu_hours'] * 3600 / 8 - duration) < 1e-5)
check('CPU192 arithmetic allocation not consumption', abs(r['allocated_cpu_hours'] * 3600 / 192 - duration) < 1e-5)
check('beforecleanup termination order', termination['start_epoch'] == r['start_epoch'] and
      r['start_epoch'] < termination['termination_epoch'] < r['end_epoch'])
check('summary exact binding and no acceptance', summary['progress_raw_sha256'] == sha(raw_path.read_bytes()) and
      summary['failure_metadata_sha256'] == sha(meta_path.read_bytes()) and
      summary['scheduler_terminal_state'] == 'UNKNOWN_PURGED_ACCOUNTING_DISABLED' and
      summary['actual_optimizer_writes_reported'] is None and
      summary['allocation_cost_is_consumed_cpu_time'] is False and
      summary['probability_consistency_accepted'] is False and summary['training_accepted'] is False)
headers = meta['headers']
check('saved header metadata all4 original node files', len(headers) == 4 and meta['host'] == 'h100-9' and
      all(x['exists'] is True and x['is_symlink'] is False and
          x['resolved'].startswith('/dataset/shaoshitong/search-r1-rl/tree-grpo/') for x in headers.values()))
header_acceptance = headers['logs/python-headers-acceptance.json']
check('headers acceptance text byte SHA', len(header_acceptance['text'].encode()) == header_acceptance['bytes'] and
      sha(header_acceptance['text'].encode()) == header_acceptance['sha256'] ==
      '83370e78f62f65d1d42c257b4f1122e1ad568fbdb3494bd431cf8bddadef2dc7')
header_json = json.loads(header_acceptance['text'])
check('headers metadata ownroot extract no system install', header_json['package_sha256_verified'] is True and
      header_json['scope'] == 'extract to ownroot only; no dpkg install/system change; only headers used')
old_source = ROOT / 'work/tree-safety110-runtime/supervisor.py'
old_ast = ast.parse(old_source.read_bytes())
include_expr = [n.value for n in ast.walk(old_ast) if isinstance(n, ast.keyword) and n.arg == 'C_INCLUDE_PATH']
check('original training one explicit header environment', len(include_expr) == 1)
original_include = eval(compile(ast.Expression(include_expr[0]), str(old_source), 'eval'),
                        {'ROOT': Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')})
fixture = meta['actual_same_node_own_cpu_only_compiler_fixture']
check('actual saved same-node CPU compiler fixture', fixture['C_INCLUDE_PATH'] == original_include and
      fixture['command'] == ['/usr/bin/gcc', '-x', 'c', '-fsyntax-only', '-'] and fixture['rc'] == 0 and
      fixture['stdout'] == fixture['stderr'] == '' and fixture['no_output_file_or_gpu_or_training'] is True and
      fixture['code'].startswith('#include <Python.h>\n'))
supervisor_ast = ast.parse((PKG / 'supervisor.py').read_bytes())
check('R2 lacks original explicit include environment', not any(isinstance(n, ast.keyword) and
      n.arg == 'C_INCLUDE_PATH' for n in ast.walk(supervisor_ast)))
child = HERE / 'failure-child/FAILURE_CHILD_RECEIPT.json'
check('exact independent child receipt', sha(child.read_bytes()) ==
      '2897f46f6b8ab6890e84e0ab9d87375d2584b308a17b02b88dbf0e3e7fc608b1')
child_data = read(child)
check('child preserves failure null and unknown', child_data['actual_optimizer_writes'] is None and
      child_data['slurm_terminal_state'] == 'UNKNOWN' and child_data['actual_gpu_probe_accepted'] is False and
      child_data['successful_rank_ready'] == child_data['successful_numeric_comparisons'] == 0)
proof = {
    'status': 'SAVED_ACTUAL_FAILURE_METADATA_CHECKS_PASSED_NO_NUMERIC_ACCEPTANCE',
    'checks_passed': len(checks), 'checks': checks, 'files': saved_files,
    'raw_sha256': sha(raw_path.read_bytes()), 'metadata_sha256': sha(meta_path.read_bytes()),
    'summary_sha256': sha(summary_path.read_bytes()), 'failure_child_sha256': sha(child.read_bytes()),
    'source_header_reference': {'path': str(old_source.relative_to(ROOT)), 'sha256': sha(old_source.read_bytes())},
    'guard_stats': {'rows': len(resources), 'first_epoch': resources[0]['epoch'], 'last_epoch': resources[-1]['epoch'],
        'max_sample_gap_seconds': max(b['epoch'] - a['epoch'] for a, b in zip(resources, resources[1:])),
        'max_foreign_memory_fraction': max(g['foreign_memory_fraction'] for x in resources for g in x['gpus']),
        'max_foreign_sm_upper': max(g['foreign_sm_upper'] for x in resources for g in x['gpus']),
        'whole_card_5second_sample_peak_bytes': max(g['total_used_memory_bytes'] for x in resources for g in x['gpus']),
        'training_capacity_proven': False},
    'process_sample_rows': len(processes),
    'cost': {'supervisor_wall_seconds': r['wall_seconds'], 'supervisor_gpu_hours': r['gpu_hours'],
        'supervisor_allocated_cpu_hours': r['allocated_cpu_hours'], 'allocated_cpus': 192, 'requested_cpus': 32,
        'before_cleanup_wall_seconds': termination['termination_epoch'] - r['start_epoch'],
        'cleanup_interval_seconds': r['end_epoch'] - termination['termination_epoch'],
        'consumed_cpu_time': None, 'complete_scheduler_allocation_cost': None,
        'scope': 'Supervisor interval only; allocation hours are not actual process CPU consumption'},
    'actual_optimizer_writes': None, 'slurm_terminal_state': 'UNKNOWN_PURGED_ACCOUNTING_DISABLED',
    'remote_calls': 0, 'actual_gpu_calls': 0, 'scheduler_calls': 0, 'pt_hf_model_reads': 0,
    'scope': 'Local independently checked saved metadata/log bytes and pure source functions; no live node recheck',
}
(HERE / 'SAVED_FAILURE_METADATA_PROOF.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
print(json.dumps({'checks_passed': len(checks), 'proof_sha256': sha((HERE / 'SAVED_FAILURE_METADATA_PROOF.json').read_bytes())}))
