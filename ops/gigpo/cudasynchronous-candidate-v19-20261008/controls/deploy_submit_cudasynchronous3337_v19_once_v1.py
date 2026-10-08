"""Fresh v19 single-use synchronous CUDA execution; never signals or resumes a failed job."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import zlib

BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
OUT = BASE / 'outputs/gigpo-e1-instruct200'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
RUN = ROOT / 'runs/gigpo-e1-fresh200-live-job3337-1791437756282298127'
STATE = ROOT / 'logs/fresh200-cudasynchronous-repair3337-once-v1'
OLD = ROOT / 'tools/fresh200-continuation-v18-jsonmetadata1'
NEW = ROOT / 'tools/fresh200-continuation-v19-cudasynchronous1'
PARENT_BINDING = ROOT / 'logs/fresh200-jsonmetadata-repair3336-once-v1/BINDING.json'
PARENT_SHA = '006c698e003aea7c629886e9bd1ab7af9628f2fc982508e1aa706e619ded8fa7'
SOURCE = 'ebe91bfebd56c3bd445219dcd94342d815f899f0'
NEW_REPO = ROOT / 'repo-v17-nonblocking1-ebe91bfe'
PYTHON = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/env/bin/python')
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def digest(rows):
    return sha(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())


def checked(path, row):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Missing/nonregular bound file: ' + str(path))
    raw = path.read_bytes()
    require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Bound bytes differ: ' + str(path))
    return raw


def record(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=sha(raw))


def once(path, raw):
    path = Path(path)
    require(path.parent.resolve() == path.parent and not path.is_symlink(), 'Noncanonical once target')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def publication_gate(row, required):
    publication = json.loads(checked(row['path'], row))
    require(publication['status'] == 'PUSHED_AND_INDEPENDENT_GIT_DOWNLOAD_VERIFIED'
            and publication['new_commit'] == publication['remote_sha']
            and publication['verification']['all_file_sha256_and_bytes_match'] is True,
            'Missing actual publication and remote file readback')
    for name, expected in required.items():
        matches = [r for r in publication['manifest'] if r['path'].endswith('/' + name)
                   and r['sha256'] == expected['sha256'] and r['bytes'] == expected['bytes']
                   and r['downloaded_bytes_match'] is True]
        require(len(matches) == 1, 'Unpublished implementation: ' + name)


def gates(path):
    gate = json.loads(Path(path).read_bytes())
    derivation = json.loads(checked(gate['derivation']['path'], gate['derivation']))
    require(derivation['parent_binding']['sha256'] == PARENT_SHA
            and derivation['source_commit'] == SOURCE and derivation['namespace'] == str(STATE),
            'Wrong exact v19 derivation')
    checked(derivation['child_binding']['local_path'], derivation['child_binding'])
    required = {Path(__file__).name: record(Path(__file__)),
        Path(gate['derivation']['path']).name: gate['derivation'],
        Path(derivation['child_binding']['local_path']).name: derivation['child_binding']}
    required.update({row['path']: row['after'] for row in derivation['runtime_manifest']['files']})
    publication_gate(gate['adapter_publication'], required)
    waiver = json.loads(checked(gate['human_review_waiver']['path'], gate['human_review_waiver']))
    policy = derivation['binding_patch']['cuda_execution_review_policy']
    require(policy['status'] == waiver['status'] == 'HUMAN_WAIVED_STATIC_REVIEW'
            and policy['human_instruction'] == waiver['human_instruction'] == '审查啥，你直接跑吧。。别浪费资源，你太慢了'
            and policy['observed_at_utc'] == waiver['observed_at_utc']
            and waiver['binding_sha256'] == derivation['child_binding']['sha256']
            and waiver['derivation_sha256'] == gate['derivation']['sha256']
            and waiver['adapter_sha256'] == {Path(__file__).name: required[Path(__file__).name]['sha256']},
            'Missing exact latest-human review waiver; never fabricate a static PASS')
    publication_gate(gate['adapter_publication'], {Path(gate['human_review_waiver']['path']).name: gate['human_review_waiver']})
    return gate, derivation


def transport(payload):
    """One local invocation; all SSH layers retain the accepted18 route and host checks."""
    action = 'submit'
    sources = {'deploy_submit_cudasynchronous3337_v19_once_v1': Path(__file__).read_bytes()}
    require(len(packed(payload)) < 500000, 'Transport only the reviewed small delta frame')
    encoded = {name: dict(b64=base64.b64encode(zlib.compress(raw)).decode(), sha256=sha(raw))
               for name, raw in sources.items()}
    receiver = """import base64,hashlib,json,sys,types,zlib
sources=json.loads(SOURCES)
for name,row in sources.items():
 raw=zlib.decompress(base64.b64decode(row['b64']))
 if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Source frame changed')
 module=types.ModuleType(name);module.__file__=name+'.py';sys.modules[name]=module
 exec(compile(raw,name+'.py','exec'),module.__dict__)
p=json.loads(sys.stdin.buffer.read())
sys.modules[TARGET].remote(p)
""".replace('SOURCES', repr(json.dumps(encoded))).replace('TARGET', repr(
        'deploy_submit_cudasynchronous3337_v19_once_v1'))
    worker = shlex.join(['env', '-u', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', str(PYTHON), '-B', '-c', receiver])
    management = """import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
if cfg['management_identity_file']!='/home/lichen/.ssh/id_ed25519' or gateway!='10.33.0.18' or ip!='10.33.0.8':raise RuntimeError('Accepted route changed')
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix+['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER],input=sys.stdin.buffer.read(),capture_output=True,timeout=220)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
""".replace('WORKER', repr(worker))
    intent = OUT / ('FRESH200-CUDASYNC-V19-REPAIR3337-' + action.upper() + '-LOCAL-INTENT-v1.json')
    once(intent, packed(dict(status='ONE_LOCAL_INVOCATION_NEVER_REPLAY', action=action,
        payload_sha256=sha(packed(payload)), source_sha256={k: v['sha256'] for k, v in encoded.items()})))
    outer = ['env', '-u', 'LD_LIBRARY_PATH', '/usr/bin/ssh', '-T', '-o', 'BatchMode=yes',
        '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
        '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host']
    try:
        result = subprocess.run(outer + ['env -u LD_LIBRARY_PATH /usr/bin/python3 -B -c ' + shlex.quote(management)],
                                input=packed(payload), capture_output=True, timeout=240)
        observation = dict(returncode=result.returncode, stdout=result.stdout.decode(errors='replace'),
                           stderr=result.stderr.decode(errors='replace'), action=action)
    except BaseException as error:
        observation = dict(status='UNCERTAIN_NEVER_REPLAY', action=action,
                           error_type=type(error).__name__, error=str(error))
    once(OUT / ('FRESH200-CUDASYNC-V19-REPAIR3337-' + action.upper() + '-TRANSPORT-v1.json'), packed(observation))
    print(json.dumps(observation))
    require(observation.get('returncode') == 0, 'Read remote intent/facts; NEVER REPLAY after an uncertain invocation')


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, env=ENV, timeout=20)
    require(result.returncode == 0, 'Unobservable factual command: ' + command[0])
    return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


OPTIONS = ['--parsable', '--partition=gpu', '--nodelist=h100-8', '--nodes=1', '--ntasks=1',
    '--gres=gpu:h100:8', '--cpus-per-task=128', '--mem=512G', '--exclusive',
    '--no-requeue', '--time=1-00:00:00', '--export=ALL']
PATCH_KEYS = {'helpers', 'runtime_files', 'runtime_version', 'runtime_variant',
    'runtime_relative_files_digest', 'runtime_canonical_files_digest', 'declared_primary_files_digest',
    'owned_submit_namespace', 'actual_runtime_static_manifest', 'cuda_execution_review_policy',
    'binding_derivation', 'cuda_synchronous_repair_parent', 'cuda_launch_blocking', 'cuda_execution_scope'}


def actual_terminal(row):
    require(Path(row['path']) == STATE / 'JOB3337-TERMINAL-FACTS-v1.json', 'Wrong exact failed-parent facts')
    facts = json.loads(checked(row['path'], row))
    require(facts['schema'] == 'ACTUAL_ORIGINAL_JOB_TERMINAL_FACTS_V1'
        and facts['job'] == 3337 and facts['node'] == 'h100-8' and facts['run'] == str(RUN)
        and facts['run_result_status'] == 'FAILED' and facts['cleanup_residual_states'] == {}
        and facts['actual_completed_outer_updates'] == 2
        and facts['actual_successful_adam_writes_per_rank'] == 40
        and facts['checkpoint_directories'] == [] and facts['exact_resume_claimed'] is False,
        'Missing truthful two-completed-update failed-parent reconciliation')
    pids = facts['original_pid_observation']; cgroup = facts['original_cgroup_observation']
    require(len(pids) == 9 and len({r['pid'] for r in pids}) == 9
        and all(r['exists'] is False and not (Path('/proc') / str(r['pid'])).exists() for r in pids)
        and cgroup['path'] == '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3337/step_0'
        and cgroup['exists'] is False and not Path(cgroup['path']).exists(), 'Original own PID/cgroup remains or PID reused')
    result_path = RUN / 'run-result.json'
    require(result_path.is_file() and not result_path.is_symlink(), 'Missing original run-result')
    terminal = json.loads(result_path.read_bytes())
    require(terminal['status'] == 'FAILED' and terminal['job_id'] == '3337'
        and terminal['source_commit'] == SOURCE and terminal['binding_sha256'] == PARENT_SHA
        and terminal['retained_training_updates_contributed'] == 2
        and terminal['cleanup_residual_states'] == {} and terminal['final_gpu']['processes'] == [],
        'Original failed run-result or actual cleanup differs')
    attempt_path = RUN / 'audit/partial-updates/attempt_000002/attempt-result.json'
    require(attempt_path.is_file() and not attempt_path.is_symlink(), 'Missing actual historical second-attempt receipt')
    actual = json.loads(attempt_path.read_bytes())['attempt']
    require(actual['counted_complete'] is True and actual['attempt_id'] == 2 and actual['completed_before'] == 1
        and actual['actual_successful_adam_writes'] == 20 and actual['actual_adam_total'] == 40
        and len(actual['rank_receipts']) == 8 and {r['rank'] for r in actual['rank_receipts']} == set(range(8))
        and all(r['counted_complete'] is True and r['actual_successful_adam_writes'] == 20
            and r['actual_adam_total'] == 40 and r['scheduler_step'] == 2
            and r['skipped_opportunities'] == 0 for r in actual['rank_receipts']),
        'Historical actual second-update eight-rank receipts differ')
    saved = facts['actual_slurm_raw']
    require(saved['returncode'] == 0 and re.search(r'\bJobId=3337\b', saved['stdout'])
        and re.search(r'\bJobState=FAILED\b', saved['stdout']) and re.search(r'\bExitCode=1:0\b', saved['stdout'])
        and PARENT_SHA in saved['stdout'], 'Missing preserved actual Slurm failure; expired live history is not zero')
    gpu = query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    require(not gpu['stdout'].strip(), 'GPU compute remains; do not affect other jobs')
    queue = query(['squeue', '-h', '--all', '-o', '%i|%j|%T|%R'])
    for line in queue['stdout'].splitlines():
        fields = line.split('|')
        require(len(fields) == 4 and not fields[1].startswith('gigpo-e1-fresh200-live'),
            'Own old/new training remains; never replay')
    return facts, gpu, queue


def remote(payload):
    require(socket.gethostname().split('.')[0] == 'h100-8' and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        'Wrong CPU-only once-submit receiver')
    require(not STATE.exists() and not STATE.is_symlink() and STATE.parent.resolve() == STATE.parent,
        'New v19 intent namespace already exists; never replay')
    STATE.mkdir()
    once(STATE / 'SUBMIT-INTENT.json', packed(dict(status='CLAIMED_NEVER_REPLAY',
        binding=payload['binding'], parent_terminal_facts=payload['parent_terminal_facts'],
        human_review_waiver=payload['human_review_waiver'], adapter_publication=payload['adapter_publication'],
        signal_sent=False, extra_validation_job=False, new_fresh_counter_initial=0)))
    for row in payload['evidence_staging']:
        path = Path(row['path'])
        require(path.parent == STATE and path.name != 'BINDING.json', 'Evidence escaped own new namespace')
        data = zlib.decompress(base64.b64decode(row['b64']))
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Evidence frame differs')
        once(path, data)
    waiver = json.loads(checked(payload['human_review_waiver']['path'], payload['human_review_waiver']))
    require(waiver['status'] == 'HUMAN_WAIVED_STATIC_REVIEW'
        and waiver['human_instruction'] == '审查啥，你直接跑吧。。别浪费资源，你太慢了'
        and waiver['binding_sha256'] == payload['binding']['sha256'], 'Latest exact human waiver differs')
    facts, gpu, queue = actual_terminal(payload['parent_terminal_facts'])
    parent = json.loads(checked(PARENT_BINDING, payload['parent_binding']))
    require(parent['code_commit'] == SOURCE and parent['helpers'] == str(OLD)
        and len(parent['files']) == 39961 and len(parent['runtime_files']) == 44,
        'Wrong immutable complete parent binding')
    require(set(payload['patch']) == PATCH_KEYS, 'Unexpected v19 binding mutation')
    child = dict(parent); child.update(payload['patch'])
    new_rows = {r['path']: r for r in child['runtime_files']}
    require(len(new_rows) == 44 and child['helpers'] == str(NEW) and child['runtime_version'] == 19
        and child['owned_submit_namespace'] == str(STATE)
        and child['cuda_synchronous_repair_parent']['terminal_facts'] == payload['parent_terminal_facts'],
        'Wrong isolated v19 runtime/facts')
    replaced = 0; rows = []
    for row in parent['files']:
        if row['path'].startswith(str(OLD) + '/'):
            require(Path(row['path']).parent == OLD, 'Unexpected runtime child path')
            rows.append(new_rows[str(NEW / Path(row['path']).name)]); replaced += 1
        else:
            rows.append(row)
    require(replaced == 44, 'All44 and only44 runtime substitutions required')
    child['files'] = rows
    child['evidence'] = parent['evidence'] + payload['additional_evidence']
    require(child['source_files'] == parent['source_files'] and child['code_commit'] == child['source_commit'] == SOURCE
        and child['repo'] == str(NEW_REPO) and child['official_training_outer_target'] == 200
        and child['human_nonblocking_diagnostics'] is True
        and child['external_diagnostic_review_blocks_training'] is False
        and child['allow_bf16_reduced_precision_reduction'] is False
        and child['cuda_launch_blocking'] == '1'
        and child['cuda_execution_scope'] == 'CUDA_SYNCHRONOUS_FORMAL_TRAINING_UNPROVEN_REPAIR_V1'
        and child['cuda_execution_review_policy']['status'] == 'HUMAN_WAIVED_STATIC_REVIEW'
        and child['cuda_execution_review_policy']['observed_at_utc'] == waiver['observed_at_utc']
        and child['initial_baseline_policy'] == parent['initial_baseline_policy'], 'Scientific/source/nonblocking policy changed')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
        'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance', 'negative_tail_limit'):
        require(child[key] is parent[key] is None, 'Unknown tolerances/tail must remain null')
    raw = packed(child)
    require(len(raw) == payload['binding']['bytes'] and sha(raw) == payload['binding']['sha256'],
        'Remote small-frame derivation differs from reviewed child binding')
    require(not NEW.exists() and not NEW.is_symlink() and NEW.parent.resolve() == NEW.parent, 'Runtime exists; never overwrite')
    NEW.mkdir()
    require(payload['changed_files'] == ['calibration_driver_v1.py', 'calibration_run_v1.py', 'calibration_worker_v1.py'] and len(payload['runtime']) == 44
        and {r['name'] for r in payload['runtime']} == {Path(k).name for k in new_rows}, 'Beyond frozen three-file execution delta')
    for row in payload['runtime']:
        name = row['name']; expected = new_rows[str(NEW / name)]
        require(Path(name).name == name and row['bytes'] == expected['bytes'] and row['sha256'] == expected['sha256'], 'Runtime frame escaped')
        if name in payload['changed_files']:
            data = zlib.decompress(base64.b64decode(row['b64']))
        else:
            require('b64' not in row, 'Only three frozen execution files receive new bytes')
            data = checked(OLD / name, row)
        require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Runtime bytes differ')
        once(NEW / name, data)
    require(subprocess.check_output(['git', '-C', str(NEW_REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
        and not subprocess.check_output(['git', '-C', str(NEW_REPO), 'status', '--porcelain'], text=True).strip(), 'Existing immutable source changed')
    storage = payload['shared_storage']
    require(storage['client_node'] == 'h100-10' and storage['mount_type'] == 'nfs4'
        and storage['source'] == '10.32.0.8:/mnt/data'
        and storage['binding_sha256'] == '2c161a96fd4f1313e1c5de935a98319bf85b9247b4c7632c73afac282ed500ec',
        'Missing retained shared-output visibility proof')
    once(STATE / 'BINDING.json', raw)
    actual_terminal(payload['parent_terminal_facts'])
    command = ['sbatch'] + OPTIONS + [str(NEW / 'calibration_v1.sbatch'), str(STATE / 'BINDING.json'),
        payload['binding']['sha256'], child['repo'], str(NEW), child['environment_root']]
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=ENV, timeout=60)
        job = result.stdout.strip()
        receipt = dict(status='SUBMITTED_EXACTLY_ONCE' if result.returncode == 0 and job.isdecimal() else 'UNCERTAIN_NEVER_REPLAY',
            command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
            job_id=job if result.returncode == 0 and job.isdecimal() else None, binding=payload['binding'],
            parent_terminal_facts=payload['parent_terminal_facts'], historical_parent_completed_outer=2,
            historical_parent_adam_writes_per_rank=40, historical_updates_stitched=False,
            new_fresh_initial_outer=0, new_fresh_initial_adam=0, signal_sent=False,
            training_success_not_yet_observed=True, extra_validation_job=False,
            queued_submission_is_not_gpu_admission=True, shared_storage=storage)
    except BaseException as error:
        receipt = dict(status='UNCERTAIN_NEVER_REPLAY', error_type=type(error).__name__, error=str(error),
            binding=payload['binding'], parent_terminal_facts=payload['parent_terminal_facts'])
    once(STATE / 'SUBMIT-RECEIPT.json', packed(receipt)); print(json.dumps(receipt))
    require(receipt['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Read own durable intent/queue; never replay')


def main():
    require(len(sys.argv) == 2, 'Supply exact human-waiver/publication gate file')
    gate, derivation = gates(sys.argv[1])
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
        staging.append(dict(path=row['path'], bytes=len(data), sha256=sha(data), b64=base64.b64encode(zlib.compress(data)).decode()))
    waiver_row = gate['human_review_waiver']
    waiver_data = checked(waiver_row['path'], waiver_row)
    remote_waiver = dict(path=str(STATE / Path(waiver_row['path']).name), bytes=len(waiver_data), sha256=sha(waiver_data))
    staging.append(dict(remote_waiver, b64=base64.b64encode(zlib.compress(waiver_data)).decode()))
    payload = dict(parent_binding=derivation['parent_binding'], binding=derivation['child_binding'],
        patch=derivation['binding_patch'], additional_evidence=derivation['additional_evidence'],
        evidence_staging=staging, parent_terminal_facts=derivation['parent_terminal_facts'],
        runtime=runtime, changed_files=derivation['changed_files'], shared_storage=derivation['shared_storage'],
        human_review_waiver=remote_waiver, adapter_publication=gate['adapter_publication'])
    transport(payload)


if __name__ == '__main__':
    main()
