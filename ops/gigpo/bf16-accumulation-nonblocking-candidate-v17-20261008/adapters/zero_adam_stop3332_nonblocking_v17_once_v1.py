"""Single-use draft: signal only the exact own3332 launcher; retain every original run file."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import zlib

BASE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
OUT = BASE / 'outputs/gigpo-e1-instruct200'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
RUN = ROOT / 'runs/gigpo-e1-fresh200-live-job3332-1791429907029168081'
STATE = ROOT / 'logs/fresh200-nonblocking-v17-repair3332-once-v1'
OLD = ROOT / 'tools/fresh200-continuation-v14-evidenceclone1'
NEW = ROOT / 'tools/fresh200-continuation-v17-bf16accum1-nonblocking1'
PARENT_BINDING = ROOT / 'logs/fresh200-evidence-repair3325-once-v1/BINDING.json'
PARENT_SHA = '2c161a96fd4f1313e1c5de935a98319bf85b9247b4c7632c73afac282ed500ec'
PARENT_SOURCE = '2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963'
SOURCE = 'ebe91bfebd56c3bd445219dcd94342d815f899f0'
OLD_REPO = ROOT / 'repo-v11-2ddbb0a'
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


def gates(path, action):
    gate = json.loads(Path(path).read_bytes())
    derivation = json.loads(checked(gate['derivation']['path'], gate['derivation']))
    require(derivation['parent_binding']['sha256'] == PARENT_SHA
            and derivation['source_commit'] == SOURCE
            and derivation['namespace'] == str(STATE), 'Wrong exact3332 derivation')
    required = {name: record(BASE / 'work' / name) for name in
                ('zero_adam_stop3332_nonblocking_v17_once_v1.py', 'deploy_submit_nonblocking3332_v17_once_v1.py')}
    required[Path(gate['derivation']['path']).name] = gate['derivation']
    required[Path(derivation['child_binding']['local_path']).name] = dict(
        bytes=derivation['child_binding']['bytes'], sha256=derivation['child_binding']['sha256'])
    checked(derivation['child_binding']['local_path'], derivation['child_binding'])
    required[Path(derivation['source_bundle']['local_path']).name] = derivation['source_bundle']
    publication_gate(gate['adapter_publication'], required)
    runtime_required = {row['path']: row['after'] for row in derivation['runtime_manifest']['files']}
    publication_gate(gate['adapter_publication'], runtime_required)
    publication_gate(gate['source_publication'], {Path(r['path']).name: r['after'] for r in derivation['source_changed_files']})
    review = json.loads(checked(gate['adapter_review']['path'], gate['adapter_review']))
    require(review['status'] == 'PASS_STATIC' and review['blocking_findings'] == []
            and review['reviewer']['candidate_author'] is False and review['reviewer']['clean_context'] is True
            and review['binding_sha256'] == derivation['child_binding']['sha256']
            and review['derivation_sha256'] == gate['derivation']['sha256']
            and review['adapter_sha256'] == {k: v['sha256'] for k, v in required.items() if k.endswith('.py')},
            'Missing exact nonauthor adapter/binding static review')
    require(action in ('stop', 'submit'), 'Unknown action')
    return gate, derivation


def transport(payload, action, stop_source, submit_source=None):
    """One local invocation; all SSH layers retain the accepted18 route and host checks."""
    sources = {'zero_adam_stop3332_nonblocking_v17_once_v1': stop_source}
    require(len(packed(payload)) < 500000, 'Transport only the reviewed small delta frame')
    if submit_source is not None:
        sources['deploy_submit_nonblocking3332_v17_once_v1'] = submit_source
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
        'zero_adam_stop3332_nonblocking_v17_once_v1' if action == 'stop' else 'deploy_submit_nonblocking3332_v17_once_v1'))
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
    intent = OUT / ('FRESH200-NONBLOCKING-V17-REPAIR3332-' + action.upper() + '-LOCAL-INTENT-v1.json')
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
    once(OUT / ('FRESH200-NONBLOCKING-V17-REPAIR3332-' + action.upper() + '-TRANSPORT-v1.json'), packed(observation))
    print(json.dumps(observation))
    require(observation.get('returncode') == 0, 'Read remote intent/facts; NEVER REPLAY after an uncertain invocation')


def proc(pid):
    directory = Path('/proc') / str(pid)
    stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    env = dict(item.split('=', 1) for item in (directory / 'environ').read_bytes().decode().split('\0') if '=' in item)
    argv = [a for a in (directory / 'cmdline').read_bytes().decode().split('\0') if a]
    groups = []
    for line in (directory / 'cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if (controllers == '' or 'freezer' in controllers.split(',')) and re.search(r'(^|/)job_3332(/|$)', relative):
            base = Path('/sys/fs/cgroup') if controllers == '' else Path('/sys/fs/cgroup/freezer')
            groups.append(base / relative.lstrip('/'))
    require(len(groups) == 1 and env['SLURM_JOB_ID'] == '3332', 'Process outside exact own3332')
    return dict(pid=pid, start_ticks=stat[19], ppid=int(stat[1]), command=argv,
                cgroup=str(groups[0]), cwd=str((directory / 'cwd').resolve()), env=env)


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, env=ENV, timeout=20)
    require(result.returncode == 0, 'Unobservable factual command: ' + command[0])
    return dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def no_updates():
    require(not (RUN / 'probability-identity-approval.json').exists()
            and not (RUN / 'probability-identity-approval.json').is_symlink()
            and not (RUN / 'probability-identity-admitted.json').exists(), 'Probability approval exists')
    require(not list(RUN.glob('calibration-attempt*.json')) and not (RUN / 'driver-result.json').exists(),
            'Actual update/driver result exists')
    if (RUN / 'live-progress.json').exists():
        progress = json.loads((RUN / 'live-progress.json').read_bytes())
        require(progress['completed_outer_updates'] == progress['actual_adam_total'] == 0,
                'Live progress is no longer zero Adam/completed0')


def remote(payload):
    require(socket.gethostname().split('.')[0] == 'h100-8'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Wrong CPU-only receiver')
    require(not STATE.is_symlink() and STATE.parent.resolve() == STATE.parent, 'Unsafe namespace')
    STATE.mkdir(exist_ok=False)
    once(STATE / 'STOP-INTENT.json', packed(dict(status='CLAIMED_NEVER_REPLAY', payload=payload)))
    expected = payload['parent']
    checked(PARENT_BINDING, expected['binding'])
    scheduler_before = query(['scontrol', 'show', 'job', '3332', '-o'])
    require(re.search(r'\bJobId=3332\b', scheduler_before['stdout'])
            and re.search(r'\bJobState=RUNNING\b', scheduler_before['stdout'])
            and 'NodeList=h100-8 ' in scheduler_before['stdout'], 'Own3332 is not currently allocated on8')
    ready_raw = checked(RUN / 'probability-identity-ready.json', expected['ready'])
    ready = json.loads(ready_raw)
    require(ready['status'] == 'WAITING_BEFORE_FIRST_ADAM' and ready['identity'] == expected['identity']
            and ready['actual_adam_writes'] == ready['scheduler'] == ready['retained_training_updates_contributed'] == 0
            and ready['data_cursor'] == 512, 'Not the bound live Adam0 waiting gate')
    no_updates()
    workers = ready['identity']['post_sampling_worker_identities']
    identities = [dict(pid=expected['driver_pid'], start_ticks=expected['driver_start_ticks'])] + workers
    processes = []
    for identity in identities:
        live = proc(identity['pid'])
        require(live['start_ticks'] == identity['start_ticks']
                and live['env']['SEARCH_R1_BASELINE_OUTPUT'] == str(RUN)
                and live['env']['SEARCH_R1_BASELINE_COMMIT'] == PARENT_SOURCE
                and live['env']['GIGPO_POLICY_NATIVE_BINDING_SHA256'] == PARENT_SHA,
                'PID/start/run/source/binding identity changed')
        processes.append({k: v for k, v in live.items() if k != 'env'})
    parent = proc(expected['driver_pid'])
    executor = None
    for _ in range(20):
        parent = proc(parent['ppid'])
        if parent['command'][1:] == [str(OLD / 'calibration_run_v1.py'), str(PARENT_BINDING), PARENT_SHA]:
            executor = parent
            break
    require(executor is not None and executor['cwd'] == str(OLD_REPO)
            and Path(executor['command'][0]).resolve() == PYTHON.resolve(), 'Exact ancestor launcher not found')
    cgroup = Path(executor['cgroup'])
    require(all(Path(row['cgroup']) == cgroup or Path(row['cgroup']).is_relative_to(cgroup) for row in processes),
            'Original Ray processes outside own executor cgroup')
    fd = os.pidfd_open(executor['pid'])
    try:
        # Recompute the saved full typed pause on CPU; no Ray/model RPC, forward or restore.
        pause_row = ready['post_sampling_pause']['raw']
        pause_path = Path(pause_row['path'])
        require(pause_path == RUN / 'post-sampling-pause-state.pt', 'Wrong typed pause path')
        checked(pause_path, pause_row)
        state_helper = expected['state_helper']
        checked(OLD / 'calibration_state_v1.py', state_helper)
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location('original3332_typed_state', OLD / 'calibration_state_v1.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pause = torch.load(pause_path, map_location='cpu', weights_only=False)
        require([row['identity'] for row in pause['workers']] == workers
                and module.state_digest(pause['driver']) == ready['identity']['post_sampling_driver_complete_state_sha256']
                and pause['driver']['pid'] == expected['driver_pid']
                and pause['driver']['actual_outer_updates'] == 0 and pause['driver']['sampler']['cursor'] == 512,
                'Saved typed driver/workers differ')
        for row in pause['workers']:
            identity = dict(row['identity'])
            declared = identity.pop('complete_state_sha256')
            require(module.state_digest(dict(identity=identity, metadata=row['metadata'])) == declared
                    and identity['scheduler_step'] == 0 and identity['learning_rates'] == [0.0]
                    and identity['optimizer_tensors_sha256'] == sha(b''), 'Typed actor is not exact Adam0')
        material = json.loads(checked(RUN / 'probability-material-manifest.json', expected['material']))
        preserved = []
        for row in material['files']:
            path = Path(row['path'])
            require(path.is_relative_to(RUN), 'Evidence escaped original run')
            checked(path, row)
            preserved.append(row)
        require(proc(executor['pid'])['start_ticks'] == executor['start_ticks'], 'Executor PID reused')
        checked(RUN / 'probability-identity-ready.json', expected['ready'])
        no_updates()
        once(STATE / 'STOP-PRECONDITION.json', packed(dict(status='ACTUAL_BOUND_ADAM0_WAITING',
            original_processes=processes, executor={k: v for k, v in executor.items() if k != 'env'},
            preserved_immutable_files=preserved, typed_pause_verified=True, original_run=str(RUN))))
        # The sole signal: original launcher handles SIGTERM and owns its existing cleanup60.
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    original_pids = [r['pid'] for r in processes] + [executor['pid']]
    deadline = time.monotonic() + 120
    terminal, scheduler, gpu = None, None, None
    while time.monotonic() < deadline:
        if not any((Path('/proc') / str(pid)).exists() for pid in original_pids) and not cgroup.exists():
            scheduler = query(['scontrol', 'show', 'job', '3332', '-o'])
            if re.search(r'\bJobId=3332\b', scheduler['stdout']) and re.search(r'\bJobState=FAILED\b', scheduler['stdout']):
                gpu = query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
                if not gpu['stdout'].strip():
                    terminal = json.loads((RUN / 'run-result.json').read_bytes())
                    break
        time.sleep(2)
    require(terminal is not None and terminal['job_id'] == '3332' and terminal['binding_sha256'] == PARENT_SHA
            and terminal['status'] == 'FAILED' and terminal['cleanup_residual_states'] == {}
            and terminal['error'] == 'Owned calibration executor interrupted by signal 15'
            and not terminal['final_gpu']['processes'], 'No actual final cleanup/Slurm/GPU receipt; no submit')
    no_updates()
    for row in preserved:
        checked(row['path'], row)
    fact = dict(status='ACTUAL_OWN3332_GRACEFUL_ZERO_ADAM_STOPPED', parent_job='3332',
        source_commit=PARENT_SOURCE, target_source_commit=SOURCE, parent_binding_sha256=PARENT_SHA, completed_outer=0, actual_adam_writes_per_rank=0,
        scheduler_step=0, data_cursor=512, original_processes=processes, executor=executor['pid'],
        original_pids_gone=True, cgroup=str(cgroup), cgroup_gone=True, actual_residual_pids=[],
        scheduler_terminal=scheduler, gpu=gpu, original_run_result=record(RUN / 'run-result.json'),
        original_pause_raw=pause_row, original_ready=expected['ready'], preserved_immutable_files=preserved,
        no_probability_approval=True, original_raw_logs_deleted=False, technical_failure_not_goal_pause=True,
        submitted_new_job=False, adapter_scope='ONE_EXACT_PARENT_SIGNAL_AND_FACTUAL_READBACK')
    once(STATE / 'STOP-FACTS.json', packed(fact))
    print(json.dumps(dict(stop_facts=record(STATE / 'STOP-FACTS.json'), status=fact['status'])))


def main():
    require(len(sys.argv) == 2, 'Supply the exact reviewed/publication gate file')
    gate, derivation = gates(sys.argv[1], 'stop')
    payload = dict(parent=derivation['parent_stop_preconditions'], child_binding=derivation['child_binding'],
        adapter_review=gate['adapter_review'], adapter_publication=gate['adapter_publication'],
        source_publication=gate['source_publication'])
    transport(payload, 'stop', Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
