"""Candidate CPU-only task archive. No training mutation, deletion or acceptance."""
import argparse
import ctypes
import datetime
import errno
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
SOURCE = '60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
NONCE = 'bab8981d9fd14afc98923eb4ebb89e33'
RUN = ROOT / ('runs/formal200-' + NONCE + '-3315')
SHARED = Path('/mnt/shared')
BASE = SHARED / 'bapo-artifacts/formal3315-60f566b3-20261008'
NFS = '10.32.0.8:/mnt/data/sihe-share5'
PROD = 'fbd0098b946e23a8eb7304912cad01b8f339e249f627c919ddec8313c2337859'
RATE = 64 * 1024 * 1024  # Aggregate source read + destination write + destination hash read.
RESERVE = 200 * 1024 ** 3
FUTURE = 2 * 60 * 1024 ** 3  # Extra checkpoints150/200 plus growth margin; checked separately.


def need(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe(path):
    path = Path(path)
    need(path.is_absolute(), 'absolute path required')
    current = Path('/')
    for part in path.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            need(not current.is_symlink(), 'symlink component: ' + str(current))
    need(path.resolve() == path, 'nonlexical path')
    return path


def fsync_dir(path):
    fd = os.open(str(safe(path)), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_json(path, value):
    path = safe(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(path.parent)


def signature(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, stat.S_IFMT(st.st_mode)]


def mount(path=SHARED):
    value = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(path)], text=True))
    rows = value['filesystems']
    need(len(rows) == 1 and rows[0]['target'] == str(SHARED)
         and rows[0]['source'] == NFS and rows[0]['fstype'] == 'nfs4', 'wrong shared mount')
    return rows[0]


def identity():
    need(not sys.flags.optimize and sys.platform == 'linux', 'unoptimized Linux required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    need(Path(sys.prefix) == ROOT / 'env', 'lexical task env required')
    need(subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'source mismatch')
    need(not subprocess.check_output(['git', '-C', str(ROOT / 'repo'), 'status', '--porcelain'], text=True).strip(), 'source dirty')
    p = ROOT / 'tools/tools-manifest.json'
    need(hashlib.sha256(p.read_bytes()).hexdigest() == PROD, 'production manifest mismatch')
    tools = json.loads(p.read_text())
    need(tools['source_commit'] == SOURCE and len(tools['files']) == 21, 'production binding mismatch')
    for name, sha in tools['files'].items():
        need(hashlib.sha256(safe(ROOT / 'tools' / name).read_bytes()).hexdigest() == sha, 'production file changed')
    receipt = json.loads(safe(ROOT / 'scheduler-head/formal200-receipt.json').read_text())
    need(receipt['source'] == SOURCE and str(receipt['job_id']) == '3315' and receipt['nonce'] == NONCE, 'run receipt mismatch')
    need(safe(RUN).is_dir(), 'missing actual run')


def checkpoint_gate(step):
    native = RUN / ('checkpoints/fresh200_state/outer_%d_COMPLETE' % step)
    hf = RUN / ('checkpoints/actor/global_step_%d' % step)
    algorithm = RUN / ('checkpoints/algorithm/global_step_%d' % step)
    for tree in [native, hf, algorithm]:
        safe(tree)
        need(tree.is_dir(), 'checkpoint group missing')
        for p in tree.rglob('*'):
            mode = p.lstat().st_mode
            need(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'checkpoint nonregular/symlink forbidden: ' + str(p))
    read = lambda p: json.loads(safe(p).read_text())
    status = read(native / 'status.json')
    need(status['status'] == 'COMPLETE' and status['actual_outer'] == step
         and status['completed_outer_updates'] == step, 'native incomplete')
    saved = read(native / 'all-rank-write-ledger.json')
    need(len(saved) == 8 and sorted(x['rank'] for x in saved) == list(range(8)), 'saved ledger ranks')
    from search_r1.baselines.fresh200_contract import validate_rank_updates
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    need(validate_rank_updates(saved, step) is True, 'native saved ledger is not an accepted complete outer')
    after = status['progress']
    need(after['completed_outers'] == step and after['attempts'] == after['recorded_attempts'] == status['attempt_id'], 'status attempt/progress mismatch')
    by_rank = {x['rank']: x for x in saved}
    before = {k: after[k] for k in ['attempts', 'completed_outers', 'actual_adam_writes_per_rank', 'skipped_opportunities_per_rank', 'incomplete_attempts', 'recorded_attempts']}
    before.update(attempts=after['attempts']-1, recorded_attempts=after['recorded_attempts']-1, completed_outers=step-1,
                  actual_adam_writes_per_rank=[after['actual_adam_writes_per_rank'][r]-by_rank[r]['optimizer_steps'] for r in range(8)],
                  skipped_opportunities_per_rank=[after['skipped_opportunities_per_rank'][r]-by_rank[r]['skipped_opportunities'] for r in range(8)])
    need(all(type(x) is int and x >= 0 for name in ['actual_adam_writes_per_rank', 'skipped_opportunities_per_rank'] for x in before[name]), 'negative or untyped cumulative writes')
    progress = Fresh200Progress(**before)
    progress.begin_attempt(step)
    need(progress.record(saved, step) is True and progress.state_dict() == after, 'native progress replay differs')
    algo = read(algorithm / 'state.json')
    need(algo['algorithm_source_commit'] == SOURCE and algo['actual_outer_updates'] == step, 'algorithm mismatch')
    driver = read(native / 'driver-reload.json')
    need(driver['step'] == step and driver['pid'] == 760122
         and driver['path'] == str(native / 'driver.pt'), 'driver mismatch')
    expected = {str(native / 'driver.pt'): driver['sha256']}
    for rank in range(8):
        r = read(native / ('ranks/rank_%d.json' % rank))
        need(r['rank'] == rank and r['world_size'] == 8 and r['step'] == step
             and r['scheduler_last_epoch'] == step and r['native_state_version'] == 3, 'native rank identity')
        need(r['roundtrip'] is (step == 5) and r['status'] == ('EXACT_FULL_STATE_RELOAD_OK' if step == 5 else 'FULL_STATE_SERIALIZED_OK'), 'roundtrip scope mismatch')
        need(r['latest_rank_event'] == by_rank[rank] and r['optimizer_state_entries'] > 0
             and r['model_dtensor_entries'] > 0, 'native state/event mismatch')
        pt = native / ('ranks/rank_%d.pt' % rank)
        need(r['path'] == str(pt), 'rank path mismatch')
        expected[str(pt)] = r['sha256']
    hf_names = {'added_tokens.json','config.json','generation_config.json','merges.txt',
                'model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors',
                'model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}
    need({str(x.relative_to(hf)) for x in hf.rglob('*') if x.is_file()} == hf_names, 'HF exact file set')
    index = read(hf / 'model.safetensors.index.json')
    need(set(index['weight_map'].values()) == {x for x in hf_names if x.endswith('.safetensors')}, 'HF shard map differs')
    if step == 5:
        p = native / 'manifest.json'
        need(hashlib.sha256(p.read_bytes()).hexdigest() == 'a960351edd3bfb712ca3c89f2e278542ac67d5cfc9750256768853117d2d9f09', 'actual5 manifest mismatch')
    return expected


def inventory():
    files, excluded = [], []
    excluded_names = {'runs', 'assets', 'assets-root', 'env', 'cache', 'lib', 'retriever-gpu-lib', 'repo'}
    # Include all own task output/control/tool trees, including every first5 tool
    # version and evidence referenced by its receipts. Other runs are not adopted.
    roots = [RUN] + [p for p in ROOT.iterdir() if p.name not in excluded_names and p.is_dir()]
    for base in roots:
        safe(base)
        need(base.is_dir(), 'required output tree missing: ' + str(base))
        for directory, dirs, names in os.walk(base, followlinks=False):
            for name in list(dirs):
                p = Path(directory) / name
                if p.is_symlink():
                    excluded.append({'path': str(p), 'reason': 'ephemeral runtime symlink; never followed', 'target': os.readlink(p)})
                    dirs.remove(name)
            for name in names:
                p = Path(directory) / name
                st = p.lstat()
                if not stat.S_ISREG(st.st_mode):
                    excluded.append({'path': str(p), 'reason': 'runtime IPC/symlink, not archived as recovery state'})
                    continue
                rel = str(p.relative_to(ROOT))
                checkpoint = any(p.is_relative_to(RUN / ('checkpoints/' + group + '/' + sub))
                                 for step in [5, 50, 100]
                                 for group, sub in [('actor', 'global_step_%d' % step),
                                                    ('fresh200_state', 'outer_%d_COMPLETE' % step),
                                                    ('algorithm', 'global_step_%d' % step)])
                # Live mutable files are evidence snapshots, never accepted as recoverable/final state.
                files.append({'relative': rel, 'path': str(p), 'stat': signature(st),
                              'bytes': st.st_size, 'scope': 'immutable-checkpoint' if checkpoint else 'bounded-evidence-snapshot'})
    for p in ROOT.iterdir():
        if stat.S_ISREG(p.lstat().st_mode):
            files.append({'relative': p.name, 'path': str(p), 'stat': signature(p.stat()),
                          'bytes': p.stat().st_size, 'scope': 'bounded-evidence-snapshot'})
    need(len({x['relative'] for x in files}) == len(files), 'duplicate inventory path')
    included = {x['path'] for x in files}
    references = {}
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values(): yield from strings(item)
        elif isinstance(value, list):
            for item in value: yield from strings(item)
    for item in files:
        p = Path(item['path'])
        if p.suffix == '.json' and item['bytes'] <= 2 * 1024 * 1024:
            try: value = json.loads(p.read_text())
            except (ValueError, UnicodeError): continue
            for text in strings(value):
                if text.startswith(str(ROOT) + '/') and '\n' not in text and '\x00' not in text:
                    reference = Path(text)
                    if text in included:
                        role = 'included-task-output-or-tool'
                    elif any(reference.is_relative_to(ROOT / name) for name in excluded_names - {'runs'}):
                        role = 'excluded-common-input-or-Git-source; covered by input/source provenance, not output archive'
                    elif not reference.exists():
                        role = 'missing-or-future-reference; not claimed backed up'
                    elif reference.is_dir():
                        role = 'directory-reference; covered only to inventory cutoff'
                    elif reference.is_symlink():
                        role = 'excluded-symlink; not followed'
                    else:
                        raise RuntimeError('Uncovered existing own output reference: ' + text)
                    references[text] = role
    return {'files': files, 'excluded': excluded, 'references': references, 'logical_bytes': sum(x['bytes'] for x in files),
            'excluded_roots': [{'path': str(ROOT / x), 'reason': 'shared input/runtime/source checkout, not task-produced output; source backed up in Git'} for x in ['assets','assets-root','env','cache','lib','retriever-gpu-lib','repo']],
            'pending': ['future checkpoint150/200', 'future output/evaluation records', 'terminal final logs', 'retained storage before NFS server October20 release']}


def reserve(required):
    st = os.statvfs(SHARED)
    available = st.f_bavail * st.f_frsize
    need(available >= required + FUTURE + RESERVE, 'insufficient shared logical-byte headroom')
    return available


def prepare():
    identity(); mount(); safe(BASE.parent)
    BASE.parent.mkdir(exist_ok=True)
    BASE.mkdir(exist_ok=False)
    data = inventory()
    expected = {}
    for step in [5, 50, 100]:
        expected.update(checkpoint_gate(step))
    available = reserve(data['logical_bytes'])
    token = uuid.uuid4().hex
    probe = {'job_id': 3315, 'source': SOURCE, 'nonce': NONCE, 'probe_uuid': token, 'created_utc': utc(), 'nfs_source': NFS}
    create_json(BASE / 'probe.json', probe)
    create_json(BASE / 'inventory.json', data)
    proof = {'job_id': 3315, 'source': SOURCE, 'nonce': NONCE, 'run': str(RUN), 'base': str(BASE), 'logical_bytes': data['logical_bytes'], 'available_bytes': available,
             'future_reserve_bytes': FUTURE, 'shared_reserve_bytes': RESERVE, 'expected_native_hashes': expected,
             'probe_sha256': hashlib.sha256((BASE / 'probe.json').read_bytes()).hexdigest(), 'probe': probe, 'prepared_utc': utc(), 'mount': mount(),
             'inventory_sha256': hashlib.sha256((BASE / 'inventory.json').read_bytes()).hexdigest(),
             'copier_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    create_json(BASE / 'prepared.json', proof)
    print(json.dumps(proof), flush=True)


def crosshost_gate():
    raw = safe(BASE / 'prepared.json').read_bytes()
    prep = json.loads(raw)
    proof = json.loads(safe(BASE / 'retained18-probe.actual.json').read_text())
    need(proof['hostname'] == 'h100-10' and proof['target'] == 'h100-10'
         and proof['probe_sha256'] == prep['probe_sha256'] and proof['probe'] == prep['probe'], 'retained18 real-read probe mismatch')
    need(proof['mount']['target'] == str(SHARED) and proof['mount']['source'] == NFS
         and proof['mount']['fstype'] == 'nfs4', 'retained18 mount mismatch')
    need(proof['prepared_sha256'] == hashlib.sha256(raw).hexdigest(), 'crosshost prepared binding differs')
    need(proof['inventory_sha256'] == prep['inventory_sha256'], 'crosshost inventory differs')
    observed = datetime.datetime.fromisoformat(proof['observed_utc'])
    prepared = datetime.datetime.fromisoformat(prep['prepared_utc'])
    need(observed >= prepared and 0 <= (datetime.datetime.now(datetime.timezone.utc) - observed).total_seconds() < 600, 'stale crosshost observation')
    need(prep['job_id'] == 3315 and prep['source'] == SOURCE and prep['nonce'] == NONCE
         and prep['run'] == str(RUN) and prep['base'] == str(BASE), 'prepared identity differs')
    need(prep['copier_code_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'copier changed after preparation')
    return prep


class Budget:
    def __init__(self):
        self.next_io = time.monotonic()
        self.bytes = 0
        self.last_guard = None

    def pressure_gate(self):
        if self.last_guard is not None and time.monotonic() - self.last_guard < 5:
            return
        while True:
            try:
                p = safe(RUN / 'resource-watch.jsonl')
                with p.open('rb') as f:
                    f.seek(max(0, p.stat().st_size - 2 * 1024 * 1024))
                    raw = f.read()
                lines = raw.splitlines()
                if not raw.endswith(b'\n'):
                    lines = lines[:-1]
                need(lines, 'missing complete resource sample')
                r = json.loads(lines[-1])
                need(r.get('slurm_job_id') == '3315' and r.get('run') == str(RUN), 'foreign resource sample')
                need(r.get('pause') is False and r.get('reason') == 'within_threshold'
                     and 0 <= time.time() - r['time'] < 120, 'resource guard pause/stale')
                mount(BASE)
                reserve(0)
                self.last_guard = time.monotonic()
                return
            except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
                print(json.dumps({'copier_only_wait': str(exc), 'utc': utc()}), flush=True)
                time.sleep(10)

    def consume(self, n):
        self.pressure_gate()
        self.bytes += n
        self.next_io = max(self.next_io, time.monotonic()) + n / RATE
        delay = self.next_io - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def no_replace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result:
        raise OSError(ctypes.get_errno(), 'renameat2 NOREPLACE failed')
    fsync_dir(target.parent)


def copy_file(entry, stage, budget):
    relative = Path(entry['relative'])
    need(not relative.is_absolute() and relative.parts and '..' not in relative.parts, 'invalid relative archive path')
    need(relative.parts[0] not in {'assets','assets-root','env','cache','lib','retriever-gpu-lib','repo'}, 'input/env/source checkout excluded from output copier')
    source = safe(entry['path']); dest = safe(stage / relative)
    need(source == ROOT / relative and dest.is_relative_to(stage) and dest != stage, 'archive/source path escape')
    need(relative.parts[0] != 'runs' or source.is_relative_to(RUN), 'foreign run rejected')
    need(entry['scope'] in {'immutable-checkpoint','bounded-evidence-snapshot'}
         and type(entry['bytes']) is int and entry['bytes'] >= 0
         and isinstance(entry['stat'], list) and len(entry['stat']) == 5, 'invalid entry structure')
    before = signature(source.stat())
    if entry['scope'] == 'immutable-checkpoint':
        need(before == entry['stat'], 'immutable source changed before copy')
    else:
        entry = dict(entry, snapshot_inventory_stat=entry['stat'], stat=before,
                     bytes=before[2], snapshot_started_utc=utc(),
                     snapshot_scope='Per-file bounded read stream; not an atomic full-run snapshot or final log')
    mount(stage)
    dest.parent.mkdir(parents=True, exist_ok=True)
    need(dest.parent.stat().st_dev == SHARED.stat().st_dev, 'destination local fallback/device mismatch')
    h = hashlib.sha256(); left = entry['bytes']
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(str(source), flags)
    try:
        need(signature(os.fstat(fd)) == before, 'source race opening')
        with os.fdopen(fd, 'rb', closefd=False) as src, dest.open('xb') as out:
            while left:
                budget.consume(min(left, 1024 * 1024) * 2)
                block = src.read(min(left, 1024 * 1024))
                need(block, 'short source read')
                out.write(block); h.update(block); left -= len(block)
            out.flush(); os.fsync(out.fileno())
        after = signature(os.fstat(fd))
        if entry['scope'] == 'immutable-checkpoint':
            need(after == before == entry['stat'], 'immutable source changed')
        else:
            need(after[:2] == before[:2] and after[2] >= entry['bytes'], 'snapshot source truncated/replaced')
    finally:
        os.close(fd)
    got = hashlib.sha256()
    with dest.open('rb') as f:
        while True:
            budget.consume(1024 * 1024)
            block = f.read(1024 * 1024)
            if not block: break
            got.update(block)
    need(dest.stat().st_size == entry['bytes'] and got.hexdigest() == h.hexdigest(), 'destination hash mismatch')
    fsync_dir(dest.parent)
    return dict(entry, sha256=h.hexdigest(), source_pre=before, source_post=after)


def copy_archive():
    identity(); mount(); prep = crosshost_gate()
    lock = safe(BASE / 'copy.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(not (BASE / 'copy-complete.json').exists(), 'archive already completed')
    os.nice(19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    inventory_raw = safe(BASE / 'inventory.json').read_bytes()
    need(hashlib.sha256(inventory_raw).hexdigest() == prep['inventory_sha256'], 'inventory changed after preparation')
    data = json.loads(inventory_raw)
    need(data['logical_bytes'] == prep['logical_bytes'] == sum(x['bytes'] for x in data['files']), 'inventory logical bytes differ')
    need(len({x['relative'] for x in data['files']}) == len(data['files']), 'duplicate prepared file')
    expected = {}
    for step in [5, 50, 100]: expected.update(checkpoint_gate(step))
    need(expected == prep['expected_native_hashes'], 'checkpoint binding changed after preparation')
    need(set(expected).issubset({x['path'] for x in data['files']}), 'native required file excluded')
    mount(BASE)
    reserve(data['logical_bytes'])
    start = time.monotonic(); cpu = time.process_time(); budget = Budget()
    stage = safe(BASE / ('.partial-' + uuid.uuid4().hex)); stage.mkdir(exist_ok=False)
    create_json(stage / 'scope.json', {'prepared': prep, 'inventory': data, 'copier_pid': os.getpid(), 'created_utc': utc(), 'final_goal_accepted': False})
    done = []
    pending_files = []
    entries = sorted(data['files'], key=lambda x: (x['scope'] != 'immutable-checkpoint', x['relative']))
    for entry in entries:
        # Runtime diagnostics may rotate or disappear. Keep a truthful pending record;
        # never downgrade failure of a recovery checkpoint to an optional snapshot.
        if entry['scope'] != 'immutable-checkpoint' and not Path(entry['path']).exists():
            pending_files.append({'path': entry['path'], 'reason': 'live evidence file disappeared since inventory'})
            continue
        item = copy_file(entry, stage, budget)
        if item['path'] in prep['expected_native_hashes']:
            need(item['sha256'] == prep['expected_native_hashes'][item['path']], 'native saved hash mismatch')
        done.append(item)
        print(json.dumps({'copied_file': item['relative'], 'bytes': item['bytes'], 'sha256': item['sha256'], 'utc': utc()}), flush=True)
    identity(); mount()
    final_expected = {}
    for step in [5, 50, 100]: final_expected.update(checkpoint_gate(step))
    need(final_expected == prep['expected_native_hashes'], 'checkpoint final binding/object kinds changed')
    for entry in done:
        if entry['scope'] == 'immutable-checkpoint':
            need(signature(safe(entry['path']).stat()) == entry['stat'], 'immutable file changed by final inventory')
    actual_checkpoint = {x['path']: x['stat'] for x in inventory()['files'] if x['scope'] == 'immutable-checkpoint'}
    expected_checkpoint = {x['path']: x['stat'] for x in done if x['scope'] == 'immutable-checkpoint'}
    need(actual_checkpoint == expected_checkpoint, 'checkpoint file set/stat changed')
    need(set(prep['expected_native_hashes']).issubset({x['path'] for x in done}), 'native required file not copied')
    manifest = {'files': done, 'scope': 'Prepared task output cutoff, immutable checkpoints5/50/100 and bounded live evidence snapshots; not final output or HF acceptance', 'pending': data['pending'],
                'excluded': data['excluded'], 'excluded_roots': data['excluded_roots'], 'pending_live_files': pending_files, 'all_task_outputs_verified': False, 'whole200_accepted': False,
                'job_id': 3315, 'source': SOURCE, 'run': str(RUN), 'completed_utc': utc(), 'copied_bytes': sum(x['bytes'] for x in done), 'io_budget_bytes': budget.bytes,
                'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - start, 'mount': mount()}
    actual = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} - {'scope.json'}
    need(actual == {x['relative'] for x in done}, 'destination set differs')
    create_json(stage / 'manifest.json', manifest)
    target = BASE / ('snapshot-' + prep['probe']['probe_uuid'] + '-COMPLETE')
    no_replace(stage, target)
    create_json(BASE / 'copy-complete.json', {'directory': str(target), 'manifest_sha256': hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest(), 'completed_utc': utc(), 'whole200_accepted': False})
    print(json.dumps({'copy_complete': str(target), 'whole200_accepted': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'copy'])
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else copy_archive)()
