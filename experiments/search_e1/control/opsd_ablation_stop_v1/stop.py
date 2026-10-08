"""Published, single-owner stop; no process-group signals or scientific edits.

Only the replaced online run is eligible. A stopped parent prevents new stages
while exact child pidfds are drained and a committed checkpoint is preserved.
On uncertainty the parent remains stopped; an error is never labeled exit zero.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import types

BASE = Path('/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008')
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CONFIG_SHA = 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
OWNER = dict(pid=720753, uid=1005, starttick=924526765)
ORACLE = dict(pid=2239529, uid=1005, starttick=914664099)
ENTRY = str(BASE / 'source/cpu-merge-51eb6c5/opsd_online_cpu_merge_recovery/recovery.py')
PYTHON = '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python'
CHECKPOINT_SHA = '9676eb662bb6c5f0ea21006e52c8bb9a4a2281a69fa235fcdc5cf2974a29e857'
HELPER_MANIFEST_SHA = '66cb43dd0830b9a7776bd644a0ef5bbf3d1661a3db3b159cc9e781db582c5233'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def proof(path):
    path = Path(path)
    def stat():
        s = path.stat()
        return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                    mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    before = stat(); digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    require(before == stat(), 'File changed while hashing: ' + str(path))
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), **before)


def checked_json(path, sha):
    actual = proof(path); require(actual['sha256'] == sha, 'JSON SHA differs')
    value = json.loads(Path(path).read_text())
    require(proof(path) == actual, 'JSON changed after verified read')
    return value, actual


def exact(current, bound):
    return current is not None and all(current.get(k) == bound[k] for k in ('pid', 'uid', 'starttick'))


def match_owner(current, command=None):
    require(exact(current, OWNER), 'Owner PID/UID/starttick differs; no signal')
    require(current.get('run_id') == RUN and current.get('cuda_visible_devices') == '', 'Owner environment differs')
    tokens = current['argv_tokens']
    require(tokens[:4] == [PYTHON, '-B', ENTRY, 'parent'], 'Owner executable/entry differs')
    for option, value in [('--config', str(BASE / 'receipts/config.json')),
                          ('--config-sha256', CONFIG_SHA), ('--run-id', RUN)]:
        require(tokens.count(option) == 1 and tokens[tokens.index(option) + 1:tokens.index(option) + 2] == [value],
                'Owner exact option differs: ' + option)
    require(tokens.count('--resume') == 1, 'Owner resume option differs')
    if command is not None:
        require(tokens == command, 'Current argv differs from actual controller binding')


def read_process(pid, proc=Path('/proc')):
    root = proc / str(pid)
    try:
        raw = (root / 'stat').read_text(); tail = raw[raw.rfind(')') + 2:].split()
        fields = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
        uid = int(fields['Uid'].split()[1])  # Kernel effective UID; not /proc directory owner.
        tokens = (root / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        environment = None
        try:
            environment = dict(item.split('=', 1) for item in (root / 'environ').read_bytes().decode(errors='replace').split('\0') if '=' in item)
        except PermissionError:
            pass
        # Recheck tick after all reads to reject a reused PID / disappearing task.
        second = (root / 'stat').read_text(); second_tail = second[second.rfind(')') + 2:].split()
        require(tail[19] == second_tail[19], 'PID changed during read')
        return dict(pid=int(pid), uid=uid, starttick=int(tail[19]), ppid=int(tail[1]),
                    pgid=int(tail[2]), state=tail[0], argv_tokens=tokens,
                    run_id=None if environment is None else environment.get('E1_RUN_ID'),
                    cuda_visible_devices=None if environment is None else environment.get('CUDA_VISIBLE_DEVICES'),
                    environment_readable=environment is not None)
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_table(reader=read_process, proc=Path('/proc')):
    table = {}
    for path in proc.iterdir():
        if path.name.isdigit():
            item = reader(int(path.name))
            if item is not None:
                table[item['pid']] = item
    return table


def descendants(table, anchors):
    found = set(anchors)
    while True:
        expanded = found | {pid for pid, row in table.items() if row['ppid'] in found}
        if expanded == found:
            return found - set(anchors)
        found = expanded


def owned_children(table, known=(), protected=()):
    # Input paths/argv substrings never establish ownership. Positive run tag,
    # exact known identity, or actual parent chain are required instead.
    live_known = {row['pid'] for row in known if exact(table.get(row['pid']), row)}
    chain = descendants(table, {OWNER['pid']} | live_known)
    selected = []
    for pid, row in table.items():
        own = pid in chain or pid in live_known or row.get('run_id') == RUN
        if pid == OWNER['pid'] or not own:
            continue
        require(row['uid'] == OWNER['uid'], 'Foreign UID in owned chain; never signal it')
        require(not any(exact(row, p) for p in [ORACLE] + list(protected)), 'Protected oracle/observer matched ownership')
        selected.append(row)
    return sorted(selected, key=lambda row: row['starttick'], reverse=True)


class Signaler:
    def __init__(self, reader=read_process, opener=None, sender=None, closer=os.close):
        self.reader = reader; self.opener = getattr(os, 'pidfd_open', None) if opener is None else opener
        self.sender = getattr(signal, 'pidfd_send_signal', None) if sender is None else sender; self.closer = closer
        require(callable(self.opener) and callable(self.sender), 'Required pidfd capability missing; no signals')

    def send(self, bound, number, journal, owner=False, command=None):
        before = self.reader(bound['pid']); require(exact(before, bound), 'Signal target absent/reused/UID changed')
        if owner: match_owner(before, command)
        require(not exact(before, ORACLE), 'Oracle is never a signal target')
        fd = self.opener(bound['pid'])
        try:
            fresh = self.reader(bound['pid']); require(exact(fresh, bound), 'PID changed after pidfd binding; no signal')
            if owner: match_owner(fresh, command)
            journal('BEFORE_SIGNAL', target=fresh, signal=signal.Signals(number).name)
            self.sender(fd, number)
            journal('SIGNAL_SENT', target=fresh, signal=signal.Signals(number).name)
        finally:
            self.closer(fd)


def drain_children(signaler, scanner, journal, protected, timeout=20, clock=time.monotonic, sleep=time.sleep):
    known = []; sent = set(); deadline = clock() + timeout
    while True:
        current = owned_children(scanner(), known, protected)
        for child in current:
            key = (child['pid'], child['uid'], child['starttick'])
            if not any(exact(child, previous) for previous in known): known.append(child)
            if key not in sent and child['state'] != 'Z':
                signaler.send(child, signal.SIGTERM, journal); sent.add(key)
        live = [c for c in current if c['state'] != 'Z']
        if not live:
            journal('CHILDREN_NOT_LIVE', children=current, signaled=known)
            return known
        if clock() >= deadline:
            for child in live:
                signaler.send(child, signal.SIGKILL, journal)
            deadline = clock() + timeout
            # No unbounded repeated kills; uncertainty leaves parent STOPPED.
            while clock() < deadline:
                remain = owned_children(scanner(), known, protected)
                if not [c for c in remain if c['state'] != 'Z']:
                    return known
                sleep(.25)
            raise RuntimeError('Owned child survived exact KILL; parent remains stopped')
        sleep(.25)


def latest_checkpoint(root, config, native):
    candidates = []
    for path in Path(root).glob('*/manifest.json'):
        match = re.fullmatch(r'update-(\d{8})-outer-(\d{5})', path.parent.name)
        if not match: continue  # Pending directories remain untouched.
        require(not path.parent.is_symlink(), 'Canonical checkpoint is a symlink')
        value = json.loads(path.read_text())
        require(value.get('run_id') == RUN and value.get('config_sha256') == CONFIG_SHA, 'Foreign checkpoint identity')
        if value.get('accepted') is True:
            require(value['optimizer_updates'] == int(match[1]) and value['cursor']['outer_index'] == int(match[2]), 'Checkpoint name/counter mismatch')
            candidates.append((int(match[1]), int(match[2]), path))
    require(candidates, 'No complete canonical checkpoint; preserve everything and HOLD')
    updates, outer, path = max(candidates)
    cp, actual = native.check_checkpoint(native.fileproof(path), config, RUN, updates, outer)
    return cp, actual


def load_native(config):
    ref = config['helper_source_manifest']
    require(ref['sha256'] == HELPER_MANIFEST_SHA, 'Original helper manifest binding changed')
    catalog, catalog_proof = checked_json(ref['path'], ref['sha256'])
    path = Path(config['paths']['helper_root']) / 'checkpoint.py'; actual = proof(path)
    row = catalog['files']['checkpoint.py']
    require(row['sha256'] == actual['sha256'] == CHECKPOINT_SHA and row['bytes'] == actual['bytes'], 'Native checkpoint source changed')
    module = types.ModuleType('stop_verified_native_checkpoint')
    module.__file__ = str(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == CHECKPOINT_SHA, 'Native bytes changed before compilation')
    exec(compile(data, str(path), 'exec'), module.__dict__)
    require(proof(path) == actual, 'Native source changed during source-only load')
    return module, dict(manifest=catalog_proof, source=actual, source_only=True, torch_imported=False)


def protected_observers():
    path = BASE / 'receipts/memory-binding.json'
    value = json.loads(path.read_text())
    def collect(item):
        if isinstance(item, dict):
            if all(k in item for k in ('pid', 'uid', 'starttick')):
                yield {k: item[k] for k in ('pid', 'uid', 'starttick')}
            for part in item.values(): yield from collect(part)
        elif isinstance(item, list):
            for part in item: yield from collect(part)
    return list(collect(value)), proof(path)


def preserve_compact(native, output, cp, cp_proof, config_proof, status):
    # Full checkpoint stays at its original canonical path; copying manifests
    # does not pretend to be a second durable copy of model/Adam bytes.
    saved = []
    for name in ('receipts/config.json', 'receipts/current-controller.json', 'runs/status.json', 'runs/child-history.jsonl'):
        source = BASE / name
        if source.is_file():
            actual = proof(source); target = output / 'original' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
            require(proof(target)['sha256'] == actual['sha256'], 'Compact preservation copy differs')
            saved.append(dict(original=actual, copy=proof(target)))
    value = dict(schema='opsd-online-authorized-stop-preservation-v1', accepted=True, run_id=RUN,
        config=config_proof, checkpoint=cp_proof, optimizer_updates=cp['optimizer_updates'],
        completed_pair_cursor=cp['cursor'], checkpoint_files_retained_in_place=True,
        independent_tensor_semantic_validation_performed=False, compact_copies=saved,
        interrupted_outer_status=status, interrupted_outer_is_not_completion=True,
        pending_checkpoints_retained=[str(p) for p in (BASE / 'runs/checkpoints').iterdir()
                                      if re.fullmatch(r'update-\d{8}-outer-\d{5}', p.name) is None])
    return native.exclusive_json(output / 'checkpoint-preservation.json', value)


def stop(args):
    require(os.geteuid() == OWNER['uid'] and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only exact UID required')
    require(os.environ.get('E1_RUN_ID') != RUN, 'Stop helper must not inherit original run tag')
    signaler = Signaler()  # Capability must exist before any mutation or signal.
    source = proof(__file__); require(source['sha256'] == args.source_sha256, 'Stop source SHA differs')
    pub, pub_proof = checked_json(args.publication, args.publication_sha256)
    require(pub.get('accepted') is True and pub.get('source_sha256') == source['sha256']
            and pub.get('config_sha256') == CONFIG_SHA and pub.get('run_id') == RUN
            and re.fullmatch('[0-9a-f]{40}', pub.get('source_commit', ''))
            and pub.get('remote_readback_sha') == pub['source_commit'], 'Stop source not published/read back for this exact run')
    config, config_proof = checked_json(BASE / 'receipts/config.json', CONFIG_SHA)
    require(config['run_id'] == RUN and config['owner']['uid'] == OWNER['uid'] and config['paths']['output_root'] == str(BASE / 'runs'), 'Config scope differs')
    config['_config_sha256'] = CONFIG_SHA
    native, native_proof = load_native(config)
    binding_path = BASE / 'receipts/current-controller.json'; binding = json.loads(binding_path.read_text())
    require(binding['accepted'] is True and binding['config_sha256'] == CONFIG_SHA and binding['run_id'] == RUN
            and exact(binding['owner'], OWNER), 'Controller binding differs')
    command = binding['command']; match_owner(read_process(OWNER['pid']), command)
    protected, observer_proof = protected_observers()
    # The journal is the only new output tree. Existing checkpoints/logs/runtime
    # and partially completed outer directories are never modified or removed.
    output = Path(args.output)
    require(output.parent.resolve() == (BASE / 'preserved').resolve() and not output.exists(), 'Output must be a fresh owned preservation directory')
    output.mkdir(parents=True, mode=0o700)
    sequence = 0
    def journal(event, **fields):
        nonlocal sequence
        sequence += 1
        return native.exclusive_json(output / ('event-%04d.json' % sequence), dict(event=event, unix=time.time(), **fields))
    journal('AUTHORIZED_OLD_RUN_STOP', source=source, publication=pub_proof, native_source=native_proof,
            config=config_proof, binding=proof(binding_path), observer=observer_proof, old_owner=OWNER,
            human_goal_created_cst='2026-10-08 23:18:19', human_goal_updated_cst='2026-10-08 23:25:26',
            scope='Stop replaced old eight-GPU online run; seven new reverse ablations remain active')
    stopped = False; terminated = False
    try:
        signaler.send(OWNER, signal.SIGSTOP, journal, owner=True, command=command); stopped = True
        children = drain_children(signaler, process_table, journal, protected)
        status = json.loads((BASE / 'runs/status.json').read_text())
        require(status['run_id'] == RUN, 'Current status run differs')
        cp, cp_proof = latest_checkpoint(BASE / 'runs/checkpoints', config, native)
        preservation = preserve_compact(native, output, cp, cp_proof, config_proof, status)
        journal('BEFORE_OWNER_TERMINATION', preservation=preservation, remaining=owned_children(process_table(), children, protected))
        require(not [r for r in owned_children(process_table(), children, protected) if r['state'] != 'Z'], 'A live child reappeared; parent remains stopped')
        signaler.send(OWNER, signal.SIGTERM, journal, owner=True, command=command)
        # TERM is pending while STOPPED; CONT delivers it, not a restart request.
        signaler.send(OWNER, signal.SIGCONT, journal, owner=True, command=command); terminated = True
        deadline = time.monotonic() + 20
        while exact(read_process(OWNER['pid']), OWNER):
            row = read_process(OWNER['pid'])
            if row is not None and row['state'] == 'Z': break
            if time.monotonic() >= deadline:
                signaler.send(OWNER, signal.SIGKILL, journal, owner=True, command=command); break
            time.sleep(.25)
        # Actual processes must be gone/reaped before declaring resource release.
        deadline = time.monotonic() + 20
        while True:
            remaining = owned_children(process_table(), children, protected)
            if not exact(read_process(OWNER['pid']), OWNER) and not remaining: break
            require(time.monotonic() < deadline, 'Owned tasks not reaped; release UNKNOWN, not rc0')
            time.sleep(.25)
        with ExitStack() as leases:
            for name in [str(BASE / 'online-experiment.lock')] + config['resources']['shared_gpu_lease_paths']:
                handle = leases.enter_context(open(name, 'a+'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
            pids = {int(line.split(',')[0]) for line in apps.splitlines() if line.strip()}
            require(pids <= {ORACLE['pid']} and (ORACLE['pid'] not in pids or exact(read_process(ORACLE['pid']), ORACLE)), 'Unreleased/foreign CUDA context; never signal it')
            pane = subprocess.check_output(['tmux', 'list-panes', '-t', binding['session'], '-F', '#{pane_dead} #{pane_dead_status}'], text=True).strip()
            require(pane in ('1 143', '1 137'), 'Actual tmux signal termination unavailable; not rc0: ' + pane)
            terminal = native.exclusive_json(output / 'terminal.json', dict(schema='opsd-online-authorized-stop-terminal-v1', accepted=True,
                run_id=RUN, config_sha256=CONFIG_SHA, owner=OWNER, old_children=children, preservation=preservation,
                actual_tmux_status=int(pane.split()[1]), signal_termination_not_exit0=True, old_run_released=True,
                gpu_apps_csv=apps, oracle_and_observer_not_signaled=True, foreign_and_transfers_not_signaled=True,
                source=source, publication=pub_proof, checked_unix=time.time()))
            print(json.dumps(dict(accepted=True, terminal=terminal, preservation=preservation)))
    except Exception as exc:
        journal('STOP_UNCERTAIN', error=str(exc), parent_stopped=stopped, parent_termination_sent=terminated,
                no_automatic_resume=True, no_scientific_files_removed=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--publication-sha256', required=True)
    parser.add_argument('--output', required=True)
    stop(parser.parse_args())
