"""External same-run recovery: only the frozen Controller.wait is overridden."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

CONFIG_SHA = 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
RUN_ID = '1d1d67b60f0e46a2afc0536a36ab69b1'
ORIGINAL_MANIFEST_SHA = '66cb43dd0830b9a7776bd644a0ef5bbf3d1661a3db3b159cc9e781db582c5233'
ORIGINAL_COMMIT = '7cf285308572af645676ae85a48fc8359f8c8ccb'
REPO_PREFIX = 'experiments/search_e1/control/opsd_online_wait_recovery/'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def fileproof(path):
    path = Path(path).resolve(); before = metadata(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    require(metadata(path) == before, 'File changed during hashing')
    return dict(path=str(path), sha256=digest.hexdigest(), **before)


def checked_json(ref):
    proof = fileproof(ref['path'])
    require(proof['sha256'] == ref['sha256'], 'Receipt SHA mismatch')
    value = json.loads(Path(proof['path']).read_text())
    require(metadata(proof['path']) == {k: proof[k] for k in metadata(proof['path'])}, 'JSON changed during read')
    return value, proof


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def exclusive_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    fsync_dir(path.parent)
    return fileproof(path)


def verify_tree(root, manifest_ref):
    root = Path(root).resolve(); manifest, proof = checked_json(manifest_ref)
    files = manifest['files']; require(files and isinstance(files, dict), 'Empty source inventory')
    actual_names = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.name != 'source-manifest.json'}
    require(actual_names == set(files), 'Source inventory differs from frozen manifest')
    actual = []
    for name, expected in sorted(files.items()):
        path = root / name
        require(path.resolve().is_relative_to(root), 'Source escapes frozen root')
        item = fileproof(path)
        require(item['sha256'] == expected['sha256'] and item['bytes'] == expected['bytes'], 'Frozen source SHA/bytes differ: ' + name)
        actual.append(dict(relative_path=name, **item))
    return dict(manifest=proof, files=actual)


def verify_external_binding(manifest_ref, publication_ref, source_root=None):
    root = Path(source_root or Path(__file__).parent).resolve()
    tree = verify_tree(root, manifest_ref)
    require(Path(tree['manifest']['path']).parent == root, 'Recovery manifest outside own directory')
    publication, pubproof = checked_json(publication_ref)
    require(publication.get('accepted') is True and publication.get('branch') == 'search-e1', 'Recovery publication not accepted')
    commit = publication.get('source_commit', '')
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None and publication.get('remote_readback_sha') == commit,
            'Recovery remote commit readback is missing/different')
    require(publication.get('source_manifest_sha256') == tree['manifest']['sha256'], 'Recovery publication manifest differs')
    require(publication.get('run_id') == RUN_ID and publication.get('config_sha256') == CONFIG_SHA, 'External publication targets another run/config')
    rows = publication.get('files', [])
    require(len({row['path'] for row in rows}) == len(rows), 'Duplicate remote readback paths')
    indexed = {row['path']: row for row in rows}
    for item in tree['files'] + [dict(relative_path='source-manifest.json', **tree['manifest'])]:
        row = indexed.get(REPO_PREFIX + item['relative_path'])
        require(row is not None and row.get('equal') is True and row.get('sha256') == item['sha256']
                and row.get('bytes') == item['bytes'], 'Recovery actual file absent/different in remote readback')
    return dict(source=tree, publication=pubproof, source_commit=commit)


def validate_old_terminal(ref, proc_root='/proc'):
    value, proof = checked_json(ref)
    require(value.get('schema') == 'opsd-online-failed-original-owner-terminal-v1' and value.get('accepted') is True,
            'Original failure terminal proof not accepted')
    require(value.get('run_id') == RUN_ID and value.get('config_sha256') == CONFIG_SHA, 'Wrong original terminal run/config')
    require(value.get('old_wait_returncodes') == 'NOT_PERSISTED_UNKNOWN', 'Do not infer old child rc from worker-complete')
    require(value.get('old_controller_tmux_exit_status') == 1, 'Original controller failure must remain explicit')
    rows = value['old_owned_processes']
    require(len(rows) == 9 and len({r['pid'] for r in rows}) == 9, 'Need original controller plus all eight workers')
    require(any(r == dict(pid=612077, uid=1005, starttick=924207665) for r in rows), 'Original controller identity differs')
    for row in rows:
        require(set(row) >= {'pid', 'uid', 'starttick'} and row['uid'] == 1005
                and type(row['starttick']) is int and row['starttick'] > 0, 'Invalid old process identity')
        require(not (Path(proc_root) / str(row['pid'])).exists(), 'Old PID is present; do not reuse historical absence')
    return dict(receipt=proof, old_owned_processes=rows, fresh_all_old_pids_absent=True,
                old_wait_returncodes='NOT_PERSISTED_UNKNOWN')


class Journal:
    """Each event is fsynced; terminal evidence is durable before any rejection."""
    def __init__(self, directory):
        # This UUID belongs exclusively to this wait invocation. Constructor I/O
        # may be retried before its first event; a nonempty existing file refuses.
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'events.jsonl'
        with self.path.open('a') as handle:
            handle.flush(); os.fsync(handle.fileno())
        require(self.path.stat().st_size == 0, 'Unexpected existing journal contents')
        fsync_dir(self.directory)
        self.durable_bytes = 0

    def record(self, event, **fields):
        value = dict(event=event, observed_unix=time.time(), **fields)
        payload = (json.dumps(value, sort_keys=True) + '\n').encode()
        with self.path.open('r+b') as handle:
            # After a failed partial append/fsync, retry restores ONLY this
            # writer's last durable boundary before appending one whole event.
            handle.truncate(self.durable_bytes); handle.seek(self.durable_bytes)
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        self.durable_bytes += len(payload)


def hold_evidence(operation, sleeper):
    """Do not let evidence/storage uncertainty unwind a live controller lease."""
    while True:
        try: return operation()
        except Exception:
            # Disk failure can also prevent status/journal writes. Retain locks,
            # keep known children registered, and retry this exact operation.
            sleeper(5)


def durable_json(path, value):
    """Idempotent per-wait evidence commit, safe after partial JSON/fsync failure."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending-evidence')
    payload = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == payload, 'Existing wait evidence differs')
    else:
        with temporary.open('wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    fsync_dir(path.parent)
    return fileproof(path)


def recovery_controller(frozen, external_proof, reverify=lambda: None, sleeper=time.sleep):
    class RecoveryController(frozen.Controller):
        def wait(self, jobs, phase):
            require(phase in ('sampling-running', 'training-running') and jobs, 'Unexpected/empty waiting phase')
            directory = Path(self.config['paths']['base']) / 'receipts' / 'wait-recovery' / uuid.uuid4().hex
            safe = lambda operation: hold_evidence(operation, sleeper)
            journal = safe(lambda: Journal(directory))
            record = lambda event, **fields: safe(lambda: journal.record(event, **fields))
            stage = lambda state, **fields: safe(lambda: self.stage(state, **fields))
            record('WAIT_BEGIN', **dict(run_id=self.run_id, config_sha256=self.config_ref['sha256'],
                owner=self.owner, child_phase=phase, children=[i for _, _, i in jobs], external_binding=external_proof,
                old_wait_returncodes_not_reused=True, signals_sent=False))
            # No sticky scientific failure bit for an admission read error.
            while True:
                try:
                    codes = [p.poll() for p, _, _ in jobs]
                    record('CHILD_POLL', returncodes=codes)
                except Exception as exc:
                    record('CHILD_POLL_UNKNOWN_HOLD', error_type=type(exc).__name__, error=str(exc))
                    stage('UNCERTAIN_WAIT_LEASE_HELD', child_phase=phase, recovery_journal=str(journal.path), signals_sent=False)
                    sleeper(5); continue
                try:
                    gate = self.gate(phase)
                    record('ADMISSION_ACCEPTED', proof=gate)
                except Exception as exc:
                    record('ADMISSION_UNCERTAIN', error_type=type(exc).__name__, error=str(exc), returncodes=codes)
                    stage('UNCERTAIN_ADMISSION_WAITING_OWNED_TERMINATION', child_phase=phase,
                               error=str(exc), exit_codes=codes, recovery_journal=str(journal.path), signals_sent=False)
                if all(c is not None for c in codes): break
                sleeper(5)
            evidence = []
            for process, handle, identity in jobs:
                while True:
                    try:
                        rc = process.wait()
                        require(type(rc) is int, 'Unknown actual wait returncode')
                        break
                    except Exception as exc:
                        record('ACTUAL_WAIT_UNKNOWN_HOLD', identity=identity, error_type=type(exc).__name__, error=str(exc))
                        stage('UNCERTAIN_WAIT_LEASE_HELD', child_phase=phase, signals_sent=False)
                        sleeper(5)
                item = dict(identity=identity, actual_wait_returncode=rc)
                record('ACTUAL_WAIT_RETURNED', **item)  # BEFORE checking rc or gates
                evidence.append(item)
                safe(lambda: None if handle.closed else handle.close())
            terminal_value = dict(
                schema='opsd-online-wait-recovery-actual-terminal-v1', run_id=self.run_id,
                config_sha256=self.config_ref['sha256'], child_phase=phase, child_exit_evidence=evidence,
                accepted_all_wait_zero=all(e['actual_wait_returncode'] == 0 for e in evidence),
                signals_sent=False, original_attempt_exit_evidence_reused=False)
            terminal = safe(lambda: durable_json(directory / 'actual-terminal.json', terminal_value))
            logproofs = []
            for _, handle, identity in jobs:
                safe(lambda: require(isinstance(handle.name, str), 'Actual child log path required'))
                logproofs.append(dict(identity=identity, file=safe(lambda: fileproof(handle.name))))
            # Even hard failures wait for the same strict release. No next CUDA phase
            # while any current admission/ownership fact is uncertain.
            while True:
                try:
                    reverify()
                    final_gate = self.gate(phase)
                    gpu = frozen.require_gpu_release(self.config, [i for _, _, i in jobs])
                    cpu_gate = self.gate('cpu-merge', [])
                    audit = getattr(self.admit, 'require_owned_release', None)
                    require(callable(audit), 'Missing original full owned-release callback')
                    owned = audit(self.config, self.run_id, self.owner, known_children=self.known)
                    require(owned.get('accepted') is True, 'Full owned release rejected')
                    # A last current gate follows the full ownership observation.
                    after_gate = self.gate('cpu-merge', [])
                    release = dict(gpu=gpu, all_owned_release=owned)
                    record('TERMINAL_RELEASE_ACCEPTED', final_gate=final_gate, cpu_gate=cpu_gate,
                                   after_gate=after_gate, release=release, actual_terminal=terminal)
                    break
                except Exception as exc:
                    record('TERMINAL_RELEASE_UNCERTAIN_HOLD', error_type=type(exc).__name__, error=str(exc), actual_terminal=terminal)
                    stage('UNCERTAIN_RELEASE_LEASE_HELD', child_phase=phase, error=str(exc),
                               recovery_journal=str(journal.path), actual_terminal=terminal,
                               signals_sent=False, recovery='Retain all leases; no next CUDA phase')
                    sleeper(5)
            accepted = all(e['actual_wait_returncode'] == 0 for e in evidence)
            record('WAIT_ACCEPTED' if accepted else 'HARD_CHILD_FAILURE', actual_terminal=terminal)
            summary = dict(schema='opsd-online-wait-recovery-phase-v1', accepted=accepted,
                run_id=self.run_id, config_sha256=self.config_ref['sha256'], child_phase=phase,
                child_exit_evidence=evidence, actual_terminal=terminal, child_logs=logproofs,
                gpu_release=release, terminal_admission=final_gate, after_release_admission=after_gate,
                events=safe(lambda: fileproof(journal.path)), external_binding=external_proof, signals_sent=False)
            summary_ref = safe(lambda: durable_json(directory / 'phase-complete.json', summary))
            require(accepted, 'Actual child nonzero returncode; preserved before rejection, no signal/restart')
            return dict(accepted=True, child_exit_evidence=evidence, gpu_release=release, wait_recovery=summary_ref)
    return RecoveryController


def load_original(config):
    require(config['helper_source_manifest']['sha256'] == ORIGINAL_MANIFEST_SHA and config['source_commit'] == ORIGINAL_COMMIT,
            'Recovery must load the original immutable helper/config')
    root = Path(config['paths']['helper_root']).resolve()
    proof = verify_tree(root, config['helper_source_manifest'])
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('opsd_online_original_controller_bound', root / 'controller.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == root / 'controller.py', 'Original controller import path differs')
    for name in ('checkpoint', 'prepare', 'sampling', 'scoring'):
        require(name in sys.modules and Path(sys.modules[name].__file__).resolve() == root / (name + '.py'),
                'Cached/shadowed original helper import: ' + name)
    verify_tree(root, config['helper_source_manifest'])
    return module, proof


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'run-id', 'recovery-manifest', 'recovery-manifest-sha256',
                 'publication', 'publication-sha256', 'original-terminal', 'original-terminal-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    require(a.resume and a.run_id == RUN_ID and a.config_sha256 == CONFIG_SHA, 'Explicit same-run immutable-config --resume only')
    config_ref = dict(path=a.config, sha256=a.config_sha256)
    config, _ = checked_json(config_ref)
    manifest_ref = dict(path=a.recovery_manifest, sha256=a.recovery_manifest_sha256)
    publication_ref = dict(path=a.publication, sha256=a.publication_sha256)
    terminal_ref = dict(path=a.original_terminal, sha256=a.original_terminal_sha256)
    external = verify_external_binding(manifest_ref, publication_ref)
    terminal = validate_old_terminal(terminal_ref)
    frozen, original = load_original(config)
    controller = None
    def reverify():
        verify_external_binding(manifest_ref, publication_ref)
        checked_json(config_ref)
        verify_tree(config['paths']['helper_root'], config['helper_source_manifest'])
    cls = recovery_controller(frozen, dict(recovery=external, original_helper=original,
                                         original_terminal=terminal), reverify=reverify)
    try:
        controller = cls(config_ref, a.run_id, resume=True)
        # Fresh old-PID absence again immediately before original gates/leases.
        validate_old_terminal(terminal_ref)
        controller.gate('cpu-merge', [])
        controller.run()  # original run/sample/trainer/resume logic, unchanged
    except Exception as exc:
        if controller is not None and controller.owns_output:
            controller.contain_failure(exc)
        raise


if __name__ == '__main__':
    main()
