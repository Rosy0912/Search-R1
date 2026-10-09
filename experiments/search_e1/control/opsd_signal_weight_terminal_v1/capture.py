"""Read-only CPU live -> same retained pane driver exit capture.

No producer import, Torch, GPU allocation, job launch, signals, locks or tmux
mutation. JSON is printed to stdout; caller preserves the exact live bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
VARIANTS = ['G_POS', 'G_NEG', 'H_POS', 'H_NEG']
STUDY_SHA = '86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e'
REPAIR_SHA = '73a5594a0dae56125cdb5f7a28f8f108454df5c8c1761d8fa629556b0cac9f4e'
LAUNCH_SHA = '87735981dd1a9f60c162d096ee0fafdf43d9fb5a7b12256e1ecdc3c4438560fb'
OLD_COMMIT = '00fec71f6e2481f9e1bc4576d956748bdda77aaa'
REPAIR_COMMIT = '98b9326a782a51261571ff24f3cbdbd87932db70'
OLD_MANIFEST = 'af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
REPAIR_MANIFEST = '3f1719bedac24844949ac05c1f42a07eb1987d45a69ddffd1d400f0b3162a880'
OWNER = dict(pid=1554337, uid=1001, starttick=687293806)
MARKER = 'signal-study-recovery-d96c7fc1e0f14ea2bf3edcb01a22bfa8'
SESSION = 'e1-signal-four-recovery-98b9326-20261010'
SERVER_PID = 1100668


class Hold(ValueError):
    pass


def require(ok, reason):
    if not ok: raise Hold(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path)
    require(not path.is_symlink(), 'Pinned input is a symlink')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Pinned input is not a regular file')
    with path.open('rb') as handle:
        raw = handle.read(); opened = os.fstat(handle.fileno())
    after = path.stat(); fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(opened, k) == getattr(after, k) for k in fields)
            and len(raw) == after.st_size, 'Pinned input changed during read')
    sha = digest(raw)
    require(expected is None or sha == expected, 'Pinned input SHA differs: ' + str(path))
    return raw, dict(path=str(path.resolve()), sha256=sha, bytes=len(raw))


def bound(reference):
    require(isinstance(reference, dict) and isinstance(reference.get('path'), str)
        and re.fullmatch('[0-9a-f]{64}', reference.get('sha256', '')), 'Explicit path/SHA required')
    raw, proof = read(reference['path'], reference['sha256'])
    if 'bytes' in reference: require(reference['bytes'] == proof['bytes'], 'Bound file byte count differs')
    return json.loads(raw), proof


def same_ref(left, right):
    return left.get('path') == right.get('path') and left.get('sha256') == right.get('sha256')


def same_process(left, right):
    return all(left.get(k) == right.get(k) for k in ('pid', 'uid', 'starttick'))


def tree(root, manifest_ref, publication_ref, expected_commit=None):
    root = Path(root).resolve(); manifest, mp = bound(manifest_ref); publication, pp = bound(publication_ref)
    require(Path(mp['path']) == root / 'source-manifest.json', 'Manifest is outside helper root')
    files = manifest.get('files'); require(isinstance(files, dict) and files, 'Empty source manifest')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'source-manifest.json'}
    require(actual == set(files), 'Frozen source inventory changed')
    proofs = []
    for name, row in sorted(files.items()):
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe manifest member')
        _, fp = read(root / rel, row['sha256'] if isinstance(row, dict) else row)
        if isinstance(row, dict): require(fp['bytes'] == row['bytes'], 'Source byte count differs')
        proofs.append(dict(relative_path=name, **fp))
    commit = publication.get('source_commit')
    require(publication.get('accepted') is True and publication.get('branch') == 'search-e1'
        and re.fullmatch('[0-9a-f]{40}', commit or '') and commit == publication.get('remote_readback_sha')
        and publication.get('source_manifest_sha256') == mp['sha256'], 'Source publication/readback not accepted')
    require(expected_commit is None or commit == expected_commit, 'Wrong source publication commit')
    prefix = 'experiments/search_e1/control/' + root.name + '/'
    rows = publication.get('files', []); indexed = {}
    for row in rows:
        # Original publications may inventory a bound helper by relative name;
        # newer publications also give the repository path. Both pin SHA/bytes.
        key = row.get('path')
        if key is None:
            name = row.get('name'); require(isinstance(name, str) and name, 'Publication source name/path missing')
            rel = Path(name); require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe publication source name')
            key = prefix + name
        elif row.get('name') is not None:
            require(key == prefix + row['name'], 'Publication source name/path disagree')
        require(key not in indexed, 'Duplicate publication source path')
        indexed[key] = row
    for fp in proofs + [dict(relative_path='source-manifest.json', **mp)]:
        row = indexed.get(prefix + fp['relative_path'])
        require(row is not None and row.get('sha256') == fp['sha256'] and row.get('bytes') == fp['bytes']
            and row.get('equal', True) is True, 'Published source byte readback differs')
    return dict(manifest=mp, publication=pp, source_commit=commit, files=proofs)


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        first = (root / 'stat').read_text(); tail = first[first.rfind(')') + 2:].split()
        status = (root / 'status').read_text()
        uids = next(row.split()[1:] for row in status.splitlines() if row.startswith('Uid:'))
        argv = [item.decode() for item in (root / 'cmdline').read_bytes().split(b'\0') if item]
        env = dict(item.decode().split('=', 1) for item in (root / 'environ').read_bytes().split(b'\0') if b'=' in item)
        last = (root / 'stat').read_text(); end = last[last.rfind(')') + 2:].split()
        last_uids = next(row.split()[1:] for row in (root / 'status').read_text().splitlines() if row.startswith('Uid:'))
    except (FileNotFoundError, ProcessLookupError):
        require(Path('/proc/self/stat').is_file() and not root.exists(), 'Opaque/raced process is not absent')
        return dict(pid=int(pid), status='ABSENT', absent_not_opaque=True)
    except PermissionError as error:
        raise Hold('Process identity is opaque: ' + str(pid)) from error
    require(tail[19] == end[19] and uids == last_uids, 'Process identity changed during read')
    return dict(pid=int(pid), uid=int(uids[0]), euid=int(uids[1]), starttick=int(tail[19]),
        ppid=int(tail[1]), pgid=int(tail[2]), state=tail[0], argv=argv, run_id=env.get('E1_RUN_ID'),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'), dont_write_bytecode=env.get('PYTHONDONTWRITEBYTECODE'),
        status='PRESENT')


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, 'Read-only command failed: ' + argv[0])
    return result.stdout.rstrip('\n')


def tmux_snapshot(request, socket_path=None):
    expected = request['tmux']; prefix = ['tmux'] + (['-S', socket_path] if socket_path else [])
    names = ['session_name', 'session_id', 'pane_id', 'pane_pid', 'pane_dead', 'pane_dead_status',
             'pane_start_command', 'pid', 'socket_path']
    def query():
        raw = command(prefix + ['display-message', '-p', '-t', expected['pane_id'],
                      '\x1f'.join('#{' + name + '}' for name in names)])
        values = raw.split('\x1f'); require(len(values) == len(names), 'Ambiguous tmux record')
        value = dict(zip(names, values))
        for key in ('session_name', 'session_id', 'pane_id', 'pane_start_command'):
            require(value[key] == expected['session' if key == 'session_name' else 'start_command' if key == 'pane_start_command' else key],
                    'Original tmux session/pane/command changed')
        require(int(value['pane_pid']) == expected['pane_pid'] and int(value['pid']) == expected['server_pid'],
                'Original pane/server PID changed')
        return value
    first = query()
    panes = command(prefix + ['list-panes', '-s', '-t', expected['session'], '-F', '#{pane_id}']).splitlines()
    require(panes == [expected['pane_id']], 'Study session is no longer exactly one retained foreground pane')
    retained = command(prefix + ['show-options', '-p', '-A', '-v', '-t', expected['pane_id'], 'remain-on-exit'])
    require(retained in ('on', '1'), 'Original pane is not retained after exit')
    second = query(); require(first == second, 'Tmux changed while capturing')
    first['remain_on_exit'] = True
    return first


def socket_identity(path):
    value = Path(path).stat(); require(stat.S_ISSOCK(value.st_mode), 'Original tmux socket is not a socket')
    return dict(path=str(Path(path).resolve()), device=value.st_dev, inode=value.st_ino, uid=value.st_uid,
                mode=value.st_mode, ctime_ns=value.st_ctime_ns)


def launch_contract(script_raw, expected_argv, marker):
    lines = script_raw.decode().splitlines()
    require(len(lines) == 7 and lines[:2] == ['#!/usr/bin/env bash', 'set -euo pipefail'], 'Unexpected launch script structure')
    expected = ["export CUDA_VISIBLE_DEVICES=''", 'export PYTHONDONTWRITEBYTECODE=1',
                'export PYTHONUNBUFFERED=1', 'export E1_RUN_ID=' + marker]
    require(lines[2:6] == expected and shlex.split(lines[6]) == ['exec'] + expected_argv,
            'Launch script is not one exact foreground exec with original CPU environment')
    require(not any(char in lines[6] for char in ('$', '`', ';', '|', '&', '\\')), 'Launch script has extra command or expansion')
    return dict(accepted=True, script_sha256=digest(script_raw), grammar='one-exec-fixed-cpu-env-v1', no_following_command=True)


def load_inputs(request_ref, own_manifest, own_publication):
    value, request_proof = bound(request_ref)
    require(value.get('schema') == 'opsd-signal-weight-terminal-request-v1' and value.get('accepted') is True,
            'Explicit terminal request required')
    require(value['owner'] == OWNER and value['supervisor_run_id'] == MARKER, 'Request targets another supervisor')
    tmux = value['tmux']
    require(tmux['session'] == SESSION and tmux['session_id'] == '$45' and tmux['pane_id'] == '%45'
        and tmux['pane_pid'] == OWNER['pid'] and tmux['server_pid'] == SERVER_PID, 'Request targets another tmux scope')
    study, sp = bound(value['study_manifest']); repair, rp = bound(value['recovery_manifest'])
    require(sp['sha256'] == STUDY_SHA and rp['sha256'] == REPAIR_SHA and Path(study['study_base']) == BASE,
            'Original study/recovery input differs')
    require(study['source_commit'] == OLD_COMMIT and study['helper_source_manifest']['sha256'] == OLD_MANIFEST
        and repair['source_commit'] == REPAIR_COMMIT and repair['source_manifest']['sha256'] == REPAIR_MANIFEST
        and repair['schema'] == 'opsd-signal-weight-recovery-manifest-v1' and repair['accepted'] is True
        and repair['scope'] == 'named_outer_old_fp32_before_setter_only'
        and repair['supervisor_run_id'] == MARKER and repair['study_id'] == study['study_id']
        and same_ref(repair['study_manifest'], sp), 'Original/recovery execution chain differs')
    require([group['variant'] for group in study['groups']] == VARIANTS and len(repair['original_configs']) == 4,
            'Original four-group order changed')
    configs = []
    for group, ref in zip(study['groups'], repair['original_configs']):
        cfg, cp = bound(group['config'])
        require(same_ref(ref, cp) and cfg['variant'] == group['variant'] and cfg['study_id'] == study['study_id']
            and cfg['source_commit'] == OLD_COMMIT and cfg['owner']['uid'] == 1001
            and cfg['owner']['physical_gpus'] == [4, 5, 6, 7]
            and cfg['helper_source_manifest']['sha256'] == OLD_MANIFEST,
            'Recovery no longer binds four original scientific configs')
        configs.append((cfg, cp))
    original = tree(study['helper_root'], study['helper_source_manifest'], study['publication'], OLD_COMMIT)
    recovery = tree(repair['helper_root'], repair['source_manifest'], repair['publication'], REPAIR_COMMIT)
    own = tree(Path(__file__).resolve().parent, own_manifest, own_publication)
    expected_argv = [configs[0][0]['paths']['python'], str(Path(repair['helper_root']) / 'entry.py'), '--mode', 'study',
        '--recovery-ref', rp['path'], '--recovery-sha256', rp['sha256'], '--', '--study-manifest', sp['path'],
        '--study-manifest-sha256', sp['sha256'], '--resume']
    raw, script = read(value['launch_script']['path'], value['launch_script']['sha256'])
    require(script['sha256'] == LAUNCH_SHA and Path(script['path']) == BASE / 'receipts/launch-recovery-98b9326.sh',
            'Original launch script differs')
    grammar = launch_contract(raw, expected_argv, MARKER)
    pub, pubref = bound(value['launch_input_publication'])
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
        and pub['source_commit'] == pub['remote_readback_sha'] and len(pub['source_commit']) == 40
        and pub['deployed_recovery_source_commit'] == REPAIR_COMMIT, 'Launch input publication not accepted')
    rows = {row['path']: row for row in pub['files']}
    prefix = 'experiments/search_e1/configs/signal-weight-recovery-20261010/'
    for proof in (script, rp):
        entry = rows.get(prefix + Path(proof['path']).name)
        require(entry and entry['sha256'] == proof['sha256'] and entry['bytes'] == proof['bytes'],
                'Launch/recovery input differs from published bytes')
    require(value['log_path'] == str(BASE / 'logs/recovery-98b9326.log'), 'Original foreground log differs')
    # The initial wrapper only sets retention and execs the exact verified script.
    start = "bash -lc 'tmux set-option -w remain-on-exit on; exec bash " + script['path'] + ' >' + value['log_path'] + " 2>&1'"
    require(tmux['start_command'] == start, 'Pane birth command has another foreground or trailing command')
    return value, dict(request=request_proof, study_manifest=sp, recovery_manifest=rp, launch_script=script,
        launch_input_publication=pubref, original_source=original, recovery_source=recovery, capture_source=own,
        expected_argv=expected_argv, launch_grammar=grammar), study, configs


def live(request, originals):
    pane = tmux_snapshot(request); owner = process(request['owner']['pid']); server = process(int(pane['pid']))
    socket = socket_identity(pane['socket_path'])
    require(pane['pane_dead'] == '0' and pane['pane_dead_status'] == '', 'Driver has no live bindable foreground')
    require(owner['status'] == 'PRESENT' and same_process(owner, request['owner']) and owner['state'] != 'Z'
        and owner['uid'] == owner['euid'] == socket['uid'] == server['uid'] == server['euid'] == 1001
        and owner['argv'] == originals['expected_argv'] and owner['run_id'] == request['supervisor_run_id']
        and owner['cuda_visible_devices'] == '' and owner['dont_write_bytecode'] == '1'
        and owner['ppid'] == server['pid'] and owner['pgid'] == owner['pid'], 'Live same-PID exec identity differs')
    require(server['status'] == 'PRESENT' and server['state'] != 'Z' and server['pid'] == SERVER_PID,
            'Original tmux server not live')
    stdout = str(Path(os.readlink('/proc/' + str(owner['pid']) + '/fd/1')).resolve())
    stderr = str(Path(os.readlink('/proc/' + str(owner['pid']) + '/fd/2')).resolve())
    require(stdout == stderr == request['log_path'], 'Actual foreground stdout/stderr is another log')
    require(same_process(process(owner['pid']), owner) and same_process(process(server['pid']), server),
            'Live owner/server changed during capture')
    return dict(schema='opsd-signal-weight-driver-live-binding-v1', accepted=True, mode='live', completion=False,
        observed_unix=time.time(), supervisor_run_id=request['supervisor_run_id'], originals=originals,
        owner=owner, server=server, socket=socket, tmux=pane, foreground_log=stdout,
        exec_same_pid=True, no_exit_code_claimed=True, no_signals=True, no_tmux_mutation=True,
        no_training_or_gpu_launch=True, torch_imported=False)


def absent_original(identity):
    actual = process(identity['pid'])
    require(actual['status'] == 'ABSENT' or not same_process(actual, identity), 'Original owner/child still actually present')
    return dict(identity=identity, current=actual, original_identity_released=True)


def verify_native_completion(reference, study, study_ref, configs, bound_owner):
    value, proof = bound(reference)
    require(value.get('accepted') is True and value.get('completed_groups') == 4
        and value.get('completed_outers') == value.get('full_training_em_rows') == 100
        and same_ref(value['study_manifest'], study_ref) and len(value['groups']) == 4
        and all(value['actual_cpu_supervisor'].get(key) == bound_owner.get(key)
                for key in ('pid', 'uid', 'euid', 'starttick', 'ppid', 'pgid', 'argv', 'run_id')),
            'Native four-group completion is not this driver/study')
    require([group['variant'] for group in value['groups']] == VARIANTS, 'Native four-group coverage/order differs')
    results = []
    for group, (config, cp) in zip(value['groups'], configs):
        require(group['actual_wait_returncode'] == 0 and group['release'].get('accepted') is True,
                'Native group actual controller wait/release is nonzero or unknown')
        actual, ap = bound(group['actual_terminal']); ledger, lp = bound(group['summary'])
        require(actual['actual_wait_returncode'] == 0 and actual['controller']['run_id'] == config['run_id']
            and actual['controller']['uid'] == 1001 and type(actual['controller']['starttick']) is int,
                'Bound group actual wait0 identity differs')
        release = group['release']['audit']
        require(release.get('accepted_no_owned_or_opaque') is True and not release.get('blockers', []),
                'Native group descendant release rejected')
        require(ledger.get('accepted') is True and ledger['variant'] == config['variant']
            and ledger['run_id'] == config['run_id'] and ledger['outer_iterations'] == 25
            and len(ledger['rows']) == 25, 'Bound TRAIN25 ledger belongs to another config or is partial')
        require([row['outer_index'] for row in ledger['rows']] == list(range(25))
            and all(row['run_id'] == config['run_id'] and row['config_sha256'] == cp['sha256']
                    for row in ledger['rows']), 'TRAIN25 row original run/config/order differs')
        results.append(dict(variant=config['variant'], config=cp, train_ledger=lp, actual_terminal=ap,
            actual_controller_wait_returncode=0, release=group['release'], controller_release=absent_original(actual['controller'])))
    return dict(accepted=True, native_completion=proof, groups=results, completed_groups=4,
                completed_outers=100, full_training_em_rows=100)


def gpu_inventory():
    def rows(kind, fields):
        raw = command(['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits'])
        return [[item.strip() for item in line.split(',')] for line in raw.splitlines() if line.strip()]
    devices = [dict(index=int(r[0]), uuid=r[1], actual_free_mib=int(r[2])) for r in rows('gpu', 'index,uuid,memory.free')]
    apps = [dict(gpu_uuid=r[0], pid=int(r[1]), used_mib=int(r[2])) for r in rows('compute-apps', 'gpu_uuid,pid,used_gpu_memory')]
    return devices, apps


def owned_release(configs, originals):
    config = configs[0][0]; devices, apps = gpu_inventory()
    selected = [d for d in devices if d['index'] in [4, 5, 6, 7]]
    require([d['index'] for d in selected] == [4, 5, 6, 7] and [d['uuid'] for d in selected] == config['owner']['gpu_uuids'],
            'Owned physical GPU/UUID mapping changed')
    admissions = [bound(c['gates']['retriever_admission'])[0] for c, _ in configs]
    keys = ('pid', 'uid', 'starttick', 'source_path', 'source_identity_sha256', 'retriever_url',
            'full_gpu_gate_path', 'full_gpu_gate_sha256')
    require(all(admission.get('accepted') is True and all(admission.get(key) == admissions[0].get(key) for key in keys)
                for admission in admissions), 'Groups no longer share the identical oracle identity/gate')
    admission = admissions[0]; oracle = process(admission['pid'])
    require(oracle['status'] == 'PRESENT' and same_process(oracle, admission) and oracle['uid'] == oracle['euid'] == 1001,
            'Approved GPU7 oracle identity changed')
    source = admission.get('source_path')
    scripts = [arg for arg in oracle['argv'] if arg.endswith('/retriever_stream.py')]
    require(len(scripts) == 1 and (not source or Path(source).resolve() == Path(scripts[0]).resolve()), 'Oracle entry changed')
    source_proof = read(scripts[0], admission['source_identity_sha256'])[1]
    owned_apps = [app for app in apps if app['gpu_uuid'] in config['owner']['gpu_uuids']]
    require(all(app['pid'] == oracle['pid'] and app['gpu_uuid'] == selected[-1]['uuid'] and app['used_mib'] <= 1024
                for app in owned_apps), 'Owned cards have a non-oracle GPU process')
    # A resource boundary only: foreign GPU0-3 are neither examined as owners nor touched.
    markers = {cfg['run_id'] for cfg, _ in configs} | {MARKER}
    helper_roots = [originals['original_source']['manifest']['path'].rsplit('/', 1)[0],
                    originals['recovery_source']['manifest']['path'].rsplit('/', 1)[0]]
    outputs = [cfg['paths']['output_root'] for cfg, _ in configs]
    checked = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid(): continue
        try:
            status = (path / 'status').read_text()
            uid = int(next(row.split()[1] for row in status.splitlines() if row.startswith('Uid:')))
        except (FileNotFoundError, ProcessLookupError): continue
        except PermissionError as error: raise Hold('Opaque process status in final scan') from error
        if uid != 1001: continue
        actual = process(int(path.name)); checked += 1
        if actual['status'] == 'ABSENT': continue
        belongs = actual['run_id'] in markers or any(arg == root or arg.startswith(root + '/')
            for arg in actual['argv'] for root in helper_roots + outputs)
        require(not belongs, 'Owned CPU/GPU descendant remains after driver exit')
    return dict(accepted=True, devices=selected, direct_memory_free=True, approved_oracle=oracle,
        oracle_source=source_proof, owned_gpu_applications=owned_apps, non_oracle_owned_gpu_processes=[],
        same_uid_processes_scanned=checked, no_owned_or_opaque_descendants=True, no_signals=True)


def terminal(request, originals, study, configs, binding_raw, binding_sha, native_ref):
    require(digest(binding_raw) == binding_sha, 'Exact live binding bytes SHA differs')
    binding = json.loads(binding_raw)
    require(binding.get('schema') == 'opsd-signal-weight-driver-live-binding-v1' and binding.get('accepted') is True
        and binding.get('mode') == 'live' and binding.get('completion') is False
        and binding['originals'] == originals and binding['supervisor_run_id'] == request['supervisor_run_id'],
            'Live binding belongs to another request/source/publication or is not live')
    require(same_process(binding['owner'], request['owner']) and binding['owner']['argv'] == originals['expected_argv']
        and binding['exec_same_pid'] is True, 'Live owner/source argv was not bound')
    pane = tmux_snapshot(request, binding['socket']['path']); server = process(int(pane['pid']))
    require(same_process(server, binding['server']) and server['status'] == 'PRESENT'
        and socket_identity(pane['socket_path']) == binding['socket'], 'Original tmux server/socket identity changed')
    for key in ('session_name', 'session_id', 'pane_id', 'pane_pid', 'pane_start_command', 'pid'):
        require(pane[key] == binding['tmux'][key], 'Retained pane identity differs from exact live binding')
    require(pane['pane_dead'] == '1' and pane['pane_dead_status'] == '0', 'Same retained foreground did not actually exit0')
    owner_release = absent_original(binding['owner'])
    native = verify_native_completion(native_ref, study, originals['study_manifest'], configs, binding['owner'])
    release = owned_release(configs, originals)
    require(tmux_snapshot(request, binding['socket']['path']) == pane, 'Pane changed during final completion/release checks')
    return dict(schema='opsd-signal-weight-driver-terminal-v1', accepted=True, mode='terminal',
        checked_unix=time.time(), supervisor_run_id=request['supervisor_run_id'], originals=originals,
        study_manifest=originals['study_manifest'], native_study_completion=native['native_completion'],
        live_binding_sha256=binding_sha, actual_supervisor_wait_returncode=0,
        driver_exit_evidence=dict(source='same-retained-tmux-pane', pane_dead=1, pane_dead_status=0,
            pane=pane, server=server, socket=binding['socket'], no_exit_code_inferred_from_absence=True),
        actual_cpu_supervisor=binding['owner'], owner_release=owner_release,
        completed_groups=4, completed_outers=100, full_training_em_rows=100, groups=native['groups'],
        owned_gpu_release=release, analysis_still_required=True, full_numeric_and_token_audit_not_claimed=True,
        no_signals=True, no_tmux_mutation=True, no_training_or_gpu_launch=True, torch_imported=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('live', 'terminal'), required=True)
    for name in ('request', 'source-manifest', 'publication'):
        parser.add_argument('--' + name, required=True); parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--binding'); parser.add_argument('--binding-sha256')
    parser.add_argument('--native-completion'); parser.add_argument('--native-completion-sha256')
    args = parser.parse_args(argv)
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Only independent UID1001 CPU capture is authorized')
    request, originals, study, configs = load_inputs(dict(path=args.request, sha256=args.request_sha256),
        dict(path=args.source_manifest, sha256=args.source_manifest_sha256), dict(path=args.publication, sha256=args.publication_sha256))
    if args.mode == 'live':
        require(not any((args.binding, args.binding_sha256, args.native_completion, args.native_completion_sha256)),
                'Live capture cannot include terminal inputs')
        result = live(request, originals)
    else:
        require(all((args.binding, args.binding_sha256, args.native_completion, args.native_completion_sha256)), 'Terminal requires live exact bytes and native completion')
        raw = sys.stdin.buffer.read() if args.binding == '-' else read(args.binding, args.binding_sha256)[0]
        result = terminal(request, originals, study, configs, raw, args.binding_sha256,
            dict(path=args.native_completion, sha256=args.native_completion_sha256))
    require('torch' not in sys.modules, 'Unexpected Torch import during CPU capture')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
