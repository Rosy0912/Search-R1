"""One-shot, read-only CPU provenance capture. Never infer rc from absence.

The terminal proof covers this exact foreground controller's return and owned
process release. It is not a 200-outer scientific audit or an effect evaluation,
and does not assign rc=0 to historical failed attempts. Output is private JSON
on stdout only. No shell string is executed and no tmux option is changed.
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
import types

sys.dont_write_bytecode = True
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CONFIG_SHA = 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
PLAN_SHA = '6b2af6ea4a463f82c81eb9b3dd84fb64f9d3c05ef3b92ee3b85fa024f6909945'
SHELL_SHA = '5e42402592cdb3aa8f3d28358db040116ca4104dbadec471821fbc10e2635a75'
ORIGINAL_SHA = '66cb43dd0830b9a7776bd644a0ef5bbf3d1661a3db3b159cc9e781db582c5233'
WAIT_SHA = 'baa072189d0213309d72dd9d04c67bdaf03361e7e7b452b137040d4bfee9a67f'
CPU_SHA = 'c349ab69d2939bf6bdd4e59eb32ae0023ec009796f85089a945d792b963d607e'
SCANNER_SHA = '2a66a863f7170035daea3d9a5f1d7a34106b12d12799f337021bf63747991e26'
COMMITS = ('7cf285308572af645676ae85a48fc8359f8c8ccb',
           '2cbcc1b4754eff51a52b213240579f5db0e10f87',
           '51eb6c574dd726b929dffcd6c0abfdc7b441a5d4')
PREFIX = 'experiments/search_e1/control/'
IDENTITY_KEYS = ('pid', 'uid', 'starttick')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def metadata(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, mode=s.st_mode)


def read_bound(path, expected_sha=None):
    path = Path(path).resolve(); before = metadata(path)
    require(stat.S_ISREG(before['mode']), 'Bound input is not a regular file')
    with path.open('rb') as handle:
        require(metadata(path)['inode'] == os.fstat(handle.fileno()).st_ino,
                'Input inode changed before read')
        raw = handle.read()
        fs = os.fstat(handle.fileno())
    after = metadata(path)
    require(after == before and fs.st_ino == before['inode'] and fs.st_dev == before['device']
            and fs.st_size == len(raw), 'Bound input changed while reading')
    sha = digest(raw)
    require(expected_sha is None or sha == expected_sha, 'Bound input SHA differs')
    return raw, dict(path=str(path), sha256=sha, **before)


def checked_json(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
            and re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')), 'Invalid path/SHA reference')
    raw, proof = read_bound(ref['path'], ref['sha256'])
    return json.loads(raw), proof


def verify_tree(root, manifest_ref, publication_ref, dirname, run, config_sha, expected_commit=None):
    root = Path(root).resolve(); manifest, mp = checked_json(manifest_ref)
    require(Path(mp['path']).parent == root, 'Source manifest outside source root')
    files = manifest.get('files')
    require(isinstance(files, dict) and files, 'Source inventory is missing')
    actual_names = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.name != 'source-manifest.json'}
    require(actual_names == set(files), 'Frozen source inventory differs')
    proofs = []
    for name, expected in sorted(files.items()):
        path = root / name
        require(path.resolve().is_relative_to(root), 'Frozen source escapes root')
        _, fp = read_bound(path, expected['sha256'])
        require(fp['bytes'] == expected['bytes'], 'Frozen source bytes differ')
        proofs.append(dict(relative_path=name, **fp))
    pub, pp = checked_json(publication_ref); commit = pub.get('source_commit', '')
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
            and re.fullmatch('[0-9a-f]{40}', commit) and pub.get('remote_readback_sha') == commit,
            'Source publication/readback is not accepted')
    require(expected_commit is None or commit == expected_commit, 'Wrong source commit')
    require(pub.get('source_manifest_sha256') == mp['sha256'], 'Published source manifest differs')
    # Original 7cf publication predates run binding; immutable config binds it.
    if dirname != 'opsd_online':
        require(pub.get('run_id') == run and pub.get('config_sha256') == config_sha,
                'Publication targets another run/config')
    rows = pub.get('files', []); indexed = {v['path']: v for v in rows}
    require(len(indexed) == len(rows), 'Duplicate publication readback paths')
    for fp in proofs + [dict(relative_path='source-manifest.json', **mp)]:
        row = indexed.get(PREFIX + dirname + '/' + fp['relative_path'])
        require(row is not None and row.get('equal') is True and row.get('sha256') == fp['sha256']
                and row.get('bytes') == fp['bytes'], 'Source is not equal to published remote file')
    return dict(manifest=mp, publication=pp, source_commit=commit, files=proofs)


def command_options(command):
    require(isinstance(command, list) and len(command) >= 4 and all(isinstance(v, str) for v in command)
            and command[1:4] == ['-B', command[2], 'parent'], 'Unexpected recovery command')
    result = {}; i = 4
    while i < len(command):
        key = command[i]; require(key.startswith('--') and key not in result, 'Ambiguous controller option')
        if key == '--resume':
            result[key] = True; i += 1
        else:
            require(i + 1 < len(command) and not command[i + 1].startswith('--'), 'Missing controller value')
            result[key] = command[i + 1]; i += 2
    require(result.get('--resume') is True, 'This capture targets the same-run recovery')
    return result


def parse_foreground(shell, launch):
    """Strictly parse, never execute, one env command with two fixed redirections."""
    require(not any(v in shell for v in ('$', '`', '\n', '\r', '\x00', '\\')),
            'Shell expansion/escape or multiline syntax is outside the foreground grammar')
    lexer = shlex.shlex(shell, posix=True, punctuation_chars=';&|<>()')
    lexer.whitespace_split = True; lexer.commenters = ''
    tokens = list(lexer)
    require(tokens and tokens.pop(0) == 'env', 'Expected one foreground env command')
    assignments = {}
    while tokens and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*=.*', tokens[0]):
        key, value = tokens.pop(0).split('=', 1)
        require(key not in assignments, 'Duplicate environment assignment')
        assignments[key] = value
    require(assignments == launch['environment'], 'Foreground numerical environment differs')
    expected = launch['command'] + ['>', launch['log'], '2', '>&', '1']
    require(tokens == expected, 'Not the exact single foreground Python command and fixed log redirections')
    require(digest(shell.encode()) == launch['shell_sha256'], 'Actual foreground command SHA differs')
    return dict(accepted=True, grammar='single-env-foreground-python-fixed-stdout-stderr-v1',
                shell_sha256=digest(shell.encode()), command_sha256=digest(json.dumps(launch['command']).encode()),
                environment_keys=sorted(assignments), log=launch['log'], no_following_command=True)


def compact(identity):
    return {k: identity[k] for k in ('pid', 'uid', 'starttick', 'ppid', 'pgid', 'state', 'comm',
                                    'argv_sha256', 'kernel_uids') if k in identity}


def same(left, right):
    return all(left.get(k) == right.get(k) for k in IDENTITY_KEYS)


def socket_identity(path):
    s = Path(path).stat()
    require(stat.S_ISSOCK(s.st_mode), 'tmux socket is missing or not a socket')
    return dict(path=str(Path(path).resolve()), device=s.st_dev, inode=s.st_ino, uid=s.st_uid,
                mode=s.st_mode, ctime_ns=s.st_ctime_ns)


def tmux_snapshot(session, pane, socket=None):
    require(re.fullmatch('[A-Za-z0-9_-]+', session) and re.fullmatch('%[0-9]+', pane), 'Unsafe tmux target')
    prefix = ['tmux'] + (['-S', socket] if socket else [])
    names = ['session_name', 'session_id', 'pane_id', 'pane_pid', 'pane_dead', 'pane_dead_status',
             'pane_start_command', 'pid', 'socket_path']
    def query():
        fmt = '\x1f'.join('#{' + n + '}' for n in names)
        raw = subprocess.check_output(prefix + ['display-message', '-p', '-t', pane, fmt], text=True, timeout=15)
        fields = raw.rstrip('\n').split('\x1f')
        require(len(fields) == len(names), 'Ambiguous tmux snapshot')
        value = dict(zip(names, fields))
        require(value['session_name'] == session and value['pane_id'] == pane, 'tmux session/pane differs')
        return value
    first = query()
    remain = subprocess.check_output(prefix + ['show-options', '-p', '-A', '-v', '-t', pane,
                                              'remain-on-exit'], text=True, timeout=15).strip()
    second = query(); require(first == second, 'tmux state changed during capture')
    first['remain_on_exit'] = remain
    return first


def load_scanner(tree):
    fp = next(v for v in tree['files'] if v['relative_path'] == 'owned_process_audit.py')
    raw, current = read_bound(fp['path'], SCANNER_SHA)
    require(current == {k: v for k, v in fp.items() if k != 'relative_path'}, 'Scanner changed after source audit')
    module = types.ModuleType('online_terminal_fresh_bound_scanner'); module.__file__ = fp['path']
    exec(compile(raw, fp['path'], 'exec'), module.__dict__)
    require(module.__file__ == fp['path'], 'Scanner module path differs')
    return module


def source_chain(config, launch, options, self_manifest, self_publication):
    require(config['run_id'] == RUN and launch['run_id'] == RUN and launch['config_sha256'] == CONFIG_SHA,
            'Wrong run/config identity')
    require(options.get('--config') == config['_config_path'] and options.get('--config-sha256') == CONFIG_SHA
            and options.get('--run-id') == RUN, 'Launch command does not bind immutable config')
    require(config['helper_source_manifest']['sha256'] == ORIGINAL_SHA
            and options.get('--source-manifest-sha256') == CPU_SHA
            and launch['source_manifest_sha256'] == CPU_SHA and launch['source_commit'] == COMMITS[2],
            'Three-layer source identity differs')
    original = verify_tree(config['paths']['helper_root'], config['helper_source_manifest'],
                           config['gates']['publication'], 'opsd_online', RUN, CONFIG_SHA, COMMITS[0])
    wait = verify_tree(options['--wait-root'], dict(path=str(Path(options['--wait-root']) / 'source-manifest.json'),
                                                 sha256=WAIT_SHA),
                       dict(path=options['--wait-publication'], sha256=options['--wait-publication-sha256']),
                       'opsd_online_wait_recovery', RUN, CONFIG_SHA, COMMITS[1])
    cpu = verify_tree(Path(options['--source-manifest']).parent,
                      dict(path=options['--source-manifest'], sha256=CPU_SHA),
                      dict(path=options['--publication'], sha256=options['--publication-sha256']),
                      'opsd_online_cpu_merge_recovery', RUN, CONFIG_SHA, COMMITS[2])
    require(str(Path(launch['command'][2]).resolve()) == str(Path(cpu['manifest']['path']).parent / 'recovery.py'),
            'Executed recovery is outside the bound source root')
    own = verify_tree(Path(__file__).parent, self_manifest, self_publication,
                      'opsd_online_results_v1', RUN, CONFIG_SHA)
    return dict(original=original, wait=wait, cpu_merge=cpu, capture=own)


def validate_live(config, launch, tmux, socket, scanner):
    owner = scanner.proc_snapshot(launch['owner']['pid'])
    require(same(owner, launch['owner']) and owner['uid'] == config['owner']['uid']
            and owner['state'] != 'Z' and owner['argv'] == launch['command'], 'Live controller identity/argv differs')
    env = scanner.proc_environment(owner['pid'])
    require(env.get('E1_RUN_ID') == RUN and env.get('CUDA_VISIBLE_DEVICES') == '', 'Controller run/CPU boundary differs')
    wrapper = scanner.proc_snapshot(int(tmux['pane_pid'])); server = scanner.proc_snapshot(int(tmux['pid']))
    require(wrapper['uid'] == owner['uid'] == server['uid'] == socket['uid'], 'Controller/wrapper/server/socket UID differs')
    require(owner['ppid'] == owner['pgid'] == wrapper['pid'] and wrapper['ppid'] == server['pid']
            and wrapper['pgid'] == wrapper['pid'] and wrapper['state'] != 'Z', 'Foreground parent/group chain differs')
    require(len(wrapper['argv']) == 3 and Path(wrapper['argv'][0]).name == 'bash'
            and wrapper['argv'][1] == '-c', 'Wrapper is not a single Bash -c command')
    grammar = parse_foreground(wrapper['argv'][2], launch)
    require(tmux['pane_start_command'] == wrapper['argv'][2]
            and tmux['pane_dead'] == '0' and tmux['pane_dead_status'] == ''
            and tmux['remain_on_exit'] == 'on', 'tmux live command/retain/status differs')
    for identity in (owner, wrapper, server):
        require(same(scanner.proc_snapshot(identity['pid']), identity), 'Process changed during live capture')
    return dict(controller=compact(owner), wrapper=compact(wrapper), server=compact(server), socket=socket,
                foreground=grammar, tmux=compact_tmux(tmux))


def compact_tmux(value):
    return {k: value[k] for k in ('session_name', 'session_id', 'pane_id', 'pane_pid', 'pane_dead',
                                  'pane_dead_status', 'pid', 'socket_path', 'remain_on_exit')} | {
        'pane_start_command_sha256': digest(value['pane_start_command'].encode())}


def history_inventory(config, launch, options):
    path = Path(config['paths']['output_root']) / 'child-history.jsonl'
    raw, proof = read_bound(path)
    require(raw.endswith(b'\n'), 'Child history has an incomplete final record')
    known = [launch['owner']]; phases = {}; rows = 0
    for line in raw.splitlines():
        value = json.loads(line); identity = value['identity']; phase = value['phase']
        require(phase in ('sampling-running', 'training-running', 'cpu-merge')
                and identity['run_id'] == RUN and identity['uid'] == config['owner']['uid']
                and all(type(identity[k]) is int for k in IDENTITY_KEYS), 'Child history has an unbound identity/phase')
        known.append({k: identity[k] for k in IDENTITY_KEYS}); rows += 1
        phases[phase] = phases.get(phase, 0) + 1
    # Preserve past failed/retired owners as ancestry anchors, never invent their rc.
    prior = []
    for key in ('original-terminal', 'previous-terminal'):
        value, fp = checked_json(dict(path=options['--' + key], sha256=options['--' + key + '-sha256']))
        require(value.get('accepted') is True and value.get('run_id') == RUN
                and value.get('config_sha256') == CONFIG_SHA, 'Historical owner proof differs')
        for identity in value.get('old_owned_processes', []):
            require(identity['uid'] == config['owner']['uid'], 'Historical owner UID differs')
            known.append({k: identity[k] for k in IDENTITY_KEYS})
        prior.append(fp)
    unique = {tuple(v[k] for k in IDENTITY_KEYS): {k: v[k] for k in IDENTITY_KEYS} for v in known}
    return list(unique.values()), dict(file=proof, rows=rows, phase_counts=phases, prior_owner_proofs=prior,
                                      scope='Identity/release inventory; historical return codes are not inferred')


def validate_terminal(config, launch, current, socket, binding, status, scanner, release):
    require(binding.get('schema') == 'opsd-online-controller-tmux-live-binding-v1'
            and binding.get('accepted') is True and binding.get('run_id') == RUN
            and binding.get('config_sha256') == CONFIG_SHA, 'Need an accepted same-run live binding')
    require(same(binding['controller'], launch['owner']), 'Live binding targets another controller')
    require(binding['foreground']['accepted'] is True and binding['foreground']['shell_sha256'] == SHELL_SHA
            and binding['foreground']['no_following_command'] is True, 'Live foreground grammar is unbound')
    before = binding['tmux']; now = compact_tmux(current)
    for key in ('session_name', 'session_id', 'pane_id', 'pane_pid', 'pid', 'socket_path',
                'remain_on_exit', 'pane_start_command_sha256'):
        require(now[key] == before[key], 'tmux pane/server/start command continuity is unknown')
    require(socket == binding['socket'], 'tmux socket identity changed')
    server = scanner.proc_snapshot(binding['server']['pid'])
    require(same(server, binding['server']) and server['state'] != 'Z', 'tmux server PID was reused or ended')
    require(current['pane_dead'] == '1' and current['pane_dead_status'] == '0'
            and current['remain_on_exit'] == 'on', 'Exact retained pane has not returned rc=0')
    for old in (binding['controller'], binding['wrapper']):
        try:
            scanner.proc_snapshot(old['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        raise ValueError('Old controller/wrapper PID is present or reused; terminal is unknown')
    require(status.get('schema') == 'opsd-online-controller-status-v1' and status.get('phase') == 'COMPLETE'
            and status.get('run_id') == RUN and status.get('config_sha256') == CONFIG_SHA
            and same(status['owner'], launch['owner']) and status.get('outer_iterations') == 200
            and type(status.get('actual_optimizer_updates')) is int and status['actual_optimizer_updates'] > 0
            and isinstance(status.get('latest_checkpoint'), dict), 'COMPLETE200 status is missing/unbound')
    require(release.get('accepted_no_owned_or_opaque') is True, 'Owned or opaque process remains')
    return dict(accepted=True, actual_exit_code=0, evidence='same-retained-tmux-pane-single-foreground-wrapper',
                controller=dict(binding['controller'], actual_exit_code=0), wrapper=binding['wrapper'],
                terminal_tmux=now, server=compact(server), socket=socket,
                actual_optimizer_updates=status['actual_optimizer_updates'], outer_iterations=200,
                latest_checkpoint=status['latest_checkpoint'], owned_release=release)


def capture(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES', '') == '' and os.environ.get('E1_RUN_ID') != RUN,
            'Capture must be CPU-only and must not inherit the training run environment')
    config, cp = checked_json(dict(path=args.config, sha256=args.config_sha256))
    require(cp['sha256'] == CONFIG_SHA, 'Unexpected immutable configuration')
    config['_config_path'] = args.config
    launch, lp = checked_json(dict(path=args.launch_proof, sha256=args.launch_proof_sha256))
    require(launch.get('accepted') is True and launch.get('shell_sha256') == SHELL_SHA, 'Launch proof is not accepted')
    _, plan = read_bound(args.plan, args.plan_sha256); require(plan['sha256'] == PLAN_SHA, 'Wrong published effect plan')
    options = command_options(launch['command'])
    sources = source_chain(config, launch, options, dict(path=args.source_manifest, sha256=args.source_manifest_sha256),
                           dict(path=args.publication, sha256=args.publication_sha256))
    scanner = load_scanner(sources['original'])
    binding = bp = None
    if args.mode == 'terminal':
        require(args.binding and args.binding_sha256, 'Terminal mode requires a prior live binding path/SHA')
        if args.binding == '-':
            raw = sys.stdin.buffer.read(); require(digest(raw) == args.binding_sha256, 'stdin binding SHA differs')
            binding = json.loads(raw); bp = dict(path='stdin', sha256=digest(raw), bytes=len(raw))
        else:
            binding, bp = checked_json(dict(path=args.binding, sha256=args.binding_sha256))
        require(binding['launch_proof']['sha256'] == lp['sha256'] and binding['config']['sha256'] == cp['sha256']
                and binding['plan']['sha256'] == plan['sha256'], 'Live binding launch/config/plan differs')
        for name, tree in sources.items():
            require(binding['sources'][name]['manifest']['sha256'] == tree['manifest']['sha256']
                    and binding['sources'][name]['publication']['sha256'] == tree['publication']['sha256'],
                    'Live binding source/publication chain differs')
    socket_path = binding['tmux']['socket_path'] if binding else None
    tmux = tmux_snapshot(args.session, args.pane, socket_path); socket = socket_identity(tmux['socket_path'])
    status, sp = checked_json_file(Path(config['paths']['output_root']) / 'status.json')
    require(status.get('run_id') == RUN and status.get('config_sha256') == CONFIG_SHA
            and same(status['owner'], launch['owner']), 'Current status owner/run differs')
    result = dict(run_id=RUN, config_sha256=CONFIG_SHA, config=cp, launch_proof=lp, plan=plan,
                  sources=sources, checked_unix=time.time(), status=sp,
                  no_node_writes=True, no_signals=True, no_options_changed=True, no_gpu_or_http=True)
    if args.mode == 'live':
        result.update(validate_live(config, launch, tmux, socket, scanner))
        result.update(schema='opsd-online-controller-tmux-live-binding-v1', accepted=True,
                      phase=status['phase'], no_exit_code_claimed=True)
    else:
        known, inventory = history_inventory(config, launch, options)
        self_identity = scanner.proc_snapshot(os.getpid())
        require(self_identity['uid'] == config['owner']['uid']
                and scanner.proc_environment(os.getpid()).get('E1_RUN_ID') != RUN, 'Capture ownership is ambiguous')
        roots = [str(Path(config['paths']['helper_root']) / n) for n in ('sampling.py', 'trainer.py')]
        roots += [config['paths']['output_root'], launch['command'][2]]
        release = scanner.audit_same_uid(RUN, launch['owner']['uid'], launch['owner']['starttick'], known, roots,
                    exclusions=scanner.load_exclusions(config.get('gates', {}).get('process_exclusions', [])),
                    ignored_exact=[self_identity], ancestry_anchors=[binding['controller'], binding['wrapper']])
        require(same(scanner.proc_snapshot(os.getpid()), self_identity), 'Capturer identity changed during release audit')
        result.update(validate_terminal(config, launch, tmux, socket, binding, status, scanner, release))
        result.update(schema='opsd-online-controller-terminal-v1', live_binding=bp, child_history=inventory,
                      scope='Exact current controller rc/release only; complete200 scientific audit and effect evaluation remain separate')
        for proof in (sp, inventory['file']):
            require(metadata(proof['path']) == {k: proof[k] for k in metadata(proof['path'])},
                    'Terminal status/child history changed during capture')
    require(tmux_snapshot(args.session, args.pane, socket_path) == tmux
            and socket_identity(tmux['socket_path']) == socket, 'tmux identity changed at capture end')
    # Status may advance in live mode; immutable source/config/launch/plan may not.
    for proof in [cp, lp, plan] + [p for tree in sources.values()
                                  for p in [tree['manifest'], tree['publication']] + tree['files']]:
        require(metadata(proof['path']) == {k: proof[k] for k in metadata(proof['path'])},
                'Bound provenance changed before capture ended')
    return result


def checked_json_file(path):
    raw, proof = read_bound(path)
    return json.loads(raw), proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('live', 'terminal'))
    for name in ('config', 'launch-proof', 'plan', 'source-manifest', 'publication'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--session', required=True); parser.add_argument('--pane', required=True)
    parser.add_argument('--binding'); parser.add_argument('--binding-sha256')
    args = parser.parse_args()
    try:
        result = capture(args); code = 0
    except Exception as error:
        result = dict(schema='opsd-online-controller-capture-unknown-v1', accepted=False, decision='UNKNOWN',
                      mode=args.mode, error_type=type(error).__name__, reason=str(error), checked_unix=time.time(),
                      no_exit_code_inferred=True, no_node_writes=True, no_signals=True, no_gpu_or_http=True)
        code = 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return code


if __name__ == '__main__':
    raise SystemExit(main())
