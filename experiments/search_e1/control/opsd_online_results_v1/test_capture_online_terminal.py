"""CPU fixtures only; no node, tmux mutation, GPU, HTTP, or live process work."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('capture_under_test', HERE / 'capture_online_terminal.py')
C = importlib.util.module_from_spec(SPEC)
exec(compile((HERE / 'capture_online_terminal.py').read_bytes(), str(HERE / 'capture_online_terminal.py'), 'exec'), C.__dict__)


class FakeScanner:
    def __init__(self, processes, environment=None):
        self.processes = processes; self.environment = environment or {}
        self.calls = []

    def proc_snapshot(self, pid):
        self.calls.append(pid)
        if pid not in self.processes:
            raise FileNotFoundError(pid)
        return copy.deepcopy(self.processes[pid])

    def proc_environment(self, pid):
        return self.environment.get(pid, {})


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.command = ['/env/bin/python', '-B', '/sources/cpu/recovery.py', 'parent',
                        '--config', '/base/receipts/config.json', '--config-sha256', C.CONFIG_SHA,
                        '--run-id', C.RUN, '--resume']
        self.env = {'CUDA_VISIBLE_DEVICES': '', 'E1_RUN_ID': C.RUN, 'PYTHONDONTWRITEBYTECODE': '1'}
        self.shell = 'env ' + ' '.join(k + '=' + v for k, v in self.env.items()) + ' ' + ' '.join(self.command) + ' > /base/logs/controller.log 2>&1'
        self.launch = dict(accepted=True, command=self.command, environment=self.env,
                           log='/base/logs/controller.log', shell_sha256=C.digest(self.shell.encode()),
                           owner=dict(pid=30, uid=1005, starttick=300), run_id=C.RUN, config_sha256=C.CONFIG_SHA)
        self.config = dict(run_id=C.RUN, owner={'uid': 1005},
                           paths={'base': str(self.root), 'output_root': str(self.root / 'runs')})
        def process(pid, tick, ppid, pgid, argv, comm):
            return dict(pid=pid, starttick=tick, uid=1005, ppid=ppid, pgid=pgid, argv=argv,
                        argv_sha256=C.digest(json.dumps(argv).encode()), comm=comm, state='S')
        self.procs = {
            30: process(30, 300, 20, 20, self.command, 'python'),
            20: process(20, 200, 10, 20, ['bash', '-c', self.shell], 'bash'),
            10: process(10, 100, 1, 10, ['tmux', 'new-session'], 'tmux: server')}
        self.tmux = dict(session_name='online-session', session_id='$6', pane_id='%70', pane_pid='20',
                         pane_dead='0', pane_dead_status='', pane_start_command=self.shell, pid='10',
                         socket_path='/tmp/tmux-1005/default', remain_on_exit='on')
        self.socket = dict(path='/tmp/tmux-1005/default', uid=1005, device=1, inode=2, mode=0o140600, ctime_ns=3)
        self.scanner = FakeScanner(self.procs, {30: self.env})
        self.binding = dict(schema='opsd-online-controller-tmux-live-binding-v1', accepted=True,
                            run_id=C.RUN, config_sha256=C.CONFIG_SHA)
        self.binding.update(C.validate_live(self.config, self.launch, self.tmux, self.socket, self.scanner))
        self.status = dict(schema='opsd-online-controller-status-v1', phase='COMPLETE', run_id=C.RUN,
                           config_sha256=C.CONFIG_SHA, owner=self.launch['owner'], outer_iterations=200,
                           actual_optimizer_updates=1990, latest_checkpoint={'path': '/cp/final', 'sha256': 'a' * 64})

    def tearDown(self):
        self.temp.cleanup()

    def terminal(self, **overrides):
        tmux = copy.deepcopy(self.tmux); tmux.update(pane_dead='1', pane_dead_status='0')
        scanner = FakeScanner({10: self.procs[10]})
        args = dict(config=self.config, launch=self.launch, current=tmux, socket=self.socket,
                    binding=self.binding, status=self.status, scanner=scanner,
                    release={'accepted_no_owned_or_opaque': True})
        args.update(overrides)
        with patch.object(C, 'SHELL_SHA', self.launch['shell_sha256']):
            return C.validate_terminal(**args)

    def write(self, name, value):
        path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=C.digest(raw))

    def test_exact_single_command_grammar(self):
        proof = C.parse_foreground(self.shell, self.launch)
        self.assertTrue(proof['accepted']); self.assertTrue(proof['no_following_command'])

    def test_shell_operators_and_expansions_rejected(self):
        for suffix in ['; echo done', ' && true', ' | cat', ' $(true)', ' `true`', '\necho bad', ' # extra', ' < /x']:
            shell = self.shell + suffix; launch = dict(self.launch, shell_sha256=C.digest(shell.encode()))
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                C.parse_foreground(shell, launch)

    def test_wrong_log_environment_and_command_rejected(self):
        variants = [self.shell.replace('controller.log', 'other.log'), self.shell.replace('E1_RUN_ID=', 'OTHER_RUN='),
                    self.shell.replace('/env/bin/python', '/other/python'), self.shell.replace(' 2>&1', '')]
        for shell in variants:
            with self.subTest(shell_hash=C.digest(shell.encode())), self.assertRaises(ValueError):
                C.parse_foreground(shell, dict(self.launch, shell_sha256=C.digest(shell.encode())))

    def test_live_parent_or_run_identity_rejected(self):
        for field, value in [('ppid', 99), ('pgid', 99), ('starttick', 999), ('uid', 0)]:
            procs = copy.deepcopy(self.procs); procs[30][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.validate_live(self.config, self.launch, self.tmux, self.socket, FakeScanner(procs, {30: self.env}))
        with self.assertRaises(ValueError):
            C.validate_live(self.config, self.launch, self.tmux, self.socket, FakeScanner(self.procs, {30: {'E1_RUN_ID': 'other'}}))

    def test_pane_must_retain_and_match_exact_foreground(self):
        for key, value in [('remain_on_exit', 'off'), ('pane_start_command', self.shell + ' '),
                           ('pane_dead', '1'), ('pane_dead_status', '0')]:
            tmux = dict(self.tmux); tmux[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_live(self.config, self.launch, tmux, self.socket, self.scanner)

    def test_terminal_accepts_dead_process_absence_but_requires_rc_zero(self):
        result = self.terminal()
        self.assertEqual(result['actual_exit_code'], 0)
        self.assertEqual(result['controller']['pid'], self.launch['owner']['pid'])
        self.assertEqual(result['controller']['actual_exit_code'], 0)
        self.assertEqual(result['actual_optimizer_updates'], 1990)

    def test_live_or_blank_dead_status_is_not_zero(self):
        for dead, rc in [('0', ''), ('1', ''), ('1', '1'), ('1', '143')]:
            tmux = dict(self.tmux, pane_dead=dead, pane_dead_status=rc)
            with self.subTest(dead=dead, rc=rc), self.assertRaises(ValueError):
                self.terminal(current=tmux)

    def test_pid_reuse_of_old_wrapper_is_unknown(self):
        for tick in (200, 201):
            procs = {10: self.procs[10], 20: dict(self.procs[20], starttick=tick)}
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                self.terminal(scanner=FakeScanner(procs))

    def test_server_reuse_or_missing_is_unknown(self):
        for procs in ({}, {10: dict(self.procs[10], starttick=101)}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.terminal(scanner=FakeScanner(procs))

    def test_socket_or_pane_reuse_is_unknown(self):
        with self.assertRaises(ValueError):
            self.terminal(socket=dict(self.socket, inode=3))
        for key, value in [('session_id', '$7'), ('pane_id', '%71'), ('pane_pid', '21'),
                           ('pane_start_command', self.shell + ' ')]:
            tmux = dict(self.tmux, pane_dead='1', pane_dead_status='0'); tmux[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.terminal(current=tmux)

    def test_complete_status_needs_all200_same_owner_and_actual_count(self):
        for key, value in [('phase', 'training-running'), ('outer_iterations', 199),
                           ('actual_optimizer_updates', 0), ('owner', dict(self.launch['owner'], starttick=301))]:
            status = dict(self.status); status[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.terminal(status=status)

    def test_opaque_or_owned_release_blocks_terminal(self):
        with self.assertRaises(ValueError):
            self.terminal(release={'accepted_no_owned_or_opaque': False, 'blockers': [{'reason': 'opaque'}]})

    def test_tmux_missing_cannot_turn_into_success(self):
        with patch.object(C.subprocess, 'check_output', side_effect=FileNotFoundError('tmux missing')):
            with self.assertRaises(FileNotFoundError):
                C.tmux_snapshot('online-session', '%70')

    def test_tmux_query_is_fixed_argv_not_shell(self):
        fields = [self.tmux[k] for k in ['session_name', 'session_id', 'pane_id', 'pane_pid', 'pane_dead',
                                        'pane_dead_status', 'pane_start_command', 'pid', 'socket_path']]
        responses = ['\x1f'.join(fields) + '\n', 'on\n', '\x1f'.join(fields) + '\n']
        with patch.object(C.subprocess, 'check_output', side_effect=responses) as call:
            self.assertEqual(C.tmux_snapshot('online-session', '%70'), self.tmux)
            for args, kwargs in call.call_args_list:
                self.assertIsInstance(args[0], list); self.assertNotIn('shell', kwargs)
                self.assertNotIn('set-option', args[0])

    def test_bound_json_hash_and_nonregular_rejected(self):
        ref = self.write('proof.json', {'x': 1}); self.assertEqual(C.checked_json(ref)[0]['x'], 1)
        Path(ref['path']).write_text('{"x":2}')
        with self.assertRaises(ValueError): C.checked_json(ref)
        with self.assertRaises(ValueError): C.read_bound(self.root)

    def test_socket_type_and_identity(self):
        path = self.root / 'socket'
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(path)); proof = C.socket_identity(path)
            self.assertEqual(proof['inode'], path.stat().st_ino)
        regular = self.root / 'regular'; regular.write_text('not a socket')
        with self.assertRaises(ValueError): C.socket_identity(regular)

    def test_source_tree_requires_remote_readback_equal_and_actual_bytes(self):
        source = self.root / 'source'; source.mkdir(); raw = b'value = 1\n'; (source / 'a.py').write_bytes(raw)
        manifest = self.write('source/source-manifest.json', {'files': {'a.py': {'sha256': C.digest(raw), 'bytes': len(raw)}}})
        pub = dict(accepted=True, branch='search-e1', source_commit='f' * 40, remote_readback_sha='f' * 40,
                   run_id=C.RUN, config_sha256=C.CONFIG_SHA, source_manifest_sha256=manifest['sha256'], files=[])
        for name in ('a.py', 'source-manifest.json'):
            data = (source / name).read_bytes(); pub['files'].append(dict(path=C.PREFIX + 'test/' + name,
                sha256=C.digest(data), bytes=len(data), equal=True))
        ref = self.write('pub.json', pub)
        proof = C.verify_tree(source, manifest, ref, 'test', C.RUN, C.CONFIG_SHA)
        self.assertEqual(len(proof['files']), 1)
        pub['remote_readback_sha'] = 'e' * 40; ref = self.write('pub.json', pub)
        with self.assertRaises(ValueError): C.verify_tree(source, manifest, ref, 'test', C.RUN, C.CONFIG_SHA)
        pub['remote_readback_sha'] = 'f' * 40; ref = self.write('pub.json', pub)
        (source / 'a.py').write_bytes(b'value = 2\n')
        with self.assertRaises(ValueError): C.verify_tree(source, manifest, ref, 'test', C.RUN, C.CONFIG_SHA)

    def test_fresh_scanner_compile_does_not_use_pyc_or_cached_module(self):
        raw = b'marker = 1\n'; ref = self.write('owned_process_audit.py', raw)
        _, proof = C.read_bound(ref['path']); tree = {'files': [dict(relative_path='owned_process_audit.py', **proof)]}
        with patch.object(C, 'SCANNER_SHA', ref['sha256']):
            first = C.load_scanner(tree); first.marker = 99
            self.assertEqual(C.load_scanner(tree).marker, 1)
            self.assertEqual(first.__file__, str(Path(ref['path']).resolve()))
        self.assertFalse((self.root / '__pycache__').exists())

    def test_history_is_inventory_not_inferred_allzero(self):
        records = [dict(phase='sampling-running', identity=dict(pid=40, uid=1005, starttick=400, run_id=C.RUN)),
                   dict(phase='cpu-merge', identity=dict(pid=41, uid=1005, starttick=401, run_id=C.RUN))]
        self.write('runs/child-history.jsonl', b''.join(json.dumps(v).encode() + b'\n' for v in records))
        options = {}
        for key, pid in [('original-terminal', 3), ('previous-terminal', 4)]:
            ref = self.write(key + '.json', dict(accepted=True, run_id=C.RUN, config_sha256=C.CONFIG_SHA,
                                               old_owned_processes=[dict(pid=pid, uid=1005, starttick=pid * 10)],
                                               actual_exit_code=1))
            options['--' + key] = ref['path']; options['--' + key + '-sha256'] = ref['sha256']
        known, proof = C.history_inventory(self.config, self.launch, options)
        self.assertEqual(proof['rows'], 2); self.assertEqual(len(known), 5)
        self.assertIn('not inferred', proof['scope']); self.assertNotIn('all_returncodes_zero', proof)
        path = self.root / 'runs/child-history.jsonl'; path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaises(ValueError): C.history_inventory(self.config, self.launch, options)

    def test_receipts_history_does_not_replace_actual_runs_history(self):
        # Native Controller.spawn writes self.output/'child-history.jsonl'.
        # An unrelated valid-looking file under base/receipts is not its output.
        record = dict(phase='sampling-running', identity=dict(pid=40, uid=1005, starttick=400, run_id=C.RUN))
        self.write('receipts/child-history.jsonl', json.dumps(record).encode() + b'\n')
        with self.assertRaises(FileNotFoundError):
            C.history_inventory(self.config, self.launch, {})

    def test_duplicate_launch_option_rejected(self):
        self.assertTrue(C.command_options(self.command)['--resume'])
        with self.assertRaises(ValueError): C.command_options(self.command + ['--resume'])


if __name__ == '__main__':
    unittest.main()
