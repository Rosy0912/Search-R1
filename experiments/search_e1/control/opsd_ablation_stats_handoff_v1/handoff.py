"""Mac read-only wait, then one published CPU bootstrap of the stats study.

The bootstrap takes no training lease, sends no signals, and never imports a
model. Its new study child retains the published native resource admission.
"""
import argparse
import base64
import copy
import hashlib
import importlib.abc
import importlib.util
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
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
NAMESPACE = 'experiments/search_e1/control/opsd_ablation_stats_handoff_v1/'
STATS_NAMESPACE = 'experiments/search_e1/control/opsd_ablation_eval_stats_recovery_v2/'
OLD_NAMESPACE = 'experiments/search_e1/control/opsd_ablation_eval_dtype_recovery_v1/'


class NotReady(ValueError):
    pass


def require(value, reason):
    if not value:
        raise NotReady(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, sha=None):
    path = Path(path)
    require(not path.is_symlink(), 'Input is a symlink')
    before = path.stat()
    with path.open('rb') as handle:
        require(os.fstat(handle.fileno()) == before, 'Input changed before read')
        raw = handle.read()
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) == getattr(after_fd, k) for k in keys), 'Input changed while reading')
    require(sha is None or digest(raw) == sha, 'Input SHA differs')
    return raw, dict(path=str(path.resolve()), sha256=digest(raw), bytes=len(raw))


def checked(ref):
    require(isinstance(ref, dict) and re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '') or ''), 'Missing bound path/SHA')
    raw, fp = read(ref['path'], ref['sha256'])
    return json.loads(raw), fp


def save(path, value):
    """Never replace a different existing artifact, including failed evidence."""
    path = Path(path); raw = canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        existing, fp = read(path)
        require(existing == raw, 'Existing independent output differs; preserve it')
        return fp
    with path.open('xb') as handle:
        os.chmod(path, 0o600); handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return read(path)[1]


def published(root, manifest_ref, publication_ref, commit, prefix):
    root = Path(root).resolve(); manifest, mp = checked(manifest_ref); pub, pp = checked(publication_ref)
    require(Path(mp['path']) == root / 'source-manifest.json', 'Manifest outside independent published source')
    require(re.fullmatch('[0-9a-f]{40}', commit or '') and pub.get('accepted') is True
            and pub.get('branch') == 'search-e1' and pub.get('source_commit') == pub.get('remote_readback_sha') == commit
            and pub.get('source_manifest_sha256') == mp['sha256'], 'Publication is not actual exact remote readback')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'source-manifest.json'}
    require(actual == set(manifest['files']), 'Published source inventory differs')
    for name, row in dict(manifest['files'], **{'source-manifest.json': mp}).items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe published source path')
        _, fp = read(root / name, row['sha256'])
        matches = [x for x in pub['files'] if x['path'] == prefix + name]
        require(fp['bytes'] == row['bytes'] and len(matches) == 1 and matches[0].get('equal') is True
                and matches[0]['sha256'] == fp['sha256'] and matches[0]['bytes'] == fp['bytes'], 'Source was not published byte-identically')
    return manifest, pub, dict(manifest=mp, publication=pp, source_commit=commit)


def process(pid):
    root = Path('/proc') / str(pid)
    raw = (root / 'stat').read_text(); tail = raw[raw.rindex(')') + 2:].split()
    status = (root / 'status').read_text().splitlines()
    uid = int(next(x for x in status if x.startswith('Uid:')).split()[2])
    argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
    environ = dict(x.split('=', 1) for x in (root / 'environ').read_bytes().decode().split('\0') if '=' in x)
    return dict(pid=int(pid), uid=uid, euid=uid, starttick=int(tail[19]), ppid=int(tail[1]), pgid=int(tail[2]),
                argv=argv, run_id=environ.get('E1_RUN_ID'), state=tail[0],
                cuda_visible_devices=environ.get('CUDA_VISIBLE_DEVICES'), dont_write_bytecode=environ.get('PYTHONDONTWRITEBYTECODE'))


def exact(actual, expected):
    return all(actual.get(k) == expected.get(k) for k in ('pid', 'uid', 'starttick'))


def gone(identity):
    try: actual = process(identity['pid'])
    except (FileNotFoundError, ProcessLookupError): return True
    require(exact(actual, identity), 'Old PID reused: terminal identity is UNKNOWN')
    raise NotReady('Exact old foreground process has not disappeared')


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.rstrip('\n')


def foreground(shell, start, old, python):
    tokens = shlex.split(shell)
    if tokens[:1] == ['exec']: tokens = tokens[1:]
    argv = start['owner']['argv']; prefix = ['env', 'CUDA_VISIBLE_DEVICES=', 'E1_RUN_ID=' + old['run_id'], 'PYTHONDONTWRITEBYTECODE=1']
    require(len(tokens) == len(prefix) + len(argv) + 3 and tokens[:4] == prefix
            and tokens[4:-3] == argv and tokens[-3] in ('>', '>>') and tokens[-1] == '2>&1', 'Not a single exact foreground command')
    require(argv == [python, '-B', str(Path(old['helper_root']) / 'study_runner.py'), '--study-manifest',
                     start['input_manifest']['path'], '--study-manifest-sha256', start['input_manifest']['sha256']], 'Old driver argv differs')
    require(Path(tokens[-2]).is_absolute(), 'Foreground log is not an absolute path')
    return tokens[-2]


def exec_same_pid(common, actual, start, old, source, python):
    """A literal bash pane command is accepted only with live exec proof."""
    require(common['pane_start_command'] == 'bash' and exact(actual, start['owner'])
            and actual['pid'] == common['pane_pid'] == start['owner']['pid']
            and actual['argv'] == start['owner']['argv'] and actual['run_id'] == old['run_id']
            and actual['uid'] == common['server']['uid'] and actual['ppid'] == common['server']['pid']
            and actual['pgid'] == common['pane_pid'] and actual['cuda_visible_devices'] == ''
            and actual['dont_write_bytecode'] == '1', 'Literal bash pane lacks exact live Python exec identity')
    require(actual['argv'] == [python, '-B', str(Path(old['helper_root']) / 'study_runner.py'),
            '--study-manifest', common['input_manifest']['path'], '--study-manifest-sha256', common['input_manifest']['sha256']],
            'Exec same PID is not the exact published study argv')
    return dict(common, schema='opsd-ablation-study-live-foreground-binding-v1', accepted=True, route='exec_same_pid',
                actual_owner=actual, source=source, helper_source_manifest=old['helper_source_manifest'],
                cuda_visible_devices='', pythondontwritebytecode='1', stdout_fd1=common['foreground_log'],
                pane_live_record=common['session_id']+' '+common['pane_id']+' '+str(common['pane_pid'])+' 0')


def same_socket(a, b):
    return Path(a['path']).resolve() == Path(b['path']).resolve() and all(a[k] == b[k] for k in ('device','inode','uid','mode'))


def inspect(plan, binding=None):
    """Only read files/proc/tmux; live or missing evidence never launches."""
    if binding is None and plan.get('live_foreground_binding'):
        binding, _ = checked(plan['live_foreground_binding'])
        require(binding.get('schema') == 'opsd-ablation-study-live-foreground-binding-v1' and binding.get('accepted') is True
                and binding.get('route') == 'exec_same_pid', 'Supplied live proof is not the independently accepted exact exec route')
    old, oldref = checked(plan['old_study']); start, sp = checked(read(Path(old['output_root']) / 'driver-start.json')[1])
    require(start['input_manifest']['path'] == oldref['path'] and start['input_manifest']['sha256'] == oldref['sha256']
            and start['owner']['run_id'] == old['run_id'], 'Old driver start/input differs')
    train, tf = checked(old['groups'][0]['training_config'])
    require(train['variant'] == 'C0' and train['scientific']['outer_iterations'] == 10, 'Wrong C0 scope')
    target = plan['old_session'] + ':'
    lines = command(['tmux', 'list-panes', '-t', target, '-F', '#{session_id} #{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}']).splitlines()
    require(len(lines) == 1, 'Old study session does not have exactly one retained pane')
    fields = lines[0].split(); require(len(fields) in (4, 5) and fields[1] == plan['old_pane'], 'Old pane identity differs')
    pane_pid, dead = int(fields[2]), int(fields[3])
    shell = command(['tmux', 'display-message', '-p', '-t', plan['old_pane'], '#{pane_start_command}'])
    log = binding.get('foreground_log') if shell == 'bash' and binding else (None if shell == 'bash' else foreground(shell, start, old, train['paths']['python']))
    server_pid = int(command(['tmux', 'display-message', '-p', '-t', plan['old_pane'], '#{pid}']))
    server = process(server_pid)
    socket = Path(command(['tmux', 'display-message', '-p', '-t', plan['old_pane'], '#{socket_path}']))
    st = socket.stat(); require(stat.S_ISSOCK(st.st_mode), 'tmux endpoint is not a socket')
    sock = dict(path=str(socket.resolve()), device=st.st_dev, inode=st.st_ino, uid=st.st_uid, mode=st.st_mode)
    common = dict(schema='opsd-ablation-stats-handoff-live-binding-v1', driver_start=sp, input_manifest=oldref,
                  session=plan['old_session'], session_id=fields[0], pane_id=fields[1], pane_pid=pane_pid,
                  owner=start['owner'], server=server, socket=sock, pane_start_command=shell, foreground_log=log)
    if not dead:
        owner = process(start['owner']['pid']); wrapper = process(pane_pid)
        if shell == 'bash':
            inventory, _, _ = published(old['helper_root'], old['helper_source_manifest'], old['publication'], old['source_commit'], OLD_NAMESPACE)
            source = read(Path(old['helper_root'])/'study_runner.py', inventory['files']['study_runner.py']['sha256'])[1]
            actual_log = os.readlink('/proc/'+str(owner['pid'])+'/fd/1')
            require(Path(actual_log).is_absolute() and Path(actual_log).is_file()
                    and Path(actual_log).resolve().is_relative_to(Path(oldref['path']).parent), 'Live exec stdout is not the original fixed input log')
            common['foreground_log'] = str(Path(actual_log).resolve())
            current = exec_same_pid(common, owner, start, old, source, train['paths']['python'])
            if binding:
                require(binding['route'] == 'exec_same_pid' and all(current[k] == binding[k] for k in
                    ('driver_start','input_manifest','session','session_id','pane_id','pane_pid','owner',
                     'pane_start_command','foreground_log','source','helper_source_manifest')), 'Live exec binding changed')
                require(same_socket(current['socket'],binding['socket']) and exact(current['server'], binding['server'])
                        and exact(current['actual_owner'], binding['actual_owner']), 'Live exec/server/socket identity changed')
            return dict(ready=False, state='WAITING_NATURAL_PARENT_EXIT', binding=binding or current)
        require(exact(owner, start['owner']) and owner['argv'] == start['owner']['argv']
                and owner['run_id'] == old['run_id'] and owner['ppid'] == owner['pgid'] == pane_pid, 'Live driver/wrapper chain differs')
        require(wrapper['uid'] == owner['uid'] == server['uid'] and len(wrapper['argv']) == 3
                and Path(wrapper['argv'][0]).name == 'bash' and wrapper['argv'][1:] == ['-c', shell], 'Live wrapper is not exact foreground Bash')
        common['wrapper'] = wrapper
        if binding:
            for key in ('driver_start', 'input_manifest', 'session', 'session_id', 'pane_id', 'pane_pid', 'owner', 'pane_start_command', 'foreground_log'):
                require(common[key] == binding[key], 'Live terminal binding changed')
            require(same_socket(common['socket'],binding['socket']) and exact(server, binding['server']) and exact(wrapper, binding['wrapper']), 'tmux server/socket/wrapper changed')
        return dict(ready=False, state='WAITING_NATURAL_PARENT_EXIT', binding=binding or common)
    require(binding is not None, 'Dead pane has no prior verified live binding; UNKNOWN')
    for key in ('driver_start', 'input_manifest', 'session', 'session_id', 'pane_id', 'pane_pid', 'owner', 'pane_start_command', 'foreground_log'):
        require(common[key] == binding[key], 'Retained pane identity changed after live observation')
    require(same_socket(common['socket'],binding['socket']) and exact(server, binding['server']) and len(fields) == 5 and fields[-1] == '2', 'Old parent did not naturally exit2 on the same server/socket/pane')
    gone(start['owner'])
    if binding.get('route') == 'exec_same_pid':
        require(shell == 'bash' and binding['accepted'] is True and binding['actual_owner']['pid'] == pane_pid,
                'Stored exec binding does not prove this exact pane process')
        inventory, _, _ = published(old['helper_root'], old['helper_source_manifest'], old['publication'], old['source_commit'], OLD_NAMESPACE)
        require(read(Path(old['helper_root'])/'study_runner.py',inventory['files']['study_runner.py']['sha256'])[1] == binding['source'], 'Old live-bound source changed')
    else: gone(binding['wrapper'])
    status, statusref = checked(read(Path(train['paths']['output_root']) / 'status.json')[1])
    terminal, terminalref = checked(read(Path(old['output_root']) / '00-C0/training/group-terminal.json')[1])
    checkpoint = status.get('latest_checkpoint')
    require(status.get('phase') == 'TRAIN_COMPLETE' and status.get('outer_iterations') == 10
            and status.get('run_id') == train['run_id'] and status.get('config_sha256') == tf['sha256']
            and terminal.get('accepted') is True and terminal.get('actual_exit_code') == 0
            and terminal.get('signal_termination_not_exit0') is False and terminal.get('run_id') == train['run_id']
            and terminal.get('config_sha256') == tf['sha256'] and terminal.get('controller') == status.get('owner')
            and terminal['status']['path'] == statusref['path'] and terminal['status']['sha256'] == statusref['sha256'], 'C0 actual tenth-outer terminal0 not ready')
    exit_value, _ = checked(terminal['actual_child_exit']); release, _ = checked(terminal['actual_release'])
    require(exit_value['actual_wait_returncode'] == 0 and exit_value['identity'] == terminal['controller']
            and exit_value['run_id'] == train['run_id'] and release['accepted'] is True, 'C0 wait0/release proof missing')
    cp, _ = checked(checkpoint)
    require(cp['run_id'] == train['run_id'] and cp['config_sha256'] == tf['sha256']
            and cp['cursor']['outer_index'] == 9 and cp['accepted'] is True, 'Wrong raw tenth-outer checkpoint')
    parent = dict(schema='opsd-ablation-study-actual-terminal-v1', accepted=True, actual_exit_code=2,
                  signal_termination_not_exit0=False, owner=start['owner'], driver_start=sp, input_manifest=oldref,
                  session=plan['old_session'], pane_dead_record=lines[0], pane_start_command=shell, foreground_log=log,
                  continuity=binding, observed_unix=time.time(), no_signals=True)
    return dict(ready=True, state='OLD_PARENT_RC2_AND_C0_TEN_RC0', binding=binding, parent=parent,
                driver_start=sp, training_config=tf, training_status=statusref, training_terminal=terminalref,
                raw10_checkpoint=checkpoint, old=old)


class SourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root, inventory): self.root, self.inventory = Path(root), inventory
    def find_spec(self, name, path=None, target=None):
        if name + '.py' in self.inventory: return importlib.util.spec_from_loader(name, self)
    def create_module(self, spec): return None
    def exec_module(self, module):
        name = module.__name__ + '.py'; raw, fp = read(self.root / name, self.inventory[name]['sha256'])
        module.__file__ = fp['path']; exec(compile(raw, fp['path'], 'exec'), module.__dict__)


def load_stats(plan):
    source = plan['new_source']; inventory, _, proof = published(source['helper_root'], source['manifest'], source['publication'], source['commit'], STATS_NAMESPACE)
    for name in inventory['files']:
        if name.endswith('.py'): require(name[:-3] not in sys.modules, 'Preloaded stats module is not a fresh source-only import')
    sys.meta_path.insert(0, SourceLoader(source['helper_root'], inventory['files']))
    import study_runner
    import completed_c0
    return study_runner, completed_c0, proof


def build_inputs(plan, observation):
    """Only after true terminal: independent JSON inputs, original science intact."""
    require(observation.get('ready') is True, 'No inputs/start before actual terminal')
    old = observation['old']; source = plan['new_source']; ids = plan['run_ids']; dest = Path(plan['destination'])
    root = Path(plan['study_root']).resolve()
    require(re.fullmatch('[0-9a-f]{32}', ids['driver']) and len(set(ids.values())) == 9
            and set(ids) == set(VARIANTS) | {'driver', 'zero'} and all(re.fullmatch('[0-9a-f]{32}', x) for x in ids.values()), 'Explicit nine distinct new run IDs required')
    paths = [dest, Path(plan['output_root'])] + [Path(x) for x in plan['eval_bases'].values()]
    require(set(plan['eval_bases']) == set(VARIANTS) | {'zero'} and all(p.resolve().is_relative_to(root) and p.resolve() != root for p in paths), 'New inputs/outputs escape this HKUST study')
    output_roots = [Path(plan['output_root'])] + [Path(x) / 'runs' for x in plan['eval_bases'].values()]
    require(not any(p.exists() for p in output_roots) and len(set(output_roots)) == 9, 'New output already exists; no duplicate or partial resume')
    old_roots = [Path(old['output_root']).resolve()]
    for item in old['groups']:
        for ref in (item['training_config'], item['evaluation_template']):
            value, _ = checked(ref); old_roots.append(Path(value['paths']['output_root']).resolve())
    zero_old, _ = checked(old['common_zero_template']); old_roots.append(Path(zero_old['paths']['output_root']).resolve())
    require(not any(a.resolve().is_relative_to(b) or b.is_relative_to(a.resolve())
                    for a in output_roots for b in old_roots), 'New output aliases/nests protected old training/evaluation output')
    manifest, pub, _ = published(source['helper_root'], source['manifest'], source['publication'], source['commit'], STATS_NAMESPACE)
    parent = copy.deepcopy(observation['parent'])
    live_ref = None
    if observation.get('binding',{}).get('route') == 'exec_same_pid':
        live_ref = save(dest / 'previous-driver-live-foreground.json', observation['binding'])
        parent['live_foreground_binding'] = live_ref
    parentref = save(dest / 'previous-driver-terminal.json', parent)
    groups = []; new_refs = {}
    for name in list(VARIANTS) + ['zero']:
        template_ref = old['common_zero_template'] if name == 'zero' else next(x['evaluation_template'] for x in old['groups'] if x['variant'] == name)
        original, original_ref = checked(template_ref); value = copy.deepcopy(original)
        require(value['training']['terminal'] is None and value['training']['raw10_checkpoint'] is None, 'Original template was already bound')
        base = str(Path(plan['eval_bases'][name]).resolve()); run_id = ids[name]
        value.update(run_id=run_id, source_commit=source['commit'], helper_source_manifest=source['manifest'], state='READY template: actual training terminal still bound only by study driver')
        value['paths'].update(base=base, output_root=str(Path(base) / 'runs'), helper_root=source['helper_root'])
        derived_pub = dict(pub, run_id=run_id, derived_from=source['publication'])
        value['gates']['publication'] = save(dest / ('publication-' + name + '.json'), derived_pub)
        storage, storage_ref = checked(original['gates']['storage'])
        require(storage['accepted'] is True and storage['cluster'] == 'HKUST' and storage['mode'] == 'human_hkust_local_exception'
                and storage['human_authorization']['explicit_exception'] is True, 'Explicit seven-group local output authorization missing')
        storage = copy.deepcopy(storage); storage.update(run_id=run_id, output_root=value['paths']['output_root'], derived_from=storage_ref)
        storage['scope'].update(run_id=run_id, output_root=value['paths']['output_root'])
        value['gates']['storage'] = save(dest / ('storage-' + name + '.json'), storage)
        new_refs[name] = save(dest / ('template-' + name + '.json'), value)
        if name != 'zero': groups.append(dict(variant=name, training_config=next(x['training_config'] for x in old['groups'] if x['variant'] == name), evaluation_template=new_refs[name]))
    driver_pub = save(dest / 'publication-driver.json', dict(pub, run_id=ids['driver'], derived_from=source['publication']))
    new = copy.deepcopy(old); new.pop('recovery', None)
    new.update(state='READY', run_id=ids['driver'], helper_root=source['helper_root'], helper_source_manifest=source['manifest'],
               source_commit=source['commit'], publication=driver_pub, output_root=str(Path(plan['output_root']).resolve()),
               groups=groups, common_zero_template=new_refs['zero'])
    new['completed_c0'] = dict(schema='opsd-ablation-completed-c0-eval-recovery-v1', training_config=observation['training_config'],
        completed_training_terminal=observation['training_terminal'], training_status=observation['training_status'],
        raw10_checkpoint=observation['raw10_checkpoint'], previous_driver_start=observation['driver_start'],
        previous_study_manifest=plan['old_study'], previous_driver_terminal=parentref,
        previous_session=plan['old_session'], expected_dead_pane=observation['parent']['pane_dead_record'])
    if live_ref: new['completed_c0']['live_foreground_binding'] = live_ref
    return new, save(dest / 'study-manifest.json', new)


def finalize(plan, binding, inspector=inspect, loader=load_stats, builder=build_inputs, launcher=None):
    observation = inspector(plan, binding)
    require(observation.get('ready') is True, 'Current parent/C0 terminal is not ready; no launch')
    runner, completed, _ = loader(plan)
    new, ref = builder(plan, observation)
    runner.validate_manifest(new); completed.preflight(new); completed.check_dead_parent(new, process)
    for item in new['groups']:
        value, _ = checked(item['evaluation_template']); runner.gates.publication(value); runner.gates.storage(value)
    zero, _ = checked(new['common_zero_template']); runner.gates.publication(zero); runner.gates.storage(zero)
    require(inspector(plan, binding).get('ready') is True, 'Terminal changed immediately before spawn')
    if launcher is not None: return launcher(new, ref)
    return launch(plan,new,ref,zero)


def launch(plan,new,ref,zero):
    """Set only the newly owned window to retain terminal status before exec."""
    session = plan['new_session']; require(re.fullmatch('[A-Za-z0-9_-]+', session), 'Unsafe new session name')
    probe = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True)
    require(probe.returncode == 1, 'Existing or uncertain new tmux session; never duplicate')
    receipt_path = Path(plan['destination']) / 'handoff-launch.json'
    require(not receipt_path.exists(), 'Launch already recorded; do not restart')
    argv = [zero['paths']['python'], '-B', str(Path(new['helper_root']) / 'study_runner.py'), '--study-manifest', ref['path'], '--study-manifest-sha256', ref['sha256']]
    log = str(Path(plan['destination']) / 'study-driver.log')
    shell = 'exec ' + shlex.join(['env', 'CUDA_VISIBLE_DEVICES=', 'E1_RUN_ID=' + new['run_id'], 'PYTHONDONTWRITEBYTECODE=1'] + argv) + ' > ' + shlex.quote(log) + ' 2>&1'
    intent = dict(schema='opsd-ablation-stats-handoff-launch-intent-v1', run_id=new['run_id'], input_manifest=ref,
                  session=session, command=shell, source=plan['new_source'], no_signals=True, no_training_locks=True,
                  bootstrap_owner=process(os.getpid()), bootstrap_no_training_marker=os.environ.get('E1_RUN_ID') is None,
                  bootstrap_scope='Independent readable CPU process; not known/run/training-source/output argv or phase-child descendant')
    save(Path(plan['destination']) / 'launch-intent.json', intent)
    result = subprocess.run(['tmux','new-session','-d','-s',session,'-P','-F','#{session_id} #{window_id} #{pane_id} #{pane_pid}','bash'],capture_output=True,text=True)
    save(Path(plan['destination'])/'new-session-created.json',dict(actual_tmux_returncode=result.returncode,
        stdout=result.stdout.strip(),stderr_sha256=digest(result.stderr.encode()),placeholder_only=True))
    require(result.returncode == 0, 'New own bash session creation failed; no foreground command submitted')
    fields=result.stdout.strip().split()
    require(len(fields)==4 and fields[0].startswith('$') and fields[1].startswith('@') and fields[2].startswith('%'), 'Cannot bind new own window/pane')
    session_id,window_id,pane_id,pane_pid=fields[0],fields[1],fields[2],int(fields[3])
    placeholder=process(pane_pid)
    require(placeholder['uid']==intent['bootstrap_owner']['uid'] and Path(placeholder['argv'][0]).name=='bash', 'New placeholder ownership differs')
    command(['tmux','set-option','-w','-t',window_id,'remain-on-exit','on'])
    retained=command(['tmux','show-options','-wv','-t',window_id,'remain-on-exit'])
    require(retained=='on' and command(['tmux','display-message','-p','-t',pane_id,'#{pane_dead}'])=='0'
            and exact(process(pane_pid),placeholder),'New own retained window was not ready before foreground exec')
    save(Path(plan['destination'])/'new-window-before-exec.json',dict(session=session,session_id=session_id,
        window_id=window_id,pane_id=pane_id,pane_pid=pane_pid,placeholder=placeholder,remain_on_exit=retained,
        only_new_owned_window_changed=True,old_and_global_options_unchanged=True))
    command(['tmux','send-keys','-t',pane_id,'-l',shell])
    command(['tmux','send-keys','-t',pane_id,'Enter'])
    proof = dict(intent, schema='opsd-ablation-stats-handoff-launch-v1', actual_tmux_returncode=result.returncode,
                 accepted=True, pane_start=result.stdout.strip(), actual_exec_command_submitted=True,
                 remain_on_exit_set_before_exec=True,terminal_route='exec_same_pid',stderr_sha256=digest(result.stderr.encode()))
    save(receipt_path, proof)
    return proof


def envelope(raw, binding):
    return canonical(dict(plan=base64.b64encode(raw).decode(), plan_sha256=digest(raw), binding=binding))


def remote(args, raw, binding, mode):
    argv = [args.ssh, '-B', 'en0', '-oProxyCommand=none', '-oProxyJump=none', '-oControlMaster=no', '-S', 'none', args.ssh_host]
    cli = [args.remote_python, '-B', args.remote_helper, mode, '--plan-sha256', digest(raw), '--self-manifest', args.self_manifest,
           '--self-manifest-sha256', args.self_manifest_sha256, '--self-publication', args.self_publication,
           '--self-publication-sha256', args.self_publication_sha256, '--self-commit', args.self_commit]
    return subprocess.run(argv + ['env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(cli)],
                          input=envelope(raw, binding), capture_output=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['watch', 'probe', 'finalize'])
    for name in ('plan-sha256', 'self-manifest', 'self-manifest-sha256', 'self-publication', 'self-publication-sha256', 'self-commit'): p.add_argument('--' + name, required=True)
    p.add_argument('--plan'); p.add_argument('--local-output'); p.add_argument('--ssh', default='/usr/bin/ssh'); p.add_argument('--ssh-host', default='hkust1')
    p.add_argument('--remote-python'); p.add_argument('--remote-helper'); args = p.parse_args()
    if args.mode != 'watch':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('E1_RUN_ID') is None
                and 'torch' not in sys.modules, 'Remote handoff must remain unmarked CPU-only')
        message = json.loads(sys.stdin.buffer.read()); raw = base64.b64decode(message['plan'], validate=True)
        require(digest(raw) == args.plan_sha256 == message['plan_sha256'], 'Plan transport SHA differs')
        plan = json.loads(raw); root = Path(__file__).resolve().parent
        published(root, dict(path=args.self_manifest, sha256=args.self_manifest_sha256), dict(path=args.self_publication, sha256=args.self_publication_sha256), args.self_commit, NAMESPACE)
        result = inspect(plan, message.get('binding')) if args.mode == 'probe' else finalize(plan, message.get('binding'))
        print(canonical(result).decode()); return
    require(args.plan and args.local_output and args.remote_python and args.remote_helper, 'Mac watcher requires explicit plan/destination/remote entry')
    raw, _ = read(args.plan, args.plan_sha256); json.loads(raw)
    dest = Path(args.local_output); require(not dest.exists(), 'Mac watcher output must be independent'); dest.mkdir(parents=True, mode=0o700)
    binding = None; index = 0
    while True:
        result = remote(args, raw, binding, 'probe')
        evidence = dict(actual_returncode=result.returncode, stdout_sha256=digest(result.stdout), stderr_sha256=digest(result.stderr), observed_unix=time.time())
        (dest / ('probe-%05d.stderr' % index)).write_bytes(result.stderr)
        if result.returncode == 0:
            value = json.loads(result.stdout); binding = value.get('binding', binding)
            save(dest / ('probe-%05d.json' % index), dict(evidence, observation=value))
            if binding and not (dest / 'live-binding.json').exists(): save(dest / 'live-binding.json', binding)
            if value.get('ready') is True:
                final = remote(args, raw, binding, 'finalize')
                (dest / 'finalize.stdout').write_bytes(final.stdout); (dest / 'finalize.stderr').write_bytes(final.stderr)
                save(dest / 'finalize-execution.json', dict(actual_returncode=final.returncode, stdout_sha256=digest(final.stdout), stderr_sha256=digest(final.stderr)))
                require(final.returncode == 0, 'Finalization uncertain/failed; preserve partial and do not duplicate launch')
                print(final.stdout.decode(), end=''); return
        else: save(dest / ('probe-%05d.json' % index), evidence)
        index += 1; time.sleep(15)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('NOT_READY_OR_UNKNOWN: ' + str(error), file=sys.stderr); sys.exit(2)
