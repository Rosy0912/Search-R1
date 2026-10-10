"""Read-only Linux ownership audit; unknown opaque live processes fail closed.

Never infer a controller exit code, read credentials into a receipt, or signal.
An inaccessible environ is UNKNOWN, never an empty/readable environment.
"""
import hashlib
import json
from pathlib import Path
import time


def identity_matches(left, right):
    return all(left.get(key) == right.get(key) for key in ('pid', 'uid', 'starttick'))


def proc_kernel_identity(pid):
    """Use kernel EUID, not the dumpability-dependent owner of /proc/PID."""
    path = Path('/proc') / str(pid)
    text = (path / 'stat').read_text(); fields = text.rsplit(') ', 1)[1].split()
    comm = text.split('(', 1)[1].rsplit(') ', 1)[0]
    result = dict(pid=int(pid), proc_directory_owner_uid=path.stat().st_uid, comm=comm, state=fields[0],
                  ppid=int(fields[1]), pgid=int(fields[2]), starttick=int(fields[19]))
    if (path / 'comm').read_text().rstrip('\n') != comm:
        raise RuntimeError('Kernel comm changed during process snapshot')
    status = {}
    for line in (path / 'status').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1); status[key] = value.strip()
    uids = [int(value) for value in status['Uid'].split()]
    if len(uids) != 4 or int(status['PPid']) != result['ppid']:
        raise RuntimeError('Kernel UID/PPid snapshot differs')
    result['kernel_uids'] = uids; result['uid'] = uids[1]
    again = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
    if int(again[19]) != result['starttick'] or int(again[1]) != result['ppid']:
        raise RuntimeError('PID identity changed while snapshotting kernel status')
    result['state'] = again[0]
    return result


def proc_snapshot(pid):
    path = Path('/proc') / str(pid); result = proc_kernel_identity(pid)
    raw = (path / 'cmdline').read_bytes()
    result['argv'] = [part.decode(errors='replace') for part in raw.split(b'\0') if part]
    result['argv_sha256'] = hashlib.sha256(raw).hexdigest()
    again = proc_kernel_identity(pid)
    if any(again[key] != result[key] for key in ('starttick', 'ppid', 'uid', 'comm', 'kernel_uids')):
        raise RuntimeError('PID identity changed while snapshotting')
    result['state'] = again['state']; result['proc_directory_owner_uid'] = again['proc_directory_owner_uid']
    return result


def proc_environment(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    return dict(item.decode(errors='replace').split('=', 1) for item in raw.split(b'\0') if b'=' in item)


def compact(value):
    return {key: value[key] for key in ('pid', 'uid', 'starttick', 'ppid', 'pgid', 'state', 'comm',
                                       'kernel_uids', 'proc_directory_owner_uid', 'argv_sha256') if key in value} | {'argv_count': len(value.get('argv', []))}


def matched_roots(argv, roots):
    matches = set(); inputs = set()
    for index, argument in enumerate(argv):
        value = argument.split('=', 1)[-1]
        for root in roots:
            if value == root or value.startswith(root.rstrip('/') + '/'):
                option = argument.split('=', 1)[0] if '=' in argument else argv[index - 1] if index else ''
                if Path(root).suffix == '.py' or option in ('--output', '--output-root', '--phase-output'):
                    matches.add(root)
                else:
                    inputs.add(root)
    return sorted(matches), sorted(inputs)


def parent_chain(value, reader, cache):
    chain = []; pid = value['ppid']; visited = {value['pid']}
    while pid:
        if pid in visited or len(chain) == 64:
            return chain, 'cycle_or_depth_limit'
        visited.add(pid)
        try:
            if pid not in cache:
                cache[pid] = reader(pid)
            parent = cache[pid]
        except (FileNotFoundError, ProcessLookupError):
            return chain, 'parent_vanished'
        except (PermissionError, RuntimeError, OSError) as error:
            return chain, 'parent_unreadable:' + type(error).__name__
        chain.append(parent); pid = parent['ppid']
    return chain, None


def old_infrastructure(value, chain, starttick):
    if value['starttick'] >= starttick or not chain:
        return None
    parent = chain[0]
    if value['comm'] == 'systemd' and value['ppid'] == 1 and parent['pid'] == 1 and parent['uid'] == 0 and parent['comm'] in ('systemd', 'init'):
        return 'predating_user_systemd_root_init_parent'
    if value['comm'] in ('sd-pam', '(sd-pam)') and parent['uid'] == value['uid'] and parent['comm'] == 'systemd' and parent['starttick'] < starttick and parent['ppid'] == 1:
        if len(chain) > 1 and chain[1]['pid'] == 1 and chain[1]['uid'] == 0 and chain[1]['comm'] in ('systemd', 'init'):
            return 'predating_sd_pam_user_systemd_root_init_chain'
    if value['comm'] in ('tmux: server', 'tmux:server') and parent['uid'] == 0 and parent['pid'] == 1:
        return 'predating_tmux_server_root_init_parent'
    return None


def root_ssh_service(value, chain):
    # A mutable user cmdline saying "sshd" is insufficient. These are kernel
    # UID/PPid/comm identities of the root-controlled parent service chain.
    if value['state'] != 'Z' and value['comm'] == 'sshd' and chain:
        parent = chain[0]
        if parent['uid'] == 0 and parent['comm'] == 'sshd' and parent['starttick'] <= value['starttick']:
            return 'actual_root_parent_sshd_service'
    return None


def load_exclusions(references):
    """Bound independent CPU-task provenance, never merely a PID allowlist."""
    exclusions = []
    for reference in references:
        path = Path(reference['path']); before = path.stat(); data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference['sha256'] or path.stat() != before:
            raise ValueError('Independent CPU exclusion receipt hash/stat changed')
        value = json.loads(data)
        if value.get('schema') != 'independent-cpu-process-provenance-v1' or value.get('verified_non_m1_owned_cpu_task') is not True:
            raise ValueError('Unknown independent CPU exclusion provenance')
        identity = value['identity']
        if any(type(identity.get(key)) is not int for key in ('pid', 'uid', 'starttick', 'ppid')):
            raise ValueError('Independent CPU exclusion lacks actual kernel identity')
        if value.get('no_gpu_or_training') is not True or not value.get('source_sha256') or not value.get('evidence'):
            raise ValueError('Independent CPU exclusion lacks source/rejection/launch evidence')
        evidence_proofs = []
        for evidence in value['evidence']:
            ep = Path(evidence['path']); first = ep.stat(); raw = ep.read_bytes()
            if hashlib.sha256(raw).hexdigest() != evidence['sha256'] or ep.stat() != first:
                raise ValueError('Independent CPU provenance evidence changed')
            evidence_proofs.append(dict(path=str(ep.resolve()), sha256=evidence['sha256']))
        exclusions.append(dict(identity=identity, receipt=dict(path=str(path.resolve()), sha256=reference['sha256']),
                               evidence=evidence_proofs, source_sha256=value['source_sha256'],
                               provenance='Root-bound independent CPU launch/rejection evidence; environ remains unknown'))
    return exclusions


def audit_same_uid(run_id, uid, controller_starttick, known_owned, argv_roots, exclusions=(),
                   ignored_exact=(), snapshot_reader=proc_snapshot, environment_reader=proc_environment,
                   pid_reader=None, ancestry_anchors=()):
    started = time.time(); cache = {}; excluded = []; blockers = []; vanished = []
    roots = [str(Path(root).resolve()) for root in argv_roots]
    pids = sorted(pid_reader() if pid_reader else [int(path.name) for path in Path('/proc').iterdir() if path.name.isdigit()])
    anchors = list(known_owned) + list(ancestry_anchors)
    for pid in pids:
        known_pid = next((identity for identity in known_owned if identity['pid'] == pid), None)
        try:
            if snapshot_reader is proc_snapshot:
                kernel = proc_kernel_identity(pid)
                if kernel['uid'] != uid and known_pid is None:
                    continue
            value = snapshot_reader(pid)
        except (FileNotFoundError, ProcessLookupError):
            vanished.append(dict(pid=pid, reason='vanished_during_snapshot')); continue
        except (PermissionError, RuntimeError, OSError) as error:
            # Unable to establish UID is unknown, not a silent foreign process.
            blockers.append(dict(pid=pid, reason='kernel_identity_unreadable:' + type(error).__name__)); continue
        # A still-present known PID with the same starttick cannot become a
        # reused PID merely by changing EUID or /proc directory ownership.
        if known_pid is not None and value['starttick'] == known_pid['starttick']:
            blockers.append(dict(compact(value), expected_known_identity=known_pid,
                                 reason='known_owned_pid_same_starttick_present_even_if_uid_changed')); continue
        if value['uid'] != uid:
            if known_pid is not None:
                excluded.append(dict(compact(value), expected_known_identity=known_pid,
                                     reason='known_pid_reused_different_starttick_foreign_kernel_uid'))
            continue
        cache[pid] = value
        item = compact(value); chain, ancestry_unknown = parent_chain(value, snapshot_reader, cache)
        item['kernel_parent_chain'] = [compact(parent) for parent in chain]
        item['parent_chain_unknown'] = ancestry_unknown
        try:
            env_run = environment_reader(pid).get('E1_RUN_ID'); item['environ_status'] = 'readable'
            item['E1_RUN_ID'] = env_run
        except (FileNotFoundError, ProcessLookupError):
            vanished.append(dict(item, reason='vanished_during_environment')); continue
        except (PermissionError, OSError) as error:
            env_run = None; item['environ_status'] = 'unknown:' + type(error).__name__
        try:
            current = snapshot_reader(pid)
        except (FileNotFoundError, ProcessLookupError):
            vanished.append(dict(item, reason='vanished_after_environment')); continue
        except (PermissionError, RuntimeError, OSError) as error:
            blockers.append(dict(item, reason='kernel_identity_unreadable_after_environment:' + type(error).__name__)); continue
        if not identity_matches(current, value) or any(current[key] != value[key] for key in ('ppid', 'comm', 'argv_sha256')):
            blockers.append(dict(item, reason='kernel_identity_or_args_changed_after_environment')); continue
        value['state'] = current['state']; item['state'] = current['state']
        own_identity = next((identity for identity in known_owned if identity_matches(value, identity)), None)
        own_argv, input_argv = matched_roots(value.get('argv', []), roots)
        descendant = [compact(parent) for parent in chain if any(identity_matches(parent, anchor) for anchor in anchors)]
        item.update(owned_argv_roots=own_argv, non_execution_input_argv_roots=input_argv, owned_parent_chain_matches=descendant,
                    reused_known_pid=any(value['pid'] == identity['pid'] and not identity_matches(value, identity) for identity in known_owned))
        if own_identity or env_run == run_id or own_argv or descendant:
            blockers.append(dict(item, reason='known_owned_or_run_or_argv_or_descendant_present')); continue
        if value['state'] != 'Z' and any(identity_matches(value, identity) for identity in ignored_exact):
            excluded.append(dict(item, reason='explicit_expected_live_external_auditor_or_phase_controller')); continue
        if ancestry_unknown:
            blockers.append(dict(item, reason='unresolved_kernel_parent_chain')); continue
        if value['state'] == 'Z':
            bound = next((proof for proof in exclusions if identity_matches(value, proof['identity']) and value['ppid'] == proof['identity']['ppid']), None)
            if bound:
                excluded.append(dict(item, reason='inert_independently_bound_cpu_zombie', provenance=bound,
                                     origin_environ_unknown=True)); continue
            blockers.append(dict(item, reason='inert_zombie_origin_not_proven_non_owned')); continue
        if item['environ_status'] == 'readable':
            excluded.append(dict(item, reason='readable_foreign_without_owned_identity_run_argv_or_parent_chain')); continue
        reason = old_infrastructure(value, chain, controller_starttick) or root_ssh_service(value, chain)
        if reason:
            excluded.append(dict(item, reason=reason, origin_environ_unknown=True)); continue
        blockers.append(dict(item, reason='opaque_live_process_not_proven_infrastructure'))
    return dict(schema='same-uid-owned-process-audit-v2', accepted_no_owned_or_opaque=not blockers,
                run_id=run_id, uid=uid, controller_starttick=controller_starttick,
                known_owned=known_owned, argv_roots=roots, ignored_exact=list(ignored_exact),
                exclusions=excluded, blockers=blockers, vanished=vanished, audited_same_uid=len(excluded)+len(blockers),
                started_unix=started, ended_unix=time.time(), no_signals_sent=True,
                inaccessible_environ_is_unknown=True, no_exit_code_inferred_from_absence=True)
