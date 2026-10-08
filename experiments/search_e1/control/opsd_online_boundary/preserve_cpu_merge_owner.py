"""Narrow technical recovery of one bound controller, retaining completed Adam.

Run with the system Python (pidfd support), CUDA_VISIBLE_DEVICES empty. This is
not a general stop command: PID, UID, tick, run, config and entry are fixed.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time

BASE = Path('/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008')
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CONFIG_SHA = 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
WAIT_COMMIT = '2cbcc1b4754eff51a52b213240579f5db0e10f87'
WAIT_SHA = '55ee0b95e6330530ebd3da0c0f32b7699e33fae17e4aa9e049e387387176a214'
OWNER = dict(pid=660173, uid=1005, starttick=924349236)
ORACLE = dict(pid=2239529, uid=1005, starttick=914664099)
CP11_SHA = '464585451581b432d83a97ad96b0b8cc49952c89c3cc39223b6fb5b8fae4a6ad'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def match_owner(current):
    require(current is not None and all(current[k] == OWNER[k] for k in OWNER), 'Controller PID/UID/tick changed; no signal')
    require(current['run_id'] == RUN and str(BASE / 'source/wait-recovery-2cbcc1b/opsd_online_wait_recovery/recovery.py') in current['argv']
            and CONFIG_SHA in current['argv'] and '--resume' in current['argv'], 'Controller entry/run/config changed; no signal')
    tokens = current.get('argv_tokens', shlex.split(current['argv']))
    require(tokens[:3] == [str(Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python')),
        '-B', str(BASE / 'source/wait-recovery-2cbcc1b/opsd_online_wait_recovery/recovery.py')], 'Exact controller executable differs')
    for option, value in [('--config', str(BASE / 'receipts/config.json')), ('--config-sha256', CONFIG_SHA), ('--run-id', RUN)]:
        require(tokens.count(option) == 1 and tokens.index(option) + 1 < len(tokens)
                and tokens[tokens.index(option) + 1] == value, 'Exact controller option differs: ' + option)
    require(tokens.count('--resume') == 1, 'Explicit original same-run resume missing')


def load_wait():
    path = BASE / 'source/wait-recovery-2cbcc1b/opsd_online_wait_recovery/recovery.py'
    require(sha(path) == WAIT_SHA, 'Frozen wait helper changed')
    spec = importlib.util.spec_from_file_location('bound_wait_boundary', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def preserve(args):
    require(os.getuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'System CPU-only helper required')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'No safe pidfd signaling available')
    wait = load_wait()
    config, config_ref = wait.checked_json(dict(path=str(BASE / 'receipts/config.json'), sha256=CONFIG_SHA))
    original, original_proof = wait.load_original(config)
    from checkpoint import check_outer_complete, check_checkpoint, fileproof, exclusive_json
    from acceptance import require_launch, require_owned_release
    config['_config_sha256'] = CONFIG_SHA
    require(not (BASE / 'runs/outers/outer-00001/outer-manifest.json').exists(), 'Outer1 already training; no boundary stop')
    status = json.loads((BASE / 'runs/status.json').read_text())
    require(status['run_id'] == RUN and status['phase'] == 'UNCERTAIN_RELEASE_LEASE_HELD'
            and status['error'] == 'Owned descendant still holds GPU after parent exit', 'Bound hold no longer current')
    current = original.process_identity(OWNER['pid']); match_owner(current)
    children = []
    for line in (BASE / 'runs/child-history.jsonl').read_text().splitlines():
        row = json.loads(line)
        identity = row.get('identity', row.get('child', row))
        require('pid' in identity and identity['uid'] == 1005 and identity['run_id'] == RUN, 'Child history identity unclear')
        if identity['pid'] not in {c['pid'] for c in children}:
            children.append(identity)
    require(children and all(not Path('/proc/' + str(c['pid'])).exists() for c in children), 'An original child remains; no signal')
    owned_release = require_owned_release(config, RUN, current, children)
    admission = require_launch(config, 'training-running', RUN, current, children)
    apps = original.gpu_snapshot()
    require({a['pid'] for a in apps['apps']} == {OWNER['pid'], ORACLE['pid']}, 'Unexpected GPU application; no signal')
    require(all(a['pid'] != OWNER['pid'] or (a['uuid'] == config['owner']['gpu_uuids'][0] and a['used_mib'] == 500)
                for a in apps['apps']), 'Controller context no longer matches exact held boundary')
    cp_ref = dict(path=str(BASE / 'runs/checkpoints/update-00000011-outer-00000/manifest.json'), sha256=CP11_SHA)
    cp, cp_proof = check_checkpoint(cp_ref, config, RUN, 11, 0)
    outer_manifest = json.loads((BASE / 'runs/outers/outer-00000/outer-manifest.json').read_text())
    completion, completion_proof = check_outer_complete(fileproof(BASE / 'runs/outers/outer-00000/training/outer-complete.json'),
                                                       config, RUN, 0, 0, outer_manifest['P'])
    require(completion['checkpoint']['sha256'] == CP11_SHA and cp['cumulative_pair_exposure'] == 327, 'Wrong final CP11/exposure')
    destination = BASE / 'preserved' / ('cpu-merge-parent-cuda-' + str(int(time.time())))
    destination.mkdir(parents=True, exist_ok=False)
    authorization = dict(latest_human_restart='Repair OPSD according to settings document and restart eight GPUs from GRPO100',
                         latest_choices='保留10，200，hkust存户到本地即可',
                         scope='Only exact blocked current controller; full run goal and observer continue; technical recovery, not human pause',
                         original_human_time='2026-10-08; exact message timestamp not supplied',
                         old_stops_superseded_for_this_new_run=True)
    before = exclusive_json(destination / 'before-signal.json', dict(accepted=True, run_id=RUN, config=config_ref,
        authorization=authorization, owner=current, children=children, gate=admission, release=owned_release,
        checkpoint=cp_proof, completion=completion_proof, source=fileproof(__file__), original_helper=original_proof,
        GPU=apps, planned_signal='pidfd SIGTERM exact controller only', no_child_signals=True))
    fd = os.pidfd_open(OWNER['pid'])
    try:
        # Recheck every dynamic boundary after the long CP11 byte reads and
        # after binding the kernel pidfd. A new child/phase rejects all signals.
        current = original.process_identity(OWNER['pid'])
        current['argv_tokens'] = (Path('/proc') / str(OWNER['pid']) / 'cmdline').read_bytes().decode().strip('\0').split('\0')
        match_owner(current)
        fresh_status = json.loads((BASE / 'runs/status.json').read_text())
        require(fresh_status['run_id'] == RUN and fresh_status['phase'] == 'UNCERTAIN_RELEASE_LEASE_HELD'
                and fresh_status['error'] == 'Owned descendant still holds GPU after parent exit', 'Controller left held boundary; no signal')
        require(not (BASE / 'runs/outers/outer-00001/outer-manifest.json').exists(), 'Outer1 advanced; no signal')
        fresh_children = [json.loads(line)['identity'] for line in (BASE / 'runs/child-history.jsonl').read_text().splitlines()]
        require({(c['pid'],c['uid'],c['starttick']) for c in fresh_children}
                == {(c['pid'],c['uid'],c['starttick']) for c in children}, 'New or changed child; no signal')
        require(all(not Path('/proc/' + str(c['pid'])).exists() for c in children), 'Original child reappeared; no signal')
        fresh_release = require_owned_release(config, RUN, current, children)
        fresh_gate = require_launch(config, 'training-running', RUN, current, children)
        fresh_apps = original.gpu_snapshot()
        require({a['pid'] for a in fresh_apps['apps']} == {OWNER['pid'], ORACLE['pid']}
                and all(a['pid'] != OWNER['pid'] or (a['uuid'] == config['owner']['gpu_uuids'][0] and a['used_mib'] == 500)
                        for a in fresh_apps['apps']), 'Fresh GPU boundary differs; no signal')
        exclusive_json(destination / 'fresh-before-signal.json', dict(accepted=True, owner=current, status=fresh_status,
            children=fresh_children, release=fresh_release, gate=fresh_gate, GPU=fresh_apps, checked_unix=time.time()))
        match_owner(original.process_identity(OWNER['pid']))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        until = time.monotonic() + 30
        while Path('/proc/' + str(OWNER['pid'])).exists():
            require(time.monotonic() < until, 'Controller did not exit; no escalation/kill group')
            time.sleep(.25)
    finally:
        os.close(fd)
    with ExitStack() as stack:
        for name in [str(BASE / 'online-experiment.lock')] + config['resources']['shared_gpu_lease_paths']:
            handle = stack.enter_context(open(name, 'a+')); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        own = original.process_identity(os.getpid())
        require_launch(config, 'cpu-merge', RUN, own, children)
        require_owned_release(config, RUN, own, children)
        snapshot = original.gpu_snapshot()
        require({a['pid'] for a in snapshot['apps']} == {ORACLE['pid']}, 'GPU still occupied after exact controller exit')
        saved = []
        for name in ('runs/status.json', 'runs/child-history.jsonl', 'logs/controller-wait-recovery1.log',
                     'receipts/current-controller.json', 'receipts/wait-recovery-launch.json'):
            source = BASE / name; target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
            proof = fileproof(source); shutil.copy2(source, target)
            require(sha(target) == proof['sha256'], 'Preserved compact file differs')
            saved.append(dict(original=proof, copy=fileproof(target)))
        outer = BASE / 'runs/outers/outer-00001'
        cache = outer / 'merged-model-cache'; receipt = outer / 'merged-model.json'
        require(cache.is_dir() and receipt.is_file() and not cache.is_symlink(), 'Expected old CPU merge cache absent')
        cache_files = [fileproof(p) for p in sorted(cache.rglob('*')) if p.is_file()]
        old_receipt = fileproof(receipt)
        os.rename(cache, destination / 'merged-model-cache'); os.rename(receipt, destination / 'merged-model.json')
        require(all(sha(destination / 'merged-model-cache' / Path(p['path']).relative_to(cache)) == p['sha256'] for p in cache_files),
                'Moved old merge bytes changed')
        require(sha(destination / 'merged-model.json') == old_receipt['sha256'], 'Moved old merge receipt changed')
        preservation = exclusive_json(destination / 'merge-preservation.json', dict(accepted=True, run_id=RUN, config_sha256=CONFIG_SHA,
            before_signal=before, old_merge_receipt=old_receipt, old_merge_files=cache_files, compact_files=saved,
            CPU_context_was_not_asserted_after_old_merge=True, old_actual_tensor_device_unknown=True,
            original_paths_absent=True, completed_CP11_retained=True, questions_and_train_cursor_unchanged=True))
        pane = subprocess.check_output(['tmux', 'list-panes', '-t', 'e1-opsd-online-recovery1-200-20261008',
            '-F', '#{pane_dead} #{pane_dead_status}'], text=True).strip()
        require(pane == '1 143', 'Actual tmux terminal differs: ' + pane)
        old = [OWNER] + [dict((k, c[k]) for k in ('pid', 'uid', 'starttick')) for c in children]
        require(all(not Path('/proc/' + str(c['pid'])).exists() for c in old), 'Old owned PID reappeared; hold')
        terminal = exclusive_json(BASE / 'receipts/cpu-merge-parent-terminal.json', dict(
            schema='opsd-online-cpu-merge-parent-terminal-v1', accepted=True, run_id=RUN, config_sha256=CONFIG_SHA,
            old_owner=OWNER, old_owned_processes=old, old_source_commit=WAIT_COMMIT, resume_checkpoint=cp_ref,
            merge_preservation=preservation, old_tmux_exit_status=143, exact_signal='pidfd SIGTERM only660173',
            technical_recovery_not_goal_pause=True, after_gpu=snapshot, fresh_all_old_pids_absent=True,
            observer_and_oracle_untouched=True, checked_unix=time.time()))
        print(json.dumps(dict(terminal=terminal, preservation=preservation)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); args = parser.parse_args(); preserve(args)
