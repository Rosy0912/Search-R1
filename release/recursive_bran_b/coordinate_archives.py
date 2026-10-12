"""Independently archive this B run's milestones; never control training.

Invoke through a live agent-forwarded SSH connection. Existing archive/prune
helpers enforce checkpoint hashes, ownership, lineage and latest-two retention.
Failures leave source files intact and are retried with a visible error record.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .archive_checkpoint import SOURCE_BASE
from .prune_archived_checkpoint import CURRENT, CURRENT_SOURCE, complete_guard

MILESTONES = (50, 100, 150, 200)


def pending_actions(completed, archived, pruned):
    archive = [step for step in MILESTONES if step <= completed and step not in archived]
    prune = [step for step in MILESTONES[:-1]
             if step <= completed - 2 and step in archived and step not in pruned]
    return archive, prune


def controller_live(pid, tick):
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return False
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return (proc.stat().st_uid == 1001 and int(fields[19]) == tick
            and fields[0] != 'Z' and CURRENT.encode() in (proc / 'cmdline').read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=int, required=True)
    parser.add_argument('--starttick', type=int, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if os.getuid() != 1001 or socket.gethostname().split('.')[0] != 'xie-lab-4':
        raise RuntimeError('Coordinator host/owner mismatch')
    checkpoint_root = SOURCE_BASE / ('output-speed-resume3-' + CURRENT) / 'checkpoints'
    receipts = SOURCE_BASE / 'receipts/archive-coordinator' / CURRENT
    receipts.mkdir(parents=True, exist_ok=True)
    lock = (receipts / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = (receipts / 'events.jsonl').open('a', buffering=1)

    def emit(**values):
        data = dict(at=time.time(), pid=os.getpid(), run=CURRENT, **values)
        line = json.dumps(data)
        journal.write(line + '\n')
        journal.flush()
        os.fsync(journal.fileno())
        print(line, flush=True)

    def execute(module, arguments, step):
        log = receipts / ('step-%d-%s.log' % (step, module.rsplit('.', 1)[-1]))
        with log.open('a') as output:
            subprocess.run([sys.executable, '-B', '-m', module] + arguments,
                           stdout=output, stderr=subprocess.STDOUT, check=True)

    emit(event='coordinator_started', source=CURRENT_SOURCE,
         controller=args.controller, starttick=args.starttick,
         milestones=MILESTONES, training_control=False)
    while True:
        failed = False
        try:
            steps = [int(p.parent.name.rsplit('_', 1)[1])
                     for p in (checkpoint_root / 'training_state').glob('global_step_*/manifest.json')]
            completed = max(steps)
            complete_guard(completed)
            archived = {s for s in MILESTONES if (receipts / ('archived-%d.json' % s)).exists()}
            pruned = {s for s in MILESTONES[:-1] if (receipts / ('pruned-%d.json' % s)).exists()}
            archive, prune = pending_actions(completed, archived, pruned)
            for step in archive:
                emit(event='archive_start', step=step, completed=completed)
                execute('release.recursive_bran_b.archive_checkpoint',
                        ['--run', CURRENT, '--step', str(step), '--checkpoint-root',
                         str(checkpoint_root), '--source-commit', CURRENT_SOURCE], step)
                # Helper success means remote complete SHA verification and pin release.
                marker = receipts / ('archived-%d.json' % step)
                with marker.open('x') as f:
                    json.dump(dict(step=step, source=CURRENT_SOURCE, at=time.time()), f)
                    f.flush(); os.fsync(f.fileno())
                emit(event='archive_verified', step=step)
            # Newly archived steps are considered on the next scan. Pruner independently
            # revalidates archive identity/inventory and the complete guard before deletion.
            for step in prune:
                execute('release.recursive_bran_b.prune_archived_checkpoint',
                        ['--run', CURRENT, '--step', str(step), '--guard-completed',
                         str(completed), '--apply'], step)
                marker = receipts / ('pruned-%d.json' % step)
                with marker.open('x') as f:
                    json.dump(dict(step=step, guard_completed=completed, at=time.time()), f)
                    f.flush(); os.fsync(f.fileno())
                emit(event='archived_local_copy_pruned', step=step)
            live = controller_live(args.controller, args.starttick)
            emit(event='coordinator_observation', completed=completed,
                 controller_live=live, archive_pending=archive, prune_pending=prune)
        except Exception as exc:
            failed = True
            emit(event='coordinator_error', error=repr(exc), training_control=False)
        if args.once:
            if failed:
                raise RuntimeError('Coordinator observation failed; see journal')
            break
        # A terminal producer is never restarted or signalled by this helper.
        if not failed and not live:
            emit(event='producer_terminal', training_control=False)
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
