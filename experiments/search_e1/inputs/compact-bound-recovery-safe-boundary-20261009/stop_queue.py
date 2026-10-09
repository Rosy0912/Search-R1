"""Stop only the exact owned Mac backup parent at a verified idle boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

PID = 3547
UID = 501
START = 'Fri Oct 9 05:08:24 2026'
LAUNCH_SHA = '17823334beb45b4e94bc0c216d67f72dc170a5fbe5424337896cc4c8abe5ce65'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), 'Absolute regular path required')
    before = path.stat()
    raw = path.read_bytes()
    require(path.stat() == before, 'Input changed during read')
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), raw


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def owner(launch):
    result = subprocess.run(['ps', '-p', str(PID), '-o', 'pid=,ppid=,uid=,state=,lstart=,args='],
                            capture_output=True, text=True)
    require(result.returncode == 0 and len(result.stdout.splitlines()) == 1, 'Owner absent or uncertain')
    fields = result.stdout.strip().split(None, 9)
    require(len(fields) == 10, 'Owner ps shape differs')
    argv = shlex.split(fields[9])
    value = dict(pid=int(fields[0]), ppid=int(fields[1]), uid=int(fields[2]), state=fields[3],
                 started=' '.join(fields[4:9]), argv=argv)
    require(value['pid'] == PID and value['ppid'] == 3546 and value['uid'] == UID and value['started'] == START,
            'Owner PID/parent/UID/start differs')
    require(argv[1:3] == ['-B', launch['argv'][0]] and argv[3:] == launch['argv'][1:], 'Owner argv differs')
    return value


def children():
    result = subprocess.run(['pgrep', '-P', str(PID)], capture_output=True, text=True)
    require(result.returncode in (0, 1) and not result.stderr, 'Child observation uncertain')
    values = [int(x) for x in result.stdout.split()]
    require(bool(values) == (result.returncode == 0), 'Child observation shape differs')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', required=True)
    parser.add_argument('--queue-status', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    require(sys.platform == 'darwin' and os.getuid() == os.geteuid() == UID, 'Mac own UID501 only')
    require(30 <= args.timeout_seconds <= 900, 'Bounded boundary wait required')
    launch, lp, _ = read(args.launch)
    require(lp['sha256'] == LAUNCH_SHA and launch['accepted'] and launch['pid'] == PID and launch['uid'] == UID,
            'Original actual launch differs')
    queue = Path(args.queue_status)
    expected = Path(launch['argv'][launch['argv'].index('--destination') + 1]) / 'queue-status.json'
    require(queue == expected, 'Only original queue status allowed')
    out = Path(args.output)
    require(out.is_absolute() and not out.exists(), 'New exclusive evidence directory required')
    out.mkdir(mode=0o700)
    save(out / 'intent.json', dict(pid=PID, uid=UID, launch=lp, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Technical own backup recovery for proven USTAR bound; no training or Node signals', started_unix=time.time()))
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        before = owner(launch)
        state, _, _ = read(queue)
        require(state['identity'] == launch['queue_identity'], 'Queue identity differs')
        if before['state'].startswith('S') and not children() and all(x['state'] != 'TRANSFERRING' for x in state['jobs'].values()):
            os.kill(PID, signal.SIGSTOP)
            stopped = True
            try:
                frozen = owner(launch)
                require(frozen['state'].startswith('T'), 'Owner not frozen')
                current, qp, raw = read(queue)
                if children() or any(x['state'] == 'TRANSFERRING' for x in current['jobs'].values()):
                    continue
                require(current['identity'] == launch['queue_identity'], 'Frozen queue identity differs')
                (out / 'queue-status.exact.json').write_bytes(raw)
                save(out / 'boundary-before-signal.json', dict(owner=before, frozen_owner=frozen, children=[],
                    queue=qp, launch=lp, observed_unix=time.time(), no_live_transfer=True))
                os.kill(PID, signal.SIGTERM)
                os.kill(PID, signal.SIGCONT)
                stopped = False
                save(out / 'signals-issued.json', dict(pid=PID, uid=UID, signals=['SIGSTOP', 'SIGTERM', 'SIGCONT'],
                    backup_parent_only=True, actual_owner_exit_code='NOT_OBSERVED_BY_THIS_NONPARENT', observed_unix=time.time()))
                print(json.dumps(dict(accepted=True, stage='SIGNALS_AT_IDLE_BOUNDARY', pid=PID,
                    actual_exit_code_requires_original_session=True, output=str(out))))
                return
            finally:
                if stopped:
                    os.kill(PID, signal.SIGCONT)
        time.sleep(1)
    raise RuntimeError('Verified idle boundary not reached; no stop issued')


if __name__ == '__main__':
    main()
