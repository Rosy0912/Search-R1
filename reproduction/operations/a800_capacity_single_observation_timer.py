"""Perform one bounded read at the saved deadline; never launch or retry a job."""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--not-before', type=float, required=True)
    parser.add_argument('--read-source-sha256', required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.root)
    source = pathlib.Path(__file__).with_name('a800_owned_capacity_bounded_read.py')
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.read_source_sha256
    sys.path.insert(0, str(root / 'actor-probes-ten'))
    from reproduction.a800_common.deploy import ssh_command

    print(json.dumps({
        'role': 'single_local_capacity_waiter', 'pid': os.getpid(),
        'not_before_unix': args.not_before, 'created_unix': time.time(),
        'remote_query_performed': False,
    }), flush=True)
    while True:
        remaining = args.not_before - time.time()
        if remaining <= 0:
            break
        time.sleep(min(30, remaining))

    succeeded = False
    failure = None
    try:
        payload = json.loads((root / 'preparation/a800-owned-free-training-gpu-reservation-single1-launch.json').read_text())
        payload['other_owned_reservation'] = json.loads((root / 'preparation/a800-owned-free-training-gpu-reservation-subset03-launch.json').read_text())
        result = subprocess.run(
            ssh_command(raw.decode()), input=json.dumps(payload), text=True,
            capture_output=True, timeout=90,
        )
        if result.returncode:
            failure = {'returncode': result.returncode, 'stderr_tail': result.stderr[-2000:]}
        else:
            observation = json.loads(result.stdout)
            assert isinstance(observation, dict)
            assert observation['role'] == 'actual_owned_capacity_reservation_bounded_observation'
            assert observation['checked_unix'] >= args.not_before
            succeeded = True
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, AssertionError, TypeError) as error:
        failure = {'error_type': type(error).__name__, 'error': str(error)[-2000:]}

    now = time.time()
    if succeeded:
        observation['local_monitor_recorded_unix'] = now
        observation['extra_evaluation_or_training_started'] = False
    else:
        observation = {
            'role': 'bounded_capacity_observation_failed',
            'observed_remote_state': 'unknown', 'local_checked_unix': now,
            'failure': failure, 'no_training_or_reservation_retry': True,
        }
    output = root / 'progress-checks' / ('actor-probes-ten-resource-' + str(time.time_ns()) + '.json')
    with output.open('x') as stream:
        json.dump(observation, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        'record': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'checked_unix': observation.get('checked_unix', now),
        'remote_read_success': succeeded,
        'own_ready': observation.get('actual_owned_60GiB_each_verified', False)
        and observation.get('other_actual_owned_60GiB_each_verified', False),
        'foreign_training_cuda_applications': observation.get('foreign_training_cuda_applications'),
        'no_mutations_or_auto_retry': True,
    }), flush=True)


if __name__ == '__main__':
    main()
