"""Explicit inspected replacement for a dispatcher that exited before any submission."""
import fcntl
import json
import os
from pathlib import Path
import time
from .pj_return_dispatch import DIRECT, BASE, PROFILES, ready, submit
from .pj_return import dump_new
from release.recursive_bran_a.ablation150_v1.progress import replace_json


def validate_retirement(proof, old, process_alive, has_submission):
    if proof.get('status') != 'INSPECTED_NETWORK_FAILURE_OLD_TRANSPORT_RETIRED':
        raise ValueError('Missing explicit inspected retirement')
    if old.get('pid') != 4131830 or old.get('status') != 'NEEDS_INSPECTION_NO_AUTOMATIC_REPLAY' or old.get('groups') != {}:
        raise ValueError('Previous dispatcher not an empty terminal attempt')
    if process_alive or has_submission or not proof.get('exact_scheduler_names_absent'):
        raise ValueError('Previous dispatcher alive or possible prior submission')


def main():
    source = os.environ['RECURSIVE_SOURCE_COMMIT']
    if Path(__file__).resolve().parents[2] != BASE / 'sources' / source:
        raise ValueError('Must use immutable deployed source')
    with (DIRECT / 'return-dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = json.loads((DIRECT / 'network-repair-20261017T1450.receipt.json').read_text())
        old = json.loads((DIRECT / 'return-dispatch-status-retired-20261017T1450.json').read_text())
        paths = [BASE / 'migrations' / p['stage'] / n for p in PROFILES.values()
                 for n in ('return-inputs.json', 'return-submit-intent.json', 'return-submission.json')]
        validate_retirement(proof, old, Path('/proc/4131830').exists(), any(p.exists() for p in paths))
        dump_new(DIRECT / 'return-dispatch-recovery-20261017T1450.intent.json',
                 dict(at=time.time(), pid=os.getpid(), source=source, previous_pid=4131830))
        state = dict(at=time.time(), pid=os.getpid(), source=source,
                     status='WAITING_FOR_VERIFIED_FULL_STATES', groups={}, previous_dispatcher=4131830)
        try:
            deadline = time.monotonic() + 18 * 3600
            while time.monotonic() < deadline:
                for group in ('generation-2', 'budget-5'):
                    if group not in state['groups'] and ready(group):
                        state['groups'][group] = submit(group, source)
                state['at'] = time.time()
                replace_json(DIRECT / 'return-dispatch-status.json', state)
                if len(state['groups']) == 2:
                    state['status'] = 'BOTH_UNIQUE_CONTINUATIONS_SUBMITTED'
                    break
                copy = json.loads((DIRECT / 'copy-status.json').read_text())
                if copy['state'].startswith('FAILED'):
                    raise RuntimeError('Transfer failed; another inspection required, no replay')
                time.sleep(45)
            else:
                raise TimeoutError('Transfer wait deadline expired')
        except BaseException as exc:
            state.update(status='NEEDS_INSPECTION_NO_AUTOMATIC_REPLAY', error=repr(exc))
            raise
        finally:
            state['at'] = time.time()
            replace_json(DIRECT / 'return-dispatch-status.json', state)


if __name__ == '__main__':
    main()
