"""Capture B's exact pre-update timing NameError; never signal any process."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
RUN = '5465012f82e6487fa391cd050500b169'
COMMIT = '7bef9bc735b1cb81a03a4387bf1ddd4374763642'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(path, expected):
    assert os.getuid() == 1001 and socket.gethostname().split('.')[0] == 'xie-lab-4'
    assert sha(path) == expected
    r = json.loads(Path(path).read_text())
    assert r['protocol'] == 'recursive-b-pre-update-driver-fault-v1'
    assert (r['failed_run'], r['failed_source']) == (RUN, COMMIT)
    for pid in r['absent_owned_pids']:
        assert not Path(f'/proc/{pid}').exists(), 'Failed-run PID present; inspect before retry'
    for key in ('log', 'restore', 'start', 'parent_manifest'):
        assert sha(r[key]) == r[key + '_sha256']
    pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t', '%7',
                                   '#{session_id}|#{pane_id}|#{pane_dead}|#{pane_dead_status}'], text=True).strip()
    assert pane == r['tmux'] and pane.endswith('|%7|1|1')
    proof = json.loads(Path(r['restore']).read_text())
    assert proof['driver_actual_state_equal'] and proof['next_outer'] == 4
    assert len(proof['workers']) == 4
    for w in proof['workers']:
        assert w['actual_cold_native_state_equal'] and w['actual_total_optimizer_calls'] == 60
        assert w['scheduler_step'] == 3 and w['adam_steps'] == [60]
    log = Path(r['log']).read_text()
    assert "NameError: name 'time' is not defined" in log
    assert 'with _driver_cpu_stage("prepare_behavior_mu",self.global_steps)' in log
    assert 'TREE_DRIVER_STAGE=' not in log
    assert not (ROOT / 'checkpoints/training_state/global_step_4/manifest.json').exists()
    assert r['worker_tensor_snapshot_accepted'] is False
    return r

def main():
    output = ROOT / ('output-speed-resume3-' + RUN)
    start = output / 'start.json'
    s = json.loads(start.read_text())
    assert s['run_id'] == RUN and s['source_commit'] == COMMIT
    record = dict(protocol='recursive-b-pre-update-driver-fault-v1', time=time.time(),
                  failed_run=RUN, failed_source=COMMIT,
                  absent_owned_pids=[s['pid'], 584926, 585647, 585946, 585947, 585948],
                  tmux=subprocess.check_output(['tmux', 'display-message', '-p', '-t', '%7',
                        '#{session_id}|#{pane_id}|#{pane_dead}|#{pane_dead_status}'], text=True).strip(),
                  log=str(ROOT / 'logs/infra-resume3-7bef9bc.log'),
                  restore=str(output / 'sparse-schedule-migrated.json'), start=str(start),
                  parent_manifest=str(ROOT / 'checkpoints/training_state/global_step_3/manifest.json'),
                  worker_tensor_snapshot_accepted=False,
                  failure='Programming NameError before prepare_behavior_mu and before first update_policy call',
                  no_write_evidence_scope='Matched driver traceback before update entry plus actual restored four-rank counters; no failure-time tensor snapshot')
    for key in ('log', 'restore', 'start', 'parent_manifest'):
        record[key + '_sha256'] = sha(record[key])
    path = ROOT / 'receipts/infra7bef-pre-update-driver-failure-terminal.json'
    raw = json.dumps(record, sort_keys=True).encode()
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    digest = sha(path)
    validate(path, digest)
    print(json.dumps(dict(path=str(path), sha256=digest)))

if __name__ == '__main__':
    main()
