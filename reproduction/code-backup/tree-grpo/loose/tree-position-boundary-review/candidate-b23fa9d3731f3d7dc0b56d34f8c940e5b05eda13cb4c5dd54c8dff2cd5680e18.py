"""Single CPU-only submission for reviewed position/action classification.

Every unknown sbatch outcome must be reconciled against the exclusive journal
and scheduler, never repeated. This is not a training or probability acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN = ROOT / 'runs/tree-safety300-3268'
PACKAGE = ROOT / 'instruct300/candidates/safety110-a632ba9-84e3e186'
TOOLS = ROOT / 'instruct300/context-layout-3992-b637'
POSITION_SHA = '3992c092e7cf4e42469b9e2cf7cab896a44395c70f224101ef37cb4c0fee05c9'
ACTION_SHA = 'b63740191fcfea587756526c9cb2186557c28233e43886b5de23dedb0d191bd1'
CONTEXT_SHA = '28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'
PACKAGE_SHA = '84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
ACTION_REVIEW_SHA = '6107b01f38416baaf5666f8ce6c0ece35a6b668d0f88ce00f99e93f50946b98d'
POSITION_REVIEW_SHA = '5251ce527eb0d0ebf0c4e97f5fc05947362498543ba26e50ac064fa0810d1ad2'
PAYLOADS = {'inspect_tree_position_boundaries.py': POSITION_SHA,
            'analyze_tree_retained_actions.py': ACTION_SHA,
            'analyze_tree_sampling_calibration.py': CONTEXT_SHA}


def bound(path, expected=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    return json.loads(data), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--position-receipt-sha256', required=True)
    args = parser.parse_args()
    assert args.position_receipt_sha256 == POSITION_REVIEW_SHA
    assert socket.gethostname().split('.')[0] == 'h100-9'
    for name, expected in PAYLOADS.items():
        data = (TOOLS / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        compile(data, str(TOOLS / name), 'exec')
    action_review, _ = bound(TOOLS / 'RETAINED_ACTION_SOURCE_CPU_READY_RECEIPT.json', ACTION_REVIEW_SHA)
    position_review, _ = bound(TOOLS / 'POSITION_SOURCE_CPU_READY_RECEIPT.json', args.position_receipt_sha256)
    assert action_review['status'] == 'READY_FOR_RETAINED_ACTION_SOURCE_ONLY_CPU_CLASSIFICATION'
    assert action_review['helper_sha256'] == ACTION_SHA and action_review['prior_findings_resolved'] is True
    assert action_review['source_only_reviewed'] is True and action_review['cpu_tiny_fixtures_passed'] is True
    assert action_review['actual_assets_loaded'] is False and action_review['actual_classification_result_accepted'] is False
    assert action_review['probability_consistency_accepted'] is False and action_review['actual_3b_training_accepted'] is False
    assert action_review['thresholds_calibrated'] is False and action_review['long_training_ready'] is False
    assert position_review['status'] == 'READY_FOR_READ_ONLY_SAME_NODE_CPU_POSITION_SOURCE_ANALYSIS'
    assert position_review['helper_sha256'] == POSITION_SHA and position_review['context_helper_sha256'] == CONTEXT_SHA
    assert position_review['actual_calibration_assets_loaded'] is False
    assert position_review['actual_model_forwards'] == 0 and position_review['actual_optimizer_writes'] == 0
    package, _ = bound(PACKAGE / 'manifest.json', PACKAGE_SHA)
    assert package['stage'] == 'calibration110' and package['code_commit'] == 'a632ba91840fb116aacfc0e770681a860647b295'
    ready, _ = bound(RUN / 'output/probability-calibration/ready.json', READY_SHA)
    assert ready['status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES' and ready['optimizer_committed'] == 0
    assert ready['parent_completed_outer'] == 110 and ready['new_completed_outer'] == 0 and ready['job_id'] == '3268'
    result, _ = bound(RUN / 'run-result.json')
    assert result['status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED' and result['cleanup_residual'] == {}
    actual = subprocess.run(['scontrol', 'show', 'job', '3271'], capture_output=True, text=True, timeout=15)
    assert actual.returncode == 0 and 'JobState=COMPLETED ' in actual.stdout and 'ExitCode=0:0' in actual.stdout
    active = subprocess.check_output(['squeue', '-h', '-w', 'h100-9', '-o', '%i %j %T'], text=True)
    assert not any(prefix in active for prefix in ('tree-context-analysis-3268', 'tree-position-actions-3268'))
    journal = ROOT / 'instruct300/POSITION_ACTION_ANALYSIS_ONCE-3268-3992-b637.json'
    intent = dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN', epoch=time.time(), parent_job=3268,
                  previous_cpu_job=3271, payload_sha256=PAYLOADS, ready_sha256=READY_SHA,
                  package_sha256=PACKAGE_SHA, action_review_sha256=ACTION_REVIEW_SHA,
                  position_review_sha256=args.position_receipt_sha256,
                  gpu_count=0, cpu_count=4, memory_gib=64, time_minutes=15)
    py = str(ROOT / 'env/bin/python')
    verify = 'import hashlib,pathlib;files=' + repr({str(TOOLS / n): sha for n, sha in PAYLOADS.items()}) + ';' + \
             'assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==s for p,s in files.items())'
    compile(verify, '<payload-sha-check>', 'exec')
    position = [py, '-B', str(TOOLS / 'inspect_tree_position_boundaries.py'), '--run', str(RUN),
                '--package', str(PACKAGE), '--expected-package-manifest-sha256', PACKAGE_SHA,
                '--expected-ready-sha256', READY_SHA, '--expected-context-helper-sha256', CONTEXT_SHA]
    action = [py, '-B', str(TOOLS / 'analyze_tree_retained_actions.py')]
    body = ('#!/usr/bin/env bash\nset -euo pipefail\nset -o noclobber\n'
            'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\n' +
            shlex.join([py, '-B', '-c', verify]) + '\n' +
            shlex.join(position) + ' > ' + shlex.quote(str(RUN)) + '/position-analysis-${SLURM_JOB_ID}.json\n' +
            'exec ' + shlex.join(action) + '\n')
    with journal.open('x') as stream:
        json.dump(intent, stream); stream.flush(); os.fsync(stream.fileno())
    try:
        proc = subprocess.run(['sbatch', '--parsable', '--job-name=tree-position-actions-3268', '--nodes=1',
                  '--nodelist=h100-9', '--gres=gpu:0', '--cpus-per-task=4', '--mem=64G', '--time=00:15:00',
                  '--no-requeue', '--output=' + str(RUN / 'retained-actions-%j.json'),
                  '--error=' + str(RUN / 'layout-analysis-%j.err')], input=body, capture_output=True, text=True, timeout=60)
        intent.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if proc.returncode == 0:
            job = proc.stdout.strip().split(';')[0]
            assert job.isdigit(), proc.stdout
            intent.update(status='SUBMITTED', job_id=int(job))
        else:
            intent['status'] = 'SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
    except BaseException as error:
        intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY', exception=repr(error))
        raise
    finally:
        temporary = journal.with_suffix('.final.tmp')
        with temporary.open('x') as stream:
            json.dump(intent, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, journal)
        print(json.dumps(intent, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
