"""One proper CPU allocation; never retry an uncertain sbatch outcome."""
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
TOOLS = ROOT/'instruct300/probe-selection-32593d9-source'
OUTPUT = ROOT/'instruct300/probe-selection-32593d9-result'
RUN = ROOT/'runs/tree-safety300-3268'
PACKAGE = ROOT/'instruct300/candidates/safety110-a632ba9-84e3e186'
PAYLOADS = {'freeze_tree_probability_selection.py': '32593d9e7dd0744dcd7c8ee764b5ecf516d213d2edcd05e72a240e2aa4ff62a4',
            'test_freeze_tree_probability_selection.py': '34341c1f0fa2f1f59cfa25c177830d80556609bcb2b06bab8671ff1ea13a3fa9',
            'analyze_tree_sampling_calibration.py': '28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'}
READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
TERMINAL_SHA = 'cda5c22fbbcff3ff6584a07c589175e20ba620b4013723f18a296f1d94220e78'
POSITION_SHA = '326fe29839606f1d52dfc7cb68e8858f92e3f12f1178965343d88dbdde4bfff3'
ACTION_SHA = '6a08df127b7f4cccdb0be3f88c16036966318b490cc4caa1da2fbb190a12b0c0'


def bound(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Changed metadata: '+str(path))
    return json.loads(payload)


def body():
    py = str(ROOT/'env/bin/python')
    verify = 'import hashlib,pathlib;files='+repr({str(TOOLS/k): v for k,v in PAYLOADS.items()})+';'+\
        'assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==s for p,s in files.items())'
    compile(verify, '<source-verifier>', 'exec')
    freeze = [py, '-B', str(TOOLS/'freeze_tree_probability_selection.py'), '--run', str(RUN),
        '--package', str(PACKAGE), '--cpu-reports', str(RUN), '--output', str(OUTPUT)]
    return ('#!/usr/bin/env bash\nset -euo pipefail\nset -o noclobber\n'
        'export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+
        shlex.join([py, '-B', '-c', verify])+'\n'+
        shlex.join([py, '-B', str(TOOLS/'test_freeze_tree_probability_selection.py')])+
        ' > '+shlex.quote(str(TOOLS))+'/target-cpu-tests-${SLURM_JOB_ID}.txt 2>&1\n'+
        'exec '+shlex.join(freeze)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-sha256', required=True)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'h100-9'
    assert len(args.review_sha256) == 64 and all(c in '0123456789abcdef' for c in args.review_sha256)
    for name, sha in PAYLOADS.items():
        data = (TOOLS/name).read_bytes(); assert hashlib.sha256(data).hexdigest() == sha
        compile(data, str(TOOLS/name), 'exec')
    review = bound(TOOLS/'SELECTION_DISPATCH_CPU_RECEIPT.json', args.review_sha256)
    assert review['status'] == 'SOURCE_AND_DISPATCH_REVIEWED_FOR_SAME_NODE_CPU_FREEZE_ONLY'
    assert review['helper_sha256'] == PAYLOADS['freeze_tree_probability_selection.py']
    assert review['dispatcher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['actual_selection_frozen'] is False and review['actual_gpu_accepted'] is False
    terminal = bound(TOOLS/'ACTUAL_CPU3273_RECEIPT.json', TERMINAL_SHA)
    assert terminal['status'] == 'ACTUAL_H1009_CPU3273_REPORTS_AND_TERMINAL_INDEPENDENTLY_VERIFIED'
    assert terminal['slurm_state'] == 'COMPLETED' and terminal['slurm_exit_code'] == '0:0'
    assert terminal['job_id'] == 3273 and terminal['actual_optimizer_writes'] == 0
    assert terminal['ready_sha256'] == READY_SHA
    bound(RUN/'position-analysis-3273.json', POSITION_SHA)
    bound(RUN/'retained-actions-3273.json', ACTION_SHA)
    ready = bound(RUN/'output/probability-calibration/ready.json', READY_SHA)
    assert ready['status'] == 'CALIBRATION_OBSERVED_ZERO_WRITES' and ready['optimizer_committed'] == 0
    assert not OUTPUT.exists()
    queue = subprocess.run(['squeue','-h','-w','h100-9','-o','%i %j %T'],capture_output=True,text=True,timeout=15)
    assert queue.returncode == 0 and queue.stderr == ''
    assert 'tree-probe-selection-3268' not in queue.stdout
    journal = ROOT/'instruct300/PROBE_SELECTION_ONCE-3268-32593d9.json'
    result = dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN', epoch=time.time(), source_sha256=PAYLOADS,
        review_sha256=args.review_sha256, terminal_scope='SAVED_VERIFIED_CPU3273_SUCCESSFUL_TERMINAL',
        terminal_receipt_sha256=TERMINAL_SHA, queue_snapshot=queue.stdout, output=str(OUTPUT),
        gpu_count=0,cpu_count=4,memory_gib=64,time_minutes=15)
    shell = body()
    with journal.open('x') as stream:
        json.dump(result,stream);stream.flush();os.fsync(stream.fileno())
    try:
        proc = subprocess.run(['sbatch','--parsable','--job-name=tree-probe-selection-3268','--nodes=1',
            '--nodelist=h100-9','--gres=gpu:0','--cpus-per-task=4','--mem=64G','--time=00:15:00',
            '--no-requeue','--output='+str(TOOLS/'selection-%j.json'),'--error='+str(TOOLS/'selection-%j.err')],
            input=shell,capture_output=True,text=True,timeout=60)
        result.update(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
        if proc.returncode == 0:
            job = proc.stdout.strip().split(';')[0];assert job.isdigit()
            result.update(status='SUBMITTED',job_id=int(job))
        else:
            result['status']='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
    except BaseException as error:
        result.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',error=repr(error))
    with journal.open('w') as stream:
        json.dump(result,stream);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    if result['status'] != 'SUBMITTED': raise SystemExit(1)


if __name__=='__main__':
    main()
