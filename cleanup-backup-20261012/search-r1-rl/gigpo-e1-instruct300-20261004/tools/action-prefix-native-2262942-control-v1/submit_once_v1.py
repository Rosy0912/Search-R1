"""Exclusive durable claim: one native sbatch, no retry/cancel/requeue/resume."""
import copy
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path
from deployment_contract_v1 import (ROOT, SOURCE, HELPERS, CONTROLS, READY, BINDING, INTENT, RECEIPT,
    require, sha, read_bound, write_once, validate_ready)


def submit(expected_ready_sha, run=subprocess.run, hostname=socket.gethostname):
    require(hostname().split('.')[0] == 'h100-8', 'Only h100-8 management execution')
    require(Path(__file__).resolve() == CONTROLS / 'submit_once_v1.py', 'Submission helper installed path differs')
    require(not any(path.exists() for path in (BINDING, INTENT, RECEIPT)), 'Existing claim/uncertain result: inspect, never resubmit')
    value = read_bound(dict(path=str(READY), sha256=expected_ready_sha))
    candidate = validate_ready(value)
    require(value['deployment_review']['sha256'] == sha(value['deployment_review']['path']), 'READY review changed')
    for record in value['files'] + value['evidence']:
        require(sha(record['path']) == record['sha256'] and Path(record['path']).stat().st_size == record['bytes'], 'Before-submit full closure changed')
    queue = run(['squeue', '-h', '-w', 'h100-8', '-o', '%i %j %T'], text=True, capture_output=True)
    compute = run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
                  text=True, capture_output=True)
    require(queue.returncode == compute.returncode == 0, 'Occupancy query unreadable')
    require(not queue.stdout.strip() and not compute.stdout.strip(), 'Node occupied: no claim or submission')
    # O_EXCL durable INTENT is shared with any later dispatch; never remove it.
    write_once(INTENT, dict(status='CLAIMED_NO_RESUBMISSION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ready_sha256=expected_ready_sha, code_commit=SOURCE, binding=str(BINDING), actual_formal_updates=0,
        queue=queue.stdout, gpu_compute=compute.stdout, submit_helper_sha256=sha(Path(__file__))))
    binding = copy.deepcopy(value)
    for record in (value['candidate_record'], value['physical_record'], value['deployment_review']):
        same = [r for r in binding['evidence'] if r['path'] == record['path']]
        require(not same or same == [record], 'Conflicting bound acceptance proof')
        if not same:
            binding['evidence'].append(record)
    write_once(BINDING, binding)
    try:
        done = run(['sbatch', '--parsable', str(HELPERS / 'native_safety_v1.sbatch'),
                    str(BINDING), sha(BINDING), candidate['repo'], str(HELPERS)], text=True, capture_output=True)
    except BaseException as error:
        write_once(RECEIPT, dict(status='UNCERTAIN_NO_RESUBMISSION', binding=str(BINDING), binding_sha256=sha(BINDING),
            ready_sha256=expected_ready_sha, error=repr(error), native_gpu_accepted=False))
        raise
    result = dict(status='UNCERTAIN_NO_RESUBMISSION', stdout=done.stdout, stderr=done.stderr, returncode=done.returncode,
        binding=str(BINDING), binding_sha256=sha(BINDING), ready_sha256=expected_ready_sha,
        actual_training_submitted=False, native_gpu_accepted=False)
    job = done.stdout.strip().split(';')[0]
    if done.returncode == 0 and job.isdigit():
        result.update(status='SUBMITTED_EXACTLY_ONCE', job=job)
    write_once(RECEIPT, result)
    require(result['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Uncertain submission: preserve claim/receipt, never retry')
    return result


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Need independently approved immutable native READY SHA')
    print(json.dumps(submit(sys.argv[1]), flush=True))
