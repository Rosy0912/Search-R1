"""One isolated calibration submission; uncertain response keeps an unreusable claim."""
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from deployment_contract_v9 import ROOT, HELPERS, SCOPE, validate_ready, read_bound, require, sha, write_once

INTENT = ROOT / 'logs/policy-safety-calibration-52befe9-submit-intent-v9.json'
RECEIPT = ROOT / 'logs/policy-safety-calibration-52befe9-submit-receipt-v9.json'


def main(binding_path, binding_sha, review_path, review_sha):
    require(socket.gethostname().split('.')[0]=='h100-8','Submission only on the authorized h100-8')
    require(Path(binding_path)==ROOT/'logs/policy-safety-calibration-52befe9-reviewed-binding-v9.json','Not exact v9 binding path')
    require(not os.path.lexists(INTENT) and not os.path.lexists(RECEIPT), 'Existing or uncertain claim: inspect intent/receipt/scheduler, never repeat')
    binding = read_bound(dict(path=str(binding_path), sha256=binding_sha))
    validate_ready(binding)
    review = read_bound(dict(path=str(review_path), sha256=review_sha))
    require(review['status'] == 'PASS' and review['scope'] == SCOPE and review['binding_sha256'] == binding_sha
            and review['submit_helper_sha256'] == sha(Path(__file__)), 'Missing exact final binding/single-submit review')
    # Every input and proof is checked again by the frozen launcher before GPU.
    # Avoid metadata changes between immutable READY and our Slurm command.
    for item in binding['files'] + binding['evidence']:
        require(Path(item['path']).stat().st_size == item['bytes'], 'Input bytes changed after READY')
    q = subprocess.check_output(['squeue', '-h', '-w', 'h100-8', '-o', '%i %j %T'], text=True)
    g = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True)
    require(not q.strip() and not g.strip(), 'Authorized node occupied: no submission claim created')
    write_once(INTENT, dict(status='CLAIMED_NO_RESUBMISSION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        binding_path=str(binding_path), binding_sha256=binding_sha, review_sha256=review_sha,
        scope=SCOPE, source_native_job=binding['source_native_acceptance']['job_id'],
        phase='probability_probe_then_wait', retained_training_updates_contributed=0, queue=q, gpu_compute=g))
    command = ['sbatch', '--parsable', str(HELPERS / 'calibration_v1.sbatch'), str(binding_path), binding_sha,
               binding['repo'], binding['helpers']]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=60)
        response=completed.stdout.strip()
        matched=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',response)
        job=matched.group(1) if matched else None
        record = dict(command=command, stdout=completed.stdout, stderr=completed.stderr, returncode=completed.returncode,
                      binding_sha256=binding_sha, retained_training_updates_contributed=0)
        if completed.returncode == 0 and job is not None:
            record.update(status='SUBMITTED_EXACTLY_ONCE', job_id=job)
        else:
            record['status'] = 'UNCERTAIN_OR_FAILED_NO_RETRY'
    except BaseException as error:
        write_once(RECEIPT, dict(status='UNCERTAIN_OR_FAILED_NO_RETRY', command=command,
            error_type=type(error).__name__, error=str(error), binding_sha256=binding_sha))
        raise
    write_once(RECEIPT, record)
    require(record['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Submission ambiguous: preserve claim and inspect; never retry')
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    require(len(sys.argv) == 5, 'Need immutable binding/SHA and independent single-submit review/SHA')
    main(*sys.argv[1:])
