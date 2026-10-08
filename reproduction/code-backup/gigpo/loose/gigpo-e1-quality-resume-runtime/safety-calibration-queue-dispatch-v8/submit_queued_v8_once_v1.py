"""External, versioned once dispatcher: Slurm may queue the frozen v8 job.

No frozen v8 bytes change. Occupancy is evidence, never a resource waiver.
The original durable v8 intent/receipt also excludes the old submitter.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CONTROLS = ROOT / 'tools/policy-safety-calibration-controls-52befe9-v8'
HELPERS = ROOT / 'tools/policy-safety-calibration-52befe9-v8'
INTENT = ROOT / 'logs/policy-safety-calibration-52befe9-submit-intent-v8.json'
RECEIPT = ROOT / 'logs/policy-safety-calibration-52befe9-submit-receipt-v8.json'
EXPECTED_BINDING = ROOT / 'logs/policy-safety-calibration-52befe9-reviewed-binding-v8.json'
EXPECTED_CANDIDATE_SHA = 'e18615c998a69e0212a8aac99fb2a5db32556b97a98cdcfd05fcb87418b618d1'
SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
CONTRACT_SHA = '9b15f36200fbae290ac0cdb5e01f757096820360b0aca27974d9441ad334f4f5'
OLD_ONCE_SHA = '3d04640c7ea7a2634f8990600a4aafa94a3939463affdcb06da626243a5a080c'
SUBMIT_MODE = 'QUEUED_SLURM_EXCLUSIVE_V8_ONCE'
JOB_NAME = 'gigpo-e1-safety-calibration25'
TARGET_HELPER = ROOT / 'tools/policy-safety-calibration-queue-dispatch-v8/submit_queued_v8_once_v1.py'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_contract():
    path = CONTROLS / 'deployment_contract_v8.py'
    require(digest(path) == CONTRACT_SHA, 'Frozen v8 contract changed')
    require(digest(CONTROLS / 'submit_once_v8.py') == OLD_ONCE_SHA,
            'Original v8 submitter changed; shared claim compatibility is unproven')
    spec = importlib.util.spec_from_file_location('queued_frozen_deployment_contract_v8', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(digest(path) == CONTRACT_SHA, 'Frozen contract changed during import')
    require(module.ROOT == ROOT and module.HELPERS == HELPERS and module.CONTROLS == CONTROLS,
            'Wrong frozen v8 paths')
    return module


def observe_occupancy(run=subprocess.run):
    """Observe other users' resources without mutating or judging their jobs."""
    def read(command):
        result = run(command, text=True, capture_output=True, timeout=30)
        require(result.returncode == 0, 'Read-only occupancy query failed: ' + str(command))
        return dict(command=command, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    queue = read(['squeue', '-h', '-w', 'h100-8', '-o', '%i|%j|%T'])
    gpu = read(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    duplicates = []
    allocation_present = False
    for line in queue['stdout'].splitlines():
        if not line.strip():
            continue
        fields = line.split('|')
        require(len(fields) == 3 and all(field.strip() for field in fields), 'Malformed squeue occupancy row')
        job, name, state = [field.strip() for field in fields]
        require(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job), 'Unrecognized active Slurm job ID')
        require(state in ('PENDING', 'RUNNING', 'COMPLETING', 'SUSPENDED', 'CONFIGURING'),
                'Unknown active Slurm state; do not claim or submit')
        allocation_present = allocation_present or state in ('RUNNING', 'COMPLETING', 'SUSPENDED', 'CONFIGURING')
        if name != JOB_NAME:
            continue
        detail = read(['scontrol', 'show', 'job', '-o', job])
        match = re.search(r'(?:^|\s)Command=(\S+)', detail['stdout'])
        require(match is not None, 'Same-name active job has no readable command identity')
        command_path = match.group(1)
        require(command_path.startswith('/'), 'Same-name command identity is ambiguous')
        same_script = command_path == str(HELPERS / 'calibration_v1.sbatch')
        duplicates.append(dict(job_id=job, name=name, scheduler_state=state,
                               command_path=command_path, same_v8_script=same_script, raw=detail))
        require(not same_script, 'This v8 calibration script already has an active job; inspect original intent/receipt')
    return dict(queue=queue, gpu_compute=gpu, same_name_other_script_jobs=duplicates,
                actual_slurm_allocation_present=allocation_present,
                interpretation='READ_ONLY_OCCUPANCY_NOT_RUNTIME_RESOURCE_ACCEPTANCE')


def dispatch(binding_path, binding_sha, review_path, review_sha, contract, run=subprocess.run):
    """Single attempt. The production main supplies only the verified frozen contract."""
    require(not os.path.lexists(INTENT) and not os.path.lexists(RECEIPT),
            'Existing or uncertain v8 claim: inspect intent/receipt/scheduler; never repeat')
    require(str(binding_path) == str(EXPECTED_BINDING) and re.fullmatch(r'[0-9a-f]{64}', binding_sha)
            and re.fullmatch(r'[0-9a-f]{64}', review_sha),
            'Only v8 immutable binding path and explicit binding/review SHA are authorized')
    binding = contract.read_bound(dict(path=str(binding_path), sha256=binding_sha))
    contract.validate_ready(binding)
    require(binding['candidate_sha256'] == EXPECTED_CANDIDATE_SHA and binding['code_commit'] == SOURCE,
            'Actual v8 READY must descend from the exact sealed candidate/source52')
    review = contract.read_bound(dict(path=str(review_path), sha256=review_sha))
    require(review['status'] == 'PASS' and review['scope'] == contract.SCOPE
            and review['binding_sha256'] == binding_sha
            and review['candidate_sha256'] == EXPECTED_CANDIDATE_SHA
            and review['code_commit'] == SOURCE
            and review['submit_helper_sha256'] == digest(Path(__file__))
            and review['submit_helper_path'] == str(TARGET_HELPER)
            and review['submit_mode'] == SUBMIT_MODE
            and review['shared_intent_path'] == str(INTENT)
            and review['shared_receipt_path'] == str(RECEIPT)
            and review['frozen_contract_sha256'] == CONTRACT_SHA
            and review['frozen_original_once_sha256'] == OLD_ONCE_SHA,
            'Missing exact independent queued-helper/binding/shared-claim review')
    for item in binding['files'] + binding['evidence']:
        require(Path(item['path']).stat().st_size == item['bytes'], 'Input bytes changed after READY')
    launcher = HELPERS / 'calibration_v1.sbatch'
    launch_records = [row for row in binding['files'] if row['path'] == str(launcher)]
    require(len(launch_records) == 1 and digest(launcher) == launch_records[0]['sha256'],
            'Original frozen Slurm script changed')
    script = launcher.read_text()
    for directive in ('#SBATCH --exclusive', '#SBATCH --no-requeue', '#SBATCH --nodelist=h100-8',
                      '#SBATCH --gres=gpu:h100:8'):
        require(directive in script.splitlines(), 'Frozen exclusive/h100-8/eight-GPU constraint missing')
    occupancy = observe_occupancy(run)
    occupancy_path = ROOT / 'logs' / f'policy-safety-queued-occupancy-v8-{time.time_ns()}-{os.getpid()}.json'
    contract.write_once(occupancy_path, dict(status='READ_ONLY_OCCUPANCY_ONLY_NOT_RESOURCE_OR_PROBABILITY_ACCEPTANCE',
        submit_helper_path=str(TARGET_HELPER), submit_helper_sha256=digest(Path(__file__)),
        binding_sha256=binding_sha, occupancy=occupancy))
    require(not occupancy['gpu_compute']['stdout'].strip() or occupancy['actual_slurm_allocation_present'],
            'GPU compute without an observed active Slurm allocation: preserve occupancy; no claim or submission')
    # O_EXCL in the frozen writer is the cross-dispatcher serialization point.
    # A racing old v8 once submitter claims exactly this same path.
    contract.write_once(INTENT, dict(status='CLAIMED_NO_RESUBMISSION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        binding_path=str(binding_path), binding_sha256=binding_sha, review_sha256=review_sha,
        scope=contract.SCOPE, source_native_job=binding['source_native_acceptance']['job_id'],
        phase='probability_probe_then_wait', retained_training_updates_contributed=0,
        submit_mode=SUBMIT_MODE, occupancy=occupancy, occupancy_record_path=str(occupancy_path),
        queued_submit_helper_path=str(Path(__file__)), queued_submit_helper_sha256=digest(Path(__file__))))
    command = ['sbatch', '--parsable', str(launcher), str(binding_path), binding_sha,
               binding['repo'], binding['helpers']]
    try:
        completed = run(command, text=True, capture_output=True, timeout=60)
        response = completed.stdout.strip()
        match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?', response)
        record = dict(command=command, stdout=completed.stdout, stderr=completed.stderr,
            returncode=completed.returncode, binding_sha256=binding_sha, submit_mode=SUBMIT_MODE,
            retained_training_updates_contributed=0, scheduler_state_observed=None,
            runtime_resource_accepted=False, probability_accepted=False)
        if completed.returncode == 0 and match is not None:
            record.update(status='SUBMITTED_EXACTLY_ONCE', job_id=match.group(1))
        else:
            record['status'] = 'UNCERTAIN_OR_FAILED_NO_RETRY'
    except BaseException as error:
        contract.write_once(RECEIPT, dict(status='UNCERTAIN_OR_FAILED_NO_RETRY', command=command,
            error_type=type(error).__name__, error=str(error), binding_sha256=binding_sha,
            submit_mode=SUBMIT_MODE, runtime_resource_accepted=False, probability_accepted=False))
        raise
    contract.write_once(RECEIPT, record)
    require(record['status'] == 'SUBMITTED_EXACTLY_ONCE',
            'Submission ambiguous: preserve shared claim and inspect; never retry')
    print(json.dumps(record), flush=True)
    return record


def main(binding_path, binding_sha, review_path, review_sha):
    require(socket.gethostname().split('.')[0] == 'h100-8', 'Only authorized h100-8 management host')
    require(Path(__file__).resolve() == TARGET_HELPER, 'Wrong immutable queued dispatcher installation path')
    return dispatch(binding_path, binding_sha, review_path, review_sha, load_frozen_contract())


if __name__ == '__main__':
    require(len(sys.argv) == 5, 'Need immutable v8 binding/SHA and independent queued-helper review/SHA')
    main(*sys.argv[1:])
