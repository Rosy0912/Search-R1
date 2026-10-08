"""One CPU afterany job for the owned3154; never submits training."""
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
DIRECTORY = ROOT / 'instruct300/post3154'
TOOL = DIRECTORY / 'verify-7b663927.py'
RECEIPT = DIRECTORY / 'READY_PREPARATION_RECEIPT.json'
SHA = '7b66392771d425c5246d0e459d24e4526a0f311ed3f1d6c8537d75a939bd1a86'
RECEIPT_SHA = 'a6f16b62b0e49e9982980fd843985706c3269c0e61562486e67cf4fb1d9304bb'
assert hashlib.sha256(TOOL.read_bytes()).hexdigest() == SHA
assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA
review = json.loads(RECEIPT.read_bytes())
assert review['source_sha256'] == SHA
assert review['status'] == 'READY_FOR_CPU_TERMINAL_EVIDENCE_COLLECTION_ONLY'
parent = subprocess.check_output(['scontrol', 'show', 'job', '3154'], text=True)
fields = dict(re.findall(r'(\S+?)=(\S+)', parent))
assert fields['JobId'] == '3154' and fields['JobState'] == 'RUNNING'
assert fields['NodeList'] == fields['BatchHost'] == 'h100-9' and fields['Restarts'] == '0'
intent = DIRECTORY / 'SUBMISSION_ONCE.json'
with intent.open('x') as stream:
    json.dump({'status': 'INTENT_RECORDED_DO_NOT_RESUBMIT', 'parent': 3154,
               'epoch': time.time(), 'tool_sha256': SHA, 'review_sha256': RECEIPT_SHA}, stream)
    stream.flush()
    os.fsync(stream.fileno())
execution = ('import pathlib,hashlib; p=pathlib.Path(' + repr(str(TOOL)) + '); '
             'b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()==' + repr(SHA) + '; '
             "exec(compile(b,str(p),'exec'))")
command = "export CUDA_VISIBLE_DEVICES=''; " + shlex.join([str(ROOT / 'env/bin/python'), '-c', execution])
args = ['sbatch', '--parsable', '-p', 'gpu', '-w', 'h100-9', '-N', '1', '--gres=gpu:0',
        '--ntasks=1', '--cpus-per-task=8', '--mem=32G', '--time=00:30:00',
        '--dependency=afterany:3154', '--job-name=tree3154-terminal-cpu',
        '--output=' + str(DIRECTORY / 'slurm-%j.out'), '--wrap=' + command]
result = subprocess.run(args, capture_output=True, text=True)
job = result.stdout.strip()
record = {'status': 'SUBMITTED_NOT_EXECUTED' if result.returncode == 0 and job.isdigit() else 'SUBMISSION_UNKNOWN_OR_FAILED_DO_NOT_RESUBMIT',
          'parent': 3154, 'cpu_job_id': int(job) if job.isdigit() else None,
          'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
          'args': args, 'tool_sha256': SHA, 'review_sha256': RECEIPT_SHA}
with (DIRECTORY / 'SUBMISSION_RESULT.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(record), flush=True)
assert record['status'] == 'SUBMITTED_NOT_EXECUTED', record
