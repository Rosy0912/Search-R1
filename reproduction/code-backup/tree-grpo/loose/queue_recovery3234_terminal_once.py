"""One CPU afterany job for the owned3234; never submits training."""
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
DIRECTORY = ROOT / 'instruct300/post3234'
TOOL = DIRECTORY / 'verify-4f9aa0e6.py'
RECEIPT = DIRECTORY / 'READY_PREPARATION_RECEIPT.json'
SHA = '4f9aa0e605b3c9111e56edc46fb2d601cd701bd375861c0048a826e8b022f38b'
RECEIPT_SHA = '494c4e61b93ef61a744c3bc12e25e43f59449c4c2c07432d43b47454aadeb775'
assert hashlib.sha256(TOOL.read_bytes()).hexdigest() == SHA
assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA
review = json.loads(RECEIPT.read_bytes())
assert review['tool_sha256'] == SHA
assert review['status'] == 'READY_FOR_CPU_TERMINAL_EVIDENCE_CAPTURE_ONLY'
assert review['job_id']=='3234' and review['expected_new_updates']==200
assert review['source_commit']=='502e9fafa8e3aba722dbfcb493abc2b9d3ece326'
parent = subprocess.check_output(['scontrol', 'show', 'job', '3234'], text=True)
fields = dict(re.findall(r'(\S+?)=(\S+)', parent))
assert fields['JobId'] == '3234' and fields['JobState'] == 'RUNNING'
assert fields['NodeList'] == fields['BatchHost'] == 'h100-9' and fields['Restarts'] == '0'
intent = DIRECTORY / 'SUBMISSION_ONCE.json'
with intent.open('x') as stream:
    json.dump({'status': 'INTENT_RECORDED_DO_NOT_RESUBMIT', 'parent': 3234,
               'epoch': time.time(), 'tool_sha256': SHA, 'review_sha256': RECEIPT_SHA}, stream)
    stream.flush()
    os.fsync(stream.fileno())
execution = ('import pathlib,hashlib; p=pathlib.Path(' + repr(str(TOOL)) + '); '
             'b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()==' + repr(SHA) + '; '
             "exec(compile(b,str(p),'exec'))")
command = "export CUDA_VISIBLE_DEVICES=''; " + shlex.join([str(ROOT / 'env/bin/python'), '-c', execution])
args = ['sbatch', '--parsable', '-p', 'gpu', '-w', 'h100-9', '-N', '1', '--gres=gpu:0',
        '--ntasks=1', '--cpus-per-task=8', '--mem=32G', '--time=02:00:00',
        '--dependency=afterany:3234', '--job-name=tree3234-terminal-cpu',
        '--output=' + str(DIRECTORY / 'slurm-%j.out'), '--wrap=' + command]
result = subprocess.run(args, capture_output=True, text=True)
job = result.stdout.strip()
record = {'status': 'SUBMITTED_NOT_EXECUTED' if result.returncode == 0 and job.isdigit() else 'SUBMISSION_UNKNOWN_OR_FAILED_DO_NOT_RESUBMIT',
          'parent': 3234, 'cpu_job_id': int(job) if job.isdigit() else None,
          'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
          'args': args, 'tool_sha256': SHA, 'review_sha256': RECEIPT_SHA}
with (DIRECTORY / 'SUBMISSION_RESULT.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(record), flush=True)
assert record['status'] == 'SUBMITTED_NOT_EXECUTED', record
