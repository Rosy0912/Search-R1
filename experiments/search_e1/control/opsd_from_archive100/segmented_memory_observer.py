"""Observe only the independent HKUST1 candidate and its allocated GPU7."""
from pathlib import Path
import json
import os
import subprocess
import time

b = Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
identity_path = b / 'receipts/source-identity.json'
while not identity_path.exists():
    time.sleep(1)
identity = json.loads(identity_path.read_text())
pid = identity['pid']
process = Path('/proc') / str(pid)
assert process.stat().st_uid == os.getuid()
assert str(b / 'source/server.py').encode() in (process / 'cmdline').read_bytes()
expected_uuid = 'GPU-6f04d45c-a34d-6617-030d-0635365f5395'
started = time.time()
minimum_free = float('inf')
maximum_process = 0
samples = 0
sample_log = b / 'receipts/gpu7-memory-v2-samples.jsonl'
receipt_path = b / 'receipts/gpu7-memory-v2-continuous.json'
while process.exists():
    try:
        card_rows = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,memory.reserved',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        card = next(row.split(', ') for row in card_rows if row.startswith('7, '))
        assert card[1] == expected_uuid, 'Physical allocated GPU7 changed'
        total, used, free, reserved = map(int, card[2:6])
        apps = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        current_process = 0
        for row in apps:
            fields = [part.strip() for part in row.split(',')]
            if len(fields) == 3 and fields[1] == str(pid):
                assert fields[0] == expected_uuid, 'Candidate allocated an unapproved GPU'
                current_process += int(fields[2])
        sampled_at = time.time()
        minimum_free = min(minimum_free, free)
        maximum_process = max(maximum_process, current_process)
        samples += 1
        sample = dict(time=sampled_at, candidate_pid=pid, gpu=7, uuid=expected_uuid,
                      total_mib=total, used_mib=used, free_mib=free, reserved_mib=reserved,
                      computed_total_minus_used_mib=total-used,
                      candidate_process_mib=current_process)
        with sample_log.open('a') as handle:
            handle.write(json.dumps(sample) + '\n')
        report = dict(candidate_pid=pid, source_identity_sha256=identity['source_identity_sha256'],
                      gpu=7, uuid=expected_uuid, sample_start_unix=started,
                      sample_end_unix=sampled_at, updated_at_unix=time.time(),
                      samples=samples, max_process_mib=maximum_process,
                      min_free_mib=minimum_free, current=sample,
                      free_measurement='nvidia-smi memory.free',
                      process_limit_mib=1024, free_minimum_mib=1024,
                      limits_met=maximum_process<=1024 and minimum_free>=1024,
                      scope='V2 directly sampled memory.free and candidate process on allocated GPU7; preserve V1 computed reserve separately')
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(receipt_path)
    except (subprocess.CalledProcessError, StopIteration, ValueError) as error:
        print('OBSERVATION_ERROR', repr(error), flush=True)
    time.sleep(.2)
print('CANDIDATE_PROCESS_EXITED', pid, flush=True)
