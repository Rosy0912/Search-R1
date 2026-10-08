"""Place one independently sealed zero-only review receipt; no job execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

TARGET = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/fresh200-mu-02b1-v11')
MANIFEST_SHA = 'a07925ce76951ef0966b5f03e847e28b0b2fdc115a6409e2bdede181300b00a2'
COMMIT = '02b1ea2688ff3fb032ee6699003a7e09d651f166'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(socket.gethostname().split('.')[0] == 'h100-9', 'Only h100-9 receipt metadata placement is allowed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Receipt placement requires explicit CPU visibility')
    require(len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1]), 'Sealed independent receipt SHA required')
    raw = sys.stdin.buffer.read(512 * 1024 + 1)
    require(len(raw) <= 512 * 1024 and hashlib.sha256(raw).hexdigest() == sys.argv[1], 'Receipt bytes/size differ')
    require(TARGET.resolve() == TARGET and TARGET.is_dir(), 'Existing own prepared target required')
    manifest_bytes = (TARGET / 'manifest.json').read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, 'Prepared package manifest differs')
    manifest = json.loads(manifest_bytes)
    require(manifest['code_commit'] == COMMIT and manifest['stage'] == 'fresh_mu_calibration', 'Wrong source or stage')
    receipt = json.loads(raw)
    require(receipt['status'] == 'READY_FOR_FRESH200_BEHAVIOR_MU_ZERO_WRITE_CALIBRATION', 'Only independently accepted zero stage can be placed')
    require(receipt['manifest_sha256'] == MANIFEST_SHA and receipt['code_commit'] == COMMIT, 'Receipt package/source differs')
    require(receipt['requires_zero_optimizer_writes'] is True and receipt['requires_actual_behavior_mu_binding'] is True, 'Zero-only constraints missing')
    for key in ('actual_3b_training_accepted', 'long_training_ready', 'original_sampler_actor_gate_accepted', 'production_thresholds_accepted'):
        require(receipt[key] is False, 'Receipt placement cannot imply training or probability acceptance')
    os.umask(0o077)
    destination = TARGET / 'PRE_DISPATCH_RECEIPT.json'
    with destination.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(TARGET, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({'status': 'ZERO_ONLY_REVIEW_RECEIPT_STAGED_NOT_SUBMITTED', 'path': str(destination), 'sha256': sys.argv[1], 'bytes': len(raw), 'asset_transfer': False, 'gpu_or_slurm_execution': False, 'optimizer_writes': 0}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
