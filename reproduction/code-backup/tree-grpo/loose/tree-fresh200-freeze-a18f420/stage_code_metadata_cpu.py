"""Stage a byte-bound code/metadata envelope on h100-9; execute nothing."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


DESTINATION = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/preparation-a18f420-20261007')
FILES = {
    'CODE_PACKAGE.tar.gz', 'PREPARATION_INPUTS.json', 'A18F_BOUNDED_CPU_RECEIPT.json', 'A18F_AFFECTED_SOURCE_MANIFEST.json', 'A18F_SOURCE_ONLY_REVIEW.json', 'HISTORICAL_53082E_CONTEXT_RECEIPT.json',
    'fresh200_dispatch_common.py', 'prepare_tree_fresh200_package.py',
    'submit_tree_fresh200_once.py', 'SOURCE_INTERFACE_RECEIPT.json',
    'SOURCE_CONTEXT_FREE_RECEIPT.json', 'SOURCE_BYTE_AUDIT.json',
    'HELPER_SOURCE_REVIEW.json', 'RETAINED_TERMINAL3314_PROOF_V2.json', 'RETAINED_TERMINAL3317_PROOF.json',
}


def main():
    if socket.gethostname().split('.')[0] != 'h100-9':
        raise ValueError('Only the existing h100-9 management route is allowed')
    if len(sys.argv) != 2 or len(sys.argv[1]) != 64:
        raise ValueError('Exact envelope SHA256 is required')
    raw = sys.stdin.buffer.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != sys.argv[1]:
        raise ValueError('Envelope size or SHA256 differs')
    envelope = json.loads(raw)
    if envelope['status'] != 'REVIEWED_CODE_METADATA_CPU_STAGE_ONLY' or set(envelope['files']) != FILES:
        raise ValueError('Exact code/metadata-only inventory required')
    contents = {}
    total = 0
    for name, record in envelope['files'].items():
        content = base64.b64decode(record['base64'], validate=True)
        total += len(content)
        if total > 32 * 1024 * 1024 or len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Envelope member size/SHA256 differs: ' + name)
        contents[name] = content
    if DESTINATION.parent.resolve() != DESTINATION.parent or DESTINATION.parent.name != 'instruct300':
        raise ValueError('Staging parent must be the existing task directory without symlinks')
    os.umask(0o077)
    DESTINATION.mkdir(exist_ok=False)
    for name in sorted(contents):
        with (DESTINATION / name).open('xb') as stream:
            stream.write(contents[name])
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {
        'status': 'CODE_METADATA_STAGED_ONLY_NOT_EXECUTED',
        'directory': str(DESTINATION), 'envelope_sha256': sys.argv[1],
        'files': {name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
                  for name, content in contents.items()},
        'asset_transfer': False, 'cuda_initialized': False,
        'slurm_submitted': False, 'optimizer_writes': 0,
    }
    with (DESTINATION / 'STAGING_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
