"""External transport-only small-chunk adapter; never modifies a live/frozen file."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'topd-official-resume-transfer'
ARCHIVE_SHA='f6b25cf3095917001abf468b1fa0a0869f50c3420b624e1d36840e11571e216a'
MANIFEST_SHA='f88ca6578e135bb684ced13c5c44b2dea3e265f34041693d9d8a88e3fa0047af'
TRANSFER_SHA='11f703ec21139b437eb5044703041077af310e196acc5b2b9bdcd634f0f95d0a'
DEFAULT_CHUNK_MIB=8


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_modules():
    require(sha(HERE/'frozen-candidate-v6.tar.gz')==ARCHIVE_SHA,'Bound frozen v6 archive differs')
    require(sha(FROZEN/'files.json')==MANIFEST_SHA,'Bound frozen manifest differs')
    manifest=json.loads((FROZEN/'files.json').read_text())
    for name,digest in manifest['files'].items():require(sha(FROZEN/name)==digest,'Frozen tool bytes differ: '+name)
    require(sha(FROZEN/'transfer.py')==TRANSFER_SHA,'Frozen relay differs')
    sys.path.insert(0,str(FROZEN))
    transfer=importlib.import_module('transfer');v6=importlib.import_module('downloader_v6')
    require(Path(transfer.__file__).resolve()==(FROZEN/'transfer.py').resolve() and Path(v6.__file__).resolve()==(FROZEN/'downloader_v6.py').resolve(),'Foreign imported module')
    return transfer,v6


def require_current_handoff(v6,path):
    proof=json.loads(Path(path).read_text());captured={x['pid'] for x in proof.get('legacy_local_identities',[])}
    # Frozen handoff knows original v5 IDs; this adapter additionally requires
    # the latest recorded v6 parent in the actual new exit inventory. Its child
    # subtree completeness must still be independently captured/reviewed by ROOT.
    if v6.STATE.exists():
        state=json.loads(v6.STATE.read_text());pid=state.get('pid')
        if type(pid) is int and pid!=os.getpid():
            require(pid in captured,'Current v6 parent missing from new actual handoff')
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Current v6 still alive; no adapter startup')
    return proof


def set_small_chunk(transfer,mib):
    require(type(mib) is int and mib in (1,2,4,8),'Allowed transport chunks1/2/4/8MiB only')
    size=mib*1024**2;require(size<=8*1024**2,'Maximum small transport chunk8MiB')
    # Single assignment before any worker exists; all slots share one immutable
    # chosen value. No per-thread global mutation or source/receiver alteration.
    transfer.CHUNK=size
    return dict(scope='TRANSPORT_CHUNK_ONLY_NO_BANDWIDTH_PROMISE',chunk_bytes=size,timeout_seconds=600,
        sustained_raw_rate_to_complete_within_timeout_mib_s=mib/600.,frozen_receiver_max_bytes=64*1024**2)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--chunk-mib',type=int,choices=(1,2,4,8),default=DEFAULT_CHUNK_MIB)
    custom,remaining=parser.parse_known_args()
    handoff=None
    for i,arg in enumerate(remaining):
        if arg=='--exclusive-handoff' and i+1<len(remaining):handoff=Path(remaining[i+1])
        elif arg.startswith('--exclusive-handoff='):handoff=Path(arg.split('=',1)[1])
    require(handoff is not None,'Real fresh exclusive handoff required before adapter startup')
    transfer,v6=frozen_modules();require_current_handoff(v6,handoff)
    config=set_small_chunk(transfer,custom.chunk_mib)
    print(json.dumps(dict(config,candidate_archive_sha256=ARCHIVE_SHA,frozen_transfer_sha256=TRANSFER_SHA)),flush=True)
    sys.argv=[sys.argv[0]]+remaining
    # Unchanged full24file pipeline, officialmetadataSHA/LFS/Git/noGold-independent
    # transport gates, parent download.lock, prefix/resume and state/unlink order.
    v6.main()


if __name__=='__main__':main()
