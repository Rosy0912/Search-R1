"""One immutable, already deployed batching core; never import Torch in parent."""
import hashlib,json,sys
from pathlib import Path

CORE=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/hf-batch-core-0f226d2/opsd_signal_weight_hf_batch_v1')
PUBLICATION=CORE.parents[2]/'receipts/hf-batch-core-publication-0f226d2.json'
PUBLICATION_SHA='91a38cff35a9c07c9dfb3c18949ce04dd5d5c82b0dd0f78b414b8fc2163bc7db'
COMMIT='0f226d21998ae70326b14be8587a83638b99f9ff'

def verify_core():
    raw=PUBLICATION.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PUBLICATION_SHA:raise ValueError('Batch core publication changed')
    pub=json.loads(raw);manifest=CORE/'source-manifest.json';data=manifest.read_bytes()
    if not pub['accepted'] or pub['source_commit']!=COMMIT or pub['remote_readback_sha']!=COMMIT or hashlib.sha256(data).hexdigest()!=pub['source_manifest_sha256']:
        raise ValueError('Batch core identity differs')
    for name,item in json.loads(data)['files'].items():
        if hashlib.sha256((CORE/name).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Batch core source changed: '+name)
    return dict(path=str(CORE),source_commit=COMMIT,publication=dict(path=str(PUBLICATION),sha256=PUBLICATION_SHA),
        manifest=dict(path=str(manifest),sha256=hashlib.sha256(data).hexdigest()))

def import_core(bound):
    if verify_core()!=bound:raise ValueError('Worker batch core differs from parent')
    sys.path.insert(0,str(CORE))
    from search_batch import run_questions
    from hf_backend import greedy_batch
    return run_questions,greedy_batch
