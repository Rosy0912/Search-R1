"""CPU-only final owner admission of the frozen resident candidate.

Does not initialize CUDA, send retrieval queries, signal a process or modify a
source/runtime/route. Root authorized admission after real full acceptance.
"""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path('/data/shaoshitong/search-e1-retriever-resident-20261008')
EXPECTED = dict(pid=4189880, uid=1005, starttick=922361711,
                source_identity_sha256='c1265ec3f5884e1b3cc61ca52dfedcbf21992ad50bcbf144deb0c9fd85d5152e')
RUN = 'resident-09a3c264ec76419f9c5da1c6eb14b364'
GATE_SHA = 'fe38729cdcb882e0c140f7be88737ad80bd3c43e6deac34c98c0623e9e7e5317'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_new(path, value):
    if path.exists():
        raise RuntimeError('Evidence/control already exists: ' + str(path))
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    gate_path = BASE / 'receipts/resident-full-gate-v1.json'
    if sha(gate_path) != GATE_SHA:
        raise RuntimeError('Accepted full gate bytes changed')
    gate = json.loads(gate_path.read_text())
    if gate['accepted'] is not True or gate['scope'] != 'full GPU and HTTP oracle equality':
        raise RuntimeError('Gate is unaccepted')
    if gate['candidate_pid'] != EXPECTED['pid'] or gate['source_identity_sha256'] != EXPECTED['source_identity_sha256']:
        raise RuntimeError('Gate belongs to another candidate')
    if any(gate['candidate_health'][k] != EXPECTED[k] for k in EXPECTED):
        raise RuntimeError('Gate candidate UID/starttick differs')
    checks = gate['cross_device_operator_checks']
    if len(checks) != 24:
        raise RuntimeError('All 24 actual full GPU comparisons required')
    sizes = [1, 63, 64, 65, 128, 129, 320, 2560]
    if [c['queries'] for c in checks[:8]] != sizes or [c['queries'] for c in checks[8:16]] != sizes:
        raise RuntimeError('Full query/tail shape coverage differs')
    for i, check in enumerate(checks):
        c = check['comparison']
        if any(c.get(k) is not True for k in ['scores_exact', 'ids_exact', 'shape_exact']):
            raise RuntimeError('Numerical/operator equality failed')
        if c['full_ntotal'] != 21015324 or c['vector_tiles'] != 642 or not c['actual_gemm_shapes']:
            raise RuntimeError('Missing actual complete-index operator shape trace')
        if check['topk'] != 3 + (i % 8) % 2 or check['return_scores'] is not (i < 8 or i >= 16):
            raise RuntimeError('Topk/return_scores coverage differs')
        if i >= 16 and (check['queries'] != 2560 or c['batch_requests'] != 8 or c['batch_queries'] != 20480):
            raise RuntimeError('Actual shared 8 x 2560 stress required')
    integration = gate['gpu7_http_integration']
    if integration['queries'] != 320 or integration['topk'] != 3 or integration['return_scores'] is not True or integration['complete_json_exact'] is not True or integration['sequential'] is not True:
        raise RuntimeError('One complete 320-question original HTTP equality required')
    # Legacy field is named requests_to_k4, although the bound oracle is K1 GPU7.
    if integration['requests_to_k4'] != 1:
        raise RuntimeError('Unexpected original HTTP load')
    entry = BASE / 'source/full_live_gate.py'
    prepared = json.loads((BASE / 'receipts/cpu-prepared.json').read_text())
    for relative, proof in prepared['source_runtime_files'].items():
        path = BASE / relative
        if path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
            raise RuntimeError('Frozen source/runtime changed: ' + relative)
    spec = importlib.util.spec_from_file_location('published_full_gate', entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    before = module.get_json('http://10.120.17.101:18005/health')
    if any(before[k] != EXPECTED[k] for k in EXPECTED) or not before['gpu_ready'] or before['ready']:
        raise RuntimeError('Expected activated, unadmitted candidate required')
    identity = json.loads((BASE / 'receipts/source-identity.json').read_text())
    module.verify_source(identity, before)
    environ = dict(x.split(b'=', 1) for x in (Path('/proc') / str(EXPECTED['pid']) / 'environ').read_bytes().split(b'\0') if b'=' in x)
    if environ.get(b'E1_RUN_ID', b'').decode() != RUN or environ.get(b'CUDA_VISIBLE_DEVICES') != b'0,1,2,3':
        raise RuntimeError('Actual run/GPU mask differs')
    bindings = json.loads((BASE / 'receipts/asset-bindings.json').read_text())
    for proof in bindings['files']:
        path = Path(proof['path'])
        st = path.stat()
        measured = dict(realpath=str(path.resolve()), size=st.st_size,
                        mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                        inode=st.st_ino, device=st.st_dev)
        if any(measured[k] != proof[k] for k in measured):
            raise RuntimeError('Bound asset file identity changed')
    before['gpu_inventory_uuids'] = [x['uuid'] for x in before['gpu_inventory']]
    module.verify_oracle(module.get_json('http://10.120.17.101:18004/health'), Path('/data/shaoshitong/search-e1-retriever-segmented-20261007/receipts/source-identity.json'))
    verified_at = time.time()
    memory = module.memory_covering(BASE / 'receipts/resident-memory-continuous.json', before, gate['gate_start_unix'], verified_at)
    if memory['observer']['source_sha256'] != prepared['source_runtime_files']['source/memory_observer.py']['sha256']:
        raise RuntimeError('Unreviewed observer source')
    if sha(gate_path) != GATE_SHA:
        raise RuntimeError('Gate mutated before admission')
    control = dict(**EXPECTED, accepted=True, run_id=RUN,
                   full_gpu_http_gate_path=str(gate_path), full_gpu_http_gate_sha256=GATE_SHA,
                   admitted_at_unix=time.time(),
                   scope='Owner-authorized private 18005 admission after actual full 24-check GPU/HTTP gate. Original 18004/public routes unchanged.')
    control_path = BASE / 'receipts/admit-requests.json'
    atomic_new(control_path, control)
    after = module.get_json('http://10.120.17.101:18005/health')
    if not after['ready'] or any(after[k] != EXPECTED[k] for k in EXPECTED):
        raise RuntimeError('Bound admission did not produce ready=true')
    receipt = dict(accepted=True, endpoint='http://10.120.17.101:18005', **EXPECTED, run_id=RUN,
        verified_at_unix=verified_at,
        completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_gate=dict(path=str(gate_path), sha256=GATE_SHA, checks=24,
            gate_start_unix=gate['gate_start_unix'], gate_end_unix=gate['gate_end_unix'],
            actual_gate_seconds=gate['gate_end_unix']-gate['gate_start_unix']),
        admission=dict(path=str(control_path), sha256=sha(control_path)),
        cpu_prepared=dict(path=str(BASE/'receipts/cpu-prepared.json'), sha256=sha(BASE/'receipts/cpu-prepared.json'), frozen_files_rehashed=len(prepared['source_runtime_files'])),
        memory=dict(path=str(BASE/'receipts/resident-memory-continuous.json'), snapshot=memory),
        gpu7_http_integration=integration, health_after=after,
        executor=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        scope='No retrieval query, GPU initialization, process signal, runtime/source modification, old evaluation restart or public route mutation. Publication handled serially by root.')
    receipt_path = BASE / 'receipts/admission-verified.json'
    atomic_new(receipt_path, receipt)
    print(json.dumps(dict(ready=True, endpoint=receipt['endpoint'], gate_sha256=GATE_SHA,
        admission_path=str(control_path), admission_sha256=sha(control_path),
        verification_path=str(receipt_path), verification_sha256=sha(receipt_path),
        gate_seconds=receipt['full_gate']['actual_gate_seconds'],
        process_peak_mib=memory['max_process_mib'], direct_free_min_mib=memory['min_free_mib'],
        completed_at_utc=receipt['completed_at_utc'])))


if __name__ == '__main__':
    main()
