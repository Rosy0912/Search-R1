"""Prepare an isolated execution copy; never stop or launch a live job.

Run on HKUST1 only after owner review.  Existing controller, source, runtime,
rollouts and receipts stay untouched.  Pipeline startup has a separate lock,
explicit preparation-1 stop proof and actual GPU ownership gate.
"""
import argparse
import ast
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.request


DEFAULT_BASE = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
DEFAULT_CANDIDATE = Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
CANDIDATE_PID = 2239529
CANDIDATE_DIGEST = 'b2626bd30e2c9ca12b94f17d4464ade6e012bbef36d7c2dd051c36e4f14fb614'
GPU7_UUID = 'GPU-6f04d45c-a34d-6617-030d-0635365f5395'
PATCHED_RELATIVE = 'search_e1/vllm_rollout.py'
OPSD_RELATIVE = 'search_e1/opsd.py'
ORIGINAL_VLLM_SHA = '3ab4138c6bea0b939aefbd2734304e2daccddd8ef1397182e873916e5aaefeb8'
ORIGINAL_CLAUSE = 'max_model_len=8192, enforce_eager=True, seed=seed)'
PATCHED_CLAUSE = 'max_model_len=8192, enforce_eager=True, seed=seed, num_gpu_blocks_override=22671)'
OLD_TIMEOUT = "dist.init_process_group('nccl', timeout=timedelta(hours=3))"
NEW_TIMEOUT = "dist.init_process_group('nccl', timeout=timedelta(hours=24))"


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def verify_manifest(root, manifest):
    if len(manifest) != 305:
        raise RuntimeError('Expected the complete 305-file frozen OPSD runtime')
    for relative, expected in manifest.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe frozen runtime manifest path')
        if sha(root / path) != expected:
            raise RuntimeError('Frozen runtime hash changed: ' + relative)


def assert_opsd_unstarted(base):
    if (base / 'runs/opsd').exists():
        raise RuntimeError('OPSD output already exists: never restart scientific training from zero')
    for name in ['logs/opsd.log', 'logs/preparation-2-opsd.log']:
        path = base / name
        if path.exists():
            with path.open(errors='replace') as stream:
                if any('REAL_OPSD_UPDATE' in line for line in stream):
                    raise RuntimeError('Real OPSD update evidence exists; migration is forbidden')
    # Match actual modules plus the experiment environment, not a stale state
    # file or a shell command containing an intended future OPSD stage.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if b'search_e1.opsd' not in args:
                continue
            env = dict(item.split(b'=', 1) for item in (process / 'environ').read_bytes().split(b'\0') if b'=' in item)
            if env.get(b'E1_BASE') == str(base).encode():
                raise RuntimeError('A live OPSD process exists: ' + process.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(actual_opsd_updates=0, output_absent=True, live_opsd_process_absent=True)


def validate_memory(memory, gate_start=None, gate_end=None):
    if memory.get('free_measurement') != 'nvidia-smi memory.free':
        raise RuntimeError('V2 directly measured memory.free is required, not total-used')
    if memory.get('candidate_pid') != CANDIDATE_PID or memory.get('source_identity_sha256') != CANDIDATE_DIGEST:
        raise RuntimeError('Memory evidence does not bind the approved candidate')
    start, end, updated = [float(memory[k]) for k in ['sample_start_unix', 'sample_end_unix', 'updated_at_unix']]
    peak, free = float(memory['max_process_mib']), float(memory['min_free_mib'])
    if not all(math.isfinite(x) for x in [start, end, updated, peak, free]):
        raise RuntimeError('Nonfinite memory evidence')
    if not start <= end <= updated <= time.time() + 2 or int(memory['samples']) < 2:
        raise RuntimeError('Invalid actual sample timestamps/count')
    if gate_start is not None and not start <= gate_start <= gate_end <= end:
        raise RuntimeError('Actual memory samples do not cover the full gate interval')
    if not 0 < peak <= 1024 or free < 1024:
        raise RuntimeError('Candidate peak <=1024 MiB and actual GPU free >=1024 MiB are required')


def verify_candidate(candidate, require_ready=False):
    gate_path = candidate / 'receipts/full-gpu-http-gate-v2.json'
    gate = read_json(gate_path)
    if gate.get('accepted') is not True or gate.get('scope') != 'full GPU and HTTP oracle equality':
        raise RuntimeError('V2 full GPU/HTTP gate is not accepted')
    if gate.get('candidate_pid') != CANDIDATE_PID or gate.get('source_identity_sha256') != CANDIDATE_DIGEST:
        raise RuntimeError('V2 gate has a different PID/source digest')
    checks = gate.get('same_gpu_operator_checks', [])
    if len(checks) != 24 or any(not all(c['comparison'].get(k) is True for k in ['scores_exact', 'ids_exact', 'shape_exact']) for c in checks):
        raise RuntimeError('Incomplete or failed same-embedding full GPU checks')
    stress = [c for c in checks if c['comparison'].get('batch_requests') == 8 and c['comparison'].get('batch_queries') == 20480 and c.get('queries') == 2560]
    if len(stress) != 8:
        raise RuntimeError('Actual single 8 x 2560 candidate stress scan is unproven')
    http = gate.get('k4_http_integration', {})
    if http.get('queries') != 320 or http.get('requests_to_k4') != 1 or http.get('complete_json_exact') is not True or http.get('sequential') is not True:
        raise RuntimeError('Single sequential real 320-query HTTP exact check is unproven')
    validate_memory(gate['memory'], float(gate['gate_start_unix']), float(gate['gate_end_unix']))
    current_memory_path = candidate / 'receipts/gpu7-memory-v2-continuous.json'
    current_memory = read_json(current_memory_path)
    validate_memory(current_memory, float(gate['gate_start_unix']), float(gate['gate_end_unix']))
    if time.time() - float(current_memory['updated_at_unix']) > 30:
        raise RuntimeError('Candidate V2 memory observer is stale')
    identity = read_json(candidate / 'receipts/source-identity.json')
    if identity['pid'] != CANDIDATE_PID or identity['source_identity_sha256'] != CANDIDATE_DIGEST:
        raise RuntimeError('Candidate source identity has changed')
    computed = hashlib.sha256(json.dumps(identity['identity'], sort_keys=True).encode()).hexdigest()
    if computed != CANDIDATE_DIGEST:
        raise RuntimeError('Candidate source identity digest does not recompute')
    sources = dict(identity['identity']['sources'])
    for key in ['original_scan', 'original_encoder']:
        source = identity['identity'][key]
        sources[source['path']] = source['sha256']
    for path, expected in sources.items():
        if sha(Path(path)) != expected:
            raise RuntimeError('Candidate/source runtime drift: ' + path)
    process = Path('/proc') / str(CANDIDATE_PID)
    if not process.exists() or process.stat().st_uid != os.getuid():
        raise RuntimeError('Approved owned candidate PID is not live')
    if str(candidate / 'source/server.py').encode() not in (process / 'cmdline').read_bytes():
        raise RuntimeError('Candidate PID ownership marker changed')
    with urllib.request.urlopen('http://10.120.17.101:18004/health', timeout=10) as response:
        health = json.load(response)
    if health.get('pid') != CANDIDATE_PID or health.get('source_identity_sha256') != CANDIDATE_DIGEST or health.get('gpu_ready') is not True:
        raise RuntimeError('Candidate live health identity/activation changed')
    if require_ready and health.get('ready') is not True:
        raise RuntimeError('Owner has not admitted /retrieve after the V2 gate')
    return dict(gate_path=str(gate_path), gate_sha256=sha(gate_path), memory_path=str(current_memory_path),
                candidate_pid=CANDIDATE_PID, source_identity_sha256=CANDIDATE_DIGEST,
                candidate_health=health, memory=current_memory)


def dataset_identity(path):
    with path.open('rb') as stream:
        count = sum(chunk.count(b'\n') for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''))
    return dict(path=str(path), realpath=str(path.resolve()), sha256=sha(path), rows=count, bytes=path.stat().st_size)


def unchanged_functions(before, after):
    left, right = ast.parse(before), ast.parse(after)
    proofs = {}
    for name in ['encoded_pair', 'pair_loss']:
        original = next(n for n in left.body if isinstance(n, ast.FunctionDef) and n.name == name)
        copied = next(n for n in right.body if isinstance(n, ast.FunctionDef) and n.name == name)
        original_bytes = ast.get_source_segment(before, original).encode()
        copied_bytes = ast.get_source_segment(after, copied).encode()
        original_ast = ast.dump(original, include_attributes=False)
        copied_ast = ast.dump(copied, include_attributes=False)
        if original_bytes != copied_bytes or original_ast != copied_ast:
            raise RuntimeError('Scientific OPSD function changed: ' + name)
        proofs[name] = dict(source_bytes_equal=True, ast_equal=True,
            source_sha256=hashlib.sha256(original_bytes).hexdigest(),
            ast_sha256=hashlib.sha256(original_ast.encode()).hexdigest())
    return proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--candidate-base', type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    base, candidate = args.base, args.candidate_base
    if base != DEFAULT_BASE or candidate != DEFAULT_CANDIDATE:
        raise RuntimeError('This reviewed migration is bound to the explicit HKUST1 experiment paths')
    # This preparation tool intentionally does not take experiment.lock held
    # by the still-running old preparation controller and never stops it.
    gate = verify_candidate(candidate)
    zero = assert_opsd_unstarted(base)
    old_manifest = read_json(base / 'receipts/opsd-runtime.sha256.json')
    source_root = base / 'runtime/opsd'
    verify_manifest(source_root, old_manifest)
    if old_manifest.get(PATCHED_RELATIVE) != ORIGINAL_VLLM_SHA:
        raise RuntimeError('Unexpected original vLLM source; do not patch an unknown runtime')
    old_text = (source_root / PATCHED_RELATIVE).read_text()
    if old_text.count(ORIGINAL_CLAUSE) != 1 or 'num_gpu_blocks_override' in old_text:
        raise RuntimeError('Expected one unchanged original vLLM constructor')
    old_opsd = (source_root / OPSD_RELATIVE).read_text()
    if old_opsd.count(OLD_TIMEOUT) == 1 and NEW_TIMEOUT not in old_opsd:
        new_opsd = old_opsd.replace(OLD_TIMEOUT, NEW_TIMEOUT)
        timeout_changed = True
        if new_opsd.replace(NEW_TIMEOUT, OLD_TIMEOUT) != old_opsd:
            raise RuntimeError('OPSD delta exceeds the single communication timeout clause')
        if ast.dump(ast.parse(new_opsd.replace(NEW_TIMEOUT, OLD_TIMEOUT)), include_attributes=False) != ast.dump(ast.parse(old_opsd), include_attributes=False):
            raise RuntimeError('OPSD normalized AST differs beyond the timeout')
    elif old_opsd.count(NEW_TIMEOUT) == 1 and OLD_TIMEOUT not in old_opsd:
        new_opsd, timeout_changed = old_opsd, False
    else:
        raise RuntimeError('Unexpected OPSD timeout source: refuse an unreviewed edit')
    function_proofs = unchanged_functions(old_opsd, new_opsd)
    target_root = base / 'runtime/opsd-preparation2'
    controls = base / 'source/preparation-2'
    receipt_path = base / 'receipts/preparation-2-config.json'
    reserved_outputs = [target_root, controls, receipt_path,
        base / 'receipts/preparation-2-runtime.delta.patch',
        base / 'receipts/preparation-2-runtime.sha256.json']
    if any(path.exists() for path in reserved_outputs):
        raise RuntimeError('Preparation-2 artifacts already exist; preserve them and inspect before any new attempt')
    for relative in old_manifest:
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    new_text = old_text.replace(ORIGINAL_CLAUSE, PATCHED_CLAUSE)
    (target_root / PATCHED_RELATIVE).write_text(new_text)
    (target_root / OPSD_RELATIVE).write_text(new_opsd)
    new_manifest = {relative: sha(target_root / relative) for relative in old_manifest}
    changed = [relative for relative in old_manifest if new_manifest[relative] != old_manifest[relative]]
    expected_changed = {PATCHED_RELATIVE, OPSD_RELATIVE} if timeout_changed else {PATCHED_RELATIVE}
    if set(changed) != expected_changed:
        raise RuntimeError('Unexpected runtime delta: ' + repr(changed))
    controls.mkdir(parents=True)
    for name in ['prepare_preparation2.py', 'pipeline_preparation2.py']:
        shutil.copy2(Path(__file__).with_name(name), controls / name)
    delta = '\n'.join(difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
        fromfile='runtime/opsd/' + PATCHED_RELATIVE, tofile='runtime/opsd-preparation2/' + PATCHED_RELATIVE, lineterm='')) + '\n'
    if timeout_changed:
        delta += '\n'.join(difflib.unified_diff(old_opsd.splitlines(), new_opsd.splitlines(),
            fromfile='runtime/opsd/' + OPSD_RELATIVE, tofile='runtime/opsd-preparation2/' + OPSD_RELATIVE, lineterm='')) + '\n'
    (base / 'receipts/preparation-2-runtime.delta.patch').write_text(delta)
    write_new_json(base / 'receipts/preparation-2-runtime.sha256.json', new_manifest)
    files = {relative for relative in new_manifest}
    if len(files) != 305:
        raise RuntimeError('Incomplete frozen execution copy')
    datasets = {name: dataset_identity(base / ('data/' + name + '.jsonl')) for name in ['train', 'validation', 'test']}
    if datasets['train']['rows'] != 169103 or datasets['validation']['rows'] != 512:
        raise RuntimeError('Original full training/512-validation budgets changed')
    config = dict(prepared=True, time=time.time(), preparation_only=True, live_processes_stopped=False,
        source_runtime=str(source_root), source_manifest_sha256=sha(base / 'receipts/opsd-runtime.sha256.json'),
        runtime=str(target_root), files=305, changed_files=changed,
        new_manifest_sha256=sha(base / 'receipts/preparation-2-runtime.sha256.json'),
        delta_sha256=sha(base / 'receipts/preparation-2-runtime.delta.patch'),
        controls={str(p): sha(p) for p in controls.glob('*.py')}, gate=gate, opsd_zero_evidence=zero,
        datasets=datasets, source_model=str(base / 'assets/archive100'), source_grpo_step=100,
        source_model_transfer_receipt_sha256=sha(base / 'receipts/archive100-transfer-accepted.json'),
        rollout_output=str(base / 'runs/preparation-2-rollouts.jsonl'), pair_output=str(base / 'runs/preparation-2-pairs.jsonl'),
        opsd_output=str(base / 'runs/opsd'), final_output=str(base / 'runs/preparation-2-final-test.jsonl'),
        retrievers=dict(preparation='http://10.120.17.101:18004/retrieve',
            opsd_baseline_validation='http://10.120.17.101:18004/retrieve',
            final_eight_gpu_vllm='http://10.120.17.101:18004/retrieve'),
        execution_change=dict(num_gpu_blocks_override=22671, inference_utilization=.85,
            cpu_blocks_expected_unchanged=7281, eager=True, communication_timeout_hours=24,
            communication_timeout_changed=timeout_changed,
            communication_timeout_reason='Tolerance for unequal per-rank validation/search latency; frozen global validation is eight ranks, validation[rank::world], 64 each for the unchanged 512 questions',
            description='GPU KV allocation and NCCL communication tolerance only; 24h is not training steps/epochs/budget; CPU swap/default and generation budgets unchanged'),
        unchanged_scientific_function_proofs=function_proofs,
        unchanged_scientific_settings=dict(questions=169103, siblings=5, workers=8, batch_questions=64,
            seed=42, worker_seeds=list(range(42,50)), temperature=1, top_p=1, max_tokens=500,
            context=8192, turns=5, search_calls=4, topk=3, observation_tokens=480,
            mining='unchanged original mine_pairs', opsd_epochs=1, opsd_lr=1e-5, rank=16, alpha=32,
            clip=10, chunk=32, accumulation=4, global_trajectory_batch=32, validation_questions=512,
            loss='per-trajectory valid-token mean, then global-trajectory mean', additional_grpo=False,
            restore_grpo_adam=False),
        required_stop_proof=str(base / 'receipts/preparation-1-stopped-for-preparation-2.json'),
        scope='Fresh full preparation only; no scientific OPSD optimizer update is restarted; owner must separately stop preparation-1 after scope review')
    write_new_json(receipt_path, config)
    print(json.dumps(dict(prepared=True, receipt=str(receipt_path), runtime=str(target_root),
                         controls=str(controls), changed_files=changed, live_stopped=False)))


if __name__ == '__main__':
    main()
