"""Prepare generation3 without launching GPU work or changing older evidence.

The complete305-file scientific runtime is copied byte-for-byte from the
accepted preparation2 runtime. Only the future process attention backend
changes to XFORMERS. Failed generation2 outputs are never merged or resumed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

DEFAULT_BASE = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
DEFAULT_CANDIDATE = Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
CANDIDATE_PID = 2239529
CANDIDATE_DIGEST = 'b2626bd30e2c9ca12b94f17d4464ade6e012bbef36d7c2dd051c36e4f14fb614'
GPU7_UUID = 'GPU-6f04d45c-a34d-6617-030d-0635365f5395'
FAILURE_SHA = 'b3ab7529d001280155f7b08e95c889c48e59c80e06b1e5ba08849e7aeb409738'
PREP2_CONFIG_SHA = '7ee152cb0b4ea1f68a7786dfe37704502fdb21f4dcf9f8bdd134353e29090648'
PREP2_MANIFEST_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
PREP2_RUN_ID = 'd588a1450f8343edbed9c1e1901ac836'
OLD_PIDS = [2308665,2308989,2309103,2309105,2309109,2309113,2309114,2309115,2309183,2309271]
OLD_PROGRESS_OBSERVER = 2336661
ENDPOINT = 'http://10.120.17.101:18004/retrieve'

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
    for name in ['logs/opsd.log', 'logs/preparation-2-opsd.log', 'logs/preparation-3-opsd.log']:
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


def require_failure_and_owned_gpu_release(base, require_stable_failed=True):
    proof_path = base/'receipts/preparation-2-failure-preserved-20261007.json'
    if sha(proof_path) != FAILURE_SHA:
        raise RuntimeError('Root reviewed preparation2 failure proof SHA differs')
    proof = read_json(proof_path)
    if (proof.get('failure_preserved') is not True or proof.get('run_id') != PREP2_RUN_ID
        or proof.get('controller_pid') != 2308665 or proof.get('actual_opsd_updates') != 0
        or proof.get('opsd_output_absent') is not True
        or proof.get('scope') != 'Fault exit of preparation-2 only; not human pause, not OPSD updates'):
        raise RuntimeError('Failure proof does not bind the exact zero-update generation2 fault')
    if proof.get('old_owned_pids_terminal') != {str(pid): 'ABSENT' for pid in OLD_PIDS}:
        raise RuntimeError('Failure proof must identify exactly the ten owned controller/parent/worker PIDs')
    for pid in OLD_PIDS:
        if (Path('/proc')/str(pid)).exists():
            raise RuntimeError('Old owned PID is present; no process will be stopped: '+str(pid))
    observer = Path('/proc')/str(OLD_PROGRESS_OBSERVER)
    observer_state = 'ABSENT'
    if observer.exists():
        observer_state = (observer/'stat').read_text().rsplit(')',1)[1].split()[0]
        if observer_state != 'Z':
            raise RuntimeError('Old CPU progress observer has not yet terminated on controller loss')
    for key,item in proof['files'].items():
        path = Path(item.get('preserved',item['original']))
        if not path.is_relative_to(base) or sha(path) != item['sha256']:
            raise RuntimeError('Preserved generation2 evidence changed: '+key)
    frozen = proof['files']['receipts/pipeline-stage.json']
    old_stage = read_json(Path(frozen['preserved']))
    if old_stage.get('phase') != 'FAILED_PREPARATION2' or old_stage.get('controller_pid') != 2308665 or old_stage.get('run_id') != PREP2_RUN_ID:
        raise RuntimeError('Preserved generation2 terminal stage identity differs')
    if require_stable_failed and sha(base/'receipts/pipeline-stage.json') != frozen['sha256']:
        raise RuntimeError('Stable pointer no longer matches the preserved generation2 fault')
    lines = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    apps = [[piece.strip() for piece in line.split(',')] for line in lines if line.strip()]
    if len(apps) != 1 or len(apps[0]) != 3 or apps[0][0] != GPU7_UUID or apps[0][1] != str(CANDIDATE_PID):
        raise RuntimeError('All eight GPUs must contain only the approved candidate before startup; never stop a foreign process')
    if not 0 < int(apps[0][2]) <= 1024:
        raise RuntimeError('Candidate current process memory exceeds1GiB')
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    cards = [[piece.strip() for piece in line.split(',')] for line in lines if line.strip()]
    if sorted(int(row[0]) for row in cards) != list(range(8)) or not any(row[0]=='7' and row[1]==GPU7_UUID for row in cards):
        raise RuntimeError('Allocated physical eightGPU inventory differs')
    if any(int(row[2]) < 1024 for row in cards):
        raise RuntimeError('Direct memory.free below1GiB before initialization')
    return dict(failure_proof_path=str(proof_path),failure_proof_sha256=FAILURE_SHA,
        failed_run_id=PREP2_RUN_ID,old_owned_pids_terminal=proof['old_owned_pids_terminal'],
        old_cpu_observer_state=observer_state,preserved_failed_stage=frozen,
        actual_compute_apps=apps,actual_cards=cards,no_signals_sent=True)


def verify_prepared_config(config, base):
    old_path = base/'receipts/preparation-2-config-v2.json'
    if sha(old_path) != PREP2_CONFIG_SHA:
        raise RuntimeError('Accepted preparation2 configuration changed')
    old = read_json(old_path)
    paths = dict(source_model=str(base/'assets/archive100'),
        source_runtime=str(base/'runtime/opsd-preparation2'),runtime=str(base/'runtime/opsd-preparation3-xformers'),
        rollout_output=str(base/'runs/preparation-3-rollouts.jsonl'),
        pair_output=str(base/'runs/preparation-3-pairs.jsonl'),opsd_output=str(base/'runs/opsd'),
        final_output=str(base/'runs/preparation-3-final-test.jsonl'),
        required_failure_proof=str(base/'receipts/preparation-2-failure-preserved-20261007.json'))
    if config.get('prepared') is not True or config.get('files') != 305 or config.get('changed_files') != []:
        raise RuntimeError('New complete byte-identical305-file preparation is required')
    if any(config.get(key) != value for key,value in paths.items()):
        raise RuntimeError('Preparation3 path is outside the explicit reviewed whitelist')
    for key in ['datasets','source_grpo_step','source_model_transfer_receipt_sha256',
                'retrievers','execution_change','unchanged_scientific_function_proofs','unchanged_scientific_settings']:
        if config.get(key) != old[key]:
            raise RuntimeError('Original scientific/physical setting changed: '+key)
    if config['retrievers'] != dict(preparation=ENDPOINT,opsd_baseline_validation=ENDPOINT,final_eight_gpu_vllm=ENDPOINT):
        raise RuntimeError('All three admitted18004 endpoint bindings must remain unchanged')
    if config.get('environment_additions') != {'VLLM_ATTENTION_BACKEND':'XFORMERS'}:
        raise RuntimeError('Only the reviewed XFORMERS process override is permitted')
    if config.get('required_failure_sha256') != FAILURE_SHA or config.get('source_manifest_sha256') != PREP2_MANIFEST_SHA:
        raise RuntimeError('Failure/source runtime proof binding changed')
    controls = base/'source/preparation-3-xformers'
    expected_names = {'prepare_preparation3_xformers.py','pipeline_preparation3_xformers.py'}
    if set(config.get('controls',{})) != {str(controls/name) for name in expected_names}:
        raise RuntimeError('New controller/helper source path whitelist differs')
    for path,expected in config['controls'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError('Preparation3 control source changed: '+path)
    return old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--candidate-base',type=Path,default=DEFAULT_CANDIDATE)
    args = parser.parse_args();base,candidate=args.base,args.candidate_base
    if base != DEFAULT_BASE or candidate != DEFAULT_CANDIDATE or os.getuid() != 1005:
        raise RuntimeError('This preparation is bound to owned UID1005 HKUST1 paths')
    zero = assert_opsd_unstarted(base)
    release = require_failure_and_owned_gpu_release(base)
    gate = verify_candidate(candidate,require_ready=True)
    old_path = base/'receipts/preparation-2-config-v2.json'
    if sha(old_path) != PREP2_CONFIG_SHA:
        raise RuntimeError('Original preparation2 config SHA differs')
    old = read_json(old_path)
    manifest_path = base/'receipts/preparation-2-runtime.sha256.json'
    if sha(manifest_path) != PREP2_MANIFEST_SHA:
        raise RuntimeError('Original preparation2 manifest SHA differs')
    manifest = read_json(manifest_path);source = base/'runtime/opsd-preparation2'
    verify_manifest(source,manifest)
    target = base/'runtime/opsd-preparation3-xformers'
    controls = base/'source/preparation-3-xformers'
    if Path(__file__).resolve().parent != controls:
        raise RuntimeError('Run the deployed new helper from the reviewed new source directory')
    names = ['prepare_preparation3_xformers.py','pipeline_preparation3_xformers.py']
    if not all((controls/name).is_file() for name in names):
        raise RuntimeError('New helper/controller must both be present before preparation')
    config_path = base/'receipts/preparation-3-config.json'
    new_manifest_path = base/'receipts/preparation-3-runtime.sha256.json'
    if any(p.exists() for p in [target,config_path,new_manifest_path]):
        raise RuntimeError('Preparation3 artifact already exists; never overwrite or silently resume it')
    for key in ['rollout_output','pair_output','final_output']:
        output = base/('runs/'+Path(old[key]).name.replace('preparation-2','preparation-3'))
        if output.exists() or Path(str(output)+'.parts').exists():
            raise RuntimeError('New preparation3 output already exists')
    for relative in manifest:
        dest = target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/relative,dest)
    actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    if actual != set(manifest):
        raise RuntimeError('New runtime does not contain exactly the305 manifest files')
    verify_manifest(target,manifest)
    write_new_json(new_manifest_path,manifest)
    if sha(new_manifest_path) != PREP2_MANIFEST_SHA:
        raise RuntimeError('The305-entry copied manifest bytes differ')
    for dataset in old['datasets'].values():
        if sha(Path(dataset['path'])) != dataset['sha256']:
            raise RuntimeError('Original dataset hash changed')
    config = dict(old)
    config.pop('controller_correction',None);config.pop('required_stop_proof',None)
    config.update(time=time.time(),source_runtime=str(source),source_manifest_sha256=PREP2_MANIFEST_SHA,
        runtime=str(target),files=305,changed_files=[],new_manifest_sha256=sha(new_manifest_path),
        delta_sha256=None,controls={str(controls/name):sha(controls/name) for name in names},
        gate=gate,opsd_zero_evidence=zero,
        rollout_output=str(base/'runs/preparation-3-rollouts.jsonl'),
        pair_output=str(base/'runs/preparation-3-pairs.jsonl'),opsd_output=str(base/'runs/opsd'),
        final_output=str(base/'runs/preparation-3-final-test.jsonl'),
        environment_additions={'VLLM_ATTENTION_BACKEND':'XFORMERS'},
        identity_path_only_environment_changes=['E1_PROFILE','E1_RUN_ID','PYTHONPATH'],
        required_failure_proof=str(base/'receipts/preparation-2-failure-preserved-20261007.json'),
        required_failure_sha256=FAILURE_SHA,owner_fault_and_gpu_release=release,
        source_preparation2_config_path=str(old_path),source_preparation2_config_sha256=PREP2_CONFIG_SHA,
        original_manifest_sha256=old['source_manifest_sha256'],
        runtime_byte_identity=dict(files=305,all_source_bytes_equal=True,changed_files=[],
            same_manifest_sha256=PREP2_MANIFEST_SHA),
        numerical_scope='XFORMERS kernel arithmetic may change BF16 logits and sampled trajectories. Fresh full generation3, not bitwise continuation or Adam restoration.',
        scope='Fresh full generation3 after preserved zeroOPSD preparation2 CUDA fault; no GRPO or existing OPSD optimizer restarted; no live process stopped')
    verify_prepared_config(config,base)
    write_new_json(config_path,config)
    print(json.dumps(dict(prepared=True,config_path=str(config_path),config_sha256=sha(config_path),
        runtime=str(target),manifest_path=str(new_manifest_path),manifest_sha256=sha(new_manifest_path),
        scientific_runtime_files=305,changed_files=[],live_stopped=False,gpu_started=False)))


if __name__ == '__main__':
    main()
