"""Hardware-only callable gate. Caller supplies reviewed teacher and owns lifecycle.

No Slurm submission, services, model download, optimizer or automatic approval.
All outputs remain UNREVIEWED; parent must kill its own process on a live thread.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import socket
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/topd')
SOURCE_SHA = 'fede3c23e882139a12a2b50c531177438012959774958fc39e03fb3f9d29e229'
METHOD_SHA = 'e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'
PARAMETERS = 32763876352
DATA_SHA = 'af397b98b1601b517c01663171ffe543573005b9d31d24cace482480751a6756'
CANDIDATE_FILES = {
    'pipeline.py': '8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde',
    'capacity_reference.py': '2d5da3fdd94b637c1625677b302df4df21395ef138f9de2ce484035eab7298ea',
    'native_gate.py': 'c945b54294a4e7d222aa6a8c964f2ac8aa5349c0a3d3199532fe0dbba18e95b0',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def local(path):
    path = Path(path).resolve()
    require(ROOT in path.parents, 'Evidence must be own local ROOT')
    return path


def write_json(path, payload):
    with Path(path).open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def check_hf_tensor_index(model, shards, expected_numel=PARAMETERS):
    """Inspect all actual tensor headers without full tensor/model loading."""
    from safetensors import safe_open
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    require(set(index.values()) == set(shards), 'Index shard coverage differs')
    tensors, numel = set(), 0
    for name in shards:
        with safe_open(str(model/name), framework='pt', device='cpu') as shard:
            for key in shard.keys():
                require(key not in tensors and index.get(key) == name, 'HF duplicate/unindexed tensor')
                tensor = shard.get_slice(key)
                require(tensor.get_dtype() in ('BF16', 'F32'), 'Unsupported teacher weight dtype')
                size = 1
                for dim in tensor.get_shape():
                    require(type(dim) is int and dim > 0, 'Invalid HF tensor shape')
                    size *= dim
                numel += size
                tensors.add(key)
    require(tensors == set(index) and numel == expected_numel, 'Full32B HF tensor/header inventory differs')


def check_identity(provenance, candidate_directory):
    """Reviewed provenance is prepared independently, never manufactured here."""
    require(provenance['scope'] == 'REVIEWED_32B_TEACHER_SOURCE_AND_QUALITY_NOT_GPU_ACCEPTANCE',
            'Need independent selected32B identity; old3B/toy/hardware proof forbidden')
    require(provenance['status'] == 'REVIEWED', 'Teacher provenance not reviewed')
    for key in ('independent_review', 'quality_result', 'model_inventory', 'cpu_identity'):
        ref = provenance[key]
        require(digest(local(ref['path'])) == ref['sha256'], 'Provenance artifact drift: ' + key)
    review = json.loads(local(provenance['independent_review']['path']).read_text())
    payload = {k: v for k, v in provenance.items() if k != 'independent_review'}
    canonical = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(review['status'] == 'PASS' and review['approval_scope'] == '32B_TEACHER_SOURCE_AND_QUALITY_ONLY'
            and review['reviewed_payload_sha256'] == canonical, 'Independent provenance payload mismatch')
    inventory = json.loads(local(provenance['model_inventory']['path']).read_text())
    require(inventory['status'] == 'FULL_HF_FILES_VERIFIED' and inventory['shards']
            and len(inventory['shards']) == len(set(inventory['shards'])),
            'Need actual complete32B HF shards')
    model = local(inventory['model_directory'])
    require(set(inventory['files']) == set(inventory['shards']) | {'config.json', 'generation_config.json',
            'tokenizer_config.json', 'tokenizer.json', 'merges.txt', 'vocab.json', 'model.safetensors.index.json'},
            'Need complete HF/tokenizer files, not a shard-only receipt')
    require(set(inventory['shards']) == {n for n in inventory['files'] if n.endswith('.safetensors')},
            'Incomplete HF shard identity')
    for name, value in inventory['files'].items():
        path = (model / name).resolve()
        require(model in path.parents and path.stat().st_size == value['size']
                and digest(path) == value['sha256'], 'Actual HF bytes drift: ' + name)
    check_hf_tensor_index(model, inventory['shards'])
    cpu_identity = json.loads(local(provenance['cpu_identity']['path']).read_text())
    require(cpu_identity['status'] == 'CPU_FP32_IDENTITY_CAPTURE_UNREVIEWED_NOT_TEACHER_QUALITY'
            and cpu_identity['snapshot']['parameter_count'] == PARAMETERS
            and cpu_identity['snapshot']['parameter_sha256'] == provenance['expected_fp32_parameter_sha256']
            and cpu_identity['model_directory'] == str(model)
            and cpu_identity['hf_file_sha256'] == {n:v['sha256'] for n,v in inventory['files'].items()},
            'Expected FP32 identity must derive from actual source-bound CPU preload')
    kind = provenance['teacher_kind']
    quality = json.loads(local(provenance['quality_result']['path']).read_text())
    require(quality['denominator'] == 64 and quality['metric'] == 'EM' and quality['greedy'] is True
            and quality['data_sha256'] == provenance['data_sha256'] == DATA_SHA, 'Need actual original64 greedy EM identity')
    correct = quality['correct']
    require(type(correct) is int and 0 <= correct <= 64, 'Noninteger quality result')
    if kind == 'OFFICIAL_QWEN2_5_32B_BASE':
        require(len(inventory['shards']) == 17 and len(inventory['files']) == 24,
                'Official Base requires exact17 shards plus7 metadata files')
        metadata = json.loads(local(provenance['official_metadata_path']).read_text())
        require(digest(local(provenance['official_metadata_path'])) == 'e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1', 'Official metadata bytes drift')
        official = {x['rfilename']: x for x in metadata['siblings']}
        for name, value in inventory['files'].items():
            require(name in official and value['size'] == official[name]['size'], 'Official inventory size drift')
            if 'lfs' in official[name]:
                require(value['sha256'] == official[name]['lfs']['sha256'], 'Official LFS identity drift')
            else:
                raw = (model/name).read_bytes()
                require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == official[name]['blobId'], 'Official Git blob drift')
        require(correct >= 26 and provenance['official_revision'] == '1818d35814b8319459f4bd55ed1ac8709630f003'
                and provenance['official_metadata_sha256'] == 'e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1',
                'Base32B quality/source not accepted')
    elif kind == 'SEARCH_G1_GRPO100_32B':
        base_ref = provenance['base_quality_result']
        require(digest(local(base_ref['path'])) == base_ref['sha256'], 'Prior base quality drift')
        base_quality = json.loads(local(base_ref['path']).read_text())
        require(base_quality['denominator'] == 64 and type(base_quality['correct']) is int
                and 0 <= base_quality['correct'] < 26 and base_quality['greedy'] is True
                and base_quality['metric'] == 'EM' and base_quality['data_sha256'] == provenance['data_sha256'],
                'G1 fallback must bind actual original base64 below26')
        ref = provenance['training_receipt']
        require(digest(local(ref['path'])) == ref['sha256'], 'G1 training receipt drift')
        training = json.loads(local(ref['path']).read_text())
        require(training['status'] == 'INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES'
                and training['actual_outer_updates'] == 100 and training['algorithm'] == 'SEARCH_G1_GRPO'
                and training['source_reference'] == '4e7448d66a78996467134441571fda0a6844d180'
                and training['hf_inventory_sha256'] == provenance['model_inventory']['sha256'],
                'No reviewed actual G1-100 teacher export')
        require(provenance['expected_fp32_parameter_sha256'] != provenance['base_fp32_parameter_sha256'],
                'G1 export must contain actual parameter changes')
    else:
        raise RuntimeError('Unknown teacher kind; no base fallback')
    candidate_directory = local(candidate_directory)
    require(provenance['candidate_files'] == CANDIDATE_FILES,
            'Frozen candidate scope differs')
    for name, expected in provenance['candidate_files'].items():
        require(digest(candidate_directory / name) == expected, 'Candidate byte drift: ' + name)
    return model


def check_inputs(tensors, pad):
    import torch
    require(set(tensors) == {'prompts', 'responses', 'attention', 'row_indices'}, 'Token-only/noGold')
    p, r, a, i = [tensors[k] for k in ('prompts', 'responses', 'attention', 'row_indices')]
    require(all(t.device.type == 'cpu' and t.dtype == torch.long for t in tensors.values()), 'CPU int64 only')
    require(p.shape == (16, 2048) and r.shape == a.shape == (16, 4096)
            and torch.equal(i, torch.arange(16)), 'Require full16-row ordered6144 capacity input')
    require(bool(((a == 0) | (a == 1)).all()), 'Nonbinary attention')
    lengths = (p != pad).sum(-1) + a.sum(-1)
    require(int(lengths.max()) == 6144 and bool((lengths > 0).all()), 'Actual6144 token coverage absent')
    require(bool((a == 0).any()), 'Need actual masked positions as well as full-length row')
    return lengths.tolist()


def assert_release(record, uuids):
    require(record['status'] == 'RELEASED', 'Phase did not release')
    require(record['original_method_sha256'] == METHOD_SHA, 'Original method changed')
    initial, staged, final = [record[k] for k in ('initial_cpu', 'staged_frozen', 'released_cpu')]
    require(initial['parameter_count'] == PARAMETERS and initial['parameter_bytes'] == PARAMETERS * 4,
            'Toy/3B/substitute parameter inventory forbidden')
    for snap in (initial, staged, final):
        require(snap['parameter_sha256'] == initial['parameter_sha256']
                and snap['buffer_sha256'] == initial['buffer_sha256']
                and snap['parameter_count'] == PARAMETERS
                and snap['training_modules'] == snap['requires_grad_parameters'] == snap['grad_parameters'] == 0,
                'Frozen FP32 identity differs')
    require(staged['parameter_devices'] == ['cuda:%d' % i for i in range(8)], 'Teacher not actually partitioned to8')
    require(initial['parameter_devices'] == final['parameter_devices'] == ['cpu']
            and all(d == 'cpu' for d in final['buffer_devices']), 'CPU return incomplete')
    values = record['allocators_released']
    require(len(values) == 8 and {v['uuid'] for v in values} == set(uuids)
            and {v['device'] for v in values} == set(range(8))
            and all(v['uuid'] == uuids[v['device']] for v in values)
            and all(v['allocated_bytes'] == v['reserved_bytes'] == 0 for v in values), 'All8 release0 required')
    require(record['actual_flags'] == dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                                         matmul_precision='highest', autocast_enabled=False), 'Actual flags differ')


def check_logp(value, tensors):
    import torch
    require(value.device.type == 'cpu' and value.dtype == torch.float32
            and value.shape == tensors['responses'].shape
            and bool(torch.isfinite(value).all())
            and bool((value[tensors['attention'] == 0] == 0).all()), 'Logp finite/mask/shape differs')


def actual_flags():
    import torch
    values = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_precision=torch.get_float32_matmul_precision(),
                  autocast_enabled=torch.is_autocast_enabled())
    require(values == dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                           matmul_precision='highest', autocast_enabled=False), 'Actual main flags differ')
    require(os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '0'
            and os.environ.get('NVIDIA_TF32_OVERRIDE') in (None, '0'), 'Environment overrides TF32')
    return dict(getters=values, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),
                NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'))


def thread_flags(record):
    threads = record['threads']
    expected = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    matmul_precision='highest', autocast_enabled=False)
    require(len(threads) == len({x['thread'] for x in threads}) == 8, 'Need8 actual worker contexts')
    require(all(x['context']['actual_flags'] == expected for x in threads), 'Actual worker flags differ')


def run_native_score_gate(teacher, runtime, tensors, candidate_directory, provenance, output_directory):
    """Minimal actual32B serial oracle + two consecutive eight-thread phases.

    Caller fresh-loads the verified HF teacher, passes actual lease/runtime, and
    supervises this direct child using OwnedProcesses. Never call from actors.
    """
    import torch
    from safetensors.torch import save_file
    from search_r1.baselines.distillation import FrozenTeacher
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    require(socket.gethostname() == 'h100-11' and os.environ.get('SLURM_JOB_NODELIST') == 'h100-11', 'Ownh11 only')
    require(torch.__version__ == '2.4.0+cu121', 'Pinned nativeTorch required')
    require(type(teacher) is FrozenTeacher and runtime.kind == 'NATIVE_UNVALIDATED_PROTOTYPE', 'Real native teacher/runtime only')
    require(digest(Path(inspect.getfile(inspect.unwrap(FrozenTeacher.log_probs)))) == SOURCE_SHA
            and hashlib.sha256(inspect.getsource(FrozenTeacher.log_probs).encode()).hexdigest() == METHOD_SHA,
            'Original method/source drift')
    model_directory = check_identity(provenance, candidate_directory)
    require(Path(teacher.model.config._name_or_path).resolve() == model_directory, 'Loaded teacher path differs')
    validate_inherited_gpu_leases(8)
    runtime.verify()
    require(teacher.model.config.num_hidden_layers == 64 and teacher.model.config.vocab_size == 152064
            and teacher.parameter_count == PARAMETERS, 'Actual full32B required')
    lengths = check_inputs(tensors, teacher.tokenizer.pad_token_id)
    out = local(output_directory)
    require(ROOT / 'logs' in out.parents, 'Own logs only')
    out.mkdir(exist_ok=False)
    # Candidate import happens only after its exact files and source are checked.
    import sys
    sys.path.insert(0, str(local(candidate_directory)))
    import capacity_reference as reference
    import pipeline as candidate_module
    from pipeline import PipelineScorer
    require(Path(reference.__file__).resolve() == local(candidate_directory)/'capacity_reference.py'
            and Path(candidate_module.__file__).resolve() == local(candidate_directory)/'pipeline.py',
            'Import cache selected another candidate')
    result = dict(status='RUNNING_NO_TRAINING', scope='32B_SCORER_NUMERICAL_RELEASE_ONLY_UNREVIEWED',
                  teacher_kind=provenance['teacher_kind'], provenance=provenance,
                  pid=os.getpid(), cgroup=Path('/proc/self/cgroup').read_text(),
                  original_unwrapped_file=str(Path(inspect.getfile(inspect.unwrap(FrozenTeacher.log_probs))).resolve()),
                  original_unwrapped_file_sha256=SOURCE_SHA, original_method_sha256=METHOD_SHA,
                  serial_raw_file_hash_semantics='DECORATED_METHOD_FILE_TORCH_CONTEXTLIB_NOT_DISTILLATION_IDENTITY',
                  gpu_uuids=runtime.uuids, actual_tokens_per_row=lengths,
                  coexistence=dict(local_gpu_retriever='PENDING', actor='PENDING', rollout='PENDING'),
                  teacher_cpu_load_timing='PENDING_CALLER_RECEIPT_NOT_MEASURED_HERE',
                  phase_records=[], peak_rss_kib_unit='Linux getrusage ru_maxrss process lifetime maximum')
    save_file(tensors, str(out / 'input.safetensors'))
    result['input_sha256'] = digest(out / 'input.safetensors')
    try:
        started = time.monotonic()
        serial = reference.SharedScorer(teacher, runtime, reference.device_map_32b(), teacher.tokenizer.get_vocab().values())
        require(serial.initial['parameter_sha256'] == provenance['expected_fp32_parameter_sha256'],
                'Loaded FP32 teacher differs from independently reviewed HF conversion')
        expected, rec = serial.score_round(tensors, batch_size=1)
        check_logp(expected, tensors)
        result['serial_actual_main_flags'] = actual_flags()
        save_file({'teacher_log_probs': expected}, str(out / 'serial.safetensors'))
        result['phase_records'].append(dict(kind='ORIGINAL_SERIAL_REFERENCE', wall_seconds=time.monotonic()-started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, record=rec))
        assert_release(rec, runtime.uuids)
        candidate = PipelineScorer(teacher, runtime, reference.device_map_32b(), teacher.tokenizer.get_vocab().values())
        for repeat in range(2):
            started = time.monotonic()
            actual, rec = candidate.score_round(tensors, workers=8)
            result['phase_records'].append(dict(kind='PIPELINE', repeat=repeat, wall_seconds=time.monotonic()-started,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, record=rec))
            save_file({'teacher_log_probs': actual}, str(out / ('parallel-%d.safetensors' % repeat)))
            assert_release(rec, runtime.uuids)
            check_logp(actual, tensors)
            thread_flags(rec)
            result['phase_records'][-1]['actual_main_flags'] = actual_flags()
            close = bool(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4))
            result['phase_records'][-1].update(atol=1e-4, rtol=1e-4, allclose=close,
                    max_abs_delta=float((actual-expected).abs().max()), gpu_pid_after_release=runtime.gpu_identity())
            require(close, 'OriginalGPU reference parity failed')
        result['status'] = 'SCORER_NUMERICAL_RELEASE_COMPLETE_UNREVIEWED_NOT_TRAINING_READY'
    except BaseException as exc:
        result['status'] = 'FAILED_NO_ACCEPTANCE'
        result['error'] = repr(exc)
        if isinstance(exc, RuntimeError) and exc.args and isinstance(exc.args[0], dict):
            result['failed_phase_record'] = exc.args[0]
        if 'candidate' in locals(): result['last_candidate_record'] = candidate.last_record
        raise
    finally:
        result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
        write_json(out / 'result.json', result)
    return result
