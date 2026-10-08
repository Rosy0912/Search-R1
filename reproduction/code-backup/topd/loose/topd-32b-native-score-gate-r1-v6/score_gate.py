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
HF_METADATA = {'config.json', 'generation_config.json', 'tokenizer_config.json',
               'tokenizer.json', 'merges.txt', 'vocab.json', 'model.safetensors.index.json'}
TRAINED_OPTIONAL_METADATA = {'special_tokens_map.json', 'added_tokens.json', 'teacher-manifest.json'}
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


def check_hf_file_set(inventory, model, kind):
    shards = set(inventory['shards'])
    listed = set(inventory['files'])
    optional = TRAINED_OPTIONAL_METADATA if kind == 'SEARCH_R1_GRPO100_32B' else set()
    require(shards | HF_METADATA <= listed and listed <= shards | HF_METADATA | optional,
            'Incomplete/unexpected HF metadata inventory')
    # Include every loader-relevant file, not only files claimed in a manifest.
    actual = {p.name for p in model.iterdir() if p.suffix in ('.json', '.txt', '.safetensors')}
    require(actual == listed and all(not (model/n).is_symlink() for n in actual),
            'Actual HF metadata/shard file set differs')


def check_tokenizer_config_identity(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    student = AutoTokenizer.from_pretrained(ROOT/'assets-root/assets/qwen2.5-3b', local_files_only=True)
    require(tok.get_vocab() == student.get_vocab(), 'Actual teacher/student token-ID vocab differs')
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        require(getattr(tok, key) == getattr(student, key), 'Actual tokenizer special-token ID differs')
    current = json.loads((model/'config.json').read_text())
    base = json.loads((ROOT/'assets-root/assets/qwen2.5-32b/config.json').read_text())
    ignored = {'_name_or_path', '_commit_hash', 'torch_dtype', 'transformers_version',
               '_attn_implementation', 'bos_token_id', 'eos_token_id', 'pad_token_id'}
    require({k:v for k,v in current.items() if k not in ignored}
            == {k:v for k,v in base.items() if k not in ignored}, 'Actual32B architecture config differs')
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        require(current.get(key) in (None, getattr(tok, key), base.get(key)),
                'Teacher config/tokenizer special-token override differs')


TRAINED_EVAL_MANIFEST='591b12e51b4d1457db422c9cdac33e462d7f1d73b9a0dc8acf9651fda623a28d'
TRAINED_HELPERS={'checkpoint_inventory.py': 'ebd884a65e185b4ae55b48e8c2034e3da8a9e18f866de49719926565fdefee2f', 'terminal.py': '2897fe0ad094f07cacd654e570df67ae04a473592726c672bdc880a39dc8310a', 'trained_assets.py': '1e721427de63b149200fba354f70b27c1de86b88d2600882a8abb760a427ca06', 'training_identity.py': 'b7208b37696b217287bd916319c3f23bda8dd36168f37ef2b0de7848a4f6eae0'}
BASE_FP32_SHA='a80140df56c39e1e0f99e29473bd1b6f3668d47cccfa05240514f560705d0d16'

def verify_trained_measurement(raw, supervision, quality, model):
    require(raw['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING'
            and raw['quality']['denominator']==64
            and raw['quality']['second_quality_cutoff_applied'] is False
            and raw['quality']['correct']==quality['correct'], 'Trained64 raw measurement differs')
    require(supervision['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING'
            and not supervision.get('error') and not supervision.get('cleanup_error')
            and supervision['sampler_errors']==[]
            and supervision['parent_cuda_initialized'] is False
            and supervision['foreign_baseline_missing']==[]
            and supervision['cleanup']['actual_cgroup_empty'] is True,
            'Trained64 actual runtime failed; retain it, never promote to teacher')
    require(supervision['terminal']['monitor_stop']['stopped'] is True
            and supervision['monitor_stop']['stopped'] is True,
            'Held-engine monitor was not actually drained before authorized exit')
    require(quality['teacher_model_directory']==str(model), 'Trained64 scored a different export')

def fresh_trained_evaluation(provenance,model,quality):
    # Reuse independently source-reviewed trained64 V2 gate; do not invent a
    # second numerical pipeline or use quality-before-quality circular approval.
    import importlib.util,sys
    directory=ROOT/'tools/qwen25-32b-trained-eval64-r1-v6'
    require(digest(directory/'files.json')==TRAINED_EVAL_MANIFEST,'Trained64 V2 source manifest drift')
    for name,expected in TRAINED_HELPERS.items():
        require(digest(directory/name)==expected,'Trained64 helper source drift')
    def helper(name):
        path=directory/(name+'.py')
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve()==path.resolve()
                    and digest(path)==TRAINED_HELPERS[path.name], 'Trained64 helper import shadow')
            return sys.modules[name]
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
    helper('checkpoint_inventory');assets=helper('trained_assets');terminal=helper('terminal')
    refs={}
    for key in ('trained_evaluation_binding','trained_evaluation_result','trained_evaluation_supervisor_result'):
        ref=provenance[key];path=local(ref['path']);require(digest(path)==ref['sha256'],'Trained64 artifact drift')
        refs[key]=(path,json.loads(path.read_text()))
    binding_path,binding=refs['trained_evaluation_binding']
    require(binding['model_directory']==str(model)
            and binding['model_inventory']==provenance['model_inventory']
            and binding['training_receipt']==provenance['training_receipt'], 'Search-R1 source/evaluation100 provenance differs')
    _,checked_model,_=assets.gate(binding_path,ROOT,binding['source_files'])
    require(checked_model==model,'Fresh Search-R1-100 gate selected different model')
    result_path,raw=refs['trained_evaluation_result'];_,supervision=refs['trained_evaluation_supervisor_result']
    require(supervision['evaluation_result_sha256']==digest(result_path)
            and quality['result_sha256']==digest(result_path), 'Selected quality is not this actual trained64')
    verify_trained_measurement(raw,supervision,quality,model)
    directory_result=result_path.parent
    require(raw['qa_sha256']==digest(directory_result/'qa.jsonl')
            and raw['final_sha256']==digest(directory_result/'final.safetensors'), 'Trained64 QA/token raw drift')
    qa=[json.loads(line) for line in (directory_result/'qa.jsonl').read_text().splitlines()]
    require(len(qa)==64 and len({r['uid'] for r in qa})==64
            and [r['sample_position'] for r in qa]==list(range(64))
            and all(r['em'] in (0.,1.) for r in qa)
            and sum(int(r['em']) for r in qa)==quality['correct'], 'Actual trained64 rows/count/order differ')
    ack=terminal.verify_ack(directory_result,supervision['terminal'])
    require(ack==supervision['terminal_child_ack'], 'Actual trained64 terminal ACK source drift')
    receipt=supervision['terminal']['receipt']
    require(receipt['result_sha256']==digest(result_path)
            and receipt['qa_sha256']==raw['qa_sha256']
            and receipt['final_sha256']==raw['final_sha256'], 'Held-engine receipt binds different raw evaluation')


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
    kind = provenance['teacher_kind']
    check_hf_file_set(inventory, model, kind)
    require(set(inventory['shards']) == {n for n in inventory['files'] if n.endswith('.safetensors')},
            'Incomplete HF shard identity')
    for name, value in inventory['files'].items():
        path = (model / name).resolve()
        require(model in path.parents and path.stat().st_size == value['size']
                and digest(path) == value['sha256'], 'Actual HF bytes drift: ' + name)
    check_hf_tensor_index(model, inventory['shards'])
    check_tokenizer_config_identity(model)
    cpu_identity = json.loads(local(provenance['cpu_identity']['path']).read_text())
    require(cpu_identity['status'] == 'CPU_FP32_IDENTITY_CAPTURE_UNREVIEWED_NOT_TEACHER_QUALITY'
            and cpu_identity['snapshot']['parameter_count'] == PARAMETERS
            and cpu_identity['snapshot']['parameter_sha256'] == provenance['expected_fp32_parameter_sha256']
            and cpu_identity['model_directory'] == str(model)
            and cpu_identity['hf_file_sha256'] == {n:v['sha256'] for n,v in inventory['files'].items()},
            'Expected FP32 identity must derive from actual source-bound CPU preload')
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
    elif kind == 'SEARCH_R1_GRPO100_32B':
        base_ref = provenance['base_quality_result']
        require(digest(local(base_ref['path'])) == base_ref['sha256'], 'Prior base quality drift')
        base_quality = json.loads(local(base_ref['path']).read_text())
        require(base_quality['denominator'] == 64 and type(base_quality['correct']) is int
                and 0 <= base_quality['correct'] < 26 and base_quality['greedy'] is True
                and base_quality['metric'] == 'EM' and base_quality['data_sha256'] == provenance['data_sha256'],
                'Search-R1 fallback must bind actual original base64 below26')
        ref = provenance['training_receipt']
        require(digest(local(ref['path'])) == ref['sha256'], 'Search-R1 training receipt drift')
        training = json.loads(local(ref['path']).read_text())
        require(training['status'] == 'INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES'
                and training['actual_outer_updates'] == 100 and training['algorithm'] == 'SEARCH_R1_GRPO'
                and training['source_reference'] == 'e623724532d06e975b61f3b35ec7925dc790c939'
                and training['hf_inventory_sha256'] == provenance['model_inventory']['sha256'],
                'No reviewed actual Search-R1-100 teacher export')
        require(provenance['base_fp32_parameter_sha256']==BASE_FP32_SHA
                and provenance['expected_fp32_parameter_sha256'] != BASE_FP32_SHA,
                'Search-R1 export must contain actual parameter changes')
        fresh_trained_evaluation(provenance,model,quality)
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
