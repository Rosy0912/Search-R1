"""Exclusive CPU-only raw epoch1 merge. No GPU evaluator or controller.

Run only after reviewing this file. Existing outputs are never replaced.
Inputs are verified before and after; partial output is retained on failure.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Before importing any model libraries.

BASE = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
SOURCE = BASE / 'assets/archive100'
ADAPTER = BASE / 'runs/opsd/adapter_epoch_1'
SELECTED = BASE / 'runs/opsd/merged'
BEST = BASE / 'runs/opsd/best_adapter'
OUTPUT = BASE / 'runs/opsd/raw_epoch1_merged'
RECEIPT = BASE / 'receipts/raw-epoch1-cpu-merge-20261008.json'
RUNTIME = BASE / 'runtime/opsd-preparation3-xformers'
EXPECTED_RUN = '914bfff786c84ede8d82414dff65f60e'
EXPECTED_RAW = '8eef75aafa06eb98dc21a6b0ab85195800fc373692d633faa0ecf461fa45afd0'
EXPECTED_BEST = '8cbebc6cb10203bf93a2c8b7271d31020ab1d1e7f1958a6f94f5890af3206c3d'
BOUND = {
    'runs/opsd/history.json': '44b2776b9a278518dc9c459cae0f224e5ed1a5cd47f8da1fc9ace05d836d7aa9',
    'runs/opsd/config.json': '84a6167b9216c65fe0e810d03a8489f263d7feccddface605fa0ab6e726e8640',
    'receipts/preparation-3-config.json': '3ac56bfa4455ecc93175172db6353bbc4a7cae95aee24781797428e50abeeb5b',
    'receipts/preparation-3-runtime.sha256.json': '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc',
    'receipts/archive100-transfer-accepted.json': '08b005d8788fbbbc6c57cf40af4c08202b59c0d89160a6f030fad8d555fc007d',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def snapshot(directory):
    return {str(path.relative_to(directory)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in sorted(directory.rglob('*')) if path.is_file()}


def exclusive_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def verify_inputs():
    if sha(ADAPTER / 'adapter_model.safetensors') != EXPECTED_RAW:
        raise RuntimeError('Raw epoch1 adapter differs from independent saved-state audit')
    if sha(BEST / 'adapter_model.safetensors') != EXPECTED_BEST:
        raise RuntimeError('Selected baseline adapter differs from initial saved-state audit')
    for relative, expected in BOUND.items():
        if sha(BASE / relative) != expected:
            raise RuntimeError('Bound receipt changed: ' + relative)
    manifest = read_json(BASE / 'receipts/preparation-3-runtime.sha256.json')
    if len(manifest) != 305:
        raise RuntimeError('Expected exactly 305 frozen runtime manifest entries')
    for relative, expected in manifest.items():
        path = RUNTIME / relative
        if path.resolve().is_relative_to(RUNTIME.resolve()) is False or sha(path) != expected:
            raise RuntimeError('Frozen runtime changed: ' + relative)
    transfer = read_json(BASE / 'receipts/archive100-transfer-accepted.json')
    if transfer.get('accepted') is not True or transfer['target'] != str(SOURCE) or len(transfer['files']) != 12:
        raise RuntimeError('Expected accepted twelve-file source100 transfer')
    for filename, evidence in transfer['files'].items():
        path = SOURCE / filename
        if not path.resolve().is_relative_to(SOURCE.resolve()):
            raise RuntimeError('Transfer path escape')
        if path.stat().st_size != evidence['bytes'] or sha(path) != evidence['sha256']:
            raise RuntimeError('Source100 changed: ' + filename)
    config = read_json(BASE / 'runs/opsd/config.json')
    if config['model'] != str(SOURCE) or config['epochs'] != 1 or config['world_size'] != 8:
        raise RuntimeError('Unexpected OPSD configuration')
    history = read_json(BASE / 'runs/opsd/history.json')
    if [(row['epoch'], row['validation_em']) for row in history] != [(0, 239/512), (1, 203/512)]:
        raise RuntimeError('Unexpected native validation history')
    run = read_json(BASE / 'receipts/preparation-3-run.json')
    if run['run_id'] != EXPECTED_RUN or run['process_uid'] != os.getuid():
        raise RuntimeError('Wrong prep3 ownership identity')
    return dict(bound_receipts=dict(BOUND), runtime_files=len(manifest), source_transfer=transfer,
                opsd_config=config, history=history, original_run_id=run['run_id'])


def tensors(directory):
    """Safetensors file/key mapping, with only the declared tied-head alias."""
    from safetensors import safe_open
    mapping = {}
    for path in sorted(directory.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as file:
            for name in file.keys():
                if name in mapping:
                    raise RuntimeError('Duplicate tensor: ' + name)
                mapping[name] = (path, name)
    if not mapping:
        raise RuntimeError('No saved tensors in ' + str(directory))
    return mapping


def tensor(mapping, name):
    from safetensors import safe_open
    path, key = mapping[name]
    with safe_open(path, framework='pt', device='cpu') as file:
        return file.get_tensor(key)


def canonical_weights(directory, torch):
    mapping = tensors(directory)
    if read_json(directory / 'config.json').get('tie_word_embeddings'):
        embedding, head = 'model.embed_tokens.weight', 'lm_head.weight'
        if embedding not in mapping:
            raise RuntimeError('Declared tied model lacks embedding')
        if head in mapping:
            if not torch.equal(tensor(mapping, head).to(torch.bfloat16), tensor(mapping, embedding).to(torch.bfloat16)):
                raise RuntimeError('Declared tied head differs from embedding')
            del mapping[head]
    return mapping


def semantic_json(path, ignored=()):
    data = read_json(path)
    return {key: value for key, value in data.items() if key not in ignored}


def main():
    audit_id = uuid.uuid4().hex
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    evidence = dict(status='STARTED', started_unix=time.time(), audit_id=audit_id,
                    uid=os.getuid(), pid=os.getpid(), starttick=stat[19],
                    original_run_id=EXPECTED_RUN, output=str(OUTPUT), script_sha256=sha(__file__),
                    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        if os.getuid() != 1005:
            raise RuntimeError('Expected K1 owner UID1005')
        if RECEIPT.exists():
            raise RuntimeError('Success receipt already exists; refusing duplicate merge')
        OUTPUT.mkdir(exist_ok=False)  # Never reuse/replace a successful or partial directory.
        evidence['inputs_before'] = verify_inputs()
        before = {name: snapshot(path) for name, path in [('raw_adapter', ADAPTER), ('best_adapter', BEST), ('selected', SELECTED)]}
        evidence['input_file_hashes_before'] = before

        import torch
        import peft
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        evidence['versions'] = dict(torch=torch.__version__, peft=peft.__version__, transformers=transformers.__version__)
        if not (torch.__version__ == '2.4.0+cu121' and peft.__version__ == '0.13.2' and transformers.__version__ == '4.46.3'):
            raise RuntimeError('Unreviewed model-library versions')
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized in CPU preparation')
        raw_config = read_json(ADAPTER / 'adapter_config.json')
        best_config = read_json(BEST / 'adapter_config.json')
        for settings in (raw_config, best_config):
            if (settings['base_model_name_or_path'] != str(SOURCE) or settings['r'] != 16
                    or settings['lora_alpha'] != 32 or settings['lora_dropout'] != 0
                    or set(settings['target_modules']) != {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}):
                raise RuntimeError('Adapter configuration differs from original OPSD')
        raw_config['target_modules'] = sorted(raw_config['target_modules'])
        best_config['target_modules'] = sorted(best_config['target_modules'])
        if raw_config != best_config:
            raise RuntimeError('Selected/raw adapter metadata differs')
        raw_adapter, best_adapter = tensors(ADAPTER), tensors(BEST)
        if set(raw_adapter) != set(best_adapter):
            raise RuntimeError('Adapter keys differ')
        raw_nonzero, best_b_keys = [], []
        for key in raw_adapter:
            value = tensor(raw_adapter, key)
            if not torch.isfinite(value).all().item():
                raise RuntimeError('Nonfinite raw adapter tensor: ' + key)
            if value.count_nonzero().item():
                raw_nonzero.append(key)
            best_value = tensor(best_adapter, key)
            if not torch.isfinite(best_value).all().item():
                raise RuntimeError('Nonfinite best adapter tensor: ' + key)
            if 'lora_B' in key:
                best_b_keys.append(key)
                if best_value.count_nonzero().item():
                    raise RuntimeError('Selected adapter is not zero-LoRA baseline')
        if not best_b_keys or not any('lora_B' in key for key in raw_nonzero):
            raise RuntimeError('Missing zero baseline or trained raw LoRA_B evidence')

        # Exact original CPU/BF16 PEFT merge calls; no thread or math-mode tuning.
        base = AutoModelForCausalLM.from_pretrained(SOURCE, torch_dtype=torch.bfloat16, local_files_only=True)
        if any(parameter.device.type != 'cpu' for parameter in base.parameters()):
            raise RuntimeError('Base model is not entirely CPU')
        selected = PeftModel.from_pretrained(base, ADAPTER, local_files_only=True)
        if any(parameter.device.type != 'cpu' for parameter in selected.parameters()):
            raise RuntimeError('Adapter model is not entirely CPU')
        selected.merge_and_unload().save_pretrained(OUTPUT, safe_serialization=True)
        AutoTokenizer.from_pretrained(SOURCE, use_fast=True, local_files_only=True).save_pretrained(OUTPUT)
        del selected, base
        gc.collect()

        # Preserve original source tokenizer/generation assets byte-for-byte;
        # config.json remains the original save_pretrained BF16 model config.
        metadata_names = ['generation_config.json', 'added_tokens.json', 'tokenizer_config.json',
                          'tokenizer.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt']
        for filename in metadata_names:
            shutil.copyfile(SOURCE / filename, OUTPUT / filename)
        for filename in metadata_names:
            if sha(OUTPUT / filename) != sha(SOURCE / filename):
                raise RuntimeError('Source metadata copy differs: ' + filename)
            if filename.endswith('.json'):
                same = semantic_json(OUTPUT / filename, ('_name_or_path', 'name_or_path')) == semantic_json(SELECTED / filename, ('_name_or_path', 'name_or_path'))
            else:
                same = sha(OUTPUT / filename) == sha(SELECTED / filename)
            if not same:
                raise RuntimeError('Selected/raw tokenizer or generation settings differ: ' + filename)
        if semantic_json(OUTPUT / 'config.json', ('_name_or_path',)) != semantic_json(SELECTED / 'config.json', ('_name_or_path',)):
            raise RuntimeError('Selected/raw saved model configuration differs')

        source_weights = canonical_weights(SOURCE, torch)
        selected_weights = canonical_weights(SELECTED, torch)
        raw_weights = canonical_weights(OUTPUT, torch)
        if not set(source_weights) == set(selected_weights) == set(raw_weights):
            raise RuntimeError('Whole-model tensor inventory differs after tied-alias normalization')
        changed = []
        for key in source_weights:
            expected = tensor(source_weights, key).to(torch.bfloat16)
            baseline = tensor(selected_weights, key)
            raw = tensor(raw_weights, key)
            if expected.shape != baseline.shape or raw.shape != baseline.shape:
                raise RuntimeError('Whole-model tensor shape differs: ' + key)
            if baseline.dtype != torch.bfloat16 or raw.dtype != torch.bfloat16:
                raise RuntimeError('Saved tensor is not BF16: ' + key)
            if not torch.isfinite(raw).all().item() or not torch.isfinite(baseline).all().item():
                raise RuntimeError('Nonfinite saved weights: ' + key)
            if not torch.equal(expected, baseline):
                raise RuntimeError('Selected merged differs from source100 BF16: ' + key)
            if not torch.equal(raw, baseline):
                changed.append(key)
            del expected, baseline, raw
        if not changed:
            raise RuntimeError('Raw merged weights do not differ from selected baseline')
        evidence['weight_validation'] = dict(canonical_tensor_count=len(raw_weights), all_saved_tensors_finite=True,
            selected_equals_source100_bf16=True, changed_raw_tensor_keys=changed,
            tied_alias='lm_head.weight -> model.embed_tokens.weight only when config declares tie_word_embeddings')
        evidence['inputs_after'] = verify_inputs()
        after = {name: snapshot(path) for name, path in [('raw_adapter', ADAPTER), ('best_adapter', BEST), ('selected', SELECTED)]}
        if after != before:
            raise RuntimeError('Adapter or selected baseline changed during CPU merge')
        evidence['input_file_hashes_after'] = after
        evidence['output_file_hashes'] = snapshot(OUTPUT)
        evidence['cpu_only'] = dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                                    cuda_initialized=torch.cuda.is_initialized(), models_checked_entirely_cpu=True)
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized during CPU merge')
        evidence.update(status='ACCEPTED', completed_unix=time.time(),
                        scope='CPU merged artifact preparation only; no raw seven-benchmark evaluator started')
        exclusive_json(RECEIPT, evidence)
        print('RAW_EPOCH1_CPU_MERGE_ACCEPTED', str(RECEIPT), flush=True)
    except BaseException as error:
        evidence.update(status='FAILED', completed_unix=time.time(), error_type=type(error).__name__,
                        error=str(error), traceback=traceback.format_exc(), partial_output_preserved=OUTPUT.exists())
        failed = BASE / ('receipts/raw-epoch1-cpu-merge-20261008.failed-' + audit_id + '.json')
        exclusive_json(failed, evidence)
        raise


if __name__ == '__main__':
    main()
