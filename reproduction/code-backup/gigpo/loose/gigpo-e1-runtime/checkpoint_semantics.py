#!/usr/bin/env python3
"""CPU-only receipt/split/tokenizer/semantic-weight gate for select_teacher.export.

No model forward or GPU allocation. Safetensors weights are mmap sliced, never
fully materialized. Success verifies artifacts, not undocumented training history.
"""
import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

CANDIDATES = (5, 100, 200, 300, 400, 500)
RULE = 'maximum holdout F1; ties earliest update'
CHUNK_ELEMENTS = 1024 * 1024  # <=4 MiB per FP32 comparison tensor


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fail(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, name):
    path = (root / name).resolve()
    fail(path.is_relative_to(root.resolve()) and path.is_file(), 'Missing/outside receipt file: ' + name)
    return path


def check_file_manifest(root, receipt, extra=()):
    files = receipt['files']
    fail(isinstance(files, dict) and files, 'Empty HF receipt')
    fail('config.json' in files and 'tokenizer_config.json' in files, 'Incomplete HF model/tokenizer receipt')
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    fail(actual == set(files) | set(extra), 'Unexpected/missing HF files: ' + repr(sorted(actual ^ (set(files) | set(extra)))))
    for name, expected in files.items():
        fail(digest_file(safe_path(root, name)) == expected, 'File SHA mismatch: ' + name)
    total = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    fail(total == receipt['sha256'], 'Receipt aggregate SHA mismatch')
    fail(any(name.endswith('.safetensors') for name in files), 'Require safetensors for bounded CPU mmap comparison')
    fail(not any(name.startswith('pytorch_model') and name.endswith('.bin') for name in files),
         'Pickled model weights unsupported by this mmap-only gate')
    for name in files:
        if name.endswith('.index.json'):
            mapping = json.loads(safe_path(root, name).read_text()).get('weight_map')
            fail(isinstance(mapping, dict) and mapping, 'Invalid shard index')
            for shard in set(mapping.values()):
                fail(shard in files, 'Shard absent from receipt: ' + shard)
                safe_path(root, shard)
    return total


def plain(value):
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def frame_identity(frame, prompt_key):
    questions = sorted(set(json.dumps(plain(prompt), sort_keys=True, ensure_ascii=False) for prompt in frame[prompt_key]))
    rows = sorted(json.dumps(plain(row), sort_keys=True, ensure_ascii=False) for row in frame.to_dict('records'))
    return dict(questions_sha256=hashlib.sha256('\n'.join(questions).encode()).hexdigest(),
                rows_sha256=hashlib.sha256('\n'.join(rows).encode()).hexdigest())


def check_split(path, expected_identity):
    import pandas as pd
    split = json.loads(path.read_text())
    spec = split['specification']
    fail(spec['seed'] == 42 and spec['fraction'] == .02, 'Nonstandard public split seed/fraction')
    for name in ('train.parquet', 'stage-validation.parquet'):
        fail(digest_file(safe_path(path.parent, name)) == split['sha256'][name], 'Split file SHA mismatch: ' + name)
    train = pd.read_parquet(path.parent / 'train.parquet')
    holdout = pd.read_parquet(path.parent / 'stage-validation.parquet')
    fail(len(holdout) == split['holdout_rows'] == 3393 and len(train) == split['train_rows'], 'Require complete3393-row common holdout')
    content = dict(version=1, split_version=spec['version'],
                   train_sources=sorted(item['sha256'] for item in spec['train']),
                   excluded_sources=sorted(item['sha256'] for item in spec['final']),
                   fraction=spec['fraction'], seed=spec['seed'], prompt_key=spec['prompt_key'],
                   train=frame_identity(train, spec['prompt_key']), holdout=frame_identity(holdout, spec['prompt_key']))
    content['sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    fail(content == split['content_identity'], 'Split content identity does not match actual rows')
    fail(content['sha256'] == expected_identity, 'Split differs from trusted public content identity')
    return split


def check_selection(report, identity):
    records = report['candidates']
    fail(len(records) == 6 and sorted(record['outer_update'] for record in records) == list(CANDIDATES), 'Require six candidate updates5/100/200/300/400/500')
    fail(report['split_content_identity'] == identity and report['evaluation_seed'] == 42, 'Receipt split/seed mismatch')
    fail(report['selection_rule'] == RULE and report['tokenizer_compatible'] is True, 'Receipt selection/tokenizer declaration mismatch')
    for record in records:
        fail(record['split_content_identity'] == identity and record['evaluation_split'] == 'common-holdout', 'Candidate used another split')
        fail(record['seed'] == 42 and record['evaluated_count'] == record['holdout_rows'] == 3393, 'Candidate did not evaluate all3393 questions')
        value = record['mean_answer_f1']
        fail(math.isfinite(value) and 0 <= value <= 1, 'Invalid candidate F1')
        metrics = record.get('metrics')
        fail(isinstance(metrics, dict) and 'val/qa_count/all' in metrics and 'val/f1/all' in metrics,
             'Missing actual candidate count/F1 metrics')
        count, f1 = metrics['val/qa_count/all'], metrics['val/f1/all']
        fail(isinstance(count, (int, float)) and math.isfinite(count) and count == 3393,
             'Candidate metric count must be finite and exactly3393')
        fail(isinstance(f1, (int, float)) and not isinstance(f1, bool) and math.isfinite(f1) and 0 <= f1 <= 1,
             'Candidate metric F1 must be finite within[0,1]')
        fail(value == f1, 'Candidate declared mean_answer_f1 disagrees with actual metric F1')
    winner = min(records, key=lambda record: (-record['mean_answer_f1'], record['outer_update']))
    fail(report['selected'] == winner, 'Wrong winner: maximum F1, earliest update tie')
    fail(report['teacher']['sha256'] == winner['checkpoint_sha256'], 'Selected checkpoint/teacher receipt SHA mismatch')
    return winner


def catalog(root, stack):
    from safetensors import safe_open
    mapping, handles = {}, {}
    for file in sorted(root.rglob('*.safetensors')):
        handle = stack.enter_context(safe_open(str(file), framework='pt', device='cpu'))
        handles[str(file.relative_to(root))] = handle
        for key in handle.keys():
            fail(key not in mapping, 'Duplicate tensor key: ' + key)
            mapping[key] = (handle, key)
    fail(mapping, 'No model tensors')
    for file in root.glob('*.index.json'):
        index = json.loads(file.read_text())['weight_map']
        for key, shard in index.items():
            fail(shard in handles and key in handles[shard].keys(), 'Shard index missing/misplaced tensor: ' + key)
        fail(set(index) == set(mapping), 'Shard index/tensor key mismatch')
    return mapping


def tokenizers(base, teacher, vocab_size):
    from transformers import AutoTokenizer
    first = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    second = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=False)
    fail(first.get_vocab() == second.get_vocab(), 'Token-to-id vocabulary mismatch')
    fail((first.bos_token_id, first.eos_token_id) == (second.bos_token_id, second.eos_token_id), 'BOS/EOS mismatch')
    config = json.loads((teacher / 'config.json').read_text())
    for key in ('bos_token_id', 'eos_token_id'):
        if key in config:
            fail(config[key] == getattr(second, key), 'Teacher config/tokenizer special-token mismatch: ' + key)
    ids = list(first.get_vocab().values())
    fail(ids and min(ids) >= 0 and max(ids) < vocab_size, 'Tokenizer ID exceeds output vocabulary dimension')
    return dict(vocabulary_tokens=len(ids), bos_token_id=first.bos_token_id, eos_token_id=first.eos_token_id,
                token_to_id_sha256=hashlib.sha256(json.dumps(first.get_vocab(), sort_keys=True).encode()).hexdigest())


def slices(handle, key, shape):
    """Tile any-rank tensor to <=CHUNK_ELEMENTS; do not load its full last row."""
    if not shape:
        yield handle.get_tensor(key)  # scalar only
        return
    tiles = []
    remaining = CHUNK_ELEMENTS
    for size in reversed(shape):
        width = min(size, remaining)
        fail(width > 0, 'Zero-sized model tensor')
        tiles.append(width)
        remaining = max(1, remaining // width)
    tiles.reverse()
    import itertools
    for starts in itertools.product(*(range(0, size, width) for size, width in zip(shape, tiles))):
        yield tuple(slice(start, min(start + width, size)) for start, width, size in zip(starts, tiles, shape))


def compare_weights(base, teacher):
    import torch
    from transformers import AutoConfig
    configs = [AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False).to_dict() for root in (base, teacher)]
    # FSDP explicitly overrides these three IDs from its tokenizer at loading.
    # Normalized AutoConfig also fills version-default fields absent in raw JSON.
    metadata = {'_name_or_path', 'torch_dtype', 'transformers_version', '_attn_implementation_autoset', '_attn_implementation',
                'bos_token_id', 'eos_token_id', 'pad_token_id'}
    semantic_configs = [{key: value for key, value in config.items() if key not in metadata} for config in configs]
    fail(semantic_configs[0] == semantic_configs[1], 'Teacher model architecture/configuration differs from base')
    vocab = configs[0]['vocab_size']
    changed = []
    elements = 0
    maximum = 0.
    with ExitStack() as stack:
        mappings = [catalog(root, stack) for root in (base, teacher)]
        embed, head = 'model.embed_tokens.weight', 'lm_head.weight'
        for mapping, config in zip(mappings, configs):
            fail(embed in mapping, 'Missing Qwen embedding tensor')
            fail(mapping[embed][0].get_slice(mapping[embed][1]).get_shape()[0] == vocab, 'Embedding vocabulary dimension mismatch')
            if head not in mapping:
                fail(config.get('tie_word_embeddings') is True, 'Missing untied output head')
                mapping[head] = mapping[embed]
            fail(mapping[head][0].get_slice(mapping[head][1]).get_shape()[0] == vocab, 'Output vocabulary dimension mismatch')
            if config.get('tie_word_embeddings') and mapping[head] != mapping[embed]:
                eh, ek = mapping[embed]
                hh, hk = mapping[head]
                shape = eh.get_slice(ek).get_shape()
                fail(shape == hh.get_slice(hk).get_shape(), 'Tied head/embedding shape mismatch')
                for index in slices(eh, ek, shape):
                    fail(bool(torch.equal(eh.get_slice(ek)[index].float(), hh.get_slice(hk)[index].float())),
                         'Tied output head has different values; HF loading would ignore this apparent update')
        fail(set(mappings[0]) == set(mappings[1]), 'Teacher/base tensor keys differ')
        for key in sorted(mappings[0]):
            (left, lk), (right, rk) = mappings[0][key], mappings[1][key]
            shape = left.get_slice(lk).get_shape()
            fail(shape == right.get_slice(rk).get_shape(), 'Tensor shape mismatch: ' + key)
            different = beyond_dtype_cast = False
            for index in slices(left, lk, shape):
                raw_a = index if not shape else left.get_slice(lk)[index]
                raw_b = right.get_tensor(rk) if not shape else right.get_slice(rk)[index]
                a, b = raw_a.float(), raw_b.float()
                fail(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite semantic tensor: ' + key)
                delta = (a - b).abs()
                fail(bool(torch.isfinite(delta).all()), 'Nonfinite FP32 tensor difference: ' + key)
                elements += a.numel()
                if delta.numel():
                    maximum = max(maximum, float(delta.max()))
                    different = different or bool((delta != 0).any())
                    beyond_dtype_cast = beyond_dtype_cast or bool((b != raw_a.to(raw_b.dtype).float()).any())
            if different and beyond_dtype_cast:
                changed.append(key)
    fail(changed, 'Teacher is numerically identical to base in float32 or differs only by dtype conversion; packaging/dtype changes are not training')
    return dict(semantic_comparison_dtype='float32', changed_tensor_names=changed,
                compared_elements_including_tied_aliases=elements, max_abs_difference=maximum,
                all_finite=True, change_beyond_base_to_teacher_dtype_cast=True,
                output_vocab_size=vocab, chunk_elements=CHUNK_ELEMENTS)


def verify(base, teacher, split_path, expected_identity):
    base, teacher, split_path = Path(base), Path(teacher), Path(split_path)
    report = json.loads((teacher / 'teacher-manifest.json').read_text())
    split = check_split(split_path, expected_identity)
    winner = check_selection(report, split['content_identity']['sha256'])
    check_file_manifest(base, report['student'])
    check_file_manifest(teacher, report['teacher'], extra=('teacher-manifest.json',))
    semantic = compare_weights(base, teacher)
    tokenizer = tokenizers(base, teacher, semantic['output_vocab_size'])
    return dict(status='PASS', teacher_path=str(teacher.resolve()), base_path=str(base.resolve()),
                selected_update=winner['outer_update'], holdout_rows=3393, seed=42,
                split_content_identity=expected_identity, teacher_files_sha256=report['teacher']['sha256'],
                teacher_receipt_sha256=digest_file(teacher / 'teacher-manifest.json'),
                semantic_weights=semantic, tokenizer=tokenizer,
                limitations='Artifact integrity/semantic update gate; training history and real TOP-D updates require separate evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'teacher', 'split-manifest', 'expected-content-identity', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.base, args.teacher, args.split_manifest, args.expected_content_identity)
    except Exception as error:
        result = dict(status='FAIL', error_type=type(error).__name__, error=str(error))
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        raise SystemExit(1)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
