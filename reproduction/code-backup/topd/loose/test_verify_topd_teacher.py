"""CPU-only artifact fixtures, not evidence that a real teacher exists."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

spec = importlib.util.spec_from_file_location('verify_topd_teacher', Path(__file__).with_name('verify-topd-teacher.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def receipt(path):
    import hashlib
    files = {str(file.relative_to(path)): gate.digest_file(file) for file in sorted(path.rglob('*')) if file.is_file()}
    return dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def fixture(tmp_path, update=False, teacher_dtype=torch.float32):
    base, teacher = tmp_path / 'base', tmp_path / 'teacher'
    for root in (base, teacher):
        root.mkdir()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'<bos>': 0, '<eos>': 1, 'hello': 2}, unk_token='hello')),
                                           bos_token='<bos>', eos_token='<eos>', unk_token='hello')
        tokenizer.save_pretrained(root)
        (root / 'config.json').write_text(json.dumps(dict(model_type='qwen2', vocab_size=3, hidden_size=2,
                                                         tie_word_embeddings=True)))
    weights = torch.ones(3, 2)
    save_file({'model.embed_tokens.weight': weights}, str(base / 'model.safetensors'))
    trained = weights.to(teacher_dtype)
    if update:
        trained[0, 0] += .5
    save_file({'model.embed_tokens.weight': trained}, str(teacher / 'model.safetensors'))
    split_dir = tmp_path / 'split'
    split_dir.mkdir()
    train = pd.DataFrame([dict(prompt=[dict(role='user', content='train')], extra_info={'index': 'train'})])
    holdout = pd.DataFrame([dict(prompt=[dict(role='user', content=str(i))], extra_info={'index': i}) for i in range(3393)])
    train.to_parquet(split_dir / 'train.parquet', index=False)
    holdout.to_parquet(split_dir / 'stage-validation.parquet', index=False)
    specification = dict(version=1, fraction=.02, seed=42, prompt_key='prompt', train=[dict(sha256='input-sha')], final=[dict(sha256='final-sha')])
    content = dict(version=1, split_version=1, train_sources=['input-sha'], excluded_sources=['final-sha'],
                   fraction=.02, seed=42, prompt_key='prompt', train=gate.frame_identity(train, 'prompt'), holdout=gate.frame_identity(holdout, 'prompt'))
    import hashlib
    content['sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    split = dict(specification=specification, content_identity=content, train_rows=1, holdout_rows=3393,
                 sha256={name: gate.digest_file(split_dir / name) for name in ('train.parquet', 'stage-validation.parquet')})
    split_file = split_dir / 'manifest.json'
    split_file.write_text(json.dumps(split))
    teacher_files = receipt(teacher)
    records = [dict(outer_update=step, evaluation_split='common-holdout', seed=42, evaluated_count=3393,
                    holdout_rows=3393, mean_answer_f1=.5, split_content_identity=content['sha256'],
                    checkpoint_sha256=teacher_files['sha256'], hf_directory=f'/source/actor/global_step_{step}',
                    metrics={'val/qa_count/all': 3393, 'val/f1/all': .5}) for step in gate.CANDIDATES]
    report = dict(selected=records[0], candidates=records, split_content_identity=content['sha256'],
                  selection_rule=gate.RULE, teacher=teacher_files, student=receipt(base),
                  tokenizer_compatible=True, evaluation_seed=42)
    (teacher / 'teacher-manifest.json').write_text(json.dumps(report))
    return base, teacher, split_file, content['sha256']


def test_identical_values_different_dtype_are_not_training(tmp_path):
    args = fixture(tmp_path, teacher_dtype=torch.bfloat16)
    assert gate.digest_file(args[0] / 'model.safetensors') != gate.digest_file(args[1] / 'model.safetensors')
    with pytest.raises(ValueError, match='numerically identical'):
        gate.verify(*args)


def test_semantic_update_passes_and_export_added_manifest_is_allowed(tmp_path):
    args = fixture(tmp_path, update=True)
    result = gate.verify(*args)
    assert result['status'] == 'PASS' and result['selected_update'] == 5
    assert result['semantic_weights']['max_abs_difference'] == .5
    assert result['semantic_weights']['all_finite']
    assert result['tokenizer']['bos_token_id'] == 0 and result['tokenizer']['eos_token_id'] == 1


def test_tampered_hf_file_is_rejected(tmp_path):
    args = fixture(tmp_path, update=True)
    (args[1] / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='File SHA mismatch'):
        gate.verify(*args)


def test_wrong_tie_winner_and_subsample_are_rejected(tmp_path):
    args = fixture(tmp_path, update=True)
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    report['selected'] = report['candidates'][1]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Wrong winner'):
        gate.verify(*args)
    report['selected'] = report['candidates'][0]
    report['candidates'][2]['evaluated_count'] = 3392
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='all3393'):
        gate.verify(*args)


def test_missing_shard_rejected(tmp_path):
    args = fixture(tmp_path, update=True)
    (args[1] / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'model.embed_tokens.weight': 'missing.safetensors'}}))
    report_path = args[1] / 'teacher-manifest.json'
    report = json.loads(report_path.read_text())
    # Recompute receipt as if an invalid export were signed: structural checks
    # must still reject the missing shard rather than rely only on stale hashes.
    report_path.unlink()
    report['teacher'] = receipt(args[1])
    for record in report['candidates']:
        record['checkpoint_sha256'] = report['teacher']['sha256']
    report['selected'] = report['candidates'][0]
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Shard absent'):
        gate.verify(*args)


def test_mmap_tiling_does_not_materialize_wide_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'CHUNK_ELEMENTS', 4)
    weights = torch.arange(30).reshape(2, 3, 5).float()
    path = tmp_path / 'tiny.safetensors'
    save_file({'x': weights}, str(path))
    from safetensors import safe_open
    chunks = []
    with safe_open(path, framework='pt', device='cpu') as handle:
        for index in gate.slices(handle, 'x', list(weights.shape)):
            chunks.append(handle.get_slice('x')[index])
    assert sum(part.numel() for part in chunks) == 30
    assert max(part.numel() for part in chunks) <= 4


def replace_weights_and_sign(args, tensor):
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    path.unlink()
    save_file({'model.embed_tokens.weight': tensor}, str(args[1] / 'model.safetensors'))
    report['teacher'] = receipt(args[1])
    for record in report['candidates']:
        record['checkpoint_sha256'] = report['teacher']['sha256']
    report['selected'] = report['candidates'][0]
    path.write_text(json.dumps(report))


def test_nonfinite_signed_weights_are_rejected(tmp_path):
    args = fixture(tmp_path, update=True)
    weights = torch.ones(3, 2)
    weights[0, 0] = float('nan')
    replace_weights_and_sign(args, weights)
    with pytest.raises(ValueError, match='Nonfinite semantic tensor'):
        gate.verify(*args)


def test_quantization_only_change_is_not_training(tmp_path):
    args = fixture(tmp_path, teacher_dtype=torch.bfloat16)
    # Signed base differs in float32, but casting it to teacher dtype exactly
    # reproduces teacher weights; no optimization beyond dtype conversion.
    save_file({'model.embed_tokens.weight': torch.full((3, 2), 1.001)}, str(args[0] / 'model.safetensors'))
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    report['student'] = receipt(args[0])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='only by dtype conversion'):
        gate.verify(*args)


def sign_teacher_directory(args):
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    path.unlink()
    report['teacher'] = receipt(args[1])
    for record in report['candidates']:
        record['checkpoint_sha256'] = report['teacher']['sha256']
    report['selected'] = report['candidates'][0]
    path.write_text(json.dumps(report))


@pytest.mark.parametrize('head_only_update', [False, True])
def test_tied_output_head_storage_cannot_fake_ignored_update(tmp_path, head_only_update):
    args = fixture(tmp_path, update=not head_only_update)
    embedding = torch.ones(3, 2)
    if not head_only_update:
        embedding[0, 0] += .5
    head = embedding.clone()
    if head_only_update:
        head[0, 0] += .5
    save_file({'model.embed_tokens.weight': embedding, 'lm_head.weight': head}, str(args[1] / 'model.safetensors'))
    sign_teacher_directory(args)
    if head_only_update:
        with pytest.raises(ValueError, match='Tied output head has different values'):
            gate.verify(*args)
    else:
        assert gate.verify(*args)['status'] == 'PASS'


def test_token_id_mapping_tamper_even_if_resigned_is_rejected(tmp_path):
    args = fixture(tmp_path, update=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'<bos>': 2, '<eos>': 1, 'hello': 0}, unk_token='hello')),
                                       bos_token='<bos>', eos_token='<eos>', unk_token='hello')
    tokenizer.save_pretrained(args[1])
    sign_teacher_directory(args)
    with pytest.raises(ValueError, match='Token-to-id vocabulary mismatch'):
        gate.verify(*args)


@pytest.mark.parametrize('metric,value', [('val/f1/all', .1), ('val/f1/all', float('nan')),
                                         ('val/f1/all', float('inf')), ('val/qa_count/all', float('nan')),
                                         ('val/qa_count/all', float('inf')), ('val/qa_count/all', 3393.9)])
def test_resigned_selection_receipt_inconsistent_or_nonfinite_metric_rejected(tmp_path, metric, value):
    args = fixture(tmp_path, update=True)
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    report['candidates'][0]['metrics'][metric] = value
    report['selected'] = report['candidates'][0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='metric'):
        gate.verify(*args)


@pytest.mark.parametrize('missing', ['metrics', 'val/f1/all', 'val/qa_count/all'])
def test_missing_actual_candidate_metrics_rejected(tmp_path, missing):
    args = fixture(tmp_path, update=True)
    path = args[1] / 'teacher-manifest.json'
    report = json.loads(path.read_text())
    if missing == 'metrics':
        del report['candidates'][0]['metrics']
    else:
        del report['candidates'][0]['metrics'][missing]
    report['selected'] = report['candidates'][0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Missing actual'):
        gate.verify(*args)
