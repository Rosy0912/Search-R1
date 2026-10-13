"""Immutable, question-only fixed-encoder features; never extract on training GPUs.

``prepare`` writes one FP16 vector per original prompt question on remote storage.
Runtime only mmap-reads these features and checks exact QID/text/content bindings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch


OFFICIAL_ENCODER = 'Qwen/Qwen2.5-3B-Instruct'
OFFICIAL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
COMMON_MANIFEST_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
SCHEMA_VERSION = 1
QUESTION_EXTRACTION_RULE = "CandidateStream.original_question:v1:join-message-content-with-newline"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def question_text_sha(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('QPROBE_EMPTY_OR_NONSTRING_QUESTION')
    # Exact original text, unlike selector's NFKC whitespace duplicate identity.
    return hashlib.sha256(question.encode('utf-8')).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
        raise ValueError('QPROBE_INVALID_SPEC_SHA: ' + name)


def validate_spec(spec: Mapping):
    spec = dict(spec)
    required = ('encoder_id', 'encoder_revision', 'encoder_files_sha256', 'tokenizer_sha256',
                'prompt_template_sha256', 'feature_layer', 'pooling', 'feature_dtype',
                'feature_dim', 'max_prompt_tokens', 'question_extraction', 'train_manifest_sha256', 'split',
                'encoder_compute_dtype')
    if any(k not in spec for k in required):
        raise ValueError('QPROBE_INCOMPLETE_FIXED_FEATURE_SPEC')
    if spec['encoder_id'] != OFFICIAL_ENCODER or spec['encoder_revision'] != OFFICIAL_REVISION:
        raise ValueError('QPROBE_REQUIRES_FIXED_OFFICIAL_ENCODER_REVISION')
    if spec['split'] != 'train' or spec['train_manifest_sha256'] != COMMON_MANIFEST_SHA:
        raise ValueError('QPROBE_REQUIRES_COMMON_TRAIN_SPLIT')
    for name in ('encoder_files_sha256', 'tokenizer_sha256', 'prompt_template_sha256', 'train_manifest_sha256'):
        _sha(spec[name], name)
    if type(spec['feature_layer']) is not int or spec['feature_layer'] < 0:
        raise ValueError('QPROBE_INVALID_FEATURE_LAYER')
    for key in ('feature_dim', 'max_prompt_tokens'):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError('QPROBE_INVALID_FEATURE_SPEC: ' + key)
    if spec['pooling'] != 'last_nonpadding_prompt_token' or spec['feature_dtype'] != 'float16':
        raise ValueError('QPROBE_UNSUPPORTED_FEATURE_REPRESENTATION')
    if spec['encoder_compute_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('QPROBE_ENCODER_COMPUTE_DTYPE_NOT_BOUND')
    if spec['question_extraction'] != QUESTION_EXTRACTION_RULE:
        raise ValueError('QPROBE_QUESTION_EXTRACTION_SPEC_MISMATCH')
    return spec


def original_question_and_id(row, prompt_key='prompt'):
    """Match CandidateStream.one exactly, without reading answers/rewards."""
    chat = row[prompt_key]
    if not isinstance(chat, (list, tuple, np.ndarray)) or not len(chat):
        raise ValueError('QPROBE_INVALID_PROMPT_MESSAGES')
    if any(not isinstance(message, Mapping) or not isinstance(message.get('content'), str) for message in chat):
        raise ValueError('QPROBE_NONSTRING_PROMPT_CONTENT')
    question = '\n'.join(message['content'] for message in chat)
    question_text_sha(question)
    extra = row.get('extra_info', {})
    upstream_id = extra.get('index') if isinstance(extra, dict) else None
    if upstream_id is None:
        from search_g1.mixed_selection import question_identity
        upstream_id = question_identity(question)
    # Identical json.dumps formatting to the existing source stream.
    stable_id = json.dumps([str(row['data_source']), str(upstream_id)])
    return question, stable_id


def last_nonpadding(hidden: torch.Tensor, attention_mask: torch.Tensor):
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError('QPROBE_BAD_HIDDEN_OR_MASK_SHAPE')
    if bool((attention_mask.sum(-1) == 0).any()):
        raise ValueError('QPROBE_ALL_PADDING_PROMPT')
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[:2])
    last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
    features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    if not torch.isfinite(features).all():
        raise ValueError('QPROBE_NONFINITE_ENCODER_FEATURE')
    return features


def extract_question_features(model, tokenizer, questions: Sequence[str], *, feature_layer: int,
                              max_prompt_tokens: int = 4096, device='cpu'):
    """A base Transformer forward (no LM head), executed only by prepare CLI.

    Hook reads a public decoder layer output. It does not early-exit the model;
    all base layers execute. This avoids claiming unverified partial forwards.
    No full hidden-state tuple or FSDP internals are used.
    """
    if not questions:
        raise ValueError('QPROBE_EMPTY_PRECOMPUTE_BATCH')
    for question in questions: question_text_sha(question)
    layers = getattr(model, 'layers', None)
    if layers is None or not 0 <= feature_layer < len(layers):
        raise ValueError('QPROBE_LAYER_NOT_FOUND_IN_BASE_MODEL')
    encoded = tokenizer(list(questions), padding=True, truncation=False, return_tensors='pt')
    if int(encoded['attention_mask'].sum(-1).max()) > max_prompt_tokens:
        raise ValueError('QPROBE_PROMPT_TOO_LONG_NO_SILENT_TRUNCATION')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    captured = []
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(last_nonpadding(hidden, encoded['attention_mask']).detach().float().cpu())
    handle = layers[feature_layer].register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
        model.train(was_training)
    if len(captured) != 1:
        raise ValueError('QPROBE_ENCODER_HOOK_NOT_EXACTLY_ONCE')
    return captured[0]


class FeatureCacheWriter:
    """Preallocate mmap vectors, then publish a completed immutable directory."""
    def __init__(self, cache_dir, spec, count):
        self.path = Path(cache_dir)
        if self.path.exists(): raise FileExistsError('QPROBE_CACHE_ALREADY_EXISTS: ' + str(self.path))
        self.spec = validate_spec(spec)
        if type(count) is not int or count <= 0: raise ValueError('QPROBE_BAD_CACHE_COUNT')
        self.count = count; self.written = 0; self.ids = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = Path(tempfile.mkdtemp(prefix=self.path.name+'.partial-', dir=self.path.parent))
        self.features = np.lib.format.open_memmap(self.partial/'features.npy', mode='w+', dtype=np.float16,
                                                  shape=(count, self.spec['feature_dim']))
        self.index = open(self.partial/'index.jsonl', 'x', encoding='utf-8')

    def append(self, qids, questions, features):
        features = features.detach().cpu().float().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
        if features.shape != (len(qids), self.spec['feature_dim']) or len(qids) != len(questions):
            raise ValueError('QPROBE_CACHE_WRITE_SHAPE_MISMATCH')
        cast = features.astype(np.float16)
        if not np.isfinite(features).all() or not np.isfinite(cast).all():
            raise ValueError('QPROBE_NONFINITE_OR_FP16_OVERFLOW_FEATURES')
        if self.written + len(qids) > self.count: raise ValueError('QPROBE_CACHE_WRITE_OVERFLOW')
        records = []
        ids = set()
        for i, (qid, question) in enumerate(zip(qids, questions)):
            if not isinstance(qid, str) or not qid or qid in self.ids or qid in ids:
                raise ValueError('QPROBE_EMPTY_OR_DUPLICATE_CACHE_QID')
            ids.add(qid)
            records.append(dict(qid=qid, question_hash=question_text_sha(question), row=self.written+i,
                                vector_sha256=hashlib.sha256(cast[i].tobytes()).hexdigest()))
        self.features[self.written:self.written+len(qids)] = cast
        for record in records: self.index.write(canonical_json(record)+'\n')
        self.ids.update(ids); self.written += len(qids)

    def finish(self):
        if self.written != self.count: raise ValueError('QPROBE_INCOMPLETE_CACHE')
        self.features.flush(); self.index.flush(); os.fsync(self.index.fileno()); self.index.close()
        manifest = dict(schema_version=SCHEMA_VERSION, spec=self.spec, spec_sha=sha_json(self.spec),
                        count=self.count, feature_dim=self.spec['feature_dim'],
                        files={name:dict(sha256=sha_file(self.partial/name), bytes=(self.partial/name).stat().st_size)
                               for name in ('features.npy', 'index.jsonl')})
        with open(self.partial/'manifest.json', 'x', encoding='utf-8') as f:
            f.write(canonical_json(manifest)+'\n'); f.flush(); os.fsync(f.fileno())
        # os.rename refuses replacing a nonempty directory; training cache is
        # admitted only with this completed manifest and never modified in place.
        if self.path.exists(): raise FileExistsError('QPROBE_CACHE_ALREADY_EXISTS')
        os.rename(self.partial, self.path)
        return manifest

    def abort(self):
        if not self.index.closed: self.index.close()
        if self.partial.exists(): shutil.rmtree(self.partial)


def write_feature_cache(cache_dir, spec, qids, questions, features):
    writer = FeatureCacheWriter(cache_dir, spec, len(qids))
    try:
        writer.append(qids, questions, features)
        return writer.finish()
    except Exception:
        writer.abort(); raise


class FeatureCache:
    def __init__(self, cache_dir, expected_spec=None, expected_spec_sha=None):
        self.path = Path(cache_dir); self.manifest_path = self.path/'manifest.json'
        if not self.manifest_path.is_file():
            raise FileNotFoundError('QPROBE_FIXED_CACHE_MISSING_RUN_PREPARE_FIRST: ' + str(self.path))
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('QPROBE_CACHE_SCHEMA_MISMATCH')
        self.spec = validate_spec(self.manifest['spec'])
        self.spec_sha = sha_json(self.spec); self.feature_spec_sha = self.spec_sha
        if self.spec_sha != self.manifest.get('spec_sha'):
            raise ValueError('QPROBE_FEATURE_SPEC_SHA_MISMATCH')
        if expected_spec is not None and self.spec_sha != sha_json(validate_spec(expected_spec)):
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        if expected_spec_sha is not None and self.spec_sha != expected_spec_sha:
            raise ValueError('QPROBE_FEATURE_SPEC_MISMATCH')
        self.feature_dim = self.spec['feature_dim']
        for name in ('features.npy', 'index.jsonl'):
            item = self.manifest['files'][name]
            if not (self.path/name).is_file() or (self.path/name).stat().st_size != item['bytes']:
                raise ValueError('QPROBE_CACHE_FILE_SIZE_MISMATCH: ' + name)
        if sha_file(self.path/'index.jsonl') != self.manifest['files']['index.jsonl']['sha256']:
            raise ValueError('QPROBE_CACHE_INDEX_SHA_MISMATCH')
        self.features = np.load(self.path/'features.npy', mmap_mode='r', allow_pickle=False)
        if self.features.dtype != np.float16 or self.features.shape != (self.manifest['count'], self.feature_dim):
            raise ValueError('QPROBE_CACHE_ARRAY_SHAPE_DTYPE_MISMATCH')
        self.index = {}
        with open(self.path/'index.jsonl', encoding='utf-8') as f:
            for expected_row, line in enumerate(f):
                record = json.loads(line)
                if record['qid'] in self.index or record['row'] != expected_row:
                    raise ValueError('QPROBE_CACHE_DUPLICATE_OR_REORDERED_INDEX')
                _sha(record['question_hash'], 'question_hash'); _sha(record['vector_sha256'], 'vector_sha256')
                self.index[record['qid']] = record
        if len(self.index) != self.manifest['count']:
            raise ValueError('QPROBE_CACHE_INDEX_COUNT_MISMATCH')

    def get_features(self, qids: Sequence[str], questions: Sequence[str]):
        if len(qids) != len(questions) or len(qids) == 0:
            raise ValueError('QPROBE_CACHE_REQUEST_SHAPE_MISMATCH')
        result = np.empty((len(qids), self.feature_dim), dtype=np.float32)
        for i, (qid, question) in enumerate(zip(qids, questions)):
            if qid not in self.index:
                raise KeyError('QPROBE_QUESTION_NOT_PRECOMPUTED: ' + str(qid))
            record = self.index[qid]
            if record['question_hash'] != question_text_sha(question):
                raise ValueError('QPROBE_CACHED_QUESTION_TEXT_MISMATCH: ' + str(qid))
            vector = self.features[record['row']]
            if not np.isfinite(vector).all() or hashlib.sha256(vector.tobytes()).hexdigest() != record['vector_sha256']:
                raise ValueError('QPROBE_CORRUPT_OR_NONFINITE_CACHED_VECTOR: ' + str(qid))
            result[i] = vector
        return torch.from_numpy(result)

    def verify_all(self):
        if sha_file(self.path/'features.npy') != self.manifest['files']['features.npy']['sha256']:
            raise ValueError('QPROBE_CACHE_FEATURES_SHA_MISMATCH')
        if not np.isfinite(self.features).all():
            raise ValueError('QPROBE_NONFINITE_FEATURE_CACHE')
        return dict(count=len(self.index), feature_dim=self.feature_dim, spec_sha=self.spec_sha,
                    feature_file_sha256=self.manifest['files']['features.npy']['sha256'])
