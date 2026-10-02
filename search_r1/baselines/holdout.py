"""Immutable, question-grouped common training/phase-validation split."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import fcntl
import pandas as pd


def plain(value):
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def question_key(prompt):
    return json.dumps(plain(prompt), sort_keys=True, ensure_ascii=False)


def paths(value):
    return [value] if isinstance(value, str) else list(value)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_frames(train, final, fraction=.02, seed=42, prompt_key='prompt'):
    """Keep duplicate prompts together; never choose final-test questions for holdout."""
    if not 0 < fraction < 1:
        raise ValueError('Holdout fraction must lie strictly between zero and one')
    keys = train[prompt_key].map(question_key)
    final_keys = set(final[prompt_key].map(question_key))
    unique = set(keys)
    eligible = unique - final_keys
    if len(unique) < 2 or not eligible:
        raise ValueError('Not enough independent questions for holdout')
    count = min(len(unique) - 1, max(1, math.ceil(len(unique) * fraction)))
    if len(eligible) < count:
        raise ValueError('Insufficient questions disjoint from final test for requested holdout')
    selected = set(sorted(eligible, key=lambda key: hashlib.sha256(
        (str(seed) + '\0' + key).encode()).hexdigest())[:count])
    mask = keys.isin(selected)
    # UID collisions silently merge different questions in all grouped algorithms.
    if 'extra_info' not in train:
        raise ValueError('Training questions require explicit extra_info.index')
    if 'extra_info' in train:
        seen = {}
        for key, extra in zip(keys, train['extra_info']):
            extra = plain(extra)
            if not isinstance(extra, dict) or 'index' not in extra or extra['index'] is None:
                raise ValueError('Each training question requires explicit extra_info.index')
            index = extra['index']
            identity = json.dumps(index, sort_keys=True)
            if identity in seen and seen[identity] != key:
                raise ValueError('extra_info.index identifies multiple different questions')
            seen[identity] = key
    return train.loc[~mask].reset_index(drop=True), train.loc[mask].reset_index(drop=True)


def prepare_common_holdout(config):
    settings = config.baselines.holdout
    if not settings.enabled:
        raise ValueError('All new baseline/GRPO comparisons require the common holdout split')
    inputs = [str(Path(p).expanduser().resolve()) for p in paths(config.data.train_files)]
    finals = [str(Path(p).expanduser().resolve()) for p in paths(config.data.val_files)]
    specification = dict(train=[dict(path=p, sha256=file_hash(p)) for p in inputs],
                         final=[dict(path=p, sha256=file_hash(p)) for p in finals],
                         fraction=float(settings.fraction), seed=int(settings.seed),
                         prompt_key=config.data.prompt_key, version=1)
    identifier = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
    parent = Path(settings.output_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / identifier
    with open(parent / (identifier + '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            train = pd.concat([pd.read_parquet(p) for p in inputs], ignore_index=True)
            final = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            retained, holdout = split_frames(train, final, specification['fraction'],
                                             specification['seed'], specification['prompt_key'])
            with tempfile.TemporaryDirectory(dir=parent, prefix=identifier + '-') as temp:
                staging = Path(temp) / 'split'
                staging.mkdir()
                retained.to_parquet(staging / 'train.parquet', index=False)
                holdout.to_parquet(staging / 'stage-validation.parquet', index=False)
                manifest = dict(specification=specification, train_rows=len(retained), holdout_rows=len(holdout),
                    sha256={name: file_hash(staging / name) for name in ('train.parquet', 'stage-validation.parquet')})
                (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                os.rename(staging, output)
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['specification'] != specification:
            raise ValueError('Common split specification differs')
        for name, expected in manifest['sha256'].items():
            if file_hash(output / name) != expected:
                raise ValueError('Common split has been modified: ' + name)
    config.data.train_files = str(output / 'train.parquet')
    config.baselines.bapo.stage_validation_files = str(output / 'stage-validation.parquet')
    return manifest
