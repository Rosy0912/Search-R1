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


def frame_identity(frame, prompt_key='prompt'):
    questions = sorted(set(frame[prompt_key].map(question_key)))
    rows = sorted(json.dumps(plain(row), sort_keys=True, ensure_ascii=False)
                  for row in frame.to_dict('records'))
    return dict(questions_sha256=hashlib.sha256('\n'.join(questions).encode()).hexdigest(),
                rows_sha256=hashlib.sha256('\n'.join(rows).encode()).hexdigest())


def select_stage_subset(frame, size, seed=42, prompt_key='prompt'):
    """Select exactly size distinct held-out questions independent of row order."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError('Stage validation size must be a positive integer')
    candidates = []
    for position, row in enumerate(frame.to_dict('records')):
        key = question_key(row[prompt_key])
        priority = hashlib.sha256((str(seed) + '\0' + key).encode()).hexdigest()
        candidates.append((priority, key, json.dumps(plain(row.get('extra_info', {})), sort_keys=True, ensure_ascii=False), position))
    selected, seen = [], set()
    for _, key, _, position in sorted(candidates):
        if key not in seen:
            selected.append(position)
            seen.add(key)
        if len(selected) == size:
            return frame.iloc[selected].reset_index(drop=True)
    raise ValueError('Insufficient distinct held-out questions for stage validation')


def content_identity(specification, train, holdout):
    content = dict(version=1, split_version=specification['version'],
                   train_sources=sorted(item['sha256'] for item in specification['train']),
                   excluded_sources=sorted(item['sha256'] for item in specification['final']),
                   fraction=specification['fraction'], seed=specification['seed'],
                   prompt_key=specification['prompt_key'],
                   train=frame_identity(train, specification['prompt_key']),
                   holdout=frame_identity(holdout, specification['prompt_key']))
    content['sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


def prepare_common_holdout(config):
    settings = config.baselines.holdout
    mode = settings.get('input_mode', 'generated')
    if mode == 'prebuilt_manifest':
        return prepare_prebuilt_holdout(config)
    if mode != 'generated':
        raise ValueError('Unknown holdout input_mode: '+str(mode))
    if not settings.enabled:
        raise ValueError('All new baseline/GRPO comparisons require the common holdout split')
    inputs = [str(Path(p).expanduser().resolve()) for p in paths(config.data.train_files)]
    evaluation_files = paths(config.data.val_files)
    if config.data.get('final_val_files'):
        evaluation_files += paths(config.data.final_val_files)
    finals = sorted({str(Path(p).expanduser().resolve()) for p in evaluation_files})
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
                manifest = dict(content_identity=content_identity(specification, retained, holdout),
                    specification=specification, train_rows=len(retained), holdout_rows=len(holdout),
                    sha256={name: file_hash(staging / name) for name in ('train.parquet', 'stage-validation.parquet')})
                (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                os.rename(staging, output)
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['specification'] != specification:
            raise ValueError('Common split specification differs')
        for name, expected in manifest['sha256'].items():
            if file_hash(output / name) != expected:
                raise ValueError('Common split has been modified: ' + name)
    if 'content_identity' not in manifest:
        raise ValueError('Legacy split missing content_identity; use a new common-data directory')
    print('BASELINE_SPLIT_MANIFEST_JSON=' + json.dumps(manifest, sort_keys=True), flush=True)
    config.data.train_files = str(output / 'train.parquet')
    config.baselines.bapo.stage_validation_files = str(output / 'stage-validation.parquet')
    size = config.baselines.bapo.get('stage_validation_size')
    if (config.baselines.get('method') == 'bapo' and size is not None
            and not config.baselines.get('holdout_eval_only', False)):
        subset_spec = dict(version=1, parent_content_identity=manifest['content_identity']['sha256'],
                           source_sha256=manifest['sha256']['stage-validation.parquet'],
                           size=size, seed=int(settings.seed), prompt_key=config.data.prompt_key)
        subset_id = hashlib.sha256(json.dumps(subset_spec, sort_keys=True).encode()).hexdigest()[:20]
        subset_parent = output / 'stage-subsets'
        subset_parent.mkdir(exist_ok=True)
        subset_path = subset_parent / subset_id
        with open(subset_parent / (subset_id + '.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            selected = select_stage_subset(pd.read_parquet(output / 'stage-validation.parquet'),
                                           size, int(settings.seed), config.data.prompt_key)
            identity = frame_identity(selected, config.data.prompt_key)
            if not subset_path.exists():
                with tempfile.TemporaryDirectory(dir=subset_parent, prefix=subset_id + '-') as temp:
                    staging = Path(temp) / 'subset'
                    staging.mkdir()
                    selected.to_parquet(staging / 'stage-validation.parquet', index=False)
                    record = dict(specification=subset_spec, rows=len(selected),
                                  content_identity=identity,
                                  question_identities=[hashlib.sha256(question_key(prompt).encode()).hexdigest()
                                      for prompt in selected[config.data.prompt_key]],
                                  sha256=file_hash(staging / 'stage-validation.parquet'))
                    (staging / 'manifest.json').write_text(json.dumps(record, indent=2))
                    os.rename(staging, subset_path)
            record = json.loads((subset_path / 'manifest.json').read_text())
            if (record['specification'] != subset_spec or record['rows'] != size
                    or record['content_identity'] != identity
                    or file_hash(subset_path / 'stage-validation.parquet') != record['sha256']
                    or frame_identity(pd.read_parquet(subset_path / 'stage-validation.parquet'),
                                      config.data.prompt_key) != identity):
                raise ValueError('Stage validation subset has been modified')
        config.baselines.bapo.stage_validation_files = str(subset_path / 'stage-validation.parquet')
        print('BAPO_STAGE_SUBSET_MANIFEST_JSON=' + json.dumps(record, sort_keys=True), flush=True)
    return manifest


E1_SOURCES = ('nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle')
E1_FINAL_COUNTS = {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267, 'hotpotqa': 7405,
                   '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}
E1_INSTRUCTION = ('Answer the given question. You must conduct reasoning inside <think> and </think> '
               'first every time you get new information. After reasoning, if you lack knowledge, '
               'call a search engine by <search> query </search>; results appear between '
               '<information> and </information>. You may search at most four times. '
               'Provide the final answer inside <answer> and </answer>, without detailed illustrations. '
               'For example, <answer> Beijing </answer>.')


def _sha256_field(value, label):
    import re
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Prebuilt manifest requires SHA256: ' + label)
    return value


def _logical_path(data_root, logical):
    """Resolve local storage without rewriting the shared manifest."""
    if not isinstance(logical, str) or not logical or '\\' in logical:
        raise ValueError('Prebuilt manifest requires a relative logical path')
    relative = Path(logical)
    if relative.is_absolute() or any(part in ('..', '.') for part in relative.parts):
        raise ValueError('Prebuilt manifest path escapes data root')
    result = (data_root / relative).resolve()
    if data_root not in result.parents or not result.is_file():
        raise ValueError('Prebuilt manifest data file missing or outside data root: ' + logical)
    return result


def ordered_question_ids_hash(frame):
    identifiers = [plain(value)['index'] for value in frame['extra_info']]
    return hashlib.sha256(json.dumps(identifiers, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def canonical_question_key(prompt):
    """Approved E1 overlap rule; legacy question_key/frame_identity stay unchanged."""
    value = plain(prompt)
    prefix = E1_INSTRUCTION + '\nQuestion: '
    if (not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],dict)
            or value[0].get('role')!='user' or not isinstance(value[0].get('content'),str)
            or not value[0]['content'].startswith(prefix)):
        raise ValueError('Prebuilt prompt differs from approved core.INSTRUCTION')
    question=value[0]['content'][len(prefix):]
    canonical=' '.join(question.casefold().rstrip('?').split())
    if not canonical: raise ValueError('Prebuilt question content is empty')
    return canonical


def canonical_questions_hash(frame,prompt_key='prompt'):
    values=sorted(set(canonical_question_key(prompt) for prompt in frame[prompt_key]))
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def _prebuilt_frame(path, record, prompt_key, split):
    import re
    expected = _sha256_field(record.get('sha256'), split + '.sha256')
    if file_hash(path) != expected:
        raise ValueError('Prebuilt split byte hash differs: ' + split)
    frame = pd.read_parquet(path)
    if type(record.get('rows')) is not int or record['rows'] != len(frame):
        raise ValueError('Prebuilt split row count differs: ' + split)
    required = {prompt_key, 'extra_info', 'data_source', 'reward_model'}
    if not required.issubset(frame.columns):
        raise ValueError('Prebuilt split lacks training columns: ' + split)
    ids, questions = [], []
    for row in frame.to_dict('records'):
        extra = plain(row['extra_info'])
        source = row['data_source']
        if not isinstance(extra, dict) or not isinstance(extra.get('index'), str):
            raise ValueError('Prebuilt row requires string source question ID: ' + split)
        identifier = extra['index']
        match = re.fullmatch(r'([^:]+):(train|test|dev):(\d+)', identifier)
        if not match or match[1] != source or source not in E1_SOURCES:
            raise ValueError('Prebuilt source/question ID mismatch: ' + split)
        if split != 'final' and match[2] != 'train':
            raise ValueError('Prebuilt training holdout ID must originate from train')
        final_split = 'test' if source in ('nq','triviaqa','popqa','bamboogle') else 'dev'
        if split == 'final' and match[2] != final_split:
            raise ValueError('Prebuilt final ID uses wrong benchmark split')
        prompt = plain(row[prompt_key])
        prefix = E1_INSTRUCTION + '\nQuestion: '
        if (not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict)
                or prompt[0].get('role') != 'user' or not isinstance(prompt[0].get('content'), str)
                or not prompt[0]['content'].startswith(prefix) or not prompt[0]['content'][len(prefix):].strip()):
            raise ValueError('Prebuilt prompt differs from approved core.INSTRUCTION: ' + split)
        reward = plain(row['reward_model'])
        if (not isinstance(reward, dict) or reward.get('style') != 'rule'
                or not isinstance(reward.get('ground_truth'), dict)
                or not isinstance(reward['ground_truth'].get('target'), list)
                or not reward['ground_truth']['target']):
            raise ValueError('Prebuilt row lacks package target answer list')
        ids.append(identifier); questions.append(canonical_question_key(prompt))
    if len(ids) != len(set(ids)):
        raise ValueError('Prebuilt split contains duplicate question IDs: ' + split)
    source_counts = {str(key): int(value) for key, value in frame['data_source'].value_counts().items()}
    if source_counts != record.get('source_counts'):
        raise ValueError('Prebuilt split source counts differ: ' + split)
    if ordered_question_ids_hash(frame) != _sha256_field(record.get('question_ids_sha256'), split + '.question_ids_sha256'):
        raise ValueError('Prebuilt ordered question IDs differ: ' + split)
    identity = frame_identity(frame, prompt_key)
    if identity != record.get('content_identity'):
        raise ValueError('Prebuilt split content identity differs: ' + split)
    if split in ('stage','validation') and len(set(questions))!=len(frame):
        raise ValueError('Prebuilt heldout questions are not independently unique: '+split)
    if file_hash(path)!=expected:
        raise ValueError('Prebuilt split changed during validation: '+split)
    return frame, set(ids), set(questions)


def prebuilt_overlap_audit(ids, questions):
    from itertools import combinations
    overlaps = {}
    for left, right in combinations(sorted(ids), 2):
        overlaps[left+'|'+right] = {'ids':len(ids[left]&ids[right]), 'questions':len(questions[left]&questions[right])}
    if any(value['ids'] or value['questions'] for value in overlaps.values()):
        raise ValueError('Prebuilt split overlap; no automatic resplit: ' + json.dumps(overlaps,sort_keys=True))
    return overlaps


def _validate_prebuilt(config):
    settings = config.baselines.holdout
    if not settings.enabled:
        raise ValueError('Prebuilt common holdout must be enabled')
    if any(config.data.get(key) is not None for key in ('train_data_num','val_data_num')):
        raise ValueError('Prebuilt common data cannot be subsampled by train_data_num/val_data_num')
    if config.baselines.bapo.get('stage_validation_size') != 64:
        raise ValueError('Prebuilt stage validation requires configured64')
    manifest_path = Path(settings.manifest_path).expanduser().resolve()
    expected_hash = _sha256_field(settings.manifest_sha256, 'manifest_sha256')
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
        raise ValueError('Prebuilt manifest byte hash differs')
    manifest = json.loads(manifest_bytes)
    if (type(manifest.get('version')) is not int or manifest.get('version') != 1 or manifest.get('protocol') != 'search-e1-common-300'
            or manifest.get('seed') != 42 or manifest.get('prompt_key') != config.data.prompt_key
            or int(settings.seed) != 42):
        raise ValueError('Prebuilt manifest protocol/seed/prompt key differs')
    preparation = manifest.get('preparation', {})
    if (preparation.get('dataset_id') != 'RUC-NLPIR/FlashRAG_datasets'
            or not isinstance(preparation.get('dataset_revision'), str) or not preparation['dataset_revision']
            or preparation.get('selection') != 'source-order-seed42-first256-dev-next32-stage'
            or preparation.get('source_order') != list(E1_SOURCES)):
        raise ValueError('Prebuilt source provenance/selection policy differs')
    for key in ('prepare_sha256', 'core_sha256'):
        _sha256_field(preparation.get(key), 'preparation.' + key)
    data_root = Path(settings.data_root).expanduser().resolve()
    raw_sources = preparation.get('raw_sources')
    if not isinstance(raw_sources, list):
        raise ValueError('Prebuilt raw source provenance missing')
    expected_raw = {('nq','train'),('hotpotqa','train')} | {
        (source, 'test' if source in ('nq','triviaqa','popqa','bamboogle') else 'dev') for source in E1_SOURCES}
    actual_raw = []
    for record in raw_sources:
        if not isinstance(record, dict): raise ValueError('Prebuilt raw source record invalid')
        actual_raw.append((record.get('source'), record.get('split')))
        path = _logical_path(data_root, record.get('path'))
        if file_hash(path) != _sha256_field(record.get('sha256'), 'raw source'):
            raise ValueError('Prebuilt raw source byte hash differs')
        if type(record.get('rows')) is not int or record['rows'] < 1:
            raise ValueError('Prebuilt raw source rows invalid')
    if len(actual_raw) != 9 or set(actual_raw) != expected_raw:
        raise ValueError('Prebuilt raw source provenance must cover two train and seven final splits')
    records = manifest.get('splits', {})
    if set(records) != {'train','validation','stage','final'}:
        raise ValueError('Prebuilt manifest requires exactly four splits')
    counts = {'validation':512, 'stage':64, 'final':51713}
    train_count=records['train'].get('rows')
    raw_train_count = sum(record['rows'] for record in raw_sources if record['split']=='train')
    if type(train_count) is not int or not 0<train_count<=raw_train_count-counts['validation']-counts['stage']:
        raise ValueError('Prebuilt train count must reflect actual overlap-excluded retained rows')
    counts['train']=train_count
    frames, ids, questions, resolved = {}, {}, {}, {}
    for split, record in records.items():
        path = _logical_path(data_root, record.get('path'))
        frames[split], ids[split], questions[split] = _prebuilt_frame(path, record, config.data.prompt_key, split)
        resolved[split] = str(path)
        if len(frames[split]) != counts[split]:
            raise ValueError('Prebuilt approved split size differs: ' + split)
    for split, expected in [('validation', {'nq':256,'hotpotqa':256}), ('stage', {'nq':32,'hotpotqa':32}), ('final', E1_FINAL_COUNTS)]:
        if records[split]['source_counts'] != expected:
            raise ValueError('Prebuilt approved source allocation differs: ' + split)
    if set(records['train']['source_counts']) != {'nq','hotpotqa'}:
        raise ValueError('Prebuilt train must originate only from NQ/Hotpot training')
    overlaps = prebuilt_overlap_audit(ids, questions)
    if file_hash(manifest_path) != expected_hash:
        raise ValueError('Prebuilt manifest changed during validation')
    return dict(input_mode='prebuilt_manifest', manifest_path=str(manifest_path), manifest_sha256=expected_hash,
        manifest=manifest, effective_paths=resolved, overlaps=overlaps,
        content_identity={'sha256':expected_hash}, train_rows=counts['train'], holdout_rows=counts['validation'])


def prepare_prebuilt_holdout(config):
    """Read immutable shared inputs; never create, sample, overwrite, or cache splits."""
    record = _validate_prebuilt(config)
    effective = record['effective_paths']
    config.data.train_files = effective['train']
    config.data.val_files = effective['validation']
    config.data.final_val_files = effective['final']
    config.baselines.bapo.stage_validation_files = effective['stage']
    if config.baselines.bapo.get('stage_validation_size') != 64:
        raise ValueError('Prebuilt stage validation requires configured64')
    print('BASELINE_PREBUILT_MANIFEST_JSON=' + json.dumps(record,sort_keys=True),flush=True)
    return record


def assert_prebuilt_unchanged(config, previous):
    if previous.get('input_mode') != 'prebuilt_manifest': return
    expected = previous['effective_paths']
    actual = {'train':paths(config.data.train_files), 'validation':paths(config.data.val_files),
              'final':paths(config.data.final_val_files), 'stage':paths(config.baselines.bapo.stage_validation_files)}
    if any(actual[key] != [value] for key, value in expected.items()):
        raise ValueError('Effective prebuilt paths changed during execution')
    current = _validate_prebuilt(config)
    if current != previous:
        raise ValueError('Prebuilt SHA/ID/row/content identity changed during execution')
    print('BASELINE_PREBUILT_FINAL_IDENTITY_JSON=' + json.dumps(current,sort_keys=True),flush=True)
