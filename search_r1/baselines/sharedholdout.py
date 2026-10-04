"""Read-only Search-E1 prebuilt manifest consumer; no split creation or sampling."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from .holdout import plain, paths, file_hash, frame_identity

def unique_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs)

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
    if canonical_questions_hash(frame, prompt_key) != _sha256_field(record.get('canonical_questions_sha256'), split + '.canonical_questions_sha256'):
        raise ValueError('Prebuilt canonical question identity differs: ' + split)
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
    if config.baselines.opsd_e1.get('stage_num') != 64:
        raise ValueError('Prebuilt stage validation requires configured64')
    manifest_path = Path(settings.manifest_path).expanduser().resolve()
    expected_hash = _sha256_field(settings.manifest_sha256, 'manifest_sha256')
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
        raise ValueError('Prebuilt manifest byte hash differs')
    manifest = unique_json(manifest_bytes)
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
    exclusion = manifest.get('exclusion_audit', {})
    exclusion_path = _logical_path(data_root, exclusion.get('path'))
    if file_hash(exclusion_path) != _sha256_field(exclusion.get('sha256'), 'exclusion_audit.sha256'):
        raise ValueError('Prebuilt exclusion audit byte hash differs')
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
    config.baselines.opsd_e1.stage_files = effective['stage']
    if config.baselines.opsd_e1.get('stage_num') != 64:
        raise ValueError('Prebuilt stage validation requires configured64')
    print('BASELINE_PREBUILT_MANIFEST_JSON=' + json.dumps(record,sort_keys=True),flush=True)
    return record


def assert_prebuilt_unchanged(config, previous):
    if previous.get('input_mode') != 'prebuilt_manifest': return
    expected = previous['effective_paths']
    actual = {'train':paths(config.data.train_files), 'validation':paths(config.data.val_files),
              'final':paths(config.data.final_val_files), 'stage':paths(config.baselines.opsd_e1.stage_files)}
    if any(actual[key] != [value] for key, value in expected.items()):
        raise ValueError('Effective prebuilt paths changed during execution')
    current = _validate_prebuilt(config)
    if current != previous:
        raise ValueError('Prebuilt SHA/ID/row/content identity changed during execution')
    print('BASELINE_PREBUILT_FINAL_IDENTITY_JSON=' + json.dumps(current,sort_keys=True),flush=True)
