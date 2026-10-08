"""Isolated, opt-in final200 paired evaluation; never a training profile.

This candidate is blocked until a separately reviewed source, actual HF200
prerequisite verifier, allocation and stage manifest are supplied.
"""
import gzip
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import corrected_evaluation as legacy

TRAINING_SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
OFFICIAL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
OFFICIAL_MANIFEST_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
FINAL_SHA = '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'
FINAL_IDS_SHA = '4e9df91a75beba40c3316576f6da049851e736df4a1e5c873f4f0cdb883fd4b1'
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa': 12576}, musique=2417, bamboogle=125)
ROLES = ('official_base', 'actual_hf200')
encoded = legacy.encoded
loads = legacy.loads
canonical = legacy.canonical
digest = legacy.digest


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer: ' + label)
    return value


def pinned_json(path, sha):
    path = Path(path)
    before = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise ValueError('Pinned JSON must be a physical regular file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    keys = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, k) != getattr(s, k) for s in (opened, after_fd, after) for k in keys):
        raise ValueError('Pinned JSON changed while read')
    if not re.fullmatch('[0-9a-f]{64}', str(sha)) or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Pinned JSON byte SHA differs')
    return loads(raw)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(Path(path).parent)


def prepare_inputs(common_path, output):
    """Read the existing common final artifact. Never select or resplit rows."""
    from .sharedholdout import _prebuilt_frame, ordered_question_ids_hash
    common = pinned_json(common_path, COMMON_SHA)
    record = common['splits']['final']
    if (record['rows'] != 51713 or record['source_counts'] != COUNTS or
            record['sha256'] != FINAL_SHA or record['question_ids_sha256'] != FINAL_IDS_SHA):
        raise ValueError('Wrong common final artifact')
    path = (Path(common_path).parent / record['path']).resolve()
    frame, _, _ = _prebuilt_frame(path, record, common['prompt_key'], 'final')
    rows = []
    for file_row, source in enumerate(frame.to_dict('records')):
        identity = legacy.row_identity(source[common['prompt_key']], source['data_source'], source['reward_model'])
        question_id = canonical(source['extra_info'])['index']
        public = dict(ordinal=file_row, file_sha256=FINAL_SHA, file_row=file_row,
                      question_id=question_id, prompt=identity['prompt'], source=identity['source'])
        rows.append(dict(ordinal=file_row, file_sha256=FINAL_SHA, file_row=file_row,
                         question_id=question_id, question_uid=hashlib.sha256(encoded(public)).hexdigest(),
                         identity=identity, identity_sha256=hashlib.sha256(encoded(identity)).hexdigest()))
    value = dict(version=1, scope='native-scoring-view', kind='FINAL200_ORDERED_DRIVER_ONLY_INPUTS',
                 common_manifest_sha256=COMMON_SHA, prompt_key=common['prompt_key'],
                 source_counts=COUNTS, question_ids_sha256=ordered_question_ids_hash(frame),
                 inputs=[dict(path=str(path), sha256=FINAL_SHA)], records=rows)
    write_once(output, encoded(value))
    return digest(output)


def load_inputs(path, sha):
    saved = pinned_json(path, sha)
    if (saved.get('kind') != 'FINAL200_ORDERED_DRIVER_ONLY_INPUTS' or
            saved.get('common_manifest_sha256') != COMMON_SHA or saved.get('source_counts') != COUNTS or
            saved.get('question_ids_sha256') != FINAL_IDS_SHA):
        raise ValueError('Wrong final200 input identity')
    if len(saved.get('inputs', [])) != 1 or saved['inputs'][0]['sha256'] != FINAL_SHA:
        raise ValueError('Final200 must read the actual single common final parquet')
    checked = legacy.load_inputs(path, sha)
    if encoded(saved) != encoded(checked):
        raise ValueError('Input manifest changed during legacy row replay')
    if Counter(r['identity']['source'] for r in saved['records']) != COUNTS:
        raise ValueError('Final200 source counts differ')
    import pandas as pd
    frame = pd.read_parquet(saved['inputs'][0]['path'])
    ids = []
    for i, source in enumerate(frame.to_dict('records')):
        row = saved['records'][i]
        question_id = canonical(source['extra_info'])['index']
        public = dict(ordinal=i, file_sha256=FINAL_SHA, file_row=i,
                      question_id=question_id, prompt=row['identity']['prompt'], source=row['identity']['source'])
        if (row['question_id'] != question_id or row['question_uid'] != hashlib.sha256(encoded(public)).hexdigest()
                or row['file_sha256'] != FINAL_SHA or row['file_row'] != i):
            raise ValueError('Original common question identity differs')
        ids.append(question_id)
    if (len(ids) != len(set(ids)) or hashlib.sha256(json.dumps(ids, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest() != FINAL_IDS_SHA):
        raise ValueError('Original common ordered IDs differ')
    if digest(path) != sha or digest(saved['inputs'][0]['path']) != FINAL_SHA:
        raise ValueError('Input manifest changed during final200 validation')
    return saved


def enabled(config):
    return config.get('baselines', {}).get('final200_evaluation', {}).get('enabled') is True


def validate_eval_config(config):
    if (config.baselines.method != 'opsd' or config.baselines.holdout_eval_only or
            config.trainer.get('val_only') is not True or config.trainer.val_before_train is not True or
            config.baselines.opsd_e1.enabled is not False or config.baselines.opsd_e1.profile is not None or
            legacy.enabled(config)):
        raise ValueError('Final200 is a separate eval-only profile, never an E1 training profile')
    if (not config.do_search or config.data.val_batch_size != 64 or config.data.val_data_num is not None or
            config.baselines.local_runtime_only is not True or config.baselines.local_runtime_host != 'h100-10'):
        raise ValueError('Wrong full native same-node final evaluation contract')
    budget = legacy.budget_snapshot(config)
    expected = {'data.max_start_length': 2048, 'data.max_prompt_length': 4096,
                'data.max_response_length': 500, 'data.max_obs_length': 500,
                'max_turns': 4, 'force_final_answer': False, 'retriever.topk': 3,
                'trainer.n_gpus_per_node': 8, 'trainer.nnodes': 1,
                'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
                'actor_rollout_ref.rollout.n': 1}
    if any(budget[k] != v or type(budget[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('Final evaluation changed the common tool budget')
    from urllib.parse import urlsplit
    url = urlsplit(config.retriever.url)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path != '/retrieve':
        raise ValueError('Evaluation retrieval must use explicit loopback')
    if (config.actor_rollout_ref.actor.use_kl_loss or config.actor_rollout_ref.actor.kl_loss_coef != 0 or
            config.algorithm.kl_ctrl.kl_coef != 0 or config.actor_rollout_ref.actor.entropy_coeff != 0):
        raise ValueError('Evaluation must not introduce PG/reference KL/entropy updates')
    return budget


def verify_stage(provenance):
    if (provenance.get('kind') != 'REVIEWED_FINAL200_PAIRED_EVAL_STAGE' or
            provenance.get('model_role') not in ROLES or provenance.get('training_commit') != TRAINING_SOURCE or
            provenance.get('training_job_id') != '3304' or provenance.get('actual_outer_updates') != 200 or
            provenance.get('common_manifest_sha256') != COMMON_SHA or
            provenance.get('official_base_revision') != OFFICIAL_REVISION or
            provenance.get('official_base_manifest_sha256') != OFFICIAL_MANIFEST_SHA):
        raise ValueError('Wrong final200 role/source/model/data prerequisites')
    integer(provenance['actual_outer_updates'], 'actual outer', 1)
    if not isinstance(provenance.get('eval_job_id'), str) or not re.fullmatch('[1-9][0-9]*', provenance['eval_job_id']):
        raise ValueError('Actual evaluation allocation is required')
    if provenance['eval_job_id'] == '3304' or os.environ.get('SLURM_JOB_ID') != provenance['eval_job_id']:
        raise ValueError('Eval must use its actual new allocation, not impersonate training3304')
    for key in ('training_hf_receipt', 'strict_training_hf_verifier'):
        value = provenance.get(key)
        if not isinstance(value, dict) or digest(value['path']) != value['sha256']:
            raise ValueError('Blocked: actual reviewed200/HF200 prerequisites are missing')
    import sys
    verifier = provenance['strict_training_hf_verifier']
    receipt = provenance['training_hf_receipt']
    command = [sys.executable, verifier['path'], '--receipt', receipt['path'], '--receipt-sha256', receipt['sha256'],
               '--model-role', provenance['model_role'], '--model-directory', provenance['model_directory']]
    result = subprocess.run(command, check=True, capture_output=True)
    expected = dict(kind='STRICT_ACTUAL_TRAINING3304_HF200_PREREQUISITES', training_commit=TRAINING_SOURCE,
                    training_job_id='3304', actual_outer_updates=200, hf_step=200, final_questions=51713,
                    receipt_sha256=receipt['sha256'], model_role=provenance['model_role'],
                    model_directory=provenance['model_directory'])
    if loads(result.stdout) != expected:
        raise ValueError('Strict200/HF200 prerequisite verifier did not return exact bound proof')
    if digest(verifier['path']) != verifier['sha256'] or digest(receipt['path']) != receipt['sha256']:
        raise ValueError('Prerequisite source/evidence changed')


def load_stage(settings):
    sha = os.environ.get('OPSD_FINAL200_EVAL_STAGE_SHA256', '')
    if os.environ.get('OPSD_CORRECTED_EVAL_STAGE_SHA256') != sha:
        raise ValueError('Worker/driver actual stage env differs')
    value = pinned_json(settings.reviewed_stage_manifest, sha)
    verify_stage(value)
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != value['eval_commit']:
        raise ValueError('Actual eval source commit differs')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Evaluation source is not clean frozen source')
    legacy.verify_generation_sources(value)
    required = {'search_r1/baselines/final200_evaluation.py', 'search_r1/baselines/sharedholdout.py'}
    if not required <= set(value['source_files']):
        raise ValueError('Incomplete new source provenance')
    for relative, expected in value['source_files'].items():
        file = repo / relative
        if not relative.endswith('.py') or not file.resolve().is_relative_to(repo) or digest(file) != expected:
            raise ValueError('Eval source file path/SHA differs')
    for module in (__name__, 'search_r1.baselines.corrected_evaluation', 'search_r1.baselines.evaluation',
                   'search_r1.qa_protocol', 'search_r1.baselines.sharedholdout', 'search_r1.llm_agent.generation'):
        loaded = importlib.import_module(module)
        relative = module.replace('.', '/') + '.py'
        if Path(loaded.__file__).resolve() != repo / relative or digest(loaded.__file__) != value['source_files'][relative]:
            raise ValueError('Actual imported eval source differs')
    value['reviewed_stage_path'] = str(Path(settings.reviewed_stage_manifest).resolve())
    value['reviewed_stage_sha256'] = sha
    if value['input_manifest_sha256'] != settings.input_manifest_sha256:
        raise ValueError('Stage input manifest differs')
    return value


def configure(config, split_manifest):
    if not enabled(config):
        return None
    validate_eval_config(config)
    manifest = load_inputs(config.baselines.final200_evaluation.input_manifest_path,
                           config.baselines.final200_evaluation.input_manifest_sha256)
    if split_manifest['content_identity']['sha256'] != COMMON_SHA:
        raise ValueError('Actual shared split identity differs')
    from .holdout import paths
    if [str(Path(p).resolve()) for p in paths(config.data.final_val_files)] != [r['path'] for r in manifest['inputs']]:
        raise ValueError('Actual final files differ')
    stage = load_stage(config.baselines.final200_evaluation)
    legacy.verify_budget(config, stage)
    from omegaconf import open_dict
    with open_dict(config):
        config.data.val_files = config.data.final_val_files
        config.data.return_raw_chat = True
    return stage


def arrays_of(batch):
    return {name: value.detach().cpu().clone().numpy().copy() for name, value in batch.items()}


class TurnObserver:
    """Read actual loop tensors; never provide conditioning to model or scorer."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.completed = None
        self.current = None

    def event(self, name, **values):
        if name == 'begin':
            if self.current is not None or self.completed is not None:
                raise ValueError('Previous raw batch was not consumed')
            gen = values['gen_batch']
            if gen.non_tensor_batch or set(gen.batch.keys()) != {'input_ids', 'attention_mask', 'position_ids'}:
                raise ValueError('Gold/identity/non-input data entered actual generation input')
            allowed = {'eos_token_id', 'pad_token_id', 'recompute_log_prob', 'do_sample', 'validate'}
            if (set(gen.meta_info) != allowed or gen.meta_info['do_sample'] is not False or
                    gen.meta_info['validate'] is not True or gen.meta_info['recompute_log_prob'] is not False):
                raise ValueError('Unexpected generation metadata or sampling budget')
            self.current = dict(initial_input=arrays_of(gen.batch),
                                initial_start=values['initial_input_ids'].detach().cpu().clone().numpy(),
                                generation_meta=canonical(gen.meta_info), turns=[])
        elif name == 'before_generation':
            active = values['active_mask'].detach().cpu().clone().numpy()
            indices = np.flatnonzero(active).tolist()
            if len(self.current['turns']) >= 4 or values['step'] != len(self.current['turns']):
                raise ValueError('Wrong raw turn order/budget')
            self.current['turns'].append(dict(step=values['step'], active_indices=indices,
                active_mask=active, generation_context=arrays_of(values['active_batch'].batch), retrieval=[]))
        elif name == 'generated':
            turn = self.current['turns'][-1]
            turn['raw_generation'] = arrays_of(values['output'].batch)
            turn['raw_generation_meta'] = canonical(values['output'].meta_info)
            if len(turn['raw_generation']['responses']) != len(turn['active_indices']):
                raise ValueError('Padding or inactive rows were mistaken for actual questions')
        elif name == 'postprocessed':
            turn = self.current['turns'][-1]
            turn['action_token_ids'] = values['responses'].detach().cpu().clone().numpy()
            turn['action_text'] = canonical(values['texts'])
        elif name == 'retrieval':
            self.current['turns'][-1]['retrieval'].append(canonical(values))
        elif name == 'environment':
            turn = self.current['turns'][-1]
            turn.update(observation_text=canonical(values['observations']), dones=canonical(values['dones']),
                        valid_actions=canonical(values['valid_actions']), searches=canonical(values['searches']))
        elif name == 'observations':
            turn = self.current['turns'][-1]
            turn['observation_ids_untruncated'] = values['untruncated'].detach().cpu().clone().numpy()
            turn['observation_ids_retained'] = values['retained'].detach().cpu().clone().numpy()
        elif name == 'updated':
            turn = self.current['turns'][-1]
            turn['rolling_after'] = arrays_of(values['rolling'].batch)
            turn['right_after'] = arrays_of(values['right'])
        elif name == 'end':
            self.current['final'] = arrays_of(values['output'].batch)
            self.current['final_meta'] = canonical(values['output'].meta_info)
            self.completed, self.current = self.current, None
        else:
            raise ValueError('Unknown observational event')

    def consume(self, arrays):
        result = self.completed
        if result is None or self.current is not None:
            raise ValueError('No complete raw generation batch')
        for name, value in arrays.items():
            if name not in result['final'] or not np.array_equal(value, result['final'][name]):
                raise ValueError('Raw final tensors differ from actual scoring tensors')
        self.completed = None
        return result


def validate_raw_batch(records, rows, tools, tokenizer):
    """CPU content replay of actual saved turns, not another model generation."""
    if len(records) != len(rows) or tools['count'] != len(rows) or tools['start'] != records[0]['ordinal']:
        raise ValueError('Raw batch cardinality/order differs')
    pad = tokenizer.pad_token_id
    shared = {integer(t['step'], 'step'): t for t in tools['turns']}
    if list(shared) != list(range(len(shared))) or not 1 <= len(shared) <= 4:
        raise ValueError('Wrong shared tool-turn sequence')
    if len(shared) != len(tools['turns']):
        raise ValueError('Duplicate tool-turn step')
    def ids(value):
        if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
            raise ValueError('Invalid raw token/mask row')
        return np.asarray(value, dtype=np.int64)
    def unpad(value):
        a = ids(value)
        return a[a != pad].tolist()
    def encode(text):
        return tokenizer(text, add_special_tokens=False)['input_ids']
    def aligned(batch):
        if set(batch) != {'input_ids', 'attention_mask', 'position_ids'}:
            raise ValueError('Raw context keys differ')
        token, attention, positions = [ids(batch[k]) for k in ('input_ids', 'attention_mask', 'position_ids')]
        if len(token) != len(attention) or len(token) != len(positions) or not np.isin(attention, [0, 1]).all():
            raise ValueError('Raw context/mask lengths differ')
        if not np.array_equal(attention, token != pad) or not np.array_equal(positions, (attention.cumsum() - 1) * attention):
            raise ValueError('Raw attention/position provenance differs')
        return token[token != pad].tolist()
    for i, (record, row) in enumerate(zip(records, rows)):
        if (row['kind'] != 'ACTUAL_FINAL200_QUESTION_FULL_TOOL_TRACE' or row['model_role'] not in ROLES or
                encoded(row['input_record']) != encoded(record) or row['gold_location'] != 'DRIVER_SCORING_ONLY'):
            raise ValueError('Raw question/model identity differs')
        initial = aligned(row['initial_input'])
        templated = tokenizer.apply_chat_template(record['identity']['prompt'], add_generation_prompt=True, tokenize=False)
        if initial != [token for token in encode(templated) if token != pad] or len(row['initial_input']['input_ids']) > 4096:
            raise ValueError('Initial input differs from actual official template/common prompt')
        if row['initial_start'] != row['initial_input']['input_ids'][-2048:]:
            raise ValueError('Actual initial start slicing differs')
        if (row['generation_meta'].get('do_sample') is not False or row['generation_meta'].get('validate') is not True or
                set(row['generation_meta']) != {'eos_token_id', 'pad_token_id', 'recompute_log_prob', 'do_sample', 'validate'}):
            raise ValueError('Raw generation metadata differs')
        rolling = initial
        right, model_mask = [], []
        terminated = False
        if [t['step'] for t in row['turns']] != list(range(len(row['turns']))) or not 1 <= len(row['turns']) <= 4:
            raise ValueError('Raw question turn ordering differs')
        for turn in row['turns']:
            step = integer(turn['step'], 'turn step')
            if terminated or turn['retrieval_step_ref'] != step or turn['active_indices'] != shared[step]['active_indices'] or i not in turn['active_indices']:
                raise ValueError('Inactive/padding row or wrong retrieval reference')
            if aligned(turn['generation_context']) != rolling or len(turn['generation_context']['input_ids']) > 4096:
                raise ValueError('Actual history/context differs')
            generated = ids(turn['raw_generation']['responses'])
            if len(generated) > 500:
                raise ValueError('Actual action generation exceeded500')
            decoded = tokenizer.decode(generated.tolist(), skip_special_tokens=True)
            action = decoded.split('</search>')[0] + '</search>' if '</search>' in decoded else (
                decoded.split('</answer>')[0] + '</answer>' if '</answer>' in decoded else decoded)
            if turn['action_text'] != action or unpad(turn['action_token_ids']) != encode(action):
                raise ValueError('Actual action postprocessing differs')
            action_tokens = unpad(turn['action_token_ids'])
            observation_tokens = encode(turn['observation_text'])
            actual_observation = ids(turn['observation_ids_untruncated']).tolist()
            if (actual_observation[:len(observation_tokens)] != observation_tokens or
                    any(token != pad for token in actual_observation[len(observation_tokens):])):
                raise ValueError('Untruncated observation differs from actual text')
            if turn['observation_ids_retained'] != turn['observation_ids_untruncated'][:500]:
                raise ValueError('Observation retention differs')
            observation = unpad(turn['observation_ids_retained'])
            match = re.search(r'<(search|answer)>(.*?)</\1>', action, re.DOTALL)
            kind = match[1] if match else None
            for key in ('done', 'valid_action', 'search'):
                if type(turn[key]) is not int or turn[key] not in (0, 1):
                    raise ValueError('Invalid actual environment flag')
            if (turn['done'] != int(kind == 'answer') or turn['valid_action'] != int(kind is not None) or
                    turn['search'] != int(kind == 'search')):
                raise ValueError('Actual action/env flags differ')
            calls = shared[step]['retrieval']
            if len(calls) != 1:
                raise ValueError('Actual native retrieval call count differs')
            call = calls[0]
            start_ns = integer(call['started_monotonic_ns'], 'retrieval clock', 1)
            end_ns = integer(call['finished_monotonic_ns'], 'retrieval clock', 1)
            if end_ns < start_ns or call['request']['topk'] != 3 or call['request']['return_scores'] is not True:
                raise ValueError('Invalid retrieval clock/budget')
            from urllib.parse import urlsplit
            url = urlsplit(call['url'])
            if url.hostname != '127.0.0.1' or url.path != '/retrieve' or url.scheme != 'http':
                raise ValueError('Cross-node raw retrieval route')
            queries = []
            search_indices = []
            for index in shared[step]['active_indices']:
                corresponding = rows[index]['turns'][step]
                search = re.search(r'<(search|answer)>(.*?)</\1>', corresponding['action_text'], re.DOTALL)
                if search and search[1] == 'search':
                    queries.append(search[2].strip())
                    search_indices.append(index)
            if call['request']['queries'] != queries or len(call['response']['result']) != len(queries):
                raise ValueError('Actual search queries/response row mapping differs')
            if kind == 'search':
                documents = call['response']['result'][search_indices.index(i)]
                if len(documents) != 3:
                    raise ValueError('Actual top3 documents missing')
                text = ''
                for number, document in enumerate(documents):
                    content = document['document']['contents']
                    title, body = content.split('\n')[0], '\n'.join(content.split('\n')[1:])
                    text += f'Doc {number + 1}(Title: {title}) {body}\n'
                if turn['observation_text'] != '\n\n<information>' + text.strip() + '</information>\n\n':
                    raise ValueError('Actual tool observation differs from returned documents')
            elif kind == 'answer' and turn['observation_text'] != '':
                raise ValueError('Answer must not obtain observation')
            rolling = (rolling + action_tokens + observation)[-4096:]
            if aligned(turn['rolling_after']) != rolling:
                raise ValueError('Actual rolling state replay differs')
            right = (right + action_tokens + observation)[:4096]
            model_mask = (model_mask + [1] * len(action_tokens) + [0] * len(observation))[:4096]
            if (unpad(turn['right_after']['responses']) != right or
                    unpad(turn['right_after']['responses_with_info_mask']) != [t for t, m in zip(right, model_mask) if m]):
                raise ValueError('Actual right-side/model provenance replay differs')
            terminated = bool(turn['done'])
        final = row['final']
        if unpad(final['responses']) != right or final['prompts'] != row['initial_start']:
            raise ValueError('Final retained scoring tensors differ')
        response_ids = ids(final['responses'])
        response_info = ids(final['info_mask'])[-len(response_ids):]
        response_attention = ids(final['attention_mask'])[-len(response_ids):]
        if not np.isin(response_info, [0, 1]).all() or not np.array_equal(response_attention, response_ids != pad):
            raise ValueError('Final scoring masks differ')
        if response_info[response_attention.astype(bool)].tolist() != model_mask:
            raise ValueError('Final model/observation mask differs')
        if final['input_ids'] != final['prompts'] + final['responses'] or row['final_meta']['turns_stats'] != len(row['turns']):
            raise ValueError('Final tensor/meta composition differs')
        if row['final_meta']['active_mask'] != (not terminated):
            raise ValueError('Final actual finish state differs')
    return dict(scope='CPU_RAW_CONTENT_REPLAY_NOT_GPU_OR_EFFECT_ACCEPTANCE', questions=len(rows),
                turns=sum(len(row['turns']) for row in rows), original_templates_replayed=True,
                raw_queries_and_retained_scoring_view_replayed=True)


def optimizer_clock(worker):
    optimizer = worker.actor_optimizer
    if optimizer is not worker.actor.actor_optimizer:
        raise ValueError('Actual actor optimizer aliases differ')
    steps = [int(value['step'].item()) for value in optimizer.state.values() if 'step' in value]
    scheduler = worker.actor_lr_scheduler.state_dict()
    if 'step' in scheduler:
        state = getattr(worker, '_final200_zero_step', None)
        if state is None or scheduler['step'] is not state['scheduler_guard']:
            raise ValueError('Unexpected scheduler instance method in actual state')
        # LambdaLR.state_dict copies __dict__; this is the guard introduced here,
        # not a clock or training state. Verify its exact object before excluding it.
        del scheduler['step']
    return dict(state_entries=len(optimizer.state), steps=steps, scheduler=canonical(scheduler))


def zero_optimizer_callback(worker, probe):
    """Runs in each independent eval worker, not in original training worker."""
    import torch
    from verl import DataProto
    from .corrected_runtime_capture import process_identity
    phase = probe.meta_info['zero_step_phase']
    if phase == 'install':
        if hasattr(worker, '_final200_zero_step'):
            raise ValueError('Optimizer guard already installed')
        state = dict(before=optimizer_clock(worker), attempted_optimizer_steps=0,
                     attempted_scheduler_steps=0, optimizer_id=id(worker.actor_optimizer),
                     scheduler_id=id(worker.actor_lr_scheduler))
        def reject_optimizer(*args, **kwargs):
            state['attempted_optimizer_steps'] += 1
            raise RuntimeError('Eval-only optimizer.step forbidden')
        def reject_scheduler(*args, **kwargs):
            state['attempted_scheduler_steps'] += 1
            raise RuntimeError('Eval-only scheduler.step forbidden')
        worker.actor_optimizer.step = reject_optimizer
        worker.actor_lr_scheduler.step = reject_scheduler
        state['optimizer_guard'] = reject_optimizer
        state['scheduler_guard'] = reject_scheduler
        worker._final200_zero_step = state
    elif phase != 'finish':
        raise ValueError('Unknown zero-step phase')
    state = worker._final200_zero_step
    after = optimizer_clock(worker)
    if (worker.actor_optimizer.step is not state['optimizer_guard'] or
            worker.actor_lr_scheduler.step is not state['scheduler_guard'] or
            id(worker.actor_optimizer) != state['optimizer_id'] or
            id(worker.actor_lr_scheduler) != state['scheduler_id'] or
            state['attempted_optimizer_steps'] or state['attempted_scheduler_steps'] or
            encoded(state['before']) != encoded(after)):
        raise ValueError('Actual eval worker optimizer/scheduler changed or guard failed')
    record = dict(scope='INDEPENDENT_EVAL_WORKER_ZERO_STEPS_NOT_TRAINING_ADAM_RESTORE',
                  phase=phase, rank=worker.rank, world_size=worker.world_size,
                  process=process_identity(os.getpid()), before=state['before'], after=after,
                  attempted_optimizer_steps=0, attempted_scheduler_steps=0,
                  source_sha256=digest(__file__))
    return DataProto.from_dict(tensors={'audit_probe': torch.zeros((1, 1), dtype=torch.int64)},
        non_tensors={'zero_step_audit': np.array([record], dtype=object)})


def worker_zero_audit(trainer, worker_audit, phase):
    import torch
    from verl import DataProto
    probe = DataProto.from_dict(tensors={'audit_probe': torch.zeros((8, 1), dtype=torch.int64)},
                               meta_info={'zero_step_phase': phase})
    result = trainer.actor_rollout_wg.execute_with_func_generator(zero_optimizer_callback, probe)
    rows = canonical(result.non_tensor_batch['zero_step_audit'])
    expected = {row['rank']: row for row in worker_audit['rows']}
    if len(rows) != 8 or {integer(r['rank'], 'rank') for r in rows} != set(range(8)):
        raise ValueError('Actual zero-step worker cardinality differs')
    for row in rows:
        actual = expected[row['rank']]['process']
        if (row['world_size'] != 8 or row['phase'] != phase or row['source_sha256'] != digest(__file__) or
                encoded(row['process']) != encoded(actual)):
            # /proc raw RSS/cmdline can change; stable ownership fields must agree.
            keys = ('pid', 'process_start_ticks', 'uid', 'uids', 'cgroup')
            if (row['world_size'] != 8 or row['phase'] != phase or row['source_sha256'] != digest(__file__) or
                    any(row['process'][k] != actual[k] for k in keys)):
                raise ValueError('Actual zero-step worker identity/source differs')
    return rows


class FullTrajectoryWriter(legacy.ScoringViewWriter):
    def __init__(self, *args, trainer, worker_audit, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_observer = TurnObserver(self.tokenizer)
        self.trainer, self.worker_audit = trainer, worker_audit
        self.zero_start = worker_zero_audit(trainer, worker_audit, 'install')
        self.raw_files = []

    def append(self, data, online, reward_tensor):
        arrays = {name: value.detach().cpu().clone().numpy() for name, value in data.batch.items()
                  if name in ('responses', 'prompts', 'attention_mask', 'info_mask', 'position_ids')}
        raw = self.generation_observer.consume(arrays)
        start = self.cursor
        super().append(data, online, reward_tensor)
        records = self.manifest['records'][start:self.cursor]
        output = []
        for i, identity in enumerate(records):
            turns = []
            for turn in raw['turns']:
                if i not in turn['active_indices']:
                    continue
                active_i = turn['active_indices'].index(i)
                turns.append(dict(step=turn['step'], active_indices=turn['active_indices'],
                    generation_context={k: v[active_i] for k, v in turn['generation_context'].items()},
                    raw_generation={k: v[active_i] for k, v in turn['raw_generation'].items()},
                    raw_generation_meta=turn['raw_generation_meta'], action_token_ids=turn['action_token_ids'][i],
                    action_text=turn['action_text'][i], observation_text=turn['observation_text'][i],
                    observation_ids_untruncated=turn['observation_ids_untruncated'][i],
                    observation_ids_retained=turn['observation_ids_retained'][i],
                    done=turn['dones'][i], valid_action=turn['valid_actions'][i], search=turn['searches'][i],
                    retrieval_step_ref=turn['step'], rolling_after={k: v[i] for k, v in turn['rolling_after'].items()},
                    right_after={k: v[i] for k, v in turn['right_after'].items()}))
            if not turns or len(turns) > 4:
                raise ValueError('Missing/excess raw question turns')
            output.append(dict(kind='ACTUAL_FINAL200_QUESTION_FULL_TOOL_TRACE', input_record=identity,
                model_role=self.provenance['model_role'], gold_location='DRIVER_SCORING_ONLY',
                initial_input={k: v[i] for k, v in raw['initial_input'].items()}, initial_start=raw['initial_start'][i],
                generation_meta=raw['generation_meta'], turns=turns,
                final={k: v[i] for k, v in raw['final'].items()},
                final_meta={k: (v[i] if isinstance(v, list) and len(v) == len(records) else v)
                            for k, v in raw['final_meta'].items()}))
        filename = f'batch-{len(self.raw_files):06d}.full-turns.jsonl.gz'
        tools_filename = f'batch-{len(self.raw_files):06d}.tool-batches.json'
        shared = dict(start=start, count=len(records), turns=[dict(step=t['step'], active_indices=t['active_indices'],
            retrieval=t['retrieval']) for t in raw['turns']])
        write_once(self.directory / tools_filename, encoded(shared))
        uncompressed = b''.join(encoded(row) + b'\n' for row in output)
        write_once(self.directory / filename, gzip.compress(uncompressed, compresslevel=1, mtime=0))
        self.raw_files.append(dict(path=filename, sha256=digest(self.directory / filename),
                                   raw_sha256=hashlib.sha256(uncompressed).hexdigest(),
                                   bytes=(self.directory / filename).stat().st_size, raw_bytes=len(uncompressed),
                                   tool_batches=tools_filename, tool_batches_sha256=digest(self.directory / tools_filename),
                                   start=start, count=len(records)))

    def finish(self, guard, online_metrics):
        if self.generation_observer.current is not None or self.generation_observer.completed is not None:
            raise ValueError('Unconsumed raw generation evidence')
        zero_end = worker_zero_audit(self.trainer, self.worker_audit, 'finish')
        legacy.verify_model(self.provenance)
        if self.cursor != 51713 or encoded(legacy.aggregate_with_rewards(self.scores, self.rewards)) != encoded(online_metrics):
            raise ValueError('Incomplete/different full evaluation aggregate')
        value = dict(version=1, scope='FULL_FINAL200_TOOL_TURNS_AND_NATIVE_SCORING_VIEW', count=self.cursor,
                     input_manifest_sha256=self.provenance['input_manifest_sha256'], source_counts=COUNTS,
                     online_metrics=canonical(online_metrics), files=self.files, full_turn_files=self.raw_files,
                     provenance=self.provenance, zero_update_guard=guard.record(),
                     actual_eval_worker_zero_optimizer=dict(start=self.zero_start, finish=zero_end),
                     lifecycle_scope='EVALUATION_OUTPUT_COMPLETE_NOT_SCHEDULER_TERMINAL_OR_RESOURCE_RELEASE')
        write_once(self.directory / 'manifest.json', encoded(value))
        return digest(self.directory / 'manifest.json')


def start_capture(trainer):
    validate_eval_config(trainer.config)
    settings = trainer.config.baselines.final200_evaluation
    manifest = load_inputs(settings.input_manifest_path, settings.input_manifest_sha256)
    if len(trainer.val_dataset) != 51713 or trainer.val_dataloader.drop_last or trainer.val_dataloader.batch_size != 64:
        raise ValueError('Filtering/subsampling/drop changed full final evaluation')
    from torch.utils.data import SequentialSampler
    if not isinstance(trainer.val_dataloader.sampler, SequentialSampler):
        raise ValueError('Final evaluation question order changed')
    provenance = load_stage(settings)
    legacy.verify_budget(trainer.config, provenance)
    if str(Path(trainer.config.actor_rollout_ref.model.path).resolve()) != provenance['model_directory']:
        raise ValueError('Actual model role path differs')
    legacy.verify_model(provenance)
    from .corrected_runtime_capture import capture_workers, GenerationCallGuard
    workers = capture_workers(trainer, provenance)
    guard = legacy.ZeroUpdateGuard(trainer)
    guard.runtime_guard = GenerationCallGuard(trainer.actor_rollout_wg, workers)
    writer = FullTrajectoryWriter(settings.output_dir, manifest, trainer.tokenizer, provenance,
                                  trainer=trainer, worker_audit=workers)
    return writer, guard


def reject_training(config):
    if enabled(config):
        raise RuntimeError('Final200 eval-only profile cannot enter a training loop')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(prepare_inputs(args.common_manifest, args.output))
