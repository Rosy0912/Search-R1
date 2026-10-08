"""Immutable paired diagnostics, never OPSD targets or evaluation inputs.

The production entry requires an already active fixed CUDA teacher. Small CPU
models may exercise the pure generation/state primitives in unit tests only.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import stat
import time

import numpy as np
import torch

from .full_state import rng_state, restore_rng, state_digest
from .holdout import plain
from .sharedholdout import E1_INSTRUCTION

COMMON_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
META_KEY = 'opsd_trajectory_archive'
TENSOR_FIELDS = ('prompts', 'responses', 'input_ids', 'attention_mask', 'info_mask', 'position_ids', 'step_ids')
DIAGNOSTIC_INSTRUCTION = ('For this independent teacher diagnostic only, return one concise final answer '
    'inside <answer> and </answer>. Do not output reasoning or search actions. '
    'This generated answer is never a training target or student input.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def physical(path, directory=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Physical absolute archive path required')
    require(path.resolve(strict=True) == path, 'Archive path contains symlink')
    value = path.stat()
    require(value.st_uid == os.getuid() and (directory or value.st_nlink == 1) and
            (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)),
            'Archive member owner/type differs')
    return path


def read_member(path, expected=None):
    path = physical(path)
    before = path.stat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = physical(path).stat()
    attrs = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    signatures = [tuple(getattr(v, field) for field in attrs) for v in (before, opened, after, current)]
    require(all(v == signatures[0] for v in signatures) and len(raw) == before.st_size,
            'Archive member changed while reading')
    pin = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        require(pin == expected, 'Archive member SHA differs')
    return raw, dict(path=str(path), sha256=pin, bytes=len(raw))


def write_member(path, raw):
    """Exclusive same-parent publication; failed partial files are preserved."""
    path = Path(path)
    parent = physical(path.parent, True)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parent_identity = tuple(getattr(os.fstat(parent_fd), field) for field in ('st_dev', 'st_ino', 'st_uid'))
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o444, dir_fd=parent_fd)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
            saved = os.fstat(handle.fileno())
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    actual, record = read_member(path, hashlib.sha256(raw).hexdigest())
    current = physical(path).stat()
    attrs = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(actual == raw and tuple(getattr(saved, field) for field in attrs) ==
            tuple(getattr(current, field) for field in attrs) and
            parent_identity == tuple(getattr(physical(parent, True).stat(), field) for field in ('st_dev', 'st_ino', 'st_uid')),
            'Archive published inode/parent/readback differs')
    return record


def parse_json(raw):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, 'Duplicate archive JSON key')
            output[key] = value
        return output
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: require(False, 'Nonfinite archive JSON'))
    return value


def read_json(path, pin=None):
    raw, record = read_member(path, pin)
    return parse_json(raw), record


def identity_fields(row):
    return {key: plain(row[key]) for key in ('question_uid', 'question_position', 'response_index')}


def validate_questions(questions, expected=512):
    require(type(expected) is int and type(questions) is list and len(questions) == expected,
            'Archive unique-question count differs')
    seen = set()
    for i, q in enumerate(questions):
        require(q['question_position'] == i and type(q['question_position']) is int,
                'Original question positions changed')
        uid = q['question_uid']
        require(type(uid) is str and re.fullmatch(r'[^:]+:train:\d+', uid)
                and uid.split(':')[0] == q['data_source'] and uid not in seen,
                'Duplicate/foreign training question identity')
        require(type(q['question']) is str and q['question'].strip()
                and type(q['accepted_answers']) is list and q['accepted_answers']
                and all(type(answer) is str for answer in q['accepted_answers']),
                'Question or privileged training-answer provenance missing')
        seen.add(uid)
    return seen


def attach_groups(batch, questions, responses=5):
    validate_questions(questions, len(questions))
    require(type(responses) is int and responses > 0 and len(batch) == len(questions) * responses,
            'Student response group count differs')
    uids, positions, response_ids = [], [], []
    for q in questions:
        for r in range(responses):
            uids.append(q['question_uid']); positions.append(q['question_position']); response_ids.append(r)
    require(list(batch.non_tensor_batch['index']) == uids, 'Repeated student question order differs')
    batch.non_tensor_batch['opsd_question_uid'] = np.asarray(uids, dtype=object)
    batch.non_tensor_batch['opsd_question_position'] = np.asarray(positions, dtype=object)
    batch.non_tensor_batch['opsd_response_index'] = np.asarray(response_ids, dtype=object)


def tensor_row(batch, i):
    result = {key: batch.batch[key][i].detach().cpu().tolist() for key in TENSOR_FIELDS}
    width = len(result['responses'])
    require(width > 0, 'Student response tensor has no retained width')
    require(result['input_ids'] == result['prompts'] + result['responses'], 'Student input/prompt/response differ')
    require(len(result['attention_mask']) == len(result['input_ids'])
            and len(result['info_mask']) == len(result['input_ids']), 'Student masks are misaligned')
    response_attention = result['attention_mask'][-width:]
    loss = result['info_mask'][-width:]
    require(all(type(v) is int and v in (0, 1) for v in response_attention + loss)
            and all(l <= a for l, a in zip(loss, response_attention)), 'Student action/loss mask differs')
    result['loss_mask'] = loss
    if 'loss_mask' in batch.batch:
        require(batch.batch['loss_mask'][i].detach().cpu().tolist() == loss, 'Actual training loss mask differs')
    return result


def student_record(batch, i, tokenizer):
    tensors = tensor_row(batch, i)
    identity = dict(question_uid=str(batch.non_tensor_batch['opsd_question_uid'][i]),
                    question_position=int(batch.non_tensor_batch['opsd_question_position'][i]),
                    response_index=int(batch.non_tensor_batch['opsd_response_index'][i]))
    return dict(**identity, trajectory_uid=str(batch.non_tensor_batch['trajectory_uid'][i]),
                tensors=tensors, tensor_sha256=digest(tensors),
                prompt_text=tokenizer.decode(tensors['prompts'], skip_special_tokens=False),
                response_text=tokenizer.decode(tensors['responses'], skip_special_tokens=False),
                tool_turns=plain(batch.non_tensor_batch['action_records'][i]),
                training_targets_are_teacher_q_on_student_history=True)


def row_key(identity):
    return identity['question_uid'], identity['response_index']


def validate_student_turns(row, question, tokenizer=None):
    """Reconstruct the retained training sequence from actual raw tool turns."""
    t = row['tensors']
    rolling = question['original_student_input_ids']
    require(type(rolling) is list and rolling and all(type(v) is int and v >= 0 for v in rolling),
            'Original student input token provenance differs')
    response, mask, steps = [], [], []
    turns = row['tool_turns']
    require(type(turns) is list and 1 <= len(turns) <= 4, 'Student interaction budget differs')
    pad = tokenizer.pad_token_id if tokenizer is not None else None
    for index, turn in enumerate(turns):
        require(type(turn['turn_index']) is int and turn['turn_index'] == index
                and turn['generation_context_token_ids'] == rolling,
                'Actual generation history differs from original prompt/tool turns')
        raw, raw_mask = turn['raw_response_token_ids'], turn['raw_response_attention_mask']
        action = turn['postprocessed_action_token_ids']
        full_obs, obs = turn['observation_token_ids_before_truncation'], turn['retained_observation_token_ids']
        require(all(type(values) is list and all(type(v) is int and v >= 0 for v in values)
                    for values in (raw, action, full_obs, obs))
                and len(raw) == len(raw_mask) and 0 < len(raw) <= 500
                and all(type(v) is int and v in (0, 1) for v in raw_mask)
                and len(obs) <= 500 and turn['action_tokens'] == len(action)
                and turn['observation_tokens'] == len(obs), 'Raw tool-turn tokens/budget differ')
        if tokenizer is not None:
            require(turn['raw_response_text'] == tokenizer.decode(raw, skip_special_tokens=False),
                    'Raw generation text differs from original token IDs')
            processed = tokenizer.decode(raw, skip_special_tokens=True)
            processed = (processed.split('</search>')[0] + '</search>' if '</search>' in processed else
                         processed.split('</answer>')[0] + '</answer>' if '</answer>' in processed else processed)
            require(turn['text'] == processed and action == [v for v in tokenizer.encode(processed, add_special_tokens=False) if v != pad]
                    and full_obs == tokenizer.encode(turn['observation_text_before_truncation'], add_special_tokens=False)
                    and obs == [v for v in full_obs[:500] if v != pad],
                    'Actual postprocessed action/observation tokenization differs')
        response = (response + action + obs)[:4096]
        mask = (mask + [1] * len(action) + [0] * len(obs))[:4096]
        steps = (steps + [index] * len(action) + [-1] * len(obs))[:4096]
        rolling = (rolling + action + obs)[-4096:]
    require(t['responses'][:len(response)] == response and t['loss_mask'][:len(mask)] == mask
            and t['step_ids'][:len(steps)] == steps
            and t['loss_mask'][len(mask):] == [0] * (len(t['responses']) - len(mask))
            and t['step_ids'][len(steps):] == [-1] * (len(t['responses']) - len(steps)),
            'Retained student response/model mask/step IDs differ from actual raw tool turns')
    prompt = question['original_student_input_ids'][-2048:]
    require(t['prompts'][-len(prompt):] == prompt, 'Retained start prompt differs from original input')
    if tokenizer is not None:
        require(len(t['prompts']) <= 2048 and t['prompts'][:-len(prompt)] == [pad] * (len(t['prompts']) - len(prompt))
                and t['responses'][len(response):] == [pad] * (len(t['responses']) - len(response)),
                'Student left/right padding differs')


class TrajectoryArchive:
    def __init__(self, trainer):
        self.trainer = trainer
        frame = trainer.train_dataset.dataframe
        self.rows = {}
        for position, extra in enumerate(frame['extra_info']):
            uid = plain(extra)['index']
            require(uid not in self.rows, 'Common training row IDs are not unique')
            self.rows[uid] = position
        self.context = None
        self.student_rows = None
        self.completed_outer = 0
        self.last_manifest = None

    def prepare_outer(self, batch, step):
        require(type(step) is int and 1 <= step <= 200 and len(batch) == 512,
                'Raw archive requires a fresh 512-question outer')
        require(step == self.completed_outer + 1, 'Archive outer progress is not continuous')
        trainer = self.trainer
        gate = trainer.e1_gate
        root = physical(Path(trainer.config.baselines.opsd_e1.run_root), True)
        parent = root / 'trajectories'
        if not parent.exists():
            parent.mkdir(mode=0o700)
        physical(parent, True)
        directory = parent / f'outer_{step:06d}'
        directory.mkdir(mode=0o700)  # An existing partial attempt must not be overwritten.
        questions = []
        frame = trainer.train_dataset.dataframe
        for i, uid in enumerate(batch.non_tensor_batch['index']):
            require(uid in self.rows, 'Question missing from the common training artifact')
            row_position = self.rows[uid]
            source_row = frame.iloc[row_position]
            prompt = plain(source_row[trainer.config.data.prompt_key])
            prefix = E1_INSTRUCTION + '\nQuestion: '
            require(len(prompt) == 1 and prompt[0]['role'] == 'user'
                    and prompt[0]['content'].startswith(prefix), 'Original common instruction changed')
            answers = plain(source_row['reward_model'])['ground_truth']['target']
            require(plain(batch.non_tensor_batch['reward_model'][i])['ground_truth']['target'] == answers
                    and batch.non_tensor_batch['data_source'][i] == source_row['data_source'],
                    'Runtime question/answer/source differs from the original common row')
            original_ids = batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()].cpu().tolist()
            rendered = trainer.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            require(original_ids == rendered, 'Actual initial student prompt differs from common official template')
            questions.append(dict(question_uid=str(uid), question_position=i,
                dataset_row_position=row_position, data_source=str(source_row['data_source']),
                question=prompt[0]['content'][len(prefix):], original_prompt=prompt,
                original_student_input_ids=original_ids, accepted_answers=answers,
                accepted_answers_provenance='common_training_reward_model.ground_truth.target'))
        validate_questions(questions)
        identity = dict(version=1, profile='multitraj200', outer_step=step,
            job_id=str(gate.job), source_commit=gate.source, config_sha256=gate.config_sha,
            common_manifest_sha256=COMMON_SHA, teacher_revision=REVISION,
            teacher_model_path=str(trainer.config.baselines.distillation.teacher_path),
            driver_pid=os.getpid(), directory=str(directory))
        request = dict(identity=identity, questions=questions,
                       scope='TRAINING_QUESTIONS_AND_PRIVILEGED_DIAGNOSTICS_ONLY')
        record = write_member(directory / 'questions.json', canonical(request))
        self.context = dict(identity=identity, questions=questions, request=record,
                            request_sha256=record['sha256'])
        self.student_rows = None
        return self.context

    def attach_groups(self, batch):
        require(self.context is not None, 'Archive was not prepared before repetition')
        attach_groups(batch, self.context['questions'])

    def archive_student(self, batch):
        require(self.context is not None and len(batch) == 2560, 'Missing complete student outer')
        directory = physical(self.context['identity']['directory'], True)
        records = [student_record(batch, i, self.trainer.tokenizer) for i in range(len(batch))]
        expected = {(q['question_uid'], r) for q in self.context['questions'] for r in range(5)}
        keys = [row_key(record) for record in records]
        require(len(set(keys)) == len(keys) and set(keys) == expected, 'Student response identities missing/duplicated')
        require(all(record['question_position'] * 5 + record['response_index'] == i
                    for i, record in enumerate(records)), 'Student archive is not original question-major order')
        require(all(record['tool_turns'] and all('raw_response_token_ids' in turn
                    and 'observation_text_before_truncation' in turn for turn in record['tool_turns'])
                    for record in records), 'Student archive lacks raw tool-turn evidence')
        raw = b''.join(canonical(record) + b'\n' for record in records)
        member = write_member(directory / 'student.jsonl', raw)
        manifest = dict(version=1, kind='OPSD_PREBALANCE_STUDENT_TRAJECTORY_ARCHIVE',
            identity=self.context['identity'], request=self.context['request'], student=member,
            question_count=512, response_count_per_question=5, student_count=2560,
            recorded_before_balancing=True, teacher_diagnostic_answers_used_as_targets=False)
        written = write_member(directory / 'student-manifest.json', canonical(manifest))
        self.context['student_manifest'] = written
        batch.meta_info[META_KEY] = copy.deepcopy(self.context)
        self.student_rows = {row_key(record): (record['tensor_sha256'], record['trajectory_uid']) for record in records}
        return {'diagnostic/student_archived_count': len(records), 'diagnostic/student_archive_bytes': member['bytes']}

    def verify_training_rows(self, batch):
        require(self.student_rows is not None and len(batch) == len(self.student_rows), 'Student archive gate missing')
        seen = set()
        for i in range(len(batch)):
            record = student_record(batch, i, self.trainer.tokenizer)
            key = row_key(record)
            require(key not in seen and self.student_rows.get(key) == (record['tensor_sha256'], record['trajectory_uid']),
                    'Balanced actual distillation row differs from its archived student trajectory')
            seen.add(key)
        require(seen == set(self.student_rows), 'Balanced training rows were dropped')

    def finish_outer(self, metrics):
        require(metrics.get('actor/optimizer_steps') == 10, 'Paired archive requires ten real optimizer calls per rank')
        context = self.context
        directory = physical(context['identity']['directory'], True)
        members, all_ids, costs = [], [], []
        for rank in range(8):
            manifest, manifest_rec = read_json(directory / f'teacher-rank-{rank}.manifest.json')
            require(manifest['identity'] == context['identity'] and manifest['rank'] == rank
                    and manifest['world_size'] == 8 and manifest['question_count'] == 64
                    and manifest['request_sha256'] == context['request_sha256'], 'Teacher rank identity/request differs')
            raw, teacher_rec = read_member(directory / f'teacher-rank-{rank}.jsonl', manifest['teacher']['sha256'])
            records = [parse_json(line) for line in raw.splitlines()]
            assigned = context['questions'][rank * 64:(rank + 1) * 64]
            require([r['question_uid'] for r in records] == [q['question_uid'] for q in assigned]
                    and [r['question_position'] for r in records] == list(range(rank * 64, (rank + 1) * 64)),
                    'Teacher rank did not generate its exact original 64 questions')
            require(manifest['rng_restored'] is True and manifest['model_state_restored'] is True,
                    'Teacher diagnostics changed RNG/model state')
            all_ids.extend(r['question_uid'] for r in records)
            members.append(dict(rank=rank, manifest=manifest_rec, teacher=teacher_rec))
            costs.append(manifest)
        require(len(all_ids) == len(set(all_ids)) == 512, 'Teacher global diagnostic count differs')
        manifest = dict(version=1, kind='OPSD_PAIRED_STUDENT_AND_PRIVILEGED_SHORT_TEACHER_ARCHIVE',
            identity=context['identity'], request=context['request'], student_manifest=context['student_manifest'],
            teacher_rank_members=members, student_count=2560, teacher_count=512,
            scope='RAW_PAIRED_DIAGNOSTICS_NOT_EFFECT_ACCEPTANCE_OR_TEACHER_SEARCH_TRAJECTORIES',
            teacher_diagnostic_answers_used_as_targets=False,
            actual_optimizer_steps_per_rank=float(metrics['actor/optimizer_steps']))
        record = write_member(directory / 'paired-manifest.json', canonical(manifest))
        print('OPSD_PAIRED_TRAJECTORY_ARCHIVE_JSON=' + canonical(record | dict(
            outer_step=context['identity']['outer_step'], student_count=2560, teacher_count=512)).decode(), flush=True)
        require(context['identity']['outer_step'] == self.completed_outer + 1, 'Archive completion counter differs')
        self.completed_outer = context['identity']['outer_step']
        self.last_manifest = record
        os.chmod(directory, 0o555)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = {'diagnostic/teacher_archived_count': 512, 'diagnostic/paired_archive_complete': 1,
                  'diagnostic/teacher_cumulative_cost_includes_diagnostics': 1}
        # DataProto.concat retains rank0 metadata; authoritative global costs
        # therefore come from all eight immutable rank reports.
        for rank, report in enumerate(costs):
            for key in ('generated_tokens', 'input_tokens', 'host_wall_seconds', 'diagnostic_cpu_seconds',
                        'diagnostic_cuda_stream_interval_seconds', 'teacher_measure_host_interval_seconds'):
                if key in report:
                    result[f'diagnostic/teacher_rank{rank}/{key}'] = report[key]
        for key in ('generated_tokens', 'input_tokens', 'diagnostic_cpu_seconds', 'diagnostic_cuda_stream_interval_seconds'):
            if all(key in report for report in costs):
                result['diagnostic/teacher_' + key + '_sum_8ranks'] = sum(report[key] for report in costs)
        return result

    def checkpoint_inventory(self, step):
        require(type(step) is int and step == self.completed_outer and 1 <= step <= 200,
                'Archive inventory cannot include future/unfinished outers')
        root = physical(Path(self.trainer.config.baselines.opsd_e1.run_root), True)
        inventory = {}
        inodes = set()
        names = archive_member_names()
        for outer in range(1, step + 1):
            directory = physical(root / 'trajectories' / f'outer_{outer:06d}', True)
            require({p.name for p in directory.iterdir()} == names, 'Archive contains extra/missing members')
            for name in sorted(names):
                _, member = read_member(directory / name)
                value = (directory / name).stat()
                inode = value.st_dev, value.st_ino
                require(inode not in inodes, 'Archive checkpoint members alias')
                inodes.add(inode)
                inventory[f'trajectory_outer_{outer:06d}/' + name] = dict(path=member['path'], sha256=member['sha256'])
        return inventory

    def state_dict(self, step):
        require(type(step) is int and step == self.completed_outer and self.last_manifest is not None,
                'Checkpoint is missing completed diagnostic archives')
        read_member(self.last_manifest['path'], self.last_manifest['sha256'])
        return dict(version=1, completed_outer=self.completed_outer, last_manifest=dict(self.last_manifest),
                    student_per_outer=2560, teacher_per_outer=512, scope='ARCHIVE_PROGRESS_ONLY_NO_GENERATED_TARGETS')

    def load_state_dict(self, state, step):
        require(type(step) is int and state['version'] == 1 and state['completed_outer'] == step
                and state['student_per_outer'] == 2560 and state['teacher_per_outer'] == 512
                and state['scope'] == 'ARCHIVE_PROGRESS_ONLY_NO_GENERATED_TARGETS', 'Diagnostic state contract differs')
        manifest, record = read_json(state['last_manifest']['path'], state['last_manifest']['sha256'])
        root = physical(Path(self.trainer.config.baselines.opsd_e1.run_root), True)
        require(Path(record['path']) == root / 'trajectories' / f'outer_{step:06d}' / 'paired-manifest.json'
                and manifest['identity']['outer_step'] == step
                and manifest['student_count'] == 2560 and manifest['teacher_count'] == 512, 'Diagnostic recovery path/count differs')
        self.completed_outer = step
        self.last_manifest = record


@contextmanager
def isolated_rng(seed, cuda_device=None):
    before = rng_state(cuda_device)
    witness = {'before_sha256': state_digest(before), 'rng_restored': False}
    try:
        random.seed(seed); np.random.seed(seed % (2 ** 32))
        torch.random.default_generator.manual_seed(seed)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.cuda.manual_seed(seed)  # Current generator only, not manual_seed_all.
        yield witness
    finally:
        restore_rng(before)
        after = rng_state(cuda_device)
        witness['after_sha256'] = state_digest(after)
        witness['rng_restored'] = witness['before_sha256'] == witness['after_sha256']
        require(witness['rng_restored'], 'Teacher diagnostic altered training RNG')


def cache_attribute(name):
    return 'cache' in name.lower() or name in ('original_inv_freq', 'attention_scaling')


@contextmanager
def isolated_generation_state(model):
    """Restore buffer objects/values, known cache attributes, flags and configs."""
    modules = list(model.modules())
    snapshots = []
    for module in modules:
        buffers = dict(module._buffers)
        values = {name: None if value is None else value.detach().clone() for name, value in buffers.items()}
        attrs = {name: copy.deepcopy(value) for name, value in module.__dict__.items() if cache_attribute(name)}
        snapshots.append((module, module.training, buffers, values, set(module._non_persistent_buffers_set), attrs))
    configurations = [(name, getattr(model, name), copy.deepcopy(getattr(model, name).__dict__))
                      for name in ('config', 'generation_config') if hasattr(model, name)]
    parameter_signature = [(id(p), p._version, p.shape, p.dtype, p.device, p.requires_grad) for p in model.parameters()]
    def observed_state():
        return dict(modules=[dict(training=module.training, buffers=dict(module._buffers),
                nonpersistent=sorted(module._non_persistent_buffers_set),
                cache={name: value for name, value in module.__dict__.items() if cache_attribute(name)})
                for module in modules], configurations={name: getattr(model, name).__dict__ for name, _, _ in configurations})
    before_digest = state_digest(observed_state())
    witness = {'model_state_restored': False, 'before_sha256': before_digest}
    try:
        yield witness
    finally:
        require(list(model.modules()) == modules, 'Teacher generation changed module topology')
        with torch.no_grad():
            for module, training, buffers, values, nonpersistent, attrs in snapshots:
                module._buffers.clear(); module._buffers.update(buffers)
                for name, value in values.items():
                    if value is not None:
                        buffers[name].copy_(value)
                module._non_persistent_buffers_set = nonpersistent
                module.training = training
                for name in list(module.__dict__):
                    if cache_attribute(name) and name not in attrs:
                        del module.__dict__[name]
                for name, value in attrs.items():
                    module.__dict__[name] = value
            for name, original, attrs in configurations:
                setattr(model, name, original)
                original.__dict__.clear(); original.__dict__.update(copy.deepcopy(attrs))
        require(all(module.training == training and set(module._buffers) == set(buffers)
                    and all((value is None and module._buffers[name] is None) or
                            (value is not None and torch.equal(module._buffers[name], value))
                            for name, value in values.items())
                    for module, training, buffers, values, _, _ in snapshots), 'Teacher buffers/flags failed restoration')
        require(parameter_signature == [(id(p), p._version, p.shape, p.dtype, p.device, p.requires_grad) for p in model.parameters()],
                'Frozen teacher parameter identity/version changed during diagnostics')
        witness['after_sha256'] = state_digest(observed_state())
        require(witness['before_sha256'] == witness['after_sha256'], 'Teacher config/cache state differs after restoration')
        witness['model_state_restored'] = True


def teacher_diagnostic_messages(q):
    content = ('Question: ' + q['question'] + '\nTeacher-only accepted training answers: '
               + repr(q['accepted_answers']) + '\n' + DIAGNOSTIC_INSTRUCTION)
    return [dict(role='user', content=content)]


def short_teacher_records(model, tokenizer, questions, device, max_new_tokens=128):
    """Real independent generate calls; CPU execution is a small-test primitive."""
    records = []
    for q in questions:
        messages = teacher_diagnostic_messages(q)
        ordinary = tokenizer.encode(messages[0]['content'], add_special_tokens=False)
        require(not set(ordinary).intersection(tokenizer.all_special_ids), 'Teacher question/answer contains chat control IDs')
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        require(ids and len(ids) + max_new_tokens <= model.config.max_position_embeddings,
                'Teacher diagnostic input exceeds context; no truncation')
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        kwargs = dict(input_ids=input_ids, attention_mask=attention, do_sample=False,
                      max_new_tokens=max_new_tokens, num_beams=1, use_cache=False,
                      return_dict_in_generate=False, output_scores=False,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if hasattr(model, 'generation_config'):
            generation_config = copy.deepcopy(model.generation_config)
            # Neutral sampling defaults avoid misleading warnings during greedy
            # diagnostics; none of these settings reaches the student rollout.
            generation_config.do_sample = False
            generation_config.temperature = 1.
            generation_config.top_p = 1.
            generation_config.top_k = 50
            kwargs['generation_config'] = generation_config
        with torch.no_grad():
            output = model.generate(**kwargs)
        require(output.dtype == torch.long and output.ndim == 2 and output.shape[0] == 1 and output.shape[1] >= len(ids)
                and output[0, :len(ids)].cpu().tolist() == ids, 'Teacher generate output/prompt provenance differs')
        generated = output[0, len(ids):].detach().cpu().tolist()
        require(0 < len(generated) <= max_new_tokens, 'Teacher diagnostic output budget differs')
        records.append(dict(question_uid=q['question_uid'], question_position=q['question_position'],
            data_source=q['data_source'], question=q['question'], accepted_answers=q['accepted_answers'],
            accepted_answers_provenance=q['accepted_answers_provenance'], input_messages=messages,
            input_token_ids=ids, input_attention_mask=[1] * len(ids), output_token_ids=generated,
            output_attention_mask=[1] * len(generated), diagnostic_loss_mask=[0] * len(generated),
            input_text=tokenizer.decode(ids, skip_special_tokens=False),
            output_text=tokenizer.decode(generated, skip_special_tokens=False),
            decoding=dict(do_sample=False, max_new_tokens=max_new_tokens, num_beams=1, use_cache=False,
                          temperature_applies=False, independent_of_student_history=True),
            generated_by='fixed_official_teacher.model.generate', privileged_training_answers=True,
            not_used_as_student_or_distillation_or_evaluation_input=True,
            is_full_teacher_search_trajectory=False, tool_turns=[]))
    return records


def generate_rank_teacher_diagnostics(teacher, data, tokenizer):
    context = data.meta_info.get(META_KEY)
    if context is None:
        return {}
    identity = context['identity']
    require(identity['profile'] == 'multitraj200' and identity['common_manifest_sha256'] == COMMON_SHA,
            'Unexpected paired archive profile/data')
    validate_questions(context['questions'])
    require(torch.distributed.is_initialized() and torch.distributed.get_world_size() == 8,
            'Teacher diagnostic requires the real eight-rank worker group')
    rank = torch.distributed.get_rank()
    require(teacher.execution_device.type == 'cuda' and teacher.device.type == 'cuda'
            and socket.gethostname().split('.')[0] == 'h100-10'
            and os.environ.get('SLURM_JOB_ID') == identity['job_id'], 'Teacher diagnostic must run on owned local GPU')
    teacher._validate_execution(teacher.device)
    require(not teacher.model.training and not any(p.requires_grad for p in teacher.model.parameters()),
            'Teacher diagnostic model is not fixed/eval')
    directory = physical(identity['directory'], True)
    request, _ = read_json(directory / 'questions.json', context['request_sha256'])
    require(request['identity'] == identity and request['questions'] == context['questions'], 'Teacher request changed in dispatch')
    assigned = context['questions'][rank * 64:(rank + 1) * 64]
    teacher._flush_cuda_events()
    measured_before = (teacher.cpu_seconds, teacher.cuda_seconds, teacher.wall_seconds)
    start = time.perf_counter()
    with isolated_rng(42 + identity['outer_step'] * 8 + rank, torch.cuda.current_device()) as rng_witness:
        with isolated_generation_state(teacher.model) as model_witness:
            with teacher.measure():
                records = short_teacher_records(teacher.model, tokenizer, assigned, teacher.device)
    teacher._flush_cuda_events()
    measured_delta = tuple(after - before for after, before in zip(
        (teacher.cpu_seconds, teacher.cuda_seconds, teacher.wall_seconds), measured_before))
    require(all(np.isfinite(value) and value >= 0 for value in measured_delta), 'Diagnostic cost interval differs')
    member = write_member(directory / f'teacher-rank-{rank}.jsonl',
                          b''.join(canonical(row) + b'\n' for row in records))
    elapsed = time.perf_counter() - start
    report = dict(version=1, kind='OPSD_REAL_FIXED_TEACHER_SHORT_ANSWER_DIAGNOSTIC_RANK',
        identity=identity, request_sha256=context['request_sha256'], rank=rank, world_size=8,
        pid=os.getpid(), uid=os.getuid(), hostname=socket.gethostname(),
        execution_device=str(teacher.device), local_cuda_device_index=torch.cuda.current_device(),
        question_count=len(records), teacher=member,
        generated_tokens=sum(len(row['output_token_ids']) for row in records),
        input_tokens=sum(len(row['input_token_ids']) for row in records), host_wall_seconds=elapsed,
        host_wall_scope='RNG_AND_MODEL_STATE_ISOLATION_GENERATION_AND_RAW_JSONL_PUBLICATION',
        diagnostic_cpu_seconds=measured_delta[0], diagnostic_cuda_stream_interval_seconds=measured_delta[1],
        teacher_measure_host_interval_seconds=measured_delta[2],
        teacher_cumulative_cost_includes_diagnostics=True,
        cost_scope='DIAGNOSTIC_INTERVAL_INCLUDED_IN_TEACHER_CUMULATIVE_DO_NOT_ADD_TWICE',
        rng_restored=rng_witness['rng_restored'], rng_witness=rng_witness,
        model_state_restored=model_witness['model_state_restored'], model_state_witness=model_witness,
        diagnostic_answers_used_as_targets=False,
        scope='PRIVILEGED_SHORT_GENERATION_ONLY_NOT_TRAINING_Q_OR_FULL_TEACHER_SEARCH_TRAJECTORY')
    write_member(directory / f'teacher-rank-{rank}.manifest.json', canonical(report))
    return {'diagnostic/teacher_generated_questions_per_rank': len(records),
            'diagnostic/teacher_generated_tokens_per_rank': report['generated_tokens'],
            'diagnostic/teacher_input_tokens_per_rank': report['input_tokens'],
            'diagnostic/teacher_host_wall_seconds_per_rank': elapsed,
            'diagnostic/teacher_rng_restored_per_rank': int(rng_witness['rng_restored']),
            'diagnostic/teacher_model_state_restored_per_rank': int(model_witness['model_state_restored'])}


def validate_archive(directory, expected_outer, expected_identity=None, expected_worker_pids=None, tokenizer=None):
    """Read all raw records, not only manifests; functional provenance gate only.

    The caller separately supplies the actual job/source/config identity and
    externally pins this implementation. No signing or GPU work occurs here.
    """
    directory = physical(directory, True)
    paired, paired_record = read_json(directory / 'paired-manifest.json')
    identity = paired['identity']
    require(type(expected_outer) is int and type(identity['outer_step']) is int and identity['outer_step'] == expected_outer
            and identity['profile'] == 'multitraj200' and identity['common_manifest_sha256'] == COMMON_SHA
            and identity['teacher_revision'] == REVISION and identity['directory'] == str(directory),
            'Paired archive identity differs')
    if expected_identity is not None:
        require(identity == expected_identity, 'Actual archive source/job/config identity differs')
    require(paired['student_count'] == 2560 and paired['teacher_count'] == 512
            and paired['teacher_diagnostic_answers_used_as_targets'] is False,
            'Paired archive cardinality/target isolation differs')
    request, request_rec = read_json(directory / 'questions.json', paired['request']['sha256'])
    require(request['identity'] == identity, 'Raw question identity differs')
    questions = request['questions']; validate_questions(questions)
    if tokenizer is not None:
        for q in questions:
            require(q['original_prompt'] == [dict(role='user', content=E1_INSTRUCTION + '\nQuestion: ' + q['question'])]
                    and q['original_student_input_ids'] == tokenizer.apply_chat_template(q['original_prompt'], tokenize=True, add_generation_prompt=True),
                    'Common student instruction/official initial template differs')
    student_manifest, student_manifest_rec = read_json(directory / 'student-manifest.json', paired['student_manifest']['sha256'])
    require(student_manifest['identity'] == identity and student_manifest['question_count'] == 512
            and student_manifest['response_count_per_question'] == 5 and student_manifest['student_count'] == 2560
            and student_manifest['recorded_before_balancing'] is True
            and student_manifest['teacher_diagnostic_answers_used_as_targets'] is False
            and student_manifest['request'] == request_rec, 'Student manifest request/count differs')
    student_raw, student_rec = read_member(directory / 'student.jsonl', student_manifest['student']['sha256'])
    require(student_rec == student_manifest['student'] and student_raw.endswith(b'\n'), 'Student complete raw member differs')
    student_rows = [parse_json(line) for line in student_raw.splitlines()]
    require(len(student_rows) == 2560, 'Raw student count differs')
    for i, row in enumerate(student_rows):
        q = questions[i // 5]
        require(row['question_uid'] == q['question_uid'] and type(row['question_position']) is int and row['question_position'] == i // 5
                and type(row['response_index']) is int and row['response_index'] == i % 5
                and row['tensor_sha256'] == digest(row['tensors'])
                and row['training_targets_are_teacher_q_on_student_history'] is True, 'Raw student group/tensor identity differs')
        t = row['tensors']; width = len(t['responses'])
        require(set(t) == set(TENSOR_FIELDS) | {'loss_mask'}
                and len(t['position_ids']) == len(t['input_ids'])
                and len(t['step_ids']) == width
                and all(type(v) is int and v >= -1 for v in t['step_ids'])
                and all(type(v) is int and v >= 0 for v in t['position_ids'])
                and all(type(v) is int and v >= 0 for v in t['input_ids'])
                and all(type(v) is int and v in (0, 1) for v in t['attention_mask'] + t['info_mask'])
                and t['info_mask'][:-width] == t['attention_mask'][:-width],
                'Raw student attention/info/position/step alignment differs')
        if tokenizer is not None:
            require(row['prompt_text'] == tokenizer.decode(t['prompts'], skip_special_tokens=False)
                    and row['response_text'] == tokenizer.decode(t['responses'], skip_special_tokens=False),
                    'Student raw prompt/response text differs from token IDs')
        require(width > 0 and t['input_ids'] == t['prompts'] + t['responses']
                and t['loss_mask'] == t['info_mask'][-width:]
                and len(t['attention_mask']) == len(t['input_ids'])
                and len(t['info_mask']) == len(t['input_ids'])
                and all(v in (0, 1) and type(v) is int for v in t['loss_mask'])
                and all(l <= a for l, a in zip(t['loss_mask'], t['attention_mask'][-width:])),
                'Raw student token/mask relation differs')
        require(row['tool_turns'] and all('generation_context_token_ids' in turn
                and 'raw_response_token_ids' in turn and 'raw_response_attention_mask' in turn
                and 'observation_text_before_truncation' in turn
                and 'observation_token_ids_before_truncation' in turn
                and 'retained_observation_token_ids' in turn for turn in row['tool_turns']),
                'Raw student tool evidence missing')
        validate_student_turns(row, q, tokenizer)
    teacher_members = []
    if expected_worker_pids is not None:
        require(type(expected_worker_pids) is dict and all(type(key) is int for key in expected_worker_pids)
                and set(expected_worker_pids) == set(range(8))
                and all(type(pid) is int and pid > 0 for pid in expected_worker_pids.values())
                and len(set(expected_worker_pids.values())) == 8, 'Actual eight teacher worker PID map differs')
    require(len(paired['teacher_rank_members']) == 8, 'Teacher rank manifests missing')
    for rank, linked in enumerate(paired['teacher_rank_members']):
        require(type(linked['rank']) is int and linked['rank'] == rank, 'Teacher rank order differs')
        report, report_rec = read_json(directory / f'teacher-rank-{rank}.manifest.json', linked['manifest']['sha256'])
        raw, raw_rec = read_member(directory / f'teacher-rank-{rank}.jsonl', linked['teacher']['sha256'])
        require(report_rec == linked['manifest'] and raw_rec == linked['teacher']
                and report['teacher'] == raw_rec and raw.endswith(b'\n')
                and report['identity'] == identity and type(report['rank']) is int and report['rank'] == rank
                and type(report['world_size']) is int and report['world_size'] == 8
                and type(report['question_count']) is int and report['question_count'] == 64
                and report['request_sha256'] == request_rec['sha256']
                and report['rng_restored'] is True and report['model_state_restored'] is True
                and report['rng_witness']['before_sha256'] == report['rng_witness']['after_sha256']
                and report['model_state_witness']['before_sha256'] == report['model_state_witness']['after_sha256'],
                'Teacher raw report/identity/RNG differs')
        if expected_worker_pids is not None:
            require(type(report['version']) is int and report['version'] == 1
                    and report.get('kind') == 'OPSD_REAL_FIXED_TEACHER_SHORT_ANSWER_DIAGNOSTIC_RANK'
                    and type(report['pid']) is int and report['pid'] == expected_worker_pids[rank]
                    and type(report['uid']) is int and report['uid'] == os.getuid()
                    and type(report['hostname']) is str and report['hostname'].split('.')[0] == 'h100-10'
                    and type(report['execution_device']) is str and re.fullmatch(r'cuda(?::[0-7])?', report['execution_device'])
                    and type(report['local_cuda_device_index']) is int and 0 <= report['local_cuda_device_index'] < 8
                    and (':' not in report['execution_device'] or int(report['execution_device'].split(':')[1]) == report['local_cuda_device_index']),
                    'Actual teacher rank process/GPU identity differs')
        rows = [parse_json(line) for line in raw.splitlines()]
        require(len(rows) == 64, 'Raw teacher rank count differs')
        for j, row in enumerate(rows):
            q = questions[rank * 64 + j]
            require(row['question_uid'] == q['question_uid'] and type(row['question_position']) is int and row['question_position'] == q['question_position']
                    and row['question'] == q['question'] and row['data_source'] == q['data_source']
                    and row['input_messages'] == teacher_diagnostic_messages(q)
                    and row['accepted_answers'] == q['accepted_answers']
                    and row['accepted_answers_provenance'] == q['accepted_answers_provenance']
                    and row['generated_by'] == 'fixed_official_teacher.model.generate'
                    and row['privileged_training_answers'] is True
                    and row['not_used_as_student_or_distillation_or_evaluation_input'] is True
                    and row['is_full_teacher_search_trajectory'] is False
                    and 0 < len(row['output_token_ids']) <= 128
                    and all(type(v) is int and v >= 0 for v in row['input_token_ids'] + row['output_token_ids'])
                    and row['output_attention_mask'] == [1] * len(row['output_token_ids'])
                    and row['diagnostic_loss_mask'] == [0] * len(row['output_token_ids'])
                    and row['input_attention_mask'] == [1] * len(row['input_token_ids'])
                    and row['decoding'] == dict(do_sample=False, max_new_tokens=128, num_beams=1,
                        use_cache=False, temperature_applies=False, independent_of_student_history=True),
                    'Teacher true generation/condition/isolation evidence differs')
            if tokenizer is not None:
                require(row['input_token_ids'] == tokenizer.apply_chat_template(row['input_messages'], tokenize=True, add_generation_prompt=True)
                        and row['input_text'] == tokenizer.decode(row['input_token_ids'], skip_special_tokens=False)
                        and row['output_text'] == tokenizer.decode(row['output_token_ids'], skip_special_tokens=False),
                        'Teacher diagnostic tokenizer/input/output text differs')
        require(report['generated_tokens'] == sum(len(row['output_token_ids']) for row in rows)
                and report['input_tokens'] == sum(len(row['input_token_ids']) for row in rows),
                'Teacher diagnostic token counters differ')
        teacher_members.append(dict(manifest=report_rec, teacher=raw_rec))
    return dict(version=1, status=('RAW_PAIRED_DIAGNOSTIC_ARCHIVE_AND_ACTUAL_WORKER_BINDINGS_VALIDATED_NOT_EFFECT_PASS'
                if expected_worker_pids is not None else 'RAW_PAIRED_ARCHIVE_CONTENT_ONLY_NO_ACTUAL_WORKER_BINDING'),
                actual_worker_bindings_verified=expected_worker_pids is not None,
                identity=identity, paired_manifest=paired_record, request=request_rec,
                student_manifest=student_manifest_rec, student=student_rec,
                student_count=2560, teacher_count=512, teacher_rank_members=teacher_members)


def archive_member_names():
    return {'questions.json', 'student.jsonl', 'student-manifest.json', 'paired-manifest.json'} | {
        f'teacher-rank-{rank}' + suffix for rank in range(8) for suffix in ('.jsonl', '.manifest.json')}


def validate_checkpoint_archives(manifest, runroot, tokenizer=None):
    """Join the external actual first5 signed inventory and eight native PIDs."""
    root = physical(runroot, True)
    require(manifest['global_step'] == manifest['actual_outer_updates'] == 5
            and manifest['world_size'] == 8 and manifest['run_root'] == str(root),
            'Archive checkpoint verifier requires the actual first5 manifest')
    files = manifest['files']
    required = {f'trajectory_outer_{step:06d}/' + name for step in range(1, 6) for name in archive_member_names()}
    actual = {name for name in files if name.startswith('trajectory_outer_')}
    require(actual == required, 'Signed first5 trajectory inventory contains extra/missing members')
    require(tokenizer is not None, 'Actual first5 archive acceptance requires the verified official tokenizer')
    configuration, _ = read_json(files['resolved_config']['path'], files['resolved_config']['sha256'])
    require(digest(configuration) == manifest['config_sha256']
            and configuration['baselines']['opsd_e1']['profile'] == 'multitraj200'
            and configuration['baselines']['distillation']['teacher_path'] == configuration['actor_rollout_ref']['model']['path'],
            'Actual resolved configuration/official fixed teacher binding differs')
    teacher_path = configuration['baselines']['distillation']['teacher_path']
    pids = {}
    for rank in range(8):
        record = files[f'actor/training_state/rank_{rank}.json']
        report, _ = read_json(record['path'], record['sha256'])
        require(report['rank'] == rank and report['world_size'] == 8 and report['step'] == 5
                and report['status'] == 'EXACT_FULL_STATE_RELOAD_OK' and type(report['pid']) is int and report['pid'] > 0,
                'Native checkpoint rank identity differs')
        pids[rank] = report['pid']
    require(len(set(pids.values())) == 8, 'Native worker PIDs alias')
    archives = []
    inodes = set()
    for step in range(1, 6):
        directory = physical(root / 'trajectories' / f'outer_{step:06d}', True)
        require({p.name for p in directory.iterdir()} == archive_member_names(), 'First5 archive directory inventory differs')
        for name in archive_member_names():
            record = files[f'trajectory_outer_{step:06d}/' + name]
            require(set(record) == {'path', 'sha256'} and Path(record['path']) == directory / name,
                    'Signed trajectory member path escapes the actual run')
            read_member(record['path'], record['sha256'])
            value = Path(record['path']).stat(); inode = value.st_dev, value.st_ino
            require(inode not in inodes, 'Signed archive member inode aliases')
            inodes.add(inode)
        paired, _ = read_json(directory / 'paired-manifest.json', files[f'trajectory_outer_{step:06d}/paired-manifest.json']['sha256'])
        identity = paired['identity']
        require(identity['job_id'] == manifest['job_id'] and identity['source_commit'] == manifest['source_commit']
                and identity['config_sha256'] == manifest['config_sha256'] and identity['driver_pid'] == manifest['driver_pid']
                and identity['teacher_model_path'] == teacher_path,
                'First5 archive is from a different actual driver/job/config/source')
        archives.append(validate_archive(directory, step, identity, pids, tokenizer))
    return dict(status='RAW_FIRST5_PAIRED_ARCHIVES_AND_NATIVE_PID_BINDINGS_VALIDATED_NOT_EFFECT_PASS',
                actual_outers=5, student_count=12800, teacher_count=2560, archives=archives,
                official_tokenizer_replayed=tokenizer is not None)
