"""Immutable pre-update training EM snapshots, independent of BAPO rewards.

Question IDs are original dataset IDs. Source rows are only batch positions;
resampling replaces a position without changing its original sibling ordinal.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from search_r1.qa_protocol import decode_model_spans, extract_final_answer
from verl.utils.reward_score.qa_em import em_check


def _plain(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _tensor_hash(value):
    return _hash(value.detach().cpu().tolist())


def _aggregate(rows, prefix, siblings):
    groups = defaultdict(list)
    for row in rows:
        groups[row['question_id']].append(row['em'])
    if any(len(group) != siblings for group in groups.values()):
        raise ValueError('Training EM does not contain exactly the sibling count per original question')
    return {prefix + '/em': sum(r['em'] for r in rows) / len(rows),
            prefix + '/any_of_' + str(siblings) + '_em': sum(any(g) for g in groups.values()) / len(groups),
            prefix + '/correct_response_count': sum(r['em'] for r in rows),
            prefix + '/correct_question_count': sum(any(g) for g in groups.values()),
            prefix + '/response_count': len(rows), prefix + '/unique_question_count': len(groups),
            prefix + '/valid_answer_count': sum(r['answer'] is not None for r in rows)}


class TrainingEMJournal:
    def __init__(self, qa_batch, tokenizer, *, actual_outer, attempt_id, run_root,
                 policy_snapshot_id, source_commit, bindings=None, expected_questions=512, siblings=5):
        if (type(actual_outer) is not int or not 1 <= actual_outer <= 200
                or type(attempt_id) is not int or attempt_id < 1):
            raise ValueError('Training EM requires actual outer and attempt identities')
        if len(qa_batch) != expected_questions * siblings:
            raise ValueError('Training EM initial question/sibling batch dimensions differ')
        self.tokenizer = tokenizer
        self.identities, self.targets = [], []
        ordinals, question_ordinals = {}, {}
        for index, reward in zip(qa_batch.non_tensor_batch['index'], qa_batch.non_tensor_batch['reward_model']):
            question_id = _plain(index)
            if not isinstance(question_id, (str, int)) or isinstance(question_id, bool):
                raise ValueError('Training EM requires an original stable dataset question ID')
            if question_id not in question_ordinals:
                question_ordinals[question_id] = len(question_ordinals)
            ordinal = ordinals.get(question_id, 0)
            ordinals[question_id] = ordinal + 1
            self.identities.append(dict(question_id=question_id,
                                       question_batch_ordinal=question_ordinals[question_id],
                                       response_ordinal=ordinal))
            targets = reward['ground_truth']['target']
            self.targets.append([targets] if isinstance(targets, str) else list(targets))
        if len(question_ordinals) != expected_questions or set(ordinals.values()) != {siblings}:
            raise ValueError('Training EM requires distinct original questions with exact sibling multiplicity')
        self.header = dict(record_type='header', schema='bapo-training-em-v1',
            snapshot_role='rollout_before_actor_update_not_accepted_outer_evidence',
            actual_outer=actual_outer, attempt_id=attempt_id, driver_pid=os.getpid(),
            run_root=str(Path(run_root).resolve()), policy_snapshot_id=policy_snapshot_id,
            source_commit=source_commit, bindings=bindings or {}, expected_questions=expected_questions,
            siblings=siblings, response_ordinal_base=0,
            original_question_ids_sha256=_hash([i['question_id'] for i in self.identities]),
            original_targets_sha256=_hash(self.targets))
        self.events, self.latest, self.generation = [], {}, {}

    def capture(self, output, sources, phase, generation_ordinal):
        if phase not in ('initial', 'resample', 'retained'):
            raise ValueError('Unknown training EM phase')
        width = output.batch['responses'].shape[1]
        mask = output.batch['info_mask'][:, -width:].bool() & output.batch['attention_mask'][:, -width:].bool()
        trajectories = output.non_tensor_batch.get('trajectory_uid')
        events = []
        for output_row, source in enumerate(sources):
            source = int(source)
            if not 0 <= source < len(self.identities):
                raise ValueError('Training EM source row outside original repeated QA batch')
            spans = decode_model_spans(self.tokenizer, output.batch['responses'][output_row], mask[output_row])
            attended = output.batch['attention_mask'][output_row, -width:].bool()
            answer = extract_final_answer(spans)
            generation = self.generation.get(source, 0) if phase == 'retained' else generation_ordinal
            event = dict(record_type='response', phase=phase, source_row_index=source,
                         generation_ordinal=generation, **self.identities[source],
                         trajectory_uid=str(trajectories[output_row]) if trajectories is not None else None,
                         terminal_model_span=spans[-1] if spans else None,
                         model_spans=spans,
                         model_spans_sha256=_hash(spans),
                         response_token_ids_sha256=_tensor_hash(output.batch['responses'][output_row][attended]),
                         authored_mask_sha256=_tensor_hash(mask[output_row][attended]),
                         answer=answer, targets=self.targets[source],
                         em=int(em_check(answer, self.targets[source])) if answer is not None else 0)
            if phase == 'retained':
                previous = self.latest.get(source)
                keys = ('answer', 'em', 'trajectory_uid', 'terminal_model_span', 'model_spans', 'model_spans_sha256')
                if previous is None or any(event[k] != previous[k] for k in keys):
                    raise ValueError('Retained training EM response differs from latest original/resample provenance')
            else:
                if phase == 'initial' and (source in self.latest or generation_ordinal != 0):
                    raise ValueError('Duplicate initial training EM source')
                if phase == 'resample' and (source not in self.latest or generation_ordinal <= self.generation[source]):
                    raise ValueError('Resample training EM generation order differs')
                self.latest[source] = event
                self.generation[source] = generation_ordinal
            events.append(event)
        self.events.extend(events)
        return events

    def finish(self, output, sources):
        self.capture(output, sources, 'retained', 0)
        initial = [r for r in self.events if r['phase'] == 'initial']
        retained = [r for r in self.events if r['phase'] == 'retained']
        expected = self.header['expected_questions'] * self.header['siblings']
        if (len(initial) != expected or len(retained) != expected
                or {r['source_row_index'] for r in retained} != set(range(expected))):
            raise ValueError('Training EM initial/retained coverage differs')
        metrics = {**_aggregate(initial, 'train_em/initial', self.header['siblings']),
                   **_aggregate(retained, 'train_em/retained', self.header['siblings']),
                   'train_em/resampled_response_count': sum(r['phase'] == 'resample' for r in self.events),
                   'train_em/actual_outer': self.header['actual_outer'],
                   'train_em/attempt_id': self.header['attempt_id']}
        records = [self.header] + self.events + [dict(record_type='summary', metrics=metrics)]
        directory = Path(self.header['run_root']) / 'training-em-audit'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ('outer_%04d-attempt_%04d.jsonl' % (self.header['actual_outer'], self.header['attempt_id']))
        raw = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n' for row in records).encode()
        fd, temporary = tempfile.mkstemp(prefix='.training-em-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, target)  # Atomic publication, refusing overwrite even after interruption.
            os.chmod(target, 0o400)
            directory_fd = os.open(directory, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
        proof = dict(path=str(target), sha256=hashlib.sha256(raw).hexdigest(),
                     actual_outer=self.header['actual_outer'], attempt_id=self.header['attempt_id'],
                     driver_pid=self.header['driver_pid'], policy_snapshot_id=self.header['policy_snapshot_id'],
                     source_commit=self.header['source_commit'], run_root=self.header['run_root'])
        return metrics, proof


def verify_training_em_audit(path, expected_sha256, expected=None):
    """Recompute terminal EM and all aggregates from a bound immutable JSONL."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Training EM audit SHA256 differs')
    records = [json.loads(line) for line in raw.splitlines()]
    if (not records or records[0].get('schema') != 'bapo-training-em-v1'
            or records[-1].get('record_type') != 'summary'):
        raise ValueError('Unknown or incomplete training EM audit')
    header = records[0]
    if any(header.get(k) != v for k, v in (expected or {}).items()):
        raise ValueError('Training EM audit provenance binding differs')
    rows = records[1:-1]
    latest, initial, retained = {}, [], []
    last_phase = -1
    for row in rows:
        phase_order = {'initial': 0, 'resample': 1, 'retained': 2}
        phase = row.get('phase')
        if phase not in phase_order or phase_order[phase] < last_phase:
            raise ValueError('Training EM event phase order differs')
        last_phase = phase_order[phase]
        spans = row['model_spans']
        if (not isinstance(spans, list) or any(not isinstance(s, str) for s in spans)
                or row['model_spans_sha256'] != _hash(spans)
                or row['terminal_model_span'] != (spans[-1] if spans else None)):
            raise ValueError('Training EM authored model span digest/boundary differs')
        answer = extract_final_answer(spans)
        score = int(em_check(answer, row['targets'])) if answer is not None else 0
        if row['record_type'] != 'response' or row['answer'] != answer or row['em'] != score:
            raise ValueError('Training EM answer/score does not reproduce from authored terminal span')
        source, phase = row['source_row_index'], row['phase']
        if phase == 'initial':
            if source in latest or row['generation_ordinal'] != 0:
                raise ValueError('Duplicate or invalid initial training EM provenance')
            latest[source] = row; initial.append(row)
        elif phase == 'resample':
            previous = latest.get(source)
            stable = ('question_id', 'response_ordinal', 'question_batch_ordinal', 'targets')
            if (previous is None or row['generation_ordinal'] <= previous['generation_ordinal']
                    or any(row[k] != previous[k] for k in stable)):
                raise ValueError('Resampled training EM original identity/generation differs')
            latest[source] = row
        elif phase == 'retained':
            previous = latest.get(source)
            if previous is None or any(row[k] != previous[k] for k in row if k != 'phase'):
                raise ValueError('Retained training EM provenance differs')
            retained.append(row)
        else:
            raise ValueError('Unknown training EM event phase')
    count = header['expected_questions'] * header['siblings']
    if (len(initial) != count or len(retained) != count
            or {r['source_row_index'] for r in initial} != set(range(count))
            or {r['source_row_index'] for r in retained} != set(range(count))):
        raise ValueError('Training EM audit question/response coverage differs')
    ordered = sorted(initial, key=lambda r: r['source_row_index'])
    if (_hash([r['question_id'] for r in ordered]) != header['original_question_ids_sha256']
            or _hash([r['targets'] for r in ordered]) != header['original_targets_sha256']):
        raise ValueError('Training EM original dataset identity/targets digest differs')
    groups = defaultdict(list)
    for row in initial:
        groups[row['question_id']].append(row)
    for question_ordinal, question_id in enumerate(dict.fromkeys(r['question_id'] for r in ordered)):
        group = groups[question_id]
        if (sorted(r['response_ordinal'] for r in group) != list(range(header['siblings']))
                or any(r['question_batch_ordinal'] != question_ordinal for r in group)
                or any(r['targets'] != group[0]['targets'] for r in group)):
            raise ValueError('Training EM original sibling/question identity differs')
    metrics = {**_aggregate(initial, 'train_em/initial', header['siblings']),
               **_aggregate(retained, 'train_em/retained', header['siblings']),
               'train_em/resampled_response_count': sum(r['phase'] == 'resample' for r in rows),
               'train_em/actual_outer': header['actual_outer'], 'train_em/attempt_id': header['attempt_id']}
    if metrics != records[-1]['metrics']:
        raise ValueError('Training EM summary does not reproduce from response records')
    return dict(header=header, metrics=metrics, path=str(Path(path).resolve()), sha256=expected_sha256)
