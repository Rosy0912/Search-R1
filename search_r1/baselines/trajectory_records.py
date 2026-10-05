"""Detached, complete rollout evidence. Never consumes RNG or edits model inputs."""
import gzip
import hashlib
import json
import os
from pathlib import Path


def atomic_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Refusing to overwrite trajectory evidence: ' + str(path))
    partial = path.with_name(path.name + '.partial')
    with partial.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode())
        raw.flush()
        os.fsync(raw.fileno())
    partial.rename(path)
    return dict(path=str(path), rows=len(rows), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class RolloutTrace:
    def __init__(self, root, logical_step, question_ids, responses_per_question):
        self.root = Path(root) / f'iter_{logical_step:06d}'
        self.step = logical_step
        self.ids = [str(value) for value in question_ids]
        self.n = responses_per_question
        self.turn = 0
        if len(self.ids) % self.n:
            raise ValueError('Incomplete response groups in trajectory audit')
        for offset in range(0, len(self.ids), self.n):
            if len(set(self.ids[offset:offset + self.n])) != 1:
                raise ValueError('Trajectory audit cannot identify contiguous response groups')

    def row_identity(self, row):
        return dict(logical_step=self.step, trajectory_row=row, question_id=self.ids[row],
                    response_index=row % self.n, response_group=row // self.n)

    def record_turn(self, predictions, observations, active, dones, valid_action, is_search):
        if len(predictions) != len(self.ids) or len(observations) != len(self.ids):
            raise ValueError('Trajectory turn coverage changed')
        self.turn += 1
        rows = [dict(self.row_identity(i), turn=self.turn, active=bool(active[i]),
                     model_action=predictions[i],
                     raw_tool_observation_before_token_limit=observations[i],
                     done=bool(dones[i]), valid_action=bool(valid_action[i]), is_search=bool(is_search[i]))
                for i in range(len(self.ids))]
        return atomic_rows(self.root / f'turn_{self.turn:02d}.jsonl.gz', rows)

    def record_active_turn(self, row_indices, predictions, observations, dones, valid_action, is_search):
        """Expand active original rows without changing the rollout's compact batch."""
        import operator
        indices = [operator.index(row) for row in row_indices]
        vectors = (predictions, observations, dones, valid_action, is_search)
        if any(len(vector) != len(indices) for vector in vectors):
            raise ValueError('Active trajectory vector coverage changed')
        if len(set(indices)) != len(indices) or any(row < 0 or row >= len(self.ids) for row in indices):
            raise ValueError('Active trajectory row identity invalid')
        expanded = [[''] * len(self.ids), [''] * len(self.ids),
                    [True] * len(self.ids), [False] * len(self.ids), [False] * len(self.ids)]
        active = [False] * len(self.ids)
        for offset, row in enumerate(indices):
            active[row] = True
            for destination, source in zip(expanded, vectors):
                destination[row] = source[offset]
        return self.record_turn(expanded[0], expanded[1], active, *expanded[2:])

    def record_generation(self, output, tokenizer):
        if len(output) != len(self.ids):
            raise ValueError('Trajectory generation coverage changed')
        rows = []
        for i in range(len(output)):
            row = self.row_identity(i)
            for key in ('prompts', 'responses', 'attention_mask', 'info_mask', 'position_ids'):
                if key in output.batch:
                    row[key] = output.batch[key][i].detach().cpu().tolist()
            row['prompt_text'] = tokenizer.decode(row['prompts'], skip_special_tokens=False)
            row['response_with_observations_text'] = tokenizer.decode(row['responses'], skip_special_tokens=False)
            rows.append(row)
        return atomic_rows(self.root / 'generated.jsonl.gz', rows)


def record_scored_batch(root, logical_step, batch, tokenizer=None):
    """Called after row balancing and the true action mask, before optimizer work."""
    ids = batch.non_tensor_batch['audit_trajectory_row']
    if sorted(int(i) for i in ids) != list(range(len(batch))):
        raise ValueError('Trajectory row identity lost during reordering')
    rows = []
    for i in range(len(batch)):
        row = dict(logical_step=logical_step, trajectory_row=int(ids[i]),
                   response_index=int(ids[i]) % 5, response_group=int(ids[i]) // 5,
                   question_id=str(batch.non_tensor_batch['index'][i]),
                   state_group_uid=str(batch.non_tensor_batch['uid'][i]))
        for key in ('step_ids', 'loss_mask', 'policy_weights', 'advantages',
                    'token_level_scores', 'token_level_rewards', 'returns', 'old_log_probs', 'ref_log_prob'):
            if key in batch.batch:
                row[key] = batch.batch[key][i].detach().cpu().tolist()
        row['reward'] = float(batch.batch['token_level_scores'][i].sum().detach().cpu())
        if tokenizer is not None:
            row['scored_answer'] = decoded_answer(batch, i, tokenizer)
            target = batch.non_tensor_batch['reward_model'][i]['ground_truth']['target']
            row['target'] = target.tolist() if hasattr(target, 'tolist') else target
        rows.append(row)
    return atomic_rows(Path(root) / f'iter_{logical_step:06d}' / 'scored.jsonl.gz', rows)


def decoded_answer(batch, row, tokenizer):
    from verl.trainer.main_ppo import decode_model_response_spans
    from .bapo import answer_from_spans
    width = batch.batch['responses'].shape[-1]
    mask = batch.batch['attention_mask'][row, -width:].bool()
    if 'info_mask' in batch.batch:
        mask = mask & batch.batch['info_mask'][row, -width:].bool()
    spans = decode_model_response_spans(tokenizer, batch.batch['responses'][row], mask)
    return answer_from_spans(spans)


def record_validation_batch(trace, batch, tokenizer, rewards):
    from .evaluation import batch_answer_metrics
    metrics = batch_answer_metrics(batch, tokenizer)
    if len(metrics) != len(trace.ids) or len(batch) != len(trace.ids):
        raise ValueError('Validation trajectory/score coverage changed')
    rows = []
    for i, metric in enumerate(metrics):
        row = dict(trace.row_identity(i), metrics=metric,
                   reward=float(rewards[i].sum().detach().cpu()))
        target = batch.non_tensor_batch['reward_model'][i]['ground_truth']['target']
        row['target'] = target.tolist() if hasattr(target, 'tolist') else target
        row['scored_answer'] = decoded_answer(batch, i, tokenizer)
        rows.append(row)
    return atomic_rows(trace.root / 'scored.jsonl.gz', rows)


def publish_trace_completion(root, logical_step, expected_rows, status, optimizer_steps=None):
    """Completion is explicit; generated/scored files alone never mean update success."""
    folder = Path(root) / f'iter_{logical_step:06d}'
    evidence = []
    for path in sorted(folder.glob('*.jsonl.gz')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        with gzip.open(path, 'rt') as stream:
            count = sum(1 for _ in stream)
        if count != expected_rows:
            raise ValueError('Incomplete trajectory rows: ' + str(path))
        evidence.append(dict(path=str(path), rows=count, sha256=digest.hexdigest(), bytes=path.stat().st_size))
    names = {Path(item['path']).name for item in evidence}
    if not {'generated.jsonl.gz', 'scored.jsonl.gz'} <= names or not any(name.startswith('turn_') for name in names):
        raise ValueError('Missing generation/scoring/turn evidence')
    if list(folder.glob('*.partial')):
        raise ValueError('Partial trajectory files remain')
    receipt = dict(logical_step=logical_step, status=status, actual_optimizer_steps=optimizer_steps,
                   expected_rows=expected_rows, turn_count=sum(name.startswith('turn_') for name in names),
                   files=evidence)
    target = folder / 'complete.json'
    if target.exists():
        raise ValueError('Refusing to overwrite trajectory completion receipt')
    partial = target.with_name('complete.json.partial')
    with partial.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    partial.rename(target)
    return receipt
