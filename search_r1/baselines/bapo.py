"""Boundary-Aware Policy Optimization, paper-semantics Search-R1 adaptation."""
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import re
import string
import torch
from search_r1.qa_protocol import extract_final_answer, parse_model_action


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def token_f1(prediction, target):
    a, b = normalize(prediction).split(), normalize(target).split()
    if not a or not b:
        return 0.
    shared = sum((Counter(a) & Counter(b)).values())
    return 2 * shared / (len(a) + len(b))


def answer_from_spans(spans):
    return extract_final_answer(spans)


def is_idk(answer):
    return answer is not None and normalize(answer) in ('i dont know', 'i do not know')


def ordinary_answer_f1(answer, targets):
    if answer is None:
        return -1.
    if is_idk(answer):
        return 0.
    if isinstance(targets, str):
        targets = [targets]
    return max((token_f1(answer, target) for target in targets), default=0.)


def answer_f1(answer, targets):
    """Fixed BAPO author scorer: categorical answers require exact match."""
    if answer is None:
        return -1.
    if is_idk(answer):
        return 0.
    targets = [targets] if isinstance(targets, str) else targets
    prediction = normalize(answer)
    scores = []
    for target in targets:
        expected = normalize(target)
        if (prediction in ('yes', 'no', 'noanswer') or expected in ('yes', 'no', 'noanswer')) and prediction != expected:
            scores.append(0.)
        else:
            scores.append(token_f1(answer, target))
    return max(scores, default=0.)


@dataclass
class BoundaryStage:
    best: float = float('-inf')
    stalled: int = 0
    early: bool = True
    patience: int = 5

    def observe(self, validation_score):
        if validation_score > self.best:
            self.best = float(validation_score)
            self.stalled = 0
        else:
            self.stalled += 1
        if self.stalled >= self.patience:
            self.early = False

    def state_dict(self):
        # Strict JSON: an unobserved best uses null, not non-standard Infinity.
        result = asdict(self)
        if result['best'] == float('-inf'):
            result['best'] = None
        return result

    def load_state_dict(self, state):
        for key in ('best', 'stalled', 'early', 'patience'):
            setattr(self, key, state[key])
        if self.best is None:
            self.best = float('-inf')


def groups(uids):
    result = defaultdict(list)
    for i, uid in enumerate(uids):
        result[uid].append(i)
    return result.values()


def boundary_scores(answers, targets, uids, stage, alpha=.05, bonus=.5):
    """Training F1; legal IDK bonus only for all-incorrect groups with open gate."""
    base = [answer_f1(a, t) for a, t in zip(answers, targets)]
    result = list(base)
    batch_ratio = sum(map(is_idk, answers)) / max(1, len(answers))
    for rows in groups(uids):
        diversity = len({normalize(answers[i]) for i in rows if answers[i] is not None})
        # Paper threshold scales with actual G, unlike the official hard-coded G=8.
        gate = batch_ratio < alpha if stage.early else diversity < len(rows) / 2
        if gate and all(base[i] <= 0 for i in rows):
            for i in rows:
                if is_idk(answers[i]):
                    result[i] += bonus
    return result


def resample_rows(answers, targets, uids, stage):
    if stage.early:
        return []
    return [i for rows in groups(uids)
            if all(answer_f1(answers[j], targets[j]) <= 0 for j in rows)
            and not any(is_idk(answers[j]) for j in rows) for i in rows]


def legal_action_records(records):
    if not records:
        return False
    final_kind = None
    for i, record in enumerate(records):
        action = parse_model_action(record['text'], strict_prefix=True)
        if action is None:
            return False
        final_kind = action[0]
        if final_kind == 'answer' and i != len(records) - 1:
            return False
    return final_kind == 'answer'


def decode_answers(data, tokenizer, require_legal=False):
    from verl.trainer.main_ppo import decode_model_response_spans
    response_len = data.batch['responses'].shape[1]
    mask = data.batch['info_mask'][:, -response_len:] * data.batch['attention_mask'][:, -response_len:]
    answers = [answer_from_spans(decode_model_response_spans(tokenizer, ids, m))
               for ids, m in zip(data.batch['responses'], mask)]
    if require_legal:
        records = data.non_tensor_batch['action_records']
        answers = [answer if legal_action_records(record) else None for answer, record in zip(answers, records)]
    return answers


class BoundaryReward:
    def __init__(self, tokenizer, stage, alpha=.05, bonus=.5):
        self.tokenizer, self.stage, self.alpha, self.bonus = tokenizer, stage, alpha, bonus

    def __call__(self, data):
        answers = decode_answers(data, self.tokenizer, require_legal=True)
        targets = [r['ground_truth']['target'] for r in data.non_tensor_batch['reward_model']]
        scores = boundary_scores(answers, targets, data.non_tensor_batch['uid'], self.stage, self.alpha, self.bonus)
        result = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        width = result.shape[1]
        mask = data.batch['info_mask'][:, -width:] * data.batch['attention_mask'][:, -width:]
        for i, score in enumerate(scores):
            positions = torch.where(mask[i].bool())[0]
            if len(positions):
                result[i, positions[-1]] = score
        return result
