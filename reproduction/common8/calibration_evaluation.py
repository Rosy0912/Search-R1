"""Ordered fixed512 calibration measurements; never health or production admission.

Call rows_from_batch after validation generation is unioned into its batch, before
any training balance/reorder. The reward tensor is neither read nor modified.
"""
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import torch

from search_g1.signals import contexts, decode_segments
from verl.utils.reward_score import qa_em

QUESTION_COUNT = 512
SOURCE_COUNTS = {'nq': 256, 'hotpotqa': 256}
_ROW_FIELDS = frozenset(('question_id', 'data_source', 'em', 'f1', 'actual_calls',
                         'valid_action', 'turns', 'answer_sha256', 'answer_present',
                         'response_token_count', 'model_token_count'))


def _require(condition, message):
    if not condition:
        raise ValueError('Invalid calibration evaluation: ' + message)


def _integer(value, name, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and
             (maximum is None or value <= maximum), 'invalid integer ' + name)
    return value


def _actions(actual, valid, turns):
    # The bound recipe has four executable turns and one final generation whose
    # search action is recognized but deliberately not sent to the retriever.
    _integer(actual, 'actual_calls', maximum=4)
    _integer(valid, 'valid_action', maximum=5)
    _integer(turns, 'turns', minimum=1, maximum=5)
    _require(actual <= valid <= turns, 'action counts exceed generation turns')


def _metadata(batch, name, count):
    value = batch.meta_info.get(name)
    _require(isinstance(value, (list, tuple)) and len(value) == count,
             'missing or misaligned ' + name)
    return value


def _aliases(target):
    if isinstance(target, str):
        return [target]
    if hasattr(target, 'tolist'):
        target = target.tolist()
    _require(isinstance(target, (list, tuple)) and bool(target) and
             all(isinstance(value, str) for value in target), 'invalid gold aliases')
    return target


def _f1(answer, aliases):
    if answer is None:
        return 0.
    prediction = qa_em.normalize_answer(answer).split()
    scores = []
    for alias in aliases:
        gold = qa_em.normalize_answer(alias).split()
        if not prediction or not gold:
            scores.append(float(prediction == gold))
        else:
            overlap = sum((Counter(prediction) & Counter(gold)).values())
            scores.append(2. * overlap / (len(prediction) + len(gold)))
    return max(scores)


def _tokens(value, name):
    _require(isinstance(value, torch.Tensor) and value.ndim == 1 and
             value.dtype in (torch.int32, torch.int64), 'invalid token tensor ' + name)
    return value


def _mask(value, width, name):
    _require(isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel() == width,
             'misaligned mask ' + name)
    _require(bool(((value == 0) | (value == 1)).all()), 'nonbinary mask ' + name)
    return value


def rows_from_batch(batch, tokenizer):
    """Extract strict generated-span EM/F1 and actual ordered executor counts.

    Raw question, evidence and answer text stay in the existing remote trajectory
    logs. Only scalar fields, IDs and a SHA256 of the extracted answer are returned.
    Zero-token/missing/invalid answers remain questions with EM=F1=0.
    """
    count = len(batch)
    _require(count > 0, 'empty batch')
    actual = _metadata(batch, 'actual_retrieval_stats', count)
    valid = _metadata(batch, 'valid_action_stats', count)
    turns = _metadata(batch, 'turns_stats', count)
    for name in ('__g1_question_id', 'data_source', 'reward_model'):
        _require(name in batch.non_tensor_batch and len(batch.non_tensor_batch[name]) == count,
                 'missing or misaligned ' + name)
    rows = []
    for index in range(count):
        _actions(actual[index], valid[index], turns[index])
        item = batch[index]
        question_id = item.non_tensor_batch['__g1_question_id']
        source = item.non_tensor_batch['data_source']
        _require(isinstance(question_id, str) and bool(question_id), 'missing question ID')
        _require(isinstance(source, str) and source in SOURCE_COUNTS, 'unexpected data source')
        prompts = _tokens(item.batch['prompts'], 'prompts')
        responses = _tokens(item.batch['responses'], 'responses')
        plen = prompts.numel()
        attention = _mask(item.batch['attention_mask'], plen + responses.numel(), 'attention')
        authored = _mask(item.batch['info_mask'], attention.numel(), 'authorship')
        _require(bool((authored <= attention).all()), 'authorship outside valid tokens')
        valid_plen = int(attention[:plen].sum())
        valid_rlen = int(attention[plen:].sum())
        _require(valid_plen > 0 and bool((attention[:plen] ==
                 (torch.arange(plen, device=attention.device) >= plen - valid_plen)).all()),
                 'prompt mask is not a nonempty left-padded suffix')
        _require(bool((attention[plen:] ==
                 (torch.arange(responses.numel(), device=attention.device) < valid_rlen)).all()),
                 'response mask is not a right-padded prefix')
        prompt = tokenizer.decode(prompts[-valid_plen:])
        response, ranges = decode_segments(tokenizer, responses[:valid_rlen],
                                           authored[plen:plen + valid_rlen])
        ctx = contexts(prompt, response, ranges)
        answer = ctx['answer'] if ctx is not None else None
        aliases = _aliases(item.non_tensor_batch['reward_model']['ground_truth']['target'])
        em = int(answer is not None and bool(qa_em.em_check(answer, aliases)))
        rows.append(dict(question_id=question_id, data_source=source, em=em,
                         f1=_f1(answer, aliases), actual_calls=actual[index],
                         valid_action=valid[index], turns=turns[index],
                         answer_present=answer is not None,
                         answer_sha256=hashlib.sha256(answer.encode('utf-8')).hexdigest()
                         if answer is not None else None,
                         response_token_count=valid_rlen,
                         model_token_count=int(authored[plen:plen + valid_rlen].sum())))
    return rows


def aggregate(rows, expected_ids, expected_sources):
    """Match all512 ordered canonical occurrences; no resampling or deduplication.

    An identical expected ID/source sequence pairs before/after evaluations.
    This validates measurement structure only; health rules belong to admission.
    """
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and
             len(rows) == QUESTION_COUNT, 'need exactly512 measured questions')
    ids, sources = list(expected_ids), list(expected_sources)
    _require(len(ids) == len(sources) == QUESTION_COUNT, 'need exactly512 canonical occurrences')
    _require(all(isinstance(value, str) and bool(value) for value in ids), 'invalid canonical IDs')
    _require(dict(Counter(sources)) == SOURCE_COUNTS, 'need canonical NQ256/Hotpot256')
    measured = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping) and set(row) == _ROW_FIELDS, 'unexpected measurement row fields')
        _require(row['question_id'] == ids[index] and row['data_source'] == sources[index],
                 'ordered question/source differs at occurrence ' + str(index))
        _integer(row['em'], 'em', maximum=1)
        _require(type(row['f1']) in (int, float) and math.isfinite(row['f1']) and
                 0 <= row['f1'] <= 1, 'invalid F1')
        _require(row['em'] <= row['f1'], 'EM and normalized-token F1 disagree')
        _actions(row['actual_calls'], row['valid_action'], row['turns'])
        length = _integer(row['response_token_count'], 'response_token_count')
        _integer(row['model_token_count'], 'model_token_count', maximum=length)
        _require(type(row['answer_present']) is bool, 'invalid answer presence')
        sha = row['answer_sha256']
        _require((row['answer_present'] and isinstance(sha, str) and len(sha) == 64 and
                  all(value in '0123456789abcdef' for value in sha)) or
                 (not row['answer_present'] and sha is None and row['em'] == row['f1'] == 0),
                 'invalid answer hash or missing-answer scores')
        measured.append(dict(row))
    calls = sum(row['actual_calls'] for row in measured)
    searched = sum(row['actual_calls'] > 0 for row in measured)
    valid = sum(row['valid_action'] for row in measured)
    opportunities = sum(row['turns'] for row in measured)
    return dict(role='policy_safety_calibration_evaluation_measurement',
                question_count=QUESTION_COUNT, source_counts=dict(SOURCE_COUNTS),
                question_ids=ids, data_sources=sources, rows=measured,
                weighted_em=math.fsum(row['em'] for row in measured) / QUESTION_COUNT,
                weighted_f1=math.fsum(row['f1'] for row in measured) / QUESTION_COUNT,
                executed_search_question_rate=searched / QUESTION_COUNT,
                executed_search_calls_per_question=calls / QUESTION_COUNT,
                executed_search_question_count=searched, executed_search_call_count=calls,
                valid_action_rate=valid / opportunities, valid_action_count=valid,
                action_opportunity_count=opportunities,
                valid_action_rate_definition='micro average: sum(valid_action_stats)/sum(turns_stats); '
                    'recognized actions per generation turn, including final recognized but unexecuted search',
                executed_search_definition='completed logical retriever searches from actual_retrieval_stats; '
                    'excludes final do_search=False and does not count HTTP retries',
                production_admission=False,
                semantics='ordered measurement only; no health or production acceptance')
