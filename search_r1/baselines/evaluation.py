"""Generated-answer metrics, shared by holdout selection and final evaluation."""
from collections import defaultdict
import re
from .bapo import answer_from_spans, token_f1, legal_action_records
from verl.utils.reward_score.qa_em import em_check


def batch_answer_metrics(data, tokenizer):
    from verl.trainer.main_ppo import decode_model_response_spans
    output = []
    for i in range(len(data)):
        item = data[i]
        width = item.batch['responses'].shape[-1]
        mask = item.batch['attention_mask'][-width:].bool()
        if 'info_mask' in item.batch:
            mask = mask & item.batch['info_mask'][-width:].bool()
        spans = decode_model_response_spans(tokenizer, item.batch['responses'], mask)
        answer = answer_from_spans(spans)
        targets = item.non_tensor_batch['reward_model']['ground_truth']['target']
        if isinstance(targets, str):
            targets = [targets]
        output.append(dict(source=item.non_tensor_batch['data_source'],
            f1=max((token_f1(answer, t) for t in targets), default=0.) if answer is not None else 0.,
            em=float(em_check(answer, targets)) if answer is not None else 0.,
            answer_format=float(answer is not None),
            format_rate=float(answer is not None and (data.meta_info.get('baseline_aligned', False) or legal_action_records([dict(text=s) for s in spans]))),
            generated_tokens=int(mask.sum()),
            search_calls=sum(len(re.findall(r'<search>.*?</search>', s, re.DOTALL)) for s in spans)))
    return output


def aggregate_answer_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source']].append(row)
    groups['all'] = rows
    output = {}
    for source, group in groups.items():
        output[f'val/qa_count/{source}'] = len(group)
        for key in ('f1', 'em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls'):
            output[f'val/{key}/{source}'] = sum(row[key] for row in group) / len(group)
    return output
