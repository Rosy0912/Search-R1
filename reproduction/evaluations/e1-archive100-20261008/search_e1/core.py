import re
import string
from collections import defaultdict

INSTRUCTION = ('Answer the given question. You must conduct reasoning inside <think> and </think> '
               'first every time you get new information. After reasoning, if you lack knowledge, '
               'call a search engine by <search> query </search>; results appear between '
               '<information> and </information>. You may search at most four times. '
               'Provide the final answer inside <answer> and </answer>, without detailed illustrations. '
               'For example, <answer> Beijing </answer>.')

def prompt(question, reference=None):
    text = INSTRUCTION + '\nQuestion: ' + question
    if reference is not None:
        text += ('\nExpert reference trajectory:\n' + reference +
                 '\nRead the reference, then attempt the same problem in your own words.')
    return [{'role': 'user', 'content': text}]

def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def reward(trajectory, answers, policy_spans=None):
    from search_r1.qa_protocol import extract_final_answer
    if policy_spans is None:
        raise ValueError('Policy span provenance is required for Search-E1 rewards')
    previous = 0
    spans = []
    for start, end in policy_spans:
        if not previous <= start <= end <= len(trajectory):
            raise ValueError('Invalid policy span bounds')
        spans.append(trajectory[start:end]); previous = end
    answer = extract_final_answer(spans)
    if answer is None:
        return 0
    if isinstance(answers, str):
        answers = [answers]
    return int(any(normalize(answer) == normalize(a) for a in answers))


def information_spans(text):
    # Include delimiters and trailing unclosed information in the external-token mask.
    return [(m.start(), m.end()) for m in re.finditer(r'<information>.*?(?:</information>|$)', text, re.S)]

def response_encoding(tokenizer, text, policy_spans=None):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = information_spans(text)
    mask = [int(b > a and not any(a < y and b > x for x, y in spans)
                and (policy_spans is None or any(x <= a and b <= y for x, y in policy_spans)))
            for a, b in encoded['offset_mapping']]
    return encoded['input_ids'], mask

def mine_pairs(rows):
    from rapidfuzz.distance import Levenshtein
    groups = defaultdict(list)
    for row in rows:
        groups[row['question_id']].append(row)
    result = []
    for key, siblings in groups.items():
        correct = [r for r in siblings if r['reward'] == 1]
        if not correct or len(siblings) < 2:
            continue
        ref = min(correct, key=lambda r: (r['search_calls'], r.get('token_count', len(r['response']))))
        candidates = [r for r in siblings if r is not ref and r['reward'] == 0]
        if not candidates:
            candidates = [r for r in siblings if r is not ref]
        # Paper does not define character distance; normalized Levenshtein is an explicit assumption.
        stu = max(candidates, key=lambda r: Levenshtein.normalized_distance(ref['response'], r['response']))
        result.append({'question_id': key, 'question': ref['question'], 'answers': ref['answers'],
                       'reference': ref['response'], 'student': stu['response'],
                       'reference_reward': ref['reward'], 'student_reward': stu['reward'],
                       'student_policy_spans': stu.get('policy_spans')})
    return result

def clipped_forward_kl(student_logits, teacher_logits, clip=10.):
    import torch
    s = torch.log_softmax(student_logits.float(), dim=-1)
    t = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    # Eq. 7 clips EACH vocabulary contribution above, without lower-clipping negative terms.
    return (t.exp() * (t - s)).clamp(max=clip).sum(dim=-1)
