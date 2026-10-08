"""CPU-only reconstruction/transition primitives; no inference or network."""
import copy
import hashlib
import json
import re

FORCED = '\n<information>Search budget exhausted. Give the final answer now inside <answer> and </answer>.</information>\n'
OBS_START, OBS_END = '\n<information>\n', '\n</information>\n'


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cut(segment):
    ends = [segment.find(tag) + len(tag) for tag in ('</search>', '</answer>') if tag in segment]
    return segment[:min(ends)] if ends else segment


def initial(backend='hf'):
    return dict(backend=backend, next_turn=0, calls_completed=0, response_before_turn='',
                policy_spans=[], observations=[], terminal=None, forced_pending=False)


def validate(state):
    t, calls = state['next_turn'], state['calls_completed']
    if type(t) != int or type(calls) != int or not 0 <= t <= 5 or not 0 <= calls <= 4:
        raise ValueError('Invalid turn/search budget')
    if state['terminal'] is None and (t >= 5 or calls != t):
        raise ValueError('Not an active native generation state')
    if state['forced_pending'] != (t == 4 and state['terminal'] is None):
        raise ValueError('Invalid pending forced state')
    previous = 0
    for a, b in state['policy_spans']:
        if not previous <= a <= b <= len(state['response_before_turn']):
            raise ValueError('Invalid state policy provenance')
        previous = b
    return state


def context(state, prefix):
    validate(state)
    if state['terminal'] is not None:
        raise ValueError('Cannot resume terminal state')
    return prefix + state['response_before_turn'] + (FORCED if state['forced_pending'] else '')


def advance(state, segment, observation=None):
    """Apply native HF post-generation transition to a supplied kept segment.

    Observation is already recorded content, never fetched here. Returns an
    awaiting-generation or terminal state; no intermediate waiting-for-HTTP state.
    """
    validate(state)
    if state['terminal'] is not None:
        raise ValueError('Cannot advance terminal state')
    result = copy.deepcopy(state); t = result['next_turn']
    segment = cut(segment)
    result['response_before_turn'] += FORCED if result['forced_pending'] else ''
    a = len(result['response_before_turn']); result['response_before_turn'] += segment
    result['policy_spans'].append([a, len(result['response_before_turn'])])
    result['next_turn'] = t + 1
    if '</answer>' in segment:
        if observation is not None:
            raise ValueError('Terminal answer cannot have observation')
        result['terminal'] = 'answer'
    else:
        match = re.search(r'<search>(.*?)</search>', segment, re.S)
        if match is None or result['calls_completed'] == 4:
            if observation is not None:
                raise ValueError('Invalid/budget action cannot have observation')
            result['terminal'] = 'invalid_or_budget'
        else:
            if not isinstance(observation, str) or not observation.startswith(OBS_START) or not observation.endswith(OBS_END):
                raise ValueError('Missing/unaccounted original observation')
            result['response_before_turn'] += observation
            result['observations'].append(dict(turn=t, query=match.group(1).strip(), inserted_information=observation,
                                               inserted_information_sha256=hashlib.sha256(observation.encode()).hexdigest()))
            result['calls_completed'] += 1
            if result['next_turn'] == 5:
                result['terminal'] = 'budget'
    result['forced_pending'] = result['next_turn'] == 4 and result['terminal'] is None
    return validate(result)


def boundary(state, label, failure_history, reason=None):
    result = copy.deepcopy(state)
    result.update(boundary=label, failure_history=failure_history,
                  status='skipped' if reason or state['terminal'] else 'ready',
                  reason=reason or ('terminal_boundary_skip:' + state['terminal'] if state['terminal'] else None),
                  remaining_turns=max(0, 5 - state['next_turn']), remaining_searches=4 - state['calls_completed'])
    result['source_trajectory_failed'] = failure_history
    result['failure_history'] = failure_history and bool(state['policy_spans'])
    result['history_label'] = 'empty_no_failure_history' if not state['policy_spans'] else 'failed_source_prefix' if failure_history else 'successful_source_prefix'
    # Identity excludes the label/status so equivalent aliases get one execution.
    result['state_sha256'] = sha({k: state[k] for k in state})
    return result


def deduplicate(boundaries):
    ready = {}; skips = []
    for item in boundaries:
        if item['status'] != 'ready':
            skips.append(item); continue
        key = item['state_sha256']
        if key not in ready:
            ready[key] = dict(item, aliases=[item['boundary']])
        elif item['boundary'] not in ready[key]['aliases']:
            ready[key]['aliases'].append(item['boundary'])
    return dict(executable_states=list(ready.values()), skipped_boundaries=skips)


def reconstruct_original(row, backend='vllm'):
    if backend not in ('vllm', 'hf'):
        raise ValueError('Unknown original executor backend')
    response = row['response']; spans = row['policy_spans']; previous = 0
    if len(spans) > 5:
        raise ValueError('More than five generated policies')
    for a, b in spans:
        if type(a) != int or type(b) != int or not previous <= a <= b <= len(response):
            raise ValueError('Invalid original policy spans')
        previous = b
    state = initial(backend); pre = []; post = []
    for i, (a, b) in enumerate(spans):
        if state['terminal'] is not None:
            raise ValueError('Policy after terminal')
        expected_a = len(state['response_before_turn']) + (len(FORCED) if i == 4 else 0)
        if a != expected_a or response[len(state['response_before_turn']):a] != (FORCED if i == 4 else ''):
            raise ValueError('Unaccounted prefix/forced gap')
        pre.append(copy.deepcopy(state)); segment = response[a:b]
        end = spans[i + 1][0] if i + 1 < len(spans) else len(response)
        gap = response[b:end]
        if i + 1 == 4 and (i + 1 < len(spans) or backend == 'vllm' and row['termination'] == 'context_limit'):
            if not gap.endswith(FORCED):
                raise ValueError('Missing exact upcoming forced suffix')
            gap = gap[:-len(FORCED)]
        has_search = '</answer>' not in segment and re.search(r'<search>(.*?)</search>', segment, re.S) is not None and state['calls_completed'] < 4
        if not has_search and gap:
            raise ValueError('Terminal policy has trailing external bytes')
        state = advance(state, segment, gap if has_search else None)
        if state['response_before_turn'] != response[:b] + gap:
            raise ValueError('Original segment differs from native structural cut')
        post.append(copy.deepcopy(state))
    if state['calls_completed'] != row['search_calls']:
        raise ValueError('Original search count differs')
    if row['termination'] == 'context_limit':
        if state['terminal'] is not None:
            raise ValueError('Context attempt follows a terminal action')
        if not spans and response:
            raise ValueError('Context-limit with unexplained response')
        # vLLM stored forced before rejecting context; canonical preturn excludes it.
        expected_response = state['response_before_turn'] + (FORCED if backend == 'vllm' and state['next_turn'] == 4 else '')
        if response != expected_response:
            raise ValueError('Context-limit source backend gap differs')
        state['terminal'] = 'context_limit'; state['forced_pending'] = False
    elif state['terminal'] != row['termination'] or state['response_before_turn'] != response:
        raise ValueError('Original termination/response does not reconcile')
    failure = row.get('reward') == 0
    a = boundary(post[0] if post else state, 'after_first_policy_and_observation', failure,
                 'no_first_policy' if not post else None)
    # Do not relabel a non-generated context-limit attempt as last answer boundary.
    bstate = state if row['termination'] == 'context_limit' or not pre else pre[-1]
    b = boundary(bstate, 'before_last_policy_generation', failure,
                 'no_last_generated_policy' if not pre else None)
    result = deduplicate([a, b])
    result.update(original_backend=backend, original_response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                  original_termination=row['termination'], generated_turns=len(spans),
                  reconstructed=True, native_context_token_admission_not_yet_checked=True)
    return result


def reconstruct_hf(trace):
    if trace.get('kind') != 'actual_autoregressive_generation' or trace.get('teacher_forced') is not False:
        raise ValueError('Expected actual HF generation trace')
    state = initial('hf'); before = []; after = []; generated_ids_available = True
    for i, event in enumerate(trace['turns']):
        if i != event['turn'] or state['next_turn'] != i or state['terminal']:
            raise ValueError('HF event turn/state mismatch')
        if event['forced_information'] != (FORCED if i == 4 else '') or event['context_text'] != context(state, trace['prompt_text']):
            raise ValueError('HF real context differs from reconstructed state')
        before.append(copy.deepcopy(state))
        if 'skipped' in event:
            if event['skipped'] != 'context_limit_without_independent_truncation' or i != len(trace['turns']) - 1 or any(k in event for k in ('policy_span', 'generated_token_ids')):
                raise ValueError('Malformed HF skipped event')
            state['terminal'] = 'context_limit'; state['forced_pending'] = False
            break
        if 'generated_token_ids' not in event:
            generated_ids_available = False
            raise ValueError('Actual IDs missing; do not manufacture via posthoc encode')
        if not isinstance(event['generated_token_ids'], list) or any(type(x) != int or x < 0 for x in event['generated_token_ids']):
            raise ValueError('Malformed actual generation IDs')
        if cut(event['decoded_segment_before_structural_cut']) != event['policy_segment']:
            raise ValueError('HF cut provenance differs')
        state = advance(state, event['policy_segment'], event.get('inserted_information'))
        if state['policy_spans'][-1] != event['policy_span']:
            raise ValueError('HF policy span differs')
        if 'inserted_information' in event:
            if (event.get('observation_content_budget') != 480 or 'retrieval_payload' not in event or
                    len(event.get('inserted_content_token_ids', [])) > 480 or
                    len(event['retrieval_payload'].get('result', [[]])[0]) != 3):
                raise ValueError('HF observation provenance/budget missing')
            if event.get('query') != state['observations'][-1]['query']:
                raise ValueError('HF executed query differs')
        after.append(copy.deepcopy(state))
    if (state['response_before_turn'] != trace['response'] or state['calls_completed'] != trace['search_calls'] or
            state['policy_spans'] != [list(x) for x in trace['policy_spans']] or state['terminal'] != trace['termination']):
        raise ValueError('HF final artifact differs from reconstructed events')
    failure = trace.get('protocol_em') == 0
    a = boundary(after[0] if after else state, 'after_first_policy_and_observation', failure, 'no_first_policy' if not after else None)
    bstate = state if state['terminal'] == 'context_limit' or not before else before[-1]
    b = boundary(bstate, 'before_last_policy_generation', failure, 'no_last_generated_policy' if not state['policy_spans'] else None)
    result = deduplicate([a, b]); result.update(reconstructed=True, actual_generation_ids_present=generated_ids_available,
        source_response_sha256=hashlib.sha256(trace['response'].encode()).hexdigest(),
        source_tokenizer_roundtrip_and_context_token_count_not_yet_checked=True)
    for item in result['executable_states'] + result['skipped_boundaries']:
        t = item['next_turn']
        if t < len(trace['turns']):
            event = trace['turns'][t]
            item['source_event'] = dict(turn=t, context_tokens=event['context_tokens'],
                context_text_sha256=hashlib.sha256(event['context_text'].encode()).hexdigest(),
                actual_generated_ids_present='generated_token_ids' in event,
                actual_generated_ids_sha256=sha(event['generated_token_ids']) if 'generated_token_ids' in event else None,
                native_split_encoding_equivalence_not_asserted=True)
    return result
