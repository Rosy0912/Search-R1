"""Fixed-weight HF continuation; imports models/network only via injected objects."""
import copy
import re
from contextlib import contextmanager


@contextmanager
def score_tap(processor_class, capture):
    """Call original processor exactly once/order; return its original score object.

    Single-thread worker scope only. capture must not mutate input; raw clone is
    made before possibly in-place processors. No output_scores/generate changes.
    """
    original = processor_class.__call__
    def tapped(processors, ids, scores, *args, **kwargs):
        raw = scores.detach().clone()
        result = original(processors, ids, scores, *args, **kwargs)
        capture(ids, raw, result, processors)
        return result
    processor_class.__call__ = tapped
    try:
        yield
    finally:
        processor_class.__call__ = original


def run(model, tokenizer, core, rollout, state_api, state, question, answers,
        reference, retrieve, score_context, max_new_tokens=500, max_context=8192):
    """Resume canonical active state. Old observations are byte-preserved, unfetched.

    score_context(event) is a context manager installed around ORIGINAL generate;
    retrieve(query) returns the unchanged native JSON payload after live admission.
    """
    if (max_new_tokens, max_context) != (500, 8192):
        raise ValueError('Frozen native context settings required')
    current = copy.deepcopy(state)
    state_api.validate(current)
    if current['terminal'] is not None:
        raise ValueError('Terminal states cannot continue')
    prefix = tokenizer.apply_chat_template(core.prompt(question, reference), tokenize=False, add_generation_prompt=True)
    model.eval(); events = []
    while current['terminal'] is None:
        text = state_api.context(current, prefix)
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(next(model.parameters()).device)
        event = dict(turn=current['next_turn'], input_ids=ids[0].detach().cpu().tolist(),
                     context_text=text, context_tokens=int(ids.shape[1]),
                     state_before=copy.deepcopy(current), forced_information=state_api.FORCED if current['forced_pending'] else '')
        events.append(event)
        if ids.shape[1] + 500 > 8192:
            event['skipped'] = 'context_limit_without_independent_truncation'
            current['terminal'] = 'context_limit'; current['forced_pending'] = False
            break
        options = dict(max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                       stopping_criteria=rollout.StoppingCriteriaList([rollout.StopTags(tokenizer, ids.shape[1])]))
        with score_context(event):
            output = model.generate(ids, **options)
        actual_ids = output[0, ids.shape[1]:].detach().cpu().tolist()
        if 'fork_scores' in event and actual_ids and event['fork_scores']['actual_greedy_target_id'] != actual_ids[0]:
            raise ValueError('Recorded actual processed fork greedy choice differs from emitted first ID')
        decoded = tokenizer.decode(actual_ids, skip_special_tokens=True)
        segment = state_api.cut(decoded)
        event.update(generated_token_ids=actual_ids, decoded_segment_before_structural_cut=decoded,
                     policy_segment=segment, kept_segment_posthoc_token_ids=tokenizer.encode(segment, add_special_tokens=False),
                     actual_ids_not_replaced_by_posthoc_encoding=True,
                     actual_ids_equal_kept_posthoc_encoding=actual_ids == tokenizer.encode(segment, add_special_tokens=False))
        observation = None
        if '</answer>' not in segment:
            match = re.search(r'<search>(.*?)</search>', segment, re.S)
            if match is not None and current['calls_completed'] < 4:
                query = match.group(1).strip(); payload = retrieve(query)
                passages = payload['result'][0]
                if len(passages) != 3:
                    raise ValueError('Admitted retriever must return native top3')
                formatted = []
                for i, passage in enumerate(passages):
                    doc = passage.get('document', passage)
                    formatted.append(f"Doc {i+1}: {doc.get('contents', doc.get('text', ''))}")
                info = '\n'.join(formatted)
                info_ids = tokenizer.encode(info, add_special_tokens=False)[:480]
                observation = '\n<information>\n' + tokenizer.decode(info_ids) + '\n</information>\n'
                event.update(query=query, retrieval_payload=payload, formatted_information_before_truncation=info,
                             inserted_content_token_ids=info_ids, inserted_information=observation)
        current = state_api.advance(current, segment, observation)
        event['state_after'] = copy.deepcopy(current)
    response = current['response_before_turn']; spans = current['policy_spans']
    from search_r1.qa_protocol import extract_final_answer
    final_answer = extract_final_answer([response[a:b] for a,b in spans])
    return dict(kind='actual_counterfactual_continuation', teacher_forced=False,
                privileged_reference=reference is not None, execution_backend='hf', source_history_backend=state['backend'],
                prompt_text=prefix, initial_state=state,
                final_state=current, response=response, policy_spans=spans,
                search_calls=current['calls_completed'], termination=current['terminal'],
                protocol_em=core.reward(response, answers, spans), protocol_valid=final_answer is not None,
                accepted_final_answer=final_answer, turns=events,
                temperature=0, max_new_tokens_per_turn=500, max_context=8192,
                training_split_encoding_equivalence_claimed=False)
