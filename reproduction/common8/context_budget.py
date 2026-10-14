"""Plan bounded interactions before sampling; never crop sampled actor context."""

ROLE = 'common8_total_context_answer_reserve_v1'


def protocol(config):
    total = int(config.max_prompt_length)
    action = int(config.max_response_length)
    reserve = int(getattr(config, 'answer_reserve_tokens', action))
    if not 0 < action <= reserve < total:
        raise ValueError('Invalid total context / answer reservation')
    return dict(role=ROLE, total_tokens=total, action_tokens=action,
                answer_reserve_tokens=reserve,
                observation_policy='clip_before_input_preserve_information_wrapper',
                final_policy='early_no_search_when_next_action_would_consume_reserve')


def final_stage(prefix_length, ordinal, max_turns, budget):
    return (ordinal == max_turns or
            prefix_length + budget['action_tokens'] + budget['answer_reserve_tokens'] >
            budget['total_tokens'])


def observation_room(prefix_length, generated_length, budget):
    return max(0, budget['total_tokens'] - budget['answer_reserve_tokens'] -
               prefix_length - generated_length)


def observation_tokens(tokenizer, text, limit):
    """Only environment content is truncated. Model output is never re-encoded."""
    if not text or limit <= 0:
        return [], bool(text)
    encode = lambda value: list(tokenizer.encode(value, add_special_tokens=False))
    full = encode(text)
    if len(full) <= limit:
        return full, False
    start, end = text.find('<information>'), text.rfind('</information>')
    if start >= 0 and end > start:
        opening = text[:start + len('<information>')]
        closing = text[end:]
        left, right = encode(opening), encode(closing)
        available = limit - len(left) - len(right)
        if available < 0:
            return [], True
        body = encode(text[start + len('<information>'):end])
        return left + body[:available] + right, True
    return full[:limit], True
