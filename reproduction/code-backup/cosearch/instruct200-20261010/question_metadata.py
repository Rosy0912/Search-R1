"""Recover question metadata from the unchanged shared dataset prompt."""
import functools


def with_question(kwargs):
    extra = kwargs['extra_info']
    if 'question' in extra:
        if not isinstance(extra['question'], str) or not extra['question']:
            raise ValueError('Malformed explicit question metadata')
        return kwargs
    messages = list(kwargs['raw_prompt'])
    if len(messages) != 1 or messages[0]['role'] != 'user':
        raise ValueError('Question adapter requires one original user prompt')
    content = messages[0]['content']
    prefix, separator, question = content.partition('\nQuestion: ')
    if not separator or not prefix.startswith('Answer the given question.') or not question:
        raise ValueError('Original question delimiter missing')
    result = dict(kwargs)
    result['extra_info'] = dict(extra, question=question)
    return result


def install(loop):
    if getattr(loop, '_shared_question_adapter', False):
        return
    original = loop.run

    @functools.wraps(original)
    async def run(instance, sampling_params, **kwargs):
        return await original(instance, sampling_params, **with_question(kwargs))

    loop.run = run
    loop._shared_question_adapter = True
