"""Dependency-free extraction of terminal, non-nested QA actions."""
import re

_RESERVED = re.compile(r"</?(?:think|search|answer|information)\b")
_ACTION = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_EOS = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>))+$")


def parse_model_action(text, strict_prefix=False):
    """Return (kind, payload) only for a complete standalone protocol action.

    Free-text reasoning prefixes remain compatible with the executor. Structured
    reasoning must be one closed think block, with no nested protocol tags.
    """
    text = _EOS.sub('', text.strip()).strip()
    actions = list(_ACTION.finditer(text))
    if len(actions) != 1:
        return None
    action = actions[0]
    payload = action.group(2).strip()
    if not payload or _RESERVED.search(payload) or text[action.end():].strip():
        return None
    prefix = text[:action.start()].strip()
    if _RESERVED.search(prefix) or (strict_prefix and prefix):
        thought = re.fullmatch(r'<think>(.*?)</think>', prefix, re.DOTALL)
        if thought is None or _RESERVED.search(thought.group(1)):
            return None
    return action.group(1), payload


def extract_final_answer(model_spans):
    """Never join authored spans across observations or reuse an earlier answer."""
    spans = list(model_spans)
    if not spans:
        return None
    action = parse_model_action(spans[-1])
    return action[1] if action is not None and action[0] == 'answer' else None


def extract_legacy_final_answer(text):
    """Preserve the legacy prompt-example convention while rejecting nesting.

    Full-transcript callers cannot establish authorship; masked model spans are
    preferred. The last answer must nevertheless be top-level and terminal.
    """
    answers = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if len(answers) <= 1:
        return None
    candidate = answers[-1]
    stack = []
    for tag in re.finditer(r'<(/?)(think|search|answer|information)>', text[:candidate.start()]):
        closing, kind = tag.groups()
        if closing:
            if not stack or stack[-1] != kind:
                return None
            stack.pop()
        else:
            stack.append(kind)
    if stack:
        return None
    return extract_final_answer([text[candidate.start():]])
