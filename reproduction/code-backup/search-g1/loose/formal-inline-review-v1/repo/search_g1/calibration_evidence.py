"""Typed calibration context: delete retrieved evidence, retain corrections."""


def invalid_action_feedback():
    return ('\nMy previous action is invalid. '
            'If I want to search, I should put the query between <search> and </search>. '
            'If I want to give the final answer, I should put the answer between '
            '<answer> and </answer>. Let me try again.\n')


def validate_provenance(response, model_ranges, evidence_ranges, correction_ranges):
    """Require a disjoint, complete partition of the generated interaction."""
    spans = []
    for kind, ranges in [('model', model_ranges), ('evidence', evidence_ranges),
                         ('correction', correction_ranges)]:
        previous = 0
        for pair in ranges:
            if len(pair) != 2:
                raise ValueError('Invalid provenance pair')
            start, end = pair
            if not (type(start) is int and type(end) is int and
                    previous <= start <= end <= len(response)):
                raise ValueError('Invalid ' + kind + ' bounds')
            previous = end
            if start < end:
                spans.append((start, end, kind))
    cursor = 0
    for start, end, _ in sorted(spans):
        if start != cursor:
            raise ValueError('Overlapping or missing provenance segment')
        cursor = end
    if cursor != len(response):
        raise ValueError('Unattributed interaction suffix')


def evidence_deleted_prefix(response, end, model_ranges, evidence_ranges,
                            correction_ranges):
    """Keep all reasoning/corrections before the answer payload verbatim."""
    validate_provenance(response, model_ranges, evidence_ranges, correction_ranges)
    if type(end) is not int or not 0 <= end <= len(response):
        raise ValueError('Invalid answer prefix boundary')
    parts = []
    cursor = 0
    for start, finish in evidence_ranges:
        if start >= end:
            break
        parts.extend([response[cursor:start], '\n<information></information>\n'])
        cursor = min(finish, end)
    parts.append(response[cursor:end])
    return ''.join(parts)
