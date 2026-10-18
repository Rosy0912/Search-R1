"""Read exact names through both valid-name lookup and verified list filtering."""
import json

CONTROL = 'zhangjunming-ttt-b20-150-r1-54c0'
CONTROL_UID = '33ac6bbf-7e6a-4a66-bcc6-261b66371975'


def exact_list(client, name):
    result = client.list_training_jobs(filter_str='name = ' + json.dumps(name), page_size=100)
    # The inspected protobuf-JSON response omits all three zero-valued fields:
    # a complete empty exact-name result is {}. Reject unknown response shapes.
    if not isinstance(result, dict) or set(result) - {'training_jobs', 'total_size', 'next_page_token'}:
        raise ValueError('Unknown training-job list response')
    jobs = result.get('training_jobs', [])
    if (not isinstance(jobs, list) or result.get('next_page_token')
            or result.get('total_size', 0) != len(jobs)
            or any(job.get('name') != name for job in jobs)):
        raise ValueError('Exact name filter/pagination not independently established')
    return jobs


def prove_absent(client, names):
    control = exact_list(client, CONTROL)
    if len(control) != 1 or control[0]['uid'] != CONTROL_UID:
        raise ValueError('Positive control for exact-name filtering failed')
    proof = {}
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('Expected the exact three current and three legacy names')
    for name in names:
        if exact_list(client, name):
            raise ValueError('Existing exact allocation; inspect, never replay: ' + name)
        if len(name) <= 32:
            try:
                client.get_training_job(name)
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
                    raise
            else:
                raise ValueError('Exact get found an allocation: ' + name)
            proof[name] = 'EXACT_FILTER_ZERO_AND_GET_404'
        else:
            # A 400 name-validation error is never treated as evidence of absence.
            proof[name] = 'EXACT_FILTER_ZERO_LEGACY_NAME_EXCEEDS_GET_LIMIT32'
    return {'positive_control_name': CONTROL, 'positive_control_uid': CONTROL_UID,
            'exact_jobs': proof}
