import hashlib,json,pathlib

def conditional_base64(binding):
    resultpath=pathlib.Path(binding['result_path']);qapath=pathlib.Path(binding['qa_path'])
    raw=resultpath.read_bytes();qa=qapath.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['result_sha256'] or hashlib.sha256(qa).hexdigest()!=binding['qa_sha256']:raise ValueError('base64 raw bytes binding mismatch')
    result=json.loads(raw);rows=[json.loads(line) for line in qa.splitlines()]
    if result['status']!='COMPLETE64_UNREVIEWED_NO_TRAINING' or result['qa_sha256']!=binding['qa_sha256']:raise ValueError('incomplete base64 result')
    if len(rows)!=64 or len({x['uid'] for x in rows})!=64 or [x['sample_position'] for x in rows]!=list(range(64)):raise ValueError('base64 identity/order mismatch')
    if any(x['em'] not in (0,1) for x in rows):raise ValueError('nonbinary base64 EM')
    correct=sum(int(x['em']) for x in rows)
    if correct>=26 or result['quality']['correct']!=correct or result['quality']['next_path']!='SEARCH_G1_GRPO100':raise ValueError('G1 conditional branch not selected')
    return {'scope':'RAW_BASE64_ROUTING_ONLY_NOT_ALGORITHM_APPROVAL','correct':correct,'questions':64,'result_sha256':binding['result_sha256'],'qa_sha256':binding['qa_sha256']}
