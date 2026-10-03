"""Shared calibration/inference provenance and strict probe wire contract."""
import math
import re

ANSWER = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
SEARCH = re.compile(r'<search>(.*?)</search>', re.DOTALL)


def authored_match(response, model_ranges):
    last = None
    previous = 0
    for start, end in model_ranges:
        if not (isinstance(start,int) and isinstance(end,int) and previous <= start <= end <= len(response)):
            raise ValueError('Invalid authored span bounds')
        previous = end
        for match in ANSWER.finditer(response[start:end]):
            last = (start + match.start(), start + match.end(),
                    start + match.start(1), start + match.end(1), match.group(1).strip())
    return last


def contexts(prompt, response, model_ranges):
    match = authored_match(response, model_ranges)
    if match is None or not match[4]:
        return None
    begin, finish, answer_start, answer_end, answer = match
    return dict(answer=answer, commit=prompt + response[:answer_start],
                answer_state=prompt + response[:answer_end],
                prefix=response[:answer_start], start=begin, end=finish)


def model_search_count(response, model_ranges):
    return sum(len(SEARCH.findall(response[start:end])) for start,end in model_ranges)


def decode_segments(tokenizer, ids, mask):
    # Decode each complete provenance segment and retain its character bounds.
    # No answer or search tag can straddle an observation boundary.
    ids = ids.tolist() if hasattr(ids,'tolist') else ids
    mask = mask.tolist() if hasattr(mask,'tolist') else mask
    if len(ids) != len(mask):
        raise ValueError('Response/mask length mismatch')
    text = ''; ranges = []; start = 0
    while start < len(ids):
        end = start + 1
        while end < len(ids) and bool(mask[end]) == bool(mask[start]):
            end += 1
        segment = tokenizer.decode(ids[start:end]); offset = len(text);text += segment
        if mask[start]:ranges.append([offset,len(text)])
        start=end
    return text,ranges


def validate_probe_reply(payload, items):
    rows = payload.get('results') if isinstance(payload,dict) else None
    version = payload.get('version') if isinstance(payload,dict) else None
    if not isinstance(version,str) or not version or not isinstance(rows,list) or len(rows)!=len(items):
        raise ValueError('Missing probe version or incomplete result count')
    for row,item in zip(rows,items):
        if not isinstance(row,dict) or row.get('id')!=item['id'] or row.get('version')!=version:
            raise ValueError('Probe row identity/version mismatch')
        for key in ('flip','boundary'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid finite probe probability: '+key)
    return rows,version
