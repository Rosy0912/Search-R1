"""Existing response-token alignment; no generation token is supervision."""
import math

def pack(prompts,responses,attention,pad,max_length):
    rows=[];indices=[]
    for prompt,response,mask in zip(prompts,responses,attention):
        prefix=[int(t) for t in prompt if int(t)!=pad]
        if not prefix:raise ValueError('Empty prediction prefix')
        selected=[int(t) for t,valid in zip(response,mask) if valid]
        row=prefix+selected
        if len(row)+1>max_length:raise ValueError('Teacher context exceeds max; no truncation')
        rows.append(row);indices.append([len(prefix)+j for j in range(len(selected))])
    return rows,indices

def unpack(outputs,rows,indices,responses,attention):
    if len(outputs)!=len(rows):raise ValueError('Teacher row count mismatch')
    values=[]
    for out,row,positions,response,mask in zip(outputs,rows,indices,responses,attention):
        if list(out.prompt_token_ids)!=row or len(out.prompt_logprobs)!=len(row):raise ValueError('Prompt order/alignment changed')
        scored=[]
        for position in positions:
            token=row[position];entry=out.prompt_logprobs[position]
            if entry is None or token not in entry:raise ValueError('Actual prompt token logprob missing')
            value=float(entry[token].logprob)
            if not math.isfinite(value):raise ValueError('Nonfinite actual token logprob')
            scored.append(value)
        cursor=iter(scored);values.append([next(cursor) if valid else 0. for valid in mask])
    return values
