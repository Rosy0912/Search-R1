"""Exclusive, lossless recovery rollout evidence, published before optimization."""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from .training_state import atomic_save, cpu_copy, file_sha


def plain(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


def save_rollouts(batch, tokenizer, step, directory):
    """All leaves, full action/tool traces, and actual actor loss inputs.

    No sampling or truncation; publication failure aborts before actor update.
    Existing evidence is never replaced. Manifest means both payloads exist.
    """
    directory=Path(directory)/f'outer-{step:06d}'
    directory.mkdir(parents=True,exist_ok=False)
    identity=dict(outer_step=int(step),job_id=os.environ.get('SLURM_JOB_ID'),
        source=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
        config=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
        data=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'))
    tensors={k:cpu_copy(v) for k,v in batch.batch.items()}
    required={'input_ids','responses','attention_mask','info_mask','old_log_probs',
              'ref_log_prob','advantages','policy_weights','token_level_rewards','loss_mask'}
    if not required.issubset(tensors):
        raise ValueError('Incomplete recovery actor evidence: '+str(required-set(tensors)))
    rows=len(batch)
    metadata=plain(batch.non_tensor_batch)
    payload=dict(identity=identity,tensors=tensors,non_tensor_batch=metadata,
                 meta_info=plain(batch.meta_info),rows=rows)
    tensor_path=directory/'actor-inputs.pt'
    tensor_sha=atomic_save(payload,tensor_path)
    text_path=directory/'trajectories.jsonl.gz'
    with text_path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            for i in range(rows):
                tokens=tensors['input_ids'][i][tensors['attention_mask'][i].bool()].tolist()
                record=dict(identity=identity,row=i,
                    metadata={k:v[i] for k,v in metadata.items()},
                    token_ids=tokens,text=tokenizer.decode(tokens,skip_special_tokens=False),
                    response_token_ids=tensors['responses'][i].tolist(),
                    reward=float(tensors['token_level_rewards'][i].sum()))
                stream.write((json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode())
        raw.flush();os.fsync(raw.fileno())
    receipt=dict(status='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED',identity=identity,
        rows=rows,payloads={tensor_path.name:tensor_sha,text_path.name:file_sha(text_path)},
        scope='Complete all-leaf pre-optimizer inputs and action/tool traces; not optimizer replay equivalence')
    atomic_save(receipt,directory/'manifest.json',json_format=True)
    return receipt
