"""HF greedy batching for an already admitted unmerged FP32 LoRA model.

No Torch import in CPU parent. Padding does not add fake repeated EOS tokens.
All scalar mathematics/model parameters remain those of native HF generation.
"""
from contextlib import contextmanager

class PaddingAwareRepetition:
    def __init__(self,processor,width,offsets):self.processor,self.width,self.offsets=processor,width,offsets
    def __call__(self,input_ids,scores):
        import torch
        safe=input_ids.clone();offsets=torch.tensor(self.offsets,device=input_ids.device)
        first=input_ids[torch.arange(len(self.offsets),device=input_ids.device),offsets]
        pad=torch.arange(self.width,device=input_ids.device)[None,:]<offsets[:,None]
        safe[:,:self.width]=torch.where(pad,first[:,None],safe[:,:self.width])
        # Repeating an already present actual token leaves the native set penalty
        # unchanged; input_ids themselves and model attention are never changed.
        return self.processor(safe,scores)

@contextmanager
def padding_processors(model,width,offsets,record=None):
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor,LogitNormalization
    owner=model.get_base_model() if hasattr(model,'get_base_model') else model
    old=owner._get_logits_processor;own=owner.__dict__.get('_get_logits_processor')
    def wrapped(*args,**kwargs):
        processors=old(*args,**kwargs)
        for i,p in enumerate(processors):
            if isinstance(p,RepetitionPenaltyLogitsProcessor):
                if p.penalty!=1.05:raise ValueError('Native repetition penalty changed')
                processors[i]=PaddingAwareRepetition(p,width,offsets)
            elif not isinstance(p,LogitNormalization):
                raise ValueError('Unvalidated position-dependent processor: '+type(p).__name__)
        if record:record({'kind':'actual_logits_processors','types':[type(p).__name__ for p in processors],
            'padding_tokens_not_counted_as_repetition':True,'original_penalty':1.05})
        return processors
    owner._get_logits_processor=wrapped
    try:yield
    finally:
        if own is None:del owner._get_logits_processor
        else:owner._get_logits_processor=own

class BatchStopTags:
    def __init__(self,tokenizer,start,count):
        self.tokenizer,self.start=tokenizer,start;self.ends=[None]*count;self.pieces={}
        decoder=getattr(getattr(tokenizer,'backend_tokenizer',None),'decoder',None)
        self.byte_level=decoder is not None and type(decoder).__name__=='ByteLevel'
    def __call__(self,input_ids,scores,**kwargs):
        import torch
        last=input_ids[:,-1].detach().cpu().tolist()
        for i,token in enumerate(last):
            if self.ends[i] is not None:continue
            if self.byte_level:
                if token not in self.pieces:self.pieces[token]=self.tokenizer.decode([token],skip_special_tokens=True)
                # ASCII '>' cannot appear through UTF-8 byte joining unless the
                # newly appended token contains it. Decode full prefix on candidates.
                if '>' not in self.pieces[token]:continue
            text=self.tokenizer.decode(input_ids[i,self.start:],skip_special_tokens=True)
            if '</search>' in text or '</answer>' in text:self.ends[i]=input_ids.shape[1]-self.start
        return torch.tensor([n is not None for n in self.ends],dtype=torch.bool,device=input_ids.device)

def greedy_batch(model,tokenizer,texts,max_new_tokens=500,record=None):
    import torch
    from transformers import StoppingCriteriaList
    cfg=model.generation_config
    if cfg.repetition_penalty!=1.05 or cfg.eos_token_id!=[151645,151643] or cfg.num_beams!=1:
        raise ValueError('Original native greedy generation config changed')
    device=next(model.parameters()).device
    sequences=[tokenizer.encode(text,add_special_tokens=False) for text in texts]
    if not sequences or any(not ids for ids in sequences):raise ValueError('Empty generation batch/context')
    width=max(map(len,sequences));offsets=[width-len(ids) for ids in sequences];pad=tokenizer.eos_token_id
    ids=torch.tensor([[pad]*n+v for n,v in zip(offsets,sequences)],dtype=torch.long,device=device)
    mask=torch.tensor([[0]*n+[1]*len(v) for n,v in zip(offsets,sequences)],dtype=torch.long,device=device)
    stops=BatchStopTags(tokenizer,width,len(texts));options=dict(max_new_tokens=max_new_tokens,do_sample=False,
        pad_token_id=pad,stopping_criteria=StoppingCriteriaList([stops]))
    model.eval()
    with torch.no_grad(),padding_processors(model,width,offsets,record):
        generated=model.generate(ids,attention_mask=mask,**options)
    results=[]
    for i,row in enumerate(generated):
        tokens=row[width:].detach().cpu().tolist()
        limit=stops.ends[i]
        eos=next((j+1 for j,v in enumerate(tokens) if v in cfg.eos_token_id),None)
        if eos is not None:limit=eos if limit is None else min(limit,eos)
        if limit is not None:tokens=tokens[:limit]
        results.append(dict(text=tokenizer.decode(tokens,skip_special_tokens=True),token_ids=tokens,
            padding_width=width,prompt_tokens=len(sequences[i]),max_new_tokens=max_new_tokens,
            actual_generation_scores_saved=False,unmerged_FP32_LoRA=True))
    return results
