"""Bound foreach AdamW temporaries without changing per-parameter updates."""
HELPER='''def _e1_optimizer_step(optimizer):
    chunk = int(os.environ.get('E1_OPTIMIZER_CHUNK', '0'))
    if chunk < 0:
        raise ValueError('Invalid optimizer parameter chunk')
    if not chunk:
        if os.environ.get('E1_STAGE_CHUNK_STATES') == '1':
            raise ValueError('State staging requires a positive optimizer chunk')
        return optimizer.step()
    original = optimizer.param_groups
    print('E1_OPTIMIZER_MEMORY_START', dict(rank=torch.distributed.get_rank(), chunk=chunk, staged=os.environ.get('E1_STAGE_CHUNK_STATES'), allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()), flush=True)
    groups = []
    for group in original:
        parameters = group['params']
        for start in range(0, len(parameters), chunk):
            piece = dict(group)
            piece['params'] = parameters[start:start + chunk]
            groups.append(piece)
    torch.cuda.empty_cache()
    staged = os.environ.get('E1_STAGE_CHUNK_STATES') == '1'
    if staged and os.environ.get('E1_JIT_OPTIMIZER') != '1':
        raise ValueError('Chunk state staging requires JIT offload')
    try:
        if staged:
            for piece in groups:
                optimizer.param_groups = [piece]
                load_fsdp_optimizer(optimizer, torch.cuda.current_device())
                try:
                    optimizer.step()
                finally:
                    offload_fsdp_optimizer(optimizer)
                    torch.cuda.synchronize()
        else:
            optimizer.param_groups = groups
            return optimizer.step()
    finally:
        optimizer.param_groups = original
        print('E1_OPTIMIZER_MEMORY_END', dict(rank=torch.distributed.get_rank(), allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()), flush=True)


'''
def patch_optimizer_chunks(source):
 assert 'def _e1_optimizer_step(' not in source
 assert source.count('            self.actor_optimizer.step()')==1
 source='import os\n'+source
 source=source.replace('class DataParallelPPOActor(',HELPER+"print('E1_EXECUTION_OVERLAY_ACTIVE', {key: os.environ.get(key) for key in ['E1_PHYSICAL_MICRO', 'E1_PHYSICAL_SCORE', 'E1_JIT_OPTIMIZER', 'E1_LOGIT_CHUNK', 'E1_OPTIMIZER_CHUNK', 'E1_STAGE_CHUNK_STATES']}, flush=True)\n\nclass DataParallelPPOActor(",1)
 source=source.replace('            self.actor_optimizer.step()','            _e1_optimizer_step(self.actor_optimizer)',1)
 # Release unused activation cache before loading moment states, not only after.
 needle='            load_fsdp_optimizer(self.actor_optimizer, torch.cuda.current_device())'
 assert source.count(needle)==1
 source=source.replace(needle,"            if int(os.environ.get('E1_OPTIMIZER_CHUNK', '0')) > 0:\n                torch.cuda.empty_cache()\n            if os.environ.get('E1_STAGE_CHUNK_STATES') != '1':\n    "+needle,1)
 if '                for data in pieces:' in source:
  source=source.replace('                for data in pieces:',"                for data in pieces:\n                    if os.environ.get('E1_STAGE_CHUNK_STATES') == '1':\n                        torch.cuda.empty_cache()",1)
 return source
