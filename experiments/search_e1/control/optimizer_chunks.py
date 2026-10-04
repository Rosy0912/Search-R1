"""Bound foreach AdamW temporaries without changing per-parameter updates."""
HELPER='''def _e1_optimizer_step(optimizer):
    chunk = int(os.environ.get('E1_OPTIMIZER_CHUNK', '0'))
    if chunk < 0:
        raise ValueError('Invalid optimizer parameter chunk')
    if not chunk:
        return optimizer.step()
    original = optimizer.param_groups
    groups = []
    for group in original:
        parameters = group['params']
        for start in range(0, len(parameters), chunk):
            piece = dict(group)
            piece['params'] = parameters[start:start + chunk]
            groups.append(piece)
    torch.cuda.empty_cache()
    optimizer.param_groups = groups
    try:
        return optimizer.step()
    finally:
        optimizer.param_groups = original


'''
def patch_optimizer_chunks(source):
 assert 'def _e1_optimizer_step(' not in source
 assert source.count('            self.actor_optimizer.step()')==1
 source='import os\n'+source
 source=source.replace('class DataParallelPPOActor(',HELPER+'class DataParallelPPOActor(',1)
 source=source.replace('            self.actor_optimizer.step()','            _e1_optimizer_step(self.actor_optimizer)',1)
 # Release unused activation cache before loading moment states, not only after.
 needle='            load_fsdp_optimizer(self.actor_optimizer, torch.cuda.current_device())'
 assert source.count(needle)==1
 return source.replace(needle,"            if int(os.environ.get('E1_OPTIMIZER_CHUNK', '0')) > 0:\n                torch.cuda.empty_cache()\n"+needle,1)
