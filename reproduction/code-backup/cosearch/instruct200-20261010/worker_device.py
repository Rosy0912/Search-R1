"""Bind the actual CUDA device to Ray's single assigned, leased GPU."""
import json,os

def select(assigned,mapping,visible):
    if len(assigned)!=1:raise RuntimeError('Worker must have one Ray-assigned GPU')
    value=str(assigned[0])
    matches=[u for u,i in mapping.items() if value==u or value==str(i)]
    if len(matches)!=1:raise RuntimeError('Ray GPU is not uniquely leased')
    if visible.count(matches[0])!=1:raise RuntimeError('Assigned GPU not uniquely visible to CUDA')
    return visible.index(matches[0]),matches[0]

def bind():
    import ray,torch
    mapping=json.loads(os.environ['COSEARCH_LEASED_UUID_TO_INDEX'])
    leased=set(os.environ['COSEARCH_LEASED_UUIDS'].split(','))
    if set(mapping)!=leased or len(leased)!=4:raise RuntimeError('Invalid four-GPU lease')
    assigned=ray.get_runtime_context().get_accelerator_ids().get('GPU',[])
    visible=[str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
    if not set(visible)<=leased:raise RuntimeError('Foreign GPU visible to worker')
    ordinal,uuid=select(assigned,mapping,visible)
    torch.cuda.set_device(ordinal)
    if str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)!=uuid:
        raise RuntimeError('Actual CUDA binding differs from Ray assignment')
    return {'ray_assigned':assigned,'visible_uuids':visible,'ordinal':ordinal,'uuid':uuid,'mask':os.environ.get('CUDA_VISIBLE_DEVICES')}
