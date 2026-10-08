"""Complete local master/Adam equality at save and real FSDP reload.

Reads ordinary local shards only. No forward, RNG, optimizer or collective calls.
Checkpoint overhead is separate from the five training iteration measurements.
"""
import functools, hashlib, json, os
from pathlib import Path

def tree_hash(value):
    import torch
    from omegaconf import DictConfig, ListConfig
    from torch.distributed.tensor import DTensor
    digest=hashlib.sha256()
    byte_count=0
    def visit(item):
        nonlocal byte_count
        if isinstance(item,DTensor):raise TypeError('DTensor full reload digest requires a separate local-shard review')
        if torch.is_tensor(item):
            assert item.is_contiguous(),'Read-only hashing must not create full noncontiguous copies'
            digest.update(json.dumps(['tensor',str(item.dtype),list(item.shape)]).encode())
            raw=item.detach().reshape(-1).view(torch.uint8)
            for offset in range(0,raw.numel(),1048576):
                data=raw[offset:offset+1048576].cpu().numpy().tobytes()
                digest.update(data);byte_count+=len(data)
        elif isinstance(item,(dict,DictConfig)):
            digest.update(b'dict')
            for key in sorted(item,key=lambda key:str(key)):
                visit(str(key));visit(item[key])
        elif isinstance(item,(list,tuple,ListConfig)):
            digest.update(b'list')
            for element in item:visit(element)
        elif item is None or isinstance(item,(str,int,float,bool)):
            digest.update(json.dumps(item,allow_nan=False).encode())
        else:raise TypeError(f'Unsupported full reload digest type {type(item).__name__}')
    visit(value)
    return dict(sha256=digest.hexdigest(),tensor_bytes=byte_count)

def full_state_digest(manager):
    parameters={name:parameter for name,parameter in manager.model.named_parameters()}
    buffers={}
    for prefix,module in manager.model.named_modules():
        for name,value in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                buffers[prefix+'.'+name]=value
    assert all(str(value.dtype)=='torch.float32' for value in parameters.values()),'Expected actual FP32 master parameters'
    optimizer=manager.optimizer
    assert optimizer is not None and parameters
    groups=[];states=[]
    for group in optimizer.param_groups:
        groups.append({key:value for key,value in group.items() if key!='params'})
        states.append([optimizer.state.get(parameter,{}) for parameter in group['params']])
    return dict(model_master=tree_hash(parameters),persistent_buffers=tree_hash(buffers),optimizer=tree_hash(dict(groups=groups,states=states)),
        scheduler=tree_hash(manager.lr_scheduler.state_dict() if manager.lr_scheduler else None),
        parameter_count=len(parameters),master_dtypes=sorted({str(value.dtype) for value in parameters.values()}))

def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()

def install():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    from observe_worker import write_record
    if getattr(FSDPCheckpointManager,'_native_full_digest_installed',False):return
    original_save=FSDPCheckpointManager.save_checkpoint
    original_load=FSDPCheckpointManager.load_checkpoint
    @functools.wraps(original_save)
    def save(manager,local_path,*args,**kwargs):
        assert manager.should_save_model and manager.should_save_optimizer and manager.should_save_extra
        before=full_state_digest(manager)
        result=original_save(manager,local_path,*args,**kwargs)
        files={}
        for prefix in ['model','optim','extra_state']:
            path=Path(local_path)/f'{prefix}_world_size_{manager.world_size}_rank_{manager.rank}.pt'
            assert path.is_file() and path.stat().st_size>0
            files[path.name]=dict(sha256=file_hash(path),bytes=path.stat().st_size)
        value=dict(state=before,files=files,role=manager._native_observed_role,source_manifest_sha256=hashlib.sha256(Path(__file__).with_name('source-manifest.json').read_bytes()).hexdigest(),rank=manager.rank,world_size=manager.world_size,
            path=str(Path(local_path).resolve()),source_job=os.environ['SLURM_JOB_ID'])
        destination=Path(local_path)/f'native-integrity-rank-{manager.rank}.json'
        pending=destination.with_suffix('.pending')
        pending.write_text(json.dumps(value,allow_nan=False));pending.replace(destination)
        write_record('checkpoint_saved_full_digest',value)
        return result
    @functools.wraps(original_load)
    def load(manager,local_path,*args,**kwargs):
        if local_path is None:return original_load(manager,local_path,*args,**kwargs)
        assert manager.should_load_model and manager.should_load_optimizer and manager.should_load_extra
        expected=json.loads((Path(local_path)/f'native-integrity-rank-{manager.rank}.json').read_text())
        assert expected['rank']==manager.rank and expected['world_size']==manager.world_size
        assert expected['role']==manager._native_observed_role
        assert expected['source_manifest_sha256']==hashlib.sha256(Path(__file__).with_name('source-manifest.json').read_bytes()).hexdigest()
        for name,value in expected['files'].items():
            path=Path(local_path)/name
            assert path.stat().st_size==value['bytes'] and file_hash(path)==value['sha256']
        result=original_load(manager,local_path,*args,**kwargs)
        actual=full_state_digest(manager)
        assert actual==expected['state'],'Full local master/Adam/scheduler differ after actual FSDP load'
        write_record('checkpoint_loaded_full_digest',dict(role=manager._native_observed_role,rank=manager.rank,world_size=manager.world_size,
            path=str(Path(local_path).resolve()),state=actual,source_job=expected['source_job'],
            full_model_optimizer_scheduler_equal=True))
        return result
    FSDPCheckpointManager.save_checkpoint=save
    FSDPCheckpointManager.load_checkpoint=load
    FSDPCheckpointManager._native_full_digest_installed=True
