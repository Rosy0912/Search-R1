"""Same-process checkpoint roundtrip primitives; never a weights-only resume."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state(cuda_device=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=[] if cuda_device is None else
                [dict(device=int(cuda_device),state=torch.cuda.get_rng_state(cuda_device))])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda'])!=1 or state['cuda'][0]['device']!=torch.cuda.current_device():
            raise ValueError('Rank-local CUDA RNG device changed')
        torch.cuda.set_rng_state(state['cuda'][0]['state'],state['cuda'][0]['device'])


def state_digest(value):
    """Exact shard/content witness with DTensor global layout and mesh identity.

    Metadata and local bytes only: no redistribute/full_tensor/all-gather, RNG
    calls, or private mesh mutation. CPU-offloaded DTensors retain their mesh.
    """
    h = hashlib.sha256()
    def add(x):
        if hasattr(x, 'local_shards'):
            h.update(str(x.metadata()).encode())
            for shard in x.local_shards():
                h.update(str(shard.metadata).encode()); add(shard.tensor)
        elif hasattr(x, 'to_local') and hasattr(x, 'placements'):
            mesh=x.device_mesh
            h.update(b'DTENSOR_LAYOUT_V2')
            add(dict(global_shape=tuple(x.shape),global_stride=tuple(x.stride()),
                     global_dtype=str(x.dtype),placements=tuple(str(p) for p in x.placements),
                     mesh_device_type=mesh.device_type,mesh_ranks=mesh.mesh,
                     mesh_dim_names=mesh.mesh_dim_names,mesh_coordinate=mesh.get_coordinate()))
            add(x.to_local())
        elif torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape), tuple(x.stride()))).encode())
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast('B'))
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype),x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(('DICT:'+str(len(x))+':').encode())
            for k in sorted(x, key=lambda k: (str(type(k)),repr(k))): add(k);add(x[k])
        elif isinstance(x, (tuple,list)):
            h.update(str((type(x).__name__,len(x))).encode())
            for item in x:add(item)
        else:
            h.update(str((type(x).__name__,repr(x))).encode())
    add(value)
    return h.hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_save(value, path):
    """Exclusive immutable member publication; do not overwrite old checkpoints."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    owned_temporary=False
    try:
        with temporary.open('xb') as handle:
            owned_temporary=True
            torch.save(value,handle);handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)
    finally:
        if owned_temporary and temporary.exists():temporary.unlink()


class StatefulSampler(torch.utils.data.Sampler):
    """Explicit epoch permutation/cursor; num_workers=0 avoids unrecorded prefetch."""
    def __init__(self, data_source, seed=None, shuffle=True):
        self.size=data_source if type(data_source) is int else len(data_source);self.shuffle=shuffle;self.epoch=0;self.cursor=0;self.consumed_cursor=0
        if self.size < 1 or type(self.shuffle) is not bool: raise ValueError('Invalid sampler dataset/shuffle')
        self.seed=seed;self.generator=torch.Generator().manual_seed(0 if seed is None else seed);self.order=None
    def __iter__(self):
        if self.order is None:
            if self.shuffle and self.seed is None:
                # Match torch RandomSampler(generator=None), preserving prior data order.
                self.generator.manual_seed(int(torch.empty((),dtype=torch.int64).random_().item()))
            self.order=torch.randperm(self.size,generator=self.generator).tolist() if self.shuffle else list(range(self.size))
        while self.cursor < self.size:
            i=self.order[self.cursor];self.cursor+=1;yield i
        if self.shuffle:
            # RandomSampler consumes a second empty-slice permutation at epoch end.
            torch.randperm(self.size,generator=self.generator)
        self.epoch+=1;self.cursor=0;self.consumed_cursor=0;self.order=None
    def __len__(self):return self.size
    def commit_consumed(self, count):
        if type(count) is not int or count < 1 or self.consumed_cursor + count > self.cursor:
            raise ValueError('Consumed cursor exceeds emitted rows')
        self.consumed_cursor += count
    def state_dict(self, strict=True):
        if strict and self.consumed_cursor != self.cursor:
            raise ValueError('Outstanding emitted rows at checkpoint')
        return dict(size=self.size,shuffle=self.shuffle,seed=self.seed,epoch=self.epoch,cursor=self.cursor,consumed_cursor=self.consumed_cursor,
                    order=None if self.order is None else list(self.order),generator=self.generator.get_state())
    def load_state_dict(self, state):
        if state['size']!=self.size or state['shuffle']!=self.shuffle or state['seed']!=self.seed:
            raise ValueError('Sampler dataset or shuffle changed')
        if type(state['cursor']) is not int or type(state['consumed_cursor']) is not int or not 0<=state['consumed_cursor']==state['cursor']<=self.size:raise ValueError('Invalid sampler cursor')
        order=state['order']
        if order is not None and (len(order)!=self.size or set(order)!=set(range(self.size))):
            raise ValueError('Invalid sampler permutation')
        if type(state['epoch']) is not int or state['epoch'] < 0: raise ValueError('Invalid sampler epoch')
        if order is None and state['cursor'] != 0: raise ValueError('Cursor has no permutation')
        self.epoch=state['epoch'];self.cursor=state['cursor'];self.consumed_cursor=state['consumed_cursor'];self.order=order
        self.generator.set_state(state['generator'])


def resolve_warmup_steps(optim_config):
    explicit=optim_config.get('lr_warmup_steps')
    if explicit is None: explicit=optim_config.get('num_warmup_steps')
    steps=int(optim_config.get('lr_warmup_steps_ratio',0.)*optim_config.get('total_training_steps',0)) if explicit is None else explicit
    if type(steps) is not int or steps<0:raise ValueError('num_warmup_steps must be a nonnegative integer')
    return steps


TEACHER_COUNTERS=('cpu_seconds','wall_seconds','transfer_wall_seconds','cuda_update_entries',
                 'cuda_seconds','scored_tokens','load_cpu_seconds','load_wall_seconds')


def teacher_state(teacher, asset_path, student_optimizer=None):
    """Fixed teacher identity/content plus actual counters, never BAPO state."""
    import math
    teacher._flush_cuda_events()
    teacher._validate_execution('cpu')
    if teacher.device.type != 'cpu' or teacher.model.training:
        raise ValueError('State boundary requires idle eval teacher on CPU')
    if any(p.requires_grad or p.grad is not None for p in teacher.model.parameters()):
        raise ValueError('Teacher is not frozen')
    if student_optimizer is not None:
        identities={id(p) for p in teacher.model.parameters()}
        if any(id(p) in identities for group in student_optimizer.param_groups for p in group['params']):
            raise ValueError('Frozen teacher entered student optimizer')
    directory=Path(asset_path).resolve()
    files={}
    for path in sorted(directory.iterdir()):
        if path.is_symlink():raise ValueError('Teacher assets contain symlink')
        if path.is_file():files[path.name]=file_digest(path)
    if not files:raise ValueError('Teacher asset content identity missing')
    counters={name:getattr(teacher,name) for name in TEACHER_COUNTERS}
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in counters.values()):
        raise ValueError('Invalid actual teacher counters')
    return dict(version=1,asset_directory=str(directory),asset_files=files,
                model_content_digest=state_digest(dict(teacher.model.named_parameters())),
                buffers={name:value.detach().cpu().clone() for name,value in teacher.model.named_buffers()},
                buffer_content_digest=state_digest(dict(teacher.model.named_buffers())),
                caches={name:{key:int(getattr(module,key)) for key in ('max_seq_len_cached','original_max_seq_len') if hasattr(module,key)} for name,module in teacher.model.named_modules() if any(hasattr(module,key) for key in ('max_seq_len_cached','original_max_seq_len'))},
                parameter_count=teacher.parameter_count,model_bytes=teacher.model_bytes,
                execution_device=str(teacher.execution_device),storage_device=str(teacher.device),
                model_config=teacher.model.config.to_dict(),counters=counters)


def restore_teacher_state(teacher, asset_path, saved, student_optimizer=None):
    current=teacher_state(teacher,asset_path,student_optimizer)
    dynamic={'counters','buffers','buffer_content_digest','caches'}
    if {k:v for k,v in current.items() if k not in dynamic} != {k:v for k,v in saved.items() if k not in dynamic}:
        raise ValueError('Fixed teacher identity/content/device changed')
    existing=dict(teacher.model.named_buffers())
    if set(existing) != set(saved['buffers']):raise ValueError('Teacher buffer schema changed')
    persistent=set(teacher.model.state_dict())
    for name,value in saved['buffers'].items():
        parent,_,key=name.rpartition('.')
        module=teacher.model.get_submodule(parent) if parent else teacher.model
        module.register_buffer(key,value.detach().clone().to('cpu'),persistent=name in persistent)
    for name,cache in saved['caches'].items():
        module=teacher.model.get_submodule(name) if name else teacher.model
        for key,value in cache.items():setattr(module,key,value)
    if set(saved['counters']) != set(TEACHER_COUNTERS):raise ValueError('Teacher counter schema changed')
    for name,value in saved['counters'].items():setattr(teacher,name,value)
    if state_digest(teacher_state(teacher,asset_path,student_optimizer)) != state_digest(saved):
        raise ValueError('Teacher counters did not restore exactly')


def recovery_core(payload):
    """Exclude ONLY runtime identity; all algorithm/state/layout bytes remain."""
    if not {'pid', 'object_ids'}.issubset(payload):
        raise ValueError('Missing full-state runtime identity')
    return {key: value for key, value in payload.items() if key not in ('pid', 'object_ids')}


def configuration_digest(config):
    from omegaconf import OmegaConf
    def plain(value):
        if isinstance(value, dict): return {key:plain(item) for key,item in value.items()}
        if isinstance(value,(tuple,list)): return [plain(item) for item in value]
        if hasattr(value,'__dict__'): return plain(vars(value))
        return value
    return state_digest(OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else plain(config))


RECOVERY_SOURCE_MODULES = (
    'search_r1.baselines.full_state', 'search_r1.baselines.distillation',
    'search_r1.baselines.controller', 'search_r1.baselines.rollout',
    'search_r1.baselines.advantages', 'search_r1.baselines.bapo',
    'search_r1.baselines.sharedholdout',
    'verl.workers.actor.dp_actor', 'verl.trainer.ppo.core_algos',
    'verl.utils.fsdp_utils', 'verl.utils.torch_functional', 'verl.protocol',
)


def recovery_source_closure(worker_file, payload_method, checker_record=None, rank=None):
    """Actual import paths/bytes plus Git commit; no GPU/model operations.

    Explicitly imports the small pure source modules so save/reload share the
    same dependency closure even when controller hooks have not run on a rank.
    The full driver/runtime authorization is a separate outer responsibility.
    """
    import importlib
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[2]
    worker=Path(worker_file).resolve()
    expected_worker=root/'verl/workers/fsdp_workers.py'
    if worker != expected_worker or Path(payload_method.__code__.co_filename).resolve() != worker:
        raise ValueError('Executing worker source path differs')
    files={}
    for name in RECOVERY_SOURCE_MODULES:
        module=importlib.import_module(name)
        actual=Path(module.__file__).resolve()
        expected=root.joinpath(*name.split('.')).with_suffix('.py')
        if actual != expected or not actual.is_file():
            raise ValueError('Actual recovery source import origin differs: '+name)

    # Include already imported repo dependencies (including package initializers),
    # never unrelated tests/third-party modules or unexecuted GPU modules.
    for name,module in sorted(list(sys.modules.items())):
        if name not in ('search_r1','verl') and not name.startswith(('search_r1.','verl.')):
            continue
        location=getattr(module,'__file__',None)
        if location is None:
            raise ValueError('Recovery source dependency has no concrete origin: '+name)
        actual=Path(location).resolve()
        expected=root.joinpath(*name.split('.'))
        expected=expected/'__init__.py' if hasattr(module,'__path__') else expected.with_suffix('.py')
        if actual!=expected or not actual.is_file():
            raise ValueError('Recovery dependency import origin differs: '+name)

    # Deterministic complete project source inventory, independent of imports
    # performed by driver versus rank. Live origins above still fail closed.
    for namespace in ('search_r1','verl'):
        for path in sorted((root/namespace).rglob('*.py')):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError('Recovery source inventory escapes physical repository')
            relative=str(path.relative_to(root))
            files[relative]=file_digest(path)
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if len(commit)!=40 or any(ch not in '0123456789abcdef' for ch in commit):
        raise ValueError('Invalid actual Git source commit')
    if checker_record is not None:
        verify_rank_import_checker(root, checker_record, rank)
    return dict(version=1,git_source_commit=commit,files=files)


def verify_rank_import_checker(root, record, rank):
    """Run externally pinned source bytes on this rank, never trust driver proof."""
    import json
    import re
    import sys
    from omegaconf import OmegaConf
    from .sharedholdout import unique_json
    record=OmegaConf.to_container(record,resolve=True) if OmegaConf.is_config(record) else dict(record)
    required={'checker_path','checker_sha256','closure_path','closure_sha256'}
    if set(record)!=required or not all(isinstance(value,str) for value in record.values()):
        raise ValueError('Invalid rank source checker configuration')
    for key in ('checker_sha256','closure_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',record[key]):
            raise ValueError('Invalid external rank checker SHA')
    root=Path(root).resolve()
    checker=Path(record['checker_path']);closure_path=Path(record['closure_path'])
    expected=root/'reproduction/baselines/e1_import_bootstrap.py'
    if checker!=expected or checker.resolve(strict=True)!=checker or not checker.is_file():
        raise ValueError('Rank checker actual source path differs')
    if (not closure_path.is_absolute() or closure_path.resolve(strict=True)!=closure_path
            or not closure_path.is_file() or closure_path.stat().st_uid!=os.getuid()):
        raise ValueError('Rank closure must be an owned physical external file')
    source=checker.read_bytes();raw=closure_path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=record['checker_sha256'] or hashlib.sha256(raw).hexdigest()!=record['closure_sha256']:
        raise ValueError('Rank external checker/closure SHA differs')
    closure=unique_json(raw)
    expected_record={'path':str(checker),'sha256':record['checker_sha256']}
    if closure.get('modules',{}).get('reproduction.baselines.e1_import_bootstrap')!=expected_record:
        raise ValueError('External closure does not pin its rank checker source')
    namespace={'__name__':'e1_rank_runtime_import_checker','__file__':str(checker)}
    exec(compile(source,str(checker),'exec',dont_inherit=True,optimize=sys.flags.optimize),namespace)
    actual,count=namespace['verify_imports'](root,closure,required=())
    if file_digest(checker)!=record['checker_sha256'] or file_digest(closure_path)!=record['closure_sha256']:
        raise ValueError('Rank source proof changed during verification')
    audit=dict(version=1,pid=os.getpid(),uid=os.getuid(),rank=rank,
        checker=expected_record,external_closure_sha256=record['closure_sha256'],
        verified_function_count=count,actual_modules=actual['modules'])
    print('E1_RANK_ACTUAL_IMPORT_CLOSURE_JSON='+json.dumps(audit,sort_keys=True),flush=True)
    return audit
