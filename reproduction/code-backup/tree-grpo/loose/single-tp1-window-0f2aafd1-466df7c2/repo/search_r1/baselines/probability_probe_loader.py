"""Original, source-bound HF/FSDP model construction with no optimizer tail.

This module is source-only until a separate selection/runtime package is reviewed.
It never imports fsdp_workers, starts Ray, constructs an optimizer or saves weights.
"""
import ast
import copy
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch

from .probability_probe_capture import file_sha, require, tensor_digest
from .probability_probe_io import safetensor_verified


def split_original_constructor(path, expected_sha):
    """Extract original statements; fail rather than guess a changed call boundary.

    CPU HF loading is separated before the original barrier so a local load failure
    can be MAX-synchronized before any peer enters the original FSDP collectives.
    All statements from that barrier through FSDP construction are kept unchanged.
    The optimizer/scheduler conditional and following tail are never executed.
    """
    path=Path(path)
    require(file_sha(path)==expected_sha,'Original FSDP source SHA changed')
    tree=ast.parse(path.read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker']
    require(len(classes)==1,'Original worker class is ambiguous')
    methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer']
    require(len(methods)==1,'Original construction entry is ambiguous')
    method=methods[0]
    barriers=[i for i,n in enumerate(method.body) if isinstance(n,ast.Expr)
        and ast.dump(n.value,include_attributes=False)==ast.dump(ast.parse('torch.distributed.barrier()').body[0].value,include_attributes=False)]
    boundaries=[i for i,n in enumerate(method.body) if isinstance(n,ast.If)
        and ast.dump(n.test,include_attributes=False)==ast.dump(ast.parse('self._is_actor',mode='eval').body,include_attributes=False)
        and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='AdamW' for x in ast.walk(n))]
    require(len(barriers)==len(boundaries)==1 and barriers[0]<boundaries[0],'Cannot establish original barrier/optimizer boundary')
    first,last=barriers[0],boundaries[0]
    prefix=copy.deepcopy(method);prefix.name='original_hf_prefix';prefix.decorator_list=[]
    prefix.body=copy.deepcopy(method.body[:first])+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
    suffix=ast.Module(body=copy.deepcopy(method.body[first:last]),type_ignores=[])
    before=ast.Module(body=[prefix],type_ignores=[])
    for part in (before,suffix):ast.fix_missing_locations(part)
    kept=ast.Module(body=copy.deepcopy(method.body[:last]),type_ignores=[])
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        ('AdamW','backward','step','update_actor','generate_sequences') for n in ast.walk(kept)),
        'Model-construction prefix contains a forbidden training call')
    return before,suffix,dict(original_source_sha256=expected_sha,
        original_method_line=method.lineno,original_barrier_line=method.body[first].lineno,
        omitted_optimizer_conditional_line=method.body[last].lineno,
        kept_ast_sha256=hashlib.sha256(ast.dump(kept,include_attributes=False).encode()).hexdigest(),
        scope='Original CPU HF statements and original barrier/MP/FSDP statements; optimizer/scheduler tail not executed')


def original_constructor_namespace():
    # Helpers are separately SHA-bound by the caller before import. Do not import
    # the worker class: its module imports Ray/sharding-manager/vLLM machinery.
    from verl.utils import hf_tokenizer
    from verl.utils.debug import log_gpu_memory_usage
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy,init_fn,get_init_weight_context_manager
    from verl.utils.fs import copy_local_path_from_hdfs
    return dict(torch=torch,warnings=warnings,hf_tokenizer=hf_tokenizer,
        log_gpu_memory_usage=log_gpu_memory_usage,get_fsdp_wrap_policy=get_fsdp_wrap_policy,
        init_fn=init_fn,get_init_weight_context_manager=get_init_weight_context_manager,
        copy_local_path_from_hdfs=copy_local_path_from_hdfs,
        logger=logging.getLogger('tree_zero_optimizer_probe'),__builtins__=__builtins__)


def load_original_hf_prefix(source_path,source_sha,hf_path,worker_config,device_mesh,rank):
    before,suffix,receipt=split_original_constructor(source_path,source_sha)
    namespace=original_constructor_namespace()
    exec(compile(before,str(source_path),'exec'),namespace)
    self=SimpleNamespace(config=worker_config,rank=rank,device_mesh=device_mesh,
        ulysses_sequence_parallel_size=1,role='actor_rollout',_is_actor=True,_is_rollout=True,_is_ref=False)
    from omegaconf import OmegaConf
    override=OmegaConf.to_container(worker_config.model.get('override_config',OmegaConf.create()))
    values=namespace['original_hf_prefix'](self,str(hf_path),worker_config.actor.fsdp_config,None,override,
        worker_config.model.use_remove_padding,worker_config.model.get('enable_gradient_checkpointing',False),
        worker_config.model.get('trust_remote_code',False))
    namespace.update(values)
    return namespace,suffix,receipt


def wrap_original_fsdp(namespace,suffix,source_path):
    exec(compile(suffix,str(source_path),'exec'),namespace)
    require('actor_module_fsdp' in namespace,'Original construction did not produce FSDP')
    return namespace['actor_module_fsdp']


def exhaustive_hf_cast_inventory(model,hf_path,index,expected_files,expected_serialized=435,expected_unique=434):
    """Every serialized HF key and actual CPU-loaded logical tensor is compared.

    Full shape/dtype/bytes are digested, including both serialized tied keys. No
    embedding/head key is discarded to make a count fit. Cast expectation is the
    actual actor MixedPrecision BF16 parameter dtype, separately bound by runner.
    """
    from safetensors import safe_open
    parameters=dict(model.named_parameters(remove_duplicate=False))
    unique=list(model.named_parameters(remove_duplicate=True))
    weights=index['weight_map']
    require(len(weights)==expected_serialized and len(parameters)==expected_serialized,
        'Serialized HF/loaded logical parameter count changed')
    require(len(unique)==expected_unique and set(weights)==set(parameters),'Unique count or complete HF logical keys changed')
    require(set(weights.values())=={p.name for p in Path(hf_path).glob('*.safetensors')},'HF index/shard file coverage changed')
    targets=[];records={};serialized_to_unique={};identity_groups={}
    for name,param in parameters.items():identity_groups.setdefault(id(param),[]).append(name)
    for names in identity_groups.values():
        for name in names:serialized_to_unique[name]=sorted(names)[0]
    for filename in sorted(set(weights.values())):
        require(Path(filename).name==filename,'Shard name is not a local basename')
        path=Path(hf_path)/filename
        require(filename in expected_files,'Frozen expected HF shard SHA missing: '+filename)
        with safetensor_verified(path,expected_files[filename],safe_open) as shard:
            names=sorted(name for name,shardname in weights.items() if shardname==filename)
            require(set(shard.keys())==set(names),'Serialized shard contains missing/unexpected logical keys')
            for name in names:
                saved=shard.get_tensor(name);loaded=parameters[name]
                require(loaded.device.type=='cpu' and not loaded.is_meta,'Original CPU HF tensor is not materialized')
                original=tensor_digest(saved);master=tensor_digest(saved.to(loaded.dtype))
                require(master==tensor_digest(loaded),'Loaded HF master differs from full checkpoint cast bytes: '+name)
                expected=tensor_digest(saved.to(torch.bfloat16))
                module_name,attribute=name.rsplit('.',1)
                module=model.get_submodule(module_name)
                require(getattr(module,attribute) is loaded,'Logical weight attribute is not the loaded HF Parameter')
                targets.append(dict(logical_name=name,module=module,attribute=attribute,expected_cast_digest=expected))
                records[name]=dict(serialized=original,loaded_master=master,expected_effective_bf16=expected,
                    unique_parameter_key=serialized_to_unique[name])
                del saved
    return targets,dict(status='EXHAUSTIVE_HF_CAST_BYTES_VERIFIED_NO_HISTORICAL_ACTOR_ACCEPTANCE',
        serialized_count=len(records),unique_parameter_count=len(unique),logical_records=records,
        serialized_to_unique=serialized_to_unique,original_3268_effective_identity_proven=False)


class SameWindowTiedAliasTrace:
    """Retain both actual attributes simultaneously in one outer FSDP gather.

    The hook is installed on the original causal-model module, whose invocation
    occurs after its outer FSDP pre-forward gather and before embeddings/LM head.
    Its original object and both child module objects are retained before wrapping.
    It never raises inside the model: local evidence failures are deferred until
    the original actor forward returns, then the runner synchronizes refusal.
    """
    def __init__(self,causal_model,embedding,lm_head):
        self.model=causal_model;self.embedding=embedding;self.head=lm_head
        self.records=[];self.errors=[];self.handle=None;self.completed=False

    def __enter__(self):
        require(self.handle is None,'Nested same-window alias trace')
        self.records=[];self.errors=[];self.completed=False
        def hook(module,args):
            try:
                # Both live tensor references are evaluated in this same callback.
                left,right=self.embedding.weight,self.head.weight
                require(left.device.type==right.device.type,'Tied attributes have different devices')
                alias=bool(torch._C._is_alias_of(left,right))
                require(alias and left.shape==right.shape and left.dtype==right.dtype and
                    left.storage_offset()==right.storage_offset() and left.stride()==right.stride(),
                    'Embedding/head do not alias the same complete live gathered tensor')
                self.records.append(dict(same_window_alias=True,logical_names=['model.embed_tokens.weight','lm_head.weight'],
                    dtype=str(left.dtype),shape=list(left.shape),device=str(left.device),
                    scope='Simultaneous live references in original causal-model pre-hook after outer FSDP gather',
                    alias_primitive='torch._C._is_alias_of; equal shape/stride/storage offset'))
            except Exception as error:self.errors.append(type(error).__name__+': '+str(error))
        self.handle=self.model.register_forward_pre_hook(hook)
        return self

    def __exit__(self,kind,value,traceback):
        self.handle.remove();self.handle=None;self.completed=False
        if kind is None:
            require(not self.errors and len(self.records)==1,'Same-window alias evidence absent/failed: '+repr(self.errors))
            self.completed=True
        return False

    def receipt(self):
        require(self.completed and self.handle is None,'Cannot publish failed/incomplete same-window alias trace')
        return dict(status='SAME_GATHER_WINDOW_TIED_ALIAS_OBSERVED',records=list(self.records),
            tied_alias_identity_proven=True,historical_3268_identity_proven=False)
