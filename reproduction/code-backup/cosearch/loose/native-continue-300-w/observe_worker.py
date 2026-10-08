"""Read-only optimizer observations, installed in native Ray workers at startup.

No replacement forward/backward/clip/Adam/scheduler, no new collectives or RNG.
The checkpoint reload runner must call init/load only, never the training fit.
"""
import functools, hashlib, json, os, time, socket
from pathlib import Path

class NativeEvidenceFatal(BaseException):
    """Cannot become an ordinary author tool fallback reward."""

def _write_record(kind, record):
    root=os.environ.get('COSEARCH_NATIVE_EVIDENCE_ROOT')
    if not root:
        raise RuntimeError('Missing native evidence root')
    directory=Path(root)/'workers'
    directory.mkdir(parents=True,exist_ok=True)
    value=dict(record,kind=kind,pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),
               source_commit='763bf8c4d7caa363ad845d39ddfd53b81ae377bd',
               hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        with (directory/f'{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n')
    except BaseException as error:
        raise NativeEvidenceFatal('Cannot preserve native worker evidence; do not convert it to a tool penalty') from error

def write_record(kind,record):
    try:return _write_record(kind,record)
    except NativeEvidenceFatal:raise
    except BaseException as error:
        raise NativeEvidenceFatal('Native evidence failed before ordinary tool fallback handling') from error

def preserve_raw_payload(kind, payload):
    """Save original trusted CPU rollout objects before any rejection/filtering."""
    import torch
    directory=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'raw-rollouts'
    path=directory/f'{kind}-{os.getpid()}-{time.monotonic_ns()}.pt'
    try:
        directory.mkdir(parents=True,exist_ok=True)
        torch.save(payload,path)
        hasher=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk)
        digest=hasher.hexdigest()
        write_record('raw_rollout_payload',dict(stage=kind,path=str(path),
            bytes=path.stat().st_size,sha256=digest))
    except BaseException as error:
        raise NativeEvidenceFatal('Cannot save original rollout payload before rejection') from error
    return path

def install_raw_rollout_observations(base, worker):
    if getattr(worker,'_native_raw_observer_installed',False):return
    original_post=worker._postprocess
    @functools.wraps(original_post)
    def postprocess(instance,inputs):
        preserve_raw_payload('pre-postprocess',inputs)
        return original_post(instance,inputs)
    original_filter=worker._process_reranker_outputs
    @functools.wraps(original_filter)
    async def process(instance,*args,**kwargs):
        preserve_raw_payload('pre-ranker-filter',dict(args=args,kwargs=kwargs))
        result=await original_filter(instance,*args,**kwargs)
        preserve_raw_payload('post-ranker-filter',result)
        return result
    worker._postprocess=postprocess
    worker._process_reranker_outputs=process
    worker._native_raw_observer_installed=True

def snapshot(optimizer):
    """Only local ordinary FSDP shards; explicitly reject distributed tensor copies."""
    import torch
    from torch.distributed.tensor import DTensor
    for group_index,group in enumerate(optimizer.param_groups):
        for parameter_index,parameter in enumerate(group['params']):
            if isinstance(parameter,DTensor):
                raise RuntimeError('DTensor observations require separately reviewed local-shard handling')
            if parameter.numel():
                assert parameter.is_contiguous(),'Probe must not copy a full noncontiguous parameter'
                value=parameter.detach().reshape(-1)[:8].to('cpu').clone()
                state=optimizer.state.get(parameter,{})
                step=state.get('step',0)
                if torch.is_tensor(step):step=step.detach().cpu().item()
                return {'parameter':parameter,'probe':value,'step':int(step),
                        'shape':list(parameter.shape),'dtype':str(parameter.dtype),
                        'group_index':group_index,'parameter_index':parameter_index,'probe_offset':0,
                        'lr':float(group['lr']),'state_initialized':bool(state)}
    raise RuntimeError('No nonempty local optimizer parameter')

def observe_update(original, actor, data):
    """Same training call/data/result; separate per-worker evidence after return."""
    import torch
    role=getattr(actor,'_native_observed_role','UNBOUND')
    assert role in ['main','reranker'],'Missing actual role binding before optimizer call'
    binding_path=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'active-batches'/(role+'.json')
    if os.environ.get('SLURM_JOB_ID'):
        binding=json.loads(binding_path.read_text())
        assert binding['role']==role and set(str(x) for x in data.non_tensor_batch['uid'])<=set(binding['uids'])
        outer_step=int(binding['step'])
    else:outer_step=None  # CPU synthetic control does not fabricate a production step.
    optimizer=actor.actor_optimizer
    if not hasattr(actor,'_native_completed_adam_calls'):
        actor._native_completed_adam_calls=0
        def completed_step(optimizer,args,kwargs):
            actor._native_completed_adam_calls+=1
        actor._native_step_observer=optimizer.register_step_post_hook(completed_step)
    calls_before=actor._native_completed_adam_calls
    before=snapshot(actor.actor_optimizer)
    started=time.monotonic()
    result=original(actor,data)
    after=snapshot(actor.actor_optimizer)
    assert before['parameter'] is after['parameter']
    mask=data.batch['response_mask'].bool()
    adv=data.batch['advantages'].detach()
    nonzero=int(((adv.abs()>1e-5)&mask).sum().cpu().item())
    delta=float((after['probe'].double()-before['probe'].double()).abs().sum().item())
    write_record('update',dict(role=role,outer_step=outer_step,
        rank=torch.distributed.get_rank(),world_size=torch.distributed.get_world_size(),
        update_index=getattr(actor,'_native_update_index',0)+1,
        adam_state_before=before['step'],adam_state_after=after['step'],
        actual_adam_delta=actor._native_completed_adam_calls-calls_before,
        probe_parameter_adam_delta=after['step']-before['step'],lr_before=before['lr'],
        parameter_probe_delta_l1=delta,parameter_dtype=after['dtype'],
        parameter_shape=after['shape'],parameter_group_index=after['group_index'],
        parameter_index=after['parameter_index'],parameter_probe_offset=after['probe_offset'],
        parameter_probe_count=len(after['probe']),state_initialized=after['state_initialized'],
        local_uids=[str(x) for x in data.non_tensor_batch.get('uid',[])],
        local_masked_nonzero_advantage_tokens=nonzero,
        original_gradient_norms=[float(x) for x in result.get('actor/grad_norm',[])],
        original_metrics=result,wall_seconds=time.monotonic()-started,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),monotonic_ns=time.monotonic_ns()))
    actor._native_update_index=getattr(actor,'_native_update_index',0)+1
    return result

def install_vllm_uuid_mapping(platform_module=None):
    """Resolve leased UUIDs for vLLM NVML queries; never change CUDA visibility."""
    if platform_module is None:
        import vllm.platforms.interface as platform_module
    assert hashlib.sha256(Path(platform_module.__file__).read_bytes()).hexdigest()=='ff142312173c173dae20a8303cd10f9c29fa255862396655b6888d2fdb73fae5'
    cls=platform_module.Platform
    original=cls.device_id_to_physical_device_id.__func__
    if getattr(original,'_native_uuid_mapping',False):return
    @functools.wraps(original)
    def resolve(platform,device_id):
        mask=os.environ.get(platform.device_control_env_var,'')
        values=mask.split(',') if mask else []
        if values and values[device_id].startswith('GPU-'):
            mapping=json.loads(os.environ['COSEARCH_LEASED_UUID_TO_INDEX'])
            assert len(mapping)==8 and set(mapping)==set(os.environ['COSEARCH_LEASED_UUIDS'].split(','))
            assert len(set(mapping.values()))==8 and all(isinstance(v,int) and v>=0 for v in mapping.values())
            assert all(value in mapping for value in values)
            return mapping[values[device_id]]
        return original(platform,device_id)
    resolve._native_uuid_mapping=True
    cls.device_id_to_physical_device_id=classmethod(resolve)

def install():
    """Ray worker setup hook. CUDA remains uninitialized until original init_model."""
    if os.environ.get('SLURM_JOB_ID'):
        assert os.environ.get('SLURM_JOB_NODELIST')=='h100-16'
        assert not any(os.environ.get('RAY_EXPERIMENTAL_NOSET_'+suffix) for suffix in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','HABANA_VISIBLE_MODULES','NEURON_RT_VISIBLE_CORES','TPU_VISIBLE_CHIPS','ONEAPI_DEVICE_SELECTOR')), 'Ray device assignment bypass is forbidden'
        assert not any(key in os.environ for key in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')), 'Conflicting vendor device aliases'
        current_mask=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not current_mask.strip():
            assert 'RANK' not in os.environ,'FSDP rank has no Ray-assigned GPU mask'
            # Ray has already applied an empty mask to this zero-GPU process.
            # Preserve every nonempty mask, including vLLM servers without RANK.
            leased=os.environ['COSEARCH_LEASED_UUIDS'].split(',')
            assert len(leased)==8 and len(set(leased))==8
            os.environ['CUDA_VISIBLE_DEVICES']=','.join(leased)
        elif 'RANK' in os.environ:
            assert all(x.strip().isdigit() for x in current_mask.split(','))
        from verl.utils.ray_utils import ray_noset_visible_devices
        assert not ray_noset_visible_devices(), 'Original Ray bypass check failed'
    if os.environ.get('SLURM_JOB_ID'):install_vllm_uuid_mapping()
    manifest=json.loads(Path(__file__).with_name('source-manifest.json').read_text())
    repo=Path(manifest['repo'])
    for relative,expected in manifest['sha256'].items():
        assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==expected,relative
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.tools.co_search_tool import CoSearchTool
    import sys
    for cls,relative in [(DataParallelPPOActor,'verl/verl/workers/actor/dp_actor.py'),
                         (ActorRolloutRefWorker,'verl/verl/workers/fsdp_workers.py'),
                         (CoSearchTool,'verl/verl/tools/co_search_tool.py')]:
        assert Path(sys.modules[cls.__module__].__file__).resolve()==(repo/relative).resolve(),cls.__module__
    if getattr(DataParallelPPOActor,'_native_observer_installed',False):return
    from verl.experimental.agent_loop.agent_loop import SearchR1RerankerAgentLoopWorkerBase
    from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
    actual_worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
    assert issubclass(actual_worker,SearchR1RerankerAgentLoopWorkerBase)
    install_raw_rollout_observations(SearchR1RerankerAgentLoopWorkerBase,actual_worker)
    import fallback_alignment
    fallback_alignment.install(actual_worker,write_record,preserve_raw_payload)
    import empty_ranker
    empty_ranker.install_worker(actual_worker,write_record,preserve_raw_payload)
    import phase_sampling
    phase_sampling.install_sampling(write_record)
    original_update=DataParallelPPOActor.update_policy
    @functools.wraps(original_update)
    def update(actor,data):
        return observe_update(original_update,actor,data)
    DataParallelPPOActor.update_policy=update
    original_init=ActorRolloutRefWorker.init_model
    @functools.wraps(original_init)
    def init(worker,*args,**kwargs):
        result=original_init(worker,*args,**kwargs)
        if getattr(worker,'_is_actor',False):
            import torch
            role='main' if worker.config.rollout.multi_turn.enable else 'reranker'
            worker.actor._native_observed_role=role
            worker.checkpoint_manager._native_observed_role=role
            device=torch.cuda.current_device()
            properties=torch.cuda.get_device_properties(device)
            initial=snapshot(worker.actor.actor_optimizer)
            write_record('ack',dict(source_manifest_sha256=hashlib.sha256(Path(__file__).with_name('source-manifest.json').read_bytes()).hexdigest(),worker_role=str(worker.role),role=role,rank=torch.distributed.get_rank(),
                world_size=torch.distributed.get_world_size(),uuid='GPU-'+str(properties.uuid).removeprefix('GPU-'),raw_uuid=str(properties.uuid),
                hostname=socket.gethostname(),process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                process_cgroup=Path('/proc/self/cgroup').read_text(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gloo_socket_ifname=os.environ.get('GLOO_SOCKET_IFNAME'),
                nccl_socket_ifname=os.environ.get('NCCL_SOCKET_IFNAME'),
                device_aliases={key:os.environ.get(key) for key in ('ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')},
                adam_state_initial=initial['step'],lr_initial=initial['lr'],
                optimizer_horizon=int(worker.config.actor.optim.total_training_steps),
                optimizer_mini_batch_local=int(worker.config.actor.ppo_mini_batch_size),
                original_actor_file_sha256=manifest['sha256']['verl/verl/workers/actor/dp_actor.py']))
        return result
    ActorRolloutRefWorker.init_model=init
    original_rerank=CoSearchTool._call_reranker
    @functools.wraps(original_rerank)
    async def rerank(tool,reranker_manager,tokenizer,request_id,initial_query,query,documents,top_m,sampling_params,reranker_as_agent=True):
        contents=[document.get('contents') or document.get('text') or document.get('passage') or '' for document in documents]
        if any(len(text)>2000 for text in contents):
            try:
                directory=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'context-refusals'
                directory.mkdir(parents=True,exist_ok=True)
                destination=directory/f'{os.getpid()}-{time.monotonic_ns()}.json'
                destination.with_suffix('.refused').touch(exist_ok=False)
            except BaseException as error:raise NativeEvidenceFatal('Cannot preserve original per-document crop refusal') from error
            from verl.tools.utils.search import format_tool_response_with_docid_map
            from verl.tools.utils.prompts import RERANK_PROMPT_WITH_INITIAL_QUERY
            full_passages,_=format_tool_response_with_docid_map(documents,max_doc_length=max(map(len,contents)))
            full_prompt=RERANK_PROMPT_WITH_INITIAL_QUERY.format(N=len(documents),M=min(top_m,len(documents)),
                initial_query=initial_query,sub_query=query,passages_block=full_passages)
            full_ids=tokenizer.apply_chat_template([{'role':'user','content':full_prompt}],add_generation_prompt=True,tokenize=True)
            record=dict(reason='author_2000_character_document_crop',request_id=request_id,initial_query=initial_query,
                query=query,full_documents=documents,full_prompt_ids=full_ids,full_prompt_tokens=len(full_ids),
                document_character_lengths=list(map(len,contents)),candidate_count=len(documents),top_m=top_m)
            destination.write_text(json.dumps(record,allow_nan=False))
            write_record('original_document_crop_refused',record)
            raise RuntimeError('Original 2000-character document crop refused; discard entire batch before PPO')
        class TokenizerView:
            def __getattr__(self,name):return getattr(tokenizer,name)
            def apply_chat_template(self,*args,**kwargs):
                ids=tokenizer.apply_chat_template(*args,**kwargs)
                record=dict(request_id=request_id,full_prompt_tokens=len(ids),candidate_count=len(documents),
                            top_m=top_m,actual_sampling_params=sampling_params)
                if len(ids)>16384:
                    # No query hashing/serialization/logging precedes the minimal fatal marker.
                    try:
                        directory=Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT'])/'context-refusals'
                        directory.mkdir(parents=True,exist_ok=True)
                        destination=directory/f'{os.getpid()}-{time.monotonic_ns()}.json'
                        destination.with_suffix('.refused').touch(exist_ok=False)
                    except BaseException as error:
                        raise NativeEvidenceFatal('Cannot preserve context refusal sentinel') from error
                    record.update(initial_query=initial_query,query=query,full_documents=documents,full_prompt_ids=ids)
                    write_record('original_context_crop_refused',record)
                    pending=destination.with_suffix('.pending')
                    pending.write_text(json.dumps(record,allow_nan=False))
                    pending.replace(destination)
                    raise RuntimeError('Original author 16384 head-crop would remove full candidate context; preserved exact input')
                write_record('ranker_inference_contract',record)
                return ids
        result=await original_rerank(tool,reranker_manager,TokenizerView(),request_id,initial_query,query,
                                     documents,top_m,sampling_params,reranker_as_agent)
        write_record('ranker_ranking_result',dict(request_id=request_id,query=query,
            response_ids=result['agent_output']['response_ids'],candidate_count=len(documents),top_m=top_m,
            original_error_type=result.get('error_type'),actual_reranked_docs=[document['id'] for document in result.get('reranked_docs',[])],
            actual_candidate_ids=[document['id'] for document in documents]))
        return result
    CoSearchTool._call_reranker=rerank
    import checkpoint_observer
    checkpoint_observer.install()
    DataParallelPPOActor._native_observer_installed=True

def defer_install():
    """Ray 2.59 startup is earlier than GPU assignment; wait for its original assignment callback."""
    import ray, ray._private.utils as ray_utils
    assert ray.__version__=='2.59.0', 'Deferred hook is pinned to reviewed Ray runtime'
    assert hashlib.sha256(Path(ray_utils.__file__).read_bytes()).hexdigest()=='3e4841837e6e7794f411c52d25d4f2ba8a6a356fb0fc49a31e4ff79d797af5bd', 'Ray assignment source changed'
    original=ray_utils.set_visible_accelerator_ids
    if getattr(original, '_native_deferred_observer', False):return
    @functools.wraps(original)
    def assigned(*args, **kwargs):
        result=original(*args, **kwargs)
        accelerator_ids=ray.get_runtime_context().get_accelerator_ids()
        if not accelerator_ids.get('GPU'):
            assert 'RANK' not in os.environ, 'Training rank has no assigned GPU'
            leased=os.environ['COSEARCH_LEASED_UUIDS'].split(',')
            assert len(leased)==8 and len(set(leased))==8
            os.environ['CUDA_VISIBLE_DEVICES']=','.join(leased)
        if getattr(assigned, '_native_installed', False):
            import sys
            assert 'torch' not in sys.modules or os.environ.get('CUDA_VISIBLE_DEVICES')==assigned._native_mask, 'CUDA mask changed after observer imports; process reuse forbidden'
        if not getattr(assigned, '_native_installed', False):
            write_record('device_assignment', dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                assigned_accelerator_ids=accelerator_ids,
                rank=os.environ.get('RANK'),gloo_socket_ifname=os.environ.get('GLOO_SOCKET_IFNAME'),
                process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                process_cgroup=Path('/proc/self/cgroup').read_text()))
            install()
            assigned._native_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
            assigned._native_installed=True
        return result
    assigned._native_deferred_observer=True
    ray_utils.set_visible_accelerator_ids=assigned
