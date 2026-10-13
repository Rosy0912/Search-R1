"""Evaluation only: eight persistent custom vLLM TP1 native-emission engines.

Invoke under the independently reviewed supervisor's torchrun/Slurm GPU lease.
No training worker, optimizer, reference actor, or model rewriting is constructed.
"""
import argparse
from collections import Counter
import gc
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from eval_merge import (asset_manifest, atomic_immutable, canonical, committed_batches,
                        digest, files, identities, object_sha, read_prefix)

def question_sha(row, prompt_key):
    def plain(value):
        if hasattr(value, 'tolist'):
            return plain(value.tolist())
        if isinstance(value, dict):
            return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)):
            return [plain(v) for v in value]
        return value
    return object_sha(plain({k:row[k] for k in (prompt_key,'data_source','reward_model')}))

def protocol(config):
    wanted = dict(max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500)
    if config.max_turns != 4 or config.force_final_answer or not config.do_search:
        raise ValueError('Original four-round search protocol required')
    if any(config.data[k] != v for k,v in wanted.items()):
        raise ValueError('Original rollout length limits changed')
    if not config.baselines.tree.token_preservation.enabled or not config.actor_rollout_ref.rollout.tree_raw_emission:
        raise ValueError('Native token-preserving original protocol required')
    if config.actor_rollout_ref.rollout.prompt_length + config.actor_rollout_ref.rollout.response_length != 4596:
        raise ValueError('Original vLLM context must be 4596')
    if urlparse(config.retriever.url).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Retriever must be existing guarded loopback service')

def main():
    p=argparse.ArgumentParser()
    for key in ('config','prefix','output','model','manifest'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--local-batch-size', type=int, default=64)
    a=p.parse_args()
    if a.local_batch_size != 64:
        raise ValueError('Reviewed execution requires local batch 64')
    if os.uname().sysname != 'Linux' or os.environ.get('SEARCH_R1_GPU_PREFLIGHT') != '1':
        raise ValueError('Remote Linux guarded GPU allocation required')
    if os.environ.get('VLLM_ATTENTION_BACKEND') != 'XFORMERS':
        raise ValueError('Original XFORMERS backend required')
    # Supervisor checks actual cgroup and retains all eight lease descriptors.
    if not os.environ.get('SEARCH_R1_GPU_LEASES'):
        raise ValueError('Missing retained allocation leases')
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout, vllm_version
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from search_r1.baselines.rollout import SearchRollout
    from search_r1.baselines.evaluation import batch_answer_metrics
    if vllm_version != '0.6.3':
        raise ValueError('Reviewed custom native engine requires vLLM 0.6.3')
    config=OmegaConf.load(a.config); OmegaConf.resolve(config); protocol(config)
    manifest=json.loads(Path(a.manifest).read_text())
    actual=asset_manifest(a.config,a.model)
    if manifest != actual:
        raise ValueError('Immutable HF/config/dataset assets changed')
    rank=int(os.environ['RANK']); local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8 or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 8:
        raise ValueError('Exactly all eight allocated GPUs required')
    torch.cuda.set_device(local_rank)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    dataset=RLHFDataset(parquet_files=files(config), tokenizer=tokenizer,
        prompt_key=config.data.prompt_key, max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True, return_raw_chat=config.data.get('return_raw_chat',False),
        truncation='error', cache_dir=str(Path(a.output)/'cache'))
    sources=list(dataset.dataframe['data_source'])
    if len(dataset) != manifest['dataset_count'] or dict(Counter(sources)) != manifest['expected_source_counts']:
        raise ValueError('Final dataset count/source names differ')
    prefix=read_prefix(a.prefix,manifest,sources)
    for i,row in enumerate(prefix):
        if row.get('question_sha256') and row['question_sha256'] != question_sha(dataset.dataframe.iloc[i],config.data.prompt_key):
            raise ValueError('Preserved prefix question mismatch')
    # A crash between batch publication and receipt publication leaves a fully
    # written, fsynced immutable batch. Validate its question mapping before
    # completing the receipt, avoiding numerically different recomputation.
    from eval_merge import validate_row
    for orphan in sorted((Path(a.output)/'batches').glob('*.json')):
        if orphan.name.endswith('.done.json'):
            continue
        done=orphan.with_name(orphan.stem+'.done.json')
        if done.exists():
            continue
        value=json.loads(orphan.read_text())
        if value['identity'] != identities(manifest) or value['prefix_count'] != len(prefix):
            raise ValueError('Orphan batch identity mismatch')
        for row in value['rows']:
            validate_row(row)
            index=row['index']
            if not len(prefix) <= index < len(dataset) or row['source'] != sources[index]:
                raise ValueError('Orphan batch question range/order mismatch')
            if row['question_sha256'] != question_sha(dataset.dataframe.iloc[index],config.data.prompt_key):
                raise ValueError('Orphan batch question identity mismatch')
        atomic_immutable(done,dict(sha256=digest(orphan),count=len(value['rows'])))
    completed=committed_batches(a.output,manifest,len(prefix))
    for index,row in completed.items():
        if row['source'] != sources[index] or row['question_sha256'] != question_sha(dataset.dataframe.iloc[index],config.data.prompt_key):
            raise ValueError('Committed question identity mismatch')
    work=[]
    for batch_no,start in enumerate(range(len(prefix),len(dataset),a.local_batch_size)):
        indices=list(range(start,min(start+a.local_batch_size,len(dataset))))
        present=[i in completed for i in indices]
        if any(present) and not all(present):
            raise ValueError('Committed partial batch violates fixed schedule')
        if batch_no % 8 == rank and not all(present):
            work.append(indices)
    # All ranks initialize together even if a resumed rank has no remaining work.
    torch.distributed.init_process_group('nccl')
    hf=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True)
    hf.eval()
    weights=hf.state_dict()
    if not all(torch.isfinite(v).all().item() for v in weights.values() if v.is_floating_point()):
        raise ValueError('Nonfinite HF200 parameters')
    rollout_cfg=OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref.rollout,resolve=True))
    rollout_cfg.tensor_model_parallel_size=1
    rollout_cfg.dtype='bfloat16'; rollout_cfg.load_format='hf'
    rollout_cfg.free_cache_engine=False
    engine=vLLMRollout(weights,rollout_cfg,tokenizer,AutoConfig.from_pretrained(a.model,local_files_only=True),train_tp=1)
    engine.inference_engine.sync_model_weights(weights,load_format='hf')
    del weights,hf; gc.collect(); torch.cuda.empty_cache()
    torch.distributed.barrier()
    class ResidentAdapter:
        def generate_sequences(self,batch):
            # SearchRollout internally constructs CPU tensors. Native worker expects
            # device tensors, while SearchRollout consumes returned tokens on CPU.
            return engine.generate_sequences(batch.to(torch.device('cuda',local_rank))).to('cpu')
    gen=GenerationConfig(max_turns=config.max_turns,
        max_start_length=config.data.max_start_length,max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,max_obs_length=config.data.max_obs_length,
        num_gpus=1,no_think_rl=config.algorithm.no_think_rl,
        search_url=config.retriever.url,topk=config.retriever.topk,force_final_answer=config.force_final_answer)
    manager=LLMGenerationManager(tokenizer,ResidentAdapter(),gen,is_validation=True)
    execution=dict(local_batch_size=64,world_size=8,tensor_parallel_size=1,resident_weights=True,
        free_cache_engine=False,rollout_padding_divisor=1,backend='XFORMERS',dtype='bfloat16',native_emission=True)
    atomic_immutable(Path(a.output)/('rank-%d-provenance.json'%rank),dict(identity=identities(manifest),execution=execution))
    print('EVAL_RESIDENT_READY '+json.dumps(dict(rank=rank,remaining_batches=len(work))),flush=True)
    for indices in work:
        started=time.perf_counter();cpu_started=time.process_time()
        batch=DataProto.from_single_dict(collate_fn([dataset[i] for i in indices]))
        prompts=batch.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        prompts.meta_info=dict(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
            recompute_log_prob=False,do_sample=False,validate=True)
        search=SearchRollout(manager,context_alignment='error',token_preservation=config.baselines.tree.token_preservation)
        generated,_=search.run(prompts)
        batch=batch.union(generated)
        for key in batch.batch:
            batch.batch[key]=batch.batch[key].long()
        rows=batch_answer_metrics(batch,tokenizer)
        if len(rows) != len(indices):
            raise ValueError('Generated score cardinality mismatch')
        for index,row in zip(indices,rows):
            row['index']=index
            row['question_sha256']=question_sha(dataset.dataframe.iloc[index],config.data.prompt_key)
            if row['source'] != sources[index]:
                raise ValueError('Generated source order mismatch')
        stem='%05d-%05d'%(indices[0],indices[-1])
        path=Path(a.output)/'batches'/(stem+'.json')
        atomic_immutable(path,dict(identity=identities(manifest),prefix_count=len(prefix),execution=execution,rows=rows,
            cost=dict(search.cost,wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu_started),
            allocation_job=os.environ['SLURM_JOB_ID'],rank=rank))
        atomic_immutable(path.with_name(stem+'.done.json'),dict(sha256=digest(path),count=len(rows)))
        print('EVAL_BATCH_COMMITTED '+json.dumps(dict(rank=rank,start=indices[0],end=indices[-1],count=len(rows))),flush=True)
    # TP1 engines do not use the world group for generation. No ending barrier:
    # faster ranks may finish while other independent queues remain active.
    print('EVAL_RANK_COMPLETE '+str(rank),flush=True)

if __name__=='__main__':
    main()
