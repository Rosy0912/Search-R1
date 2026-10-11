"""Inference only: existing BAPO validation protocol, eight rollout ranks, no optimizer."""
import json,os,time
from pathlib import Path
import ray
from omegaconf import OmegaConf
from types import SimpleNamespace
from verl.workers.fsdp_workers import ActorRolloutRefWorker

@ray.remote(num_gpus=1,num_cpus=1)
class StaticWorker:
    def __init__(self,model,config):
        import torch
        from vllm import LLM,SamplingParams
        from torch.nn.utils.rnn import pad_sequence
        from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
        from verl.utils import hf_tokenizer
        tokenizer=hf_tokenizer(model)
        class Engine(LLM):
            def _post_process_outputs(self,outputs):
                ids=[];probs=[]
                for request in outputs:
                    for output in request.outputs:
                        ids.append(torch.tensor(output.token_ids,dtype=torch.long))
                        probs.append(torch.tensor([p[token].logprob for p,token in zip(output.logprobs,output.token_ids)],dtype=torch.float32))
                return pad_sequence(ids,batch_first=True,padding_value=tokenizer.pad_token_id),pad_sequence(probs,batch_first=True,padding_value=tokenizer.pad_token_id)
            def generate(self,*args,**kwargs):
                return self._post_process_outputs(super().generate(*args,**kwargs))
        config=OmegaConf.create(OmegaConf.to_container(config,resolve=True));config.free_cache_engine=False
        self.rollout=vLLMRollout.__new__(vLLMRollout)
        self.rollout.config=config;self.rollout.tokenizer=tokenizer;self.rollout.pad_token_id=tokenizer.pad_token_id
        self.rollout.inference_engine=Engine(model=model,tokenizer=model,dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,max_model_len=config.prompt_length+config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,max_num_seqs=config.max_num_seqs,enforce_eager=True,disable_log_stats=True)
        self.rollout.sampling_params=SamplingParams(n=1,logprobs=1,max_tokens=config.response_length,detokenize=False,
            temperature=1.0,top_k=-1,top_p=1.0,ignore_eos=False)
    def ready(self):return dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),optimizer=False)
    def generate(self,batch):
        batch.meta_info.update(eos_token_id=self.rollout.tokenizer.eos_token_id,pad_token_id=self.rollout.pad_token_id)
        return self.rollout.generate_sequences(batch).to('cpu')

class StaticGroup:
    world_size=8
    def __init__(self,workers):self.workers=workers
    def generate_sequences(self,batch):
        from verl import DataProto
        chunks=batch.chunk(8)
        return DataProto.concat(ray.get([w.generate.remote(b) for w,b in zip(self.workers,chunks)]))

@ray.remote
def evaluate(config,run):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
    from verl.utils import hf_tokenizer
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,Role,ResourcePoolManager
    from verl.single_controller.ray import RayWorkerGroup
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
    dataset=RLHFDataset(config.data.val_files,tokenizer,prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length,filter_prompts=True,truncation='error')
    assert len(dataset)==51713, f'Unexpected final dataset size {len(dataset)}'
    offset=int(config.eval_resume_completed)
    dataset.dataframe=dataset.dataframe.iloc[offset:].reset_index(drop=True)
    loader=DataLoader(dataset,batch_size=config.data.val_batch_size,shuffle=False,drop_last=False,collate_fn=collate_fn)
    def batches():
        done=offset
        for i,batch in enumerate(loader):
            start=time.monotonic()
            yield batch
            done+=len(batch['input_ids'])
            record=dict(batch=offset//config.data.val_batch_size+i+1,completed=done,total=51713,batch_size=len(batch['input_ids']),seconds=time.monotonic()-start,time=time.time())
            with (Path(run)/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('EVAL150_BATCH_COMPLETE='+json.dumps(record),flush=True)
    t=RayPPOTrainer.__new__(RayPPOTrainer)
    t.config=config;t.tokenizer=tokenizer;t.hybrid_engine=True
    t.use_reference_policy=False;t.use_rm=False
    workers=[StaticWorker.remote(config.actor_rollout_ref.model.path,config.actor_rollout_ref.rollout) for _ in range(8)]
    ray.get([w.ready.remote() for w in workers])
    t.actor_rollout_wg=StaticGroup(workers)
    t.baseline_controller=SimpleNamespace(method='bapo',ranker=None)
    score=RewardManager(tokenizer,num_examine=0)
    def reward(batch):
        from search_r1.baselines.evaluation import batch_answer_metrics
        rows=batch_answer_metrics(batch,tokenizer)
        with (Path(run)/'qa-rows.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return score(batch)
    t.val_reward_fn=reward
    t.val_dataloader=batches();t.global_steps=150
    # Only the validation metadata branch; native training initialization stays disabled.
    config.trainer.fresh200_enabled=True
    result=t._validate()
    from search_r1.baselines.evaluation import aggregate_answer_metrics
    all_rows=[json.loads(line) for line in (Path(run)/'qa-rows.jsonl').read_text().splitlines()]
    assert len(all_rows)==51713
    full=aggregate_answer_metrics(all_rows)
    (Path(run)/'final51713.json').write_text(json.dumps(dict(checkpoint_outer=150,metrics=full,suffix_runtime_metrics=result,resumed_prefix=offset),indent=2))
    print('EVAL150_COMPLETE51713='+json.dumps(full),flush=True)
    return full

def main():
    root=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
    old=root/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
    run=Path(os.environ['BAPO_RUN_ID'])
    c=OmegaConf.load(old/'resolved-config.yaml')
    c.actor_rollout_ref.model.path=str(old/'checkpoints/actor/global_step_150')
    c.actor_rollout_ref.model.revision=None
    c.actor_rollout_ref.actor.fresh200_enabled=False
    c.trainer.fresh200_enabled=False;c.trainer.fresh200_gate_enabled=False
    c.trainer.resume_full_state=None;c.trainer.resume_permission=None;c.trainer.resume_verify_only=False
    c.data.val_files=c.data.final_val_files;c.data.val_batch_size=512
    c.actor_rollout_ref.rollout.gpu_memory_utilization=0.20
    c.actor_rollout_ref.rollout.max_num_batched_tokens=16384
    c.actor_rollout_ref.rollout.max_num_seqs=128
    c.trainer.default_local_dir=str(run/'unused-training-checkpoints')
    prior=root/'runs/eval150-final51713-852d701d694d4dfe98f2c2f6985740a8-3376'
    # Completed QA rows have fixed parquet order and greedy checkpoint150 semantics.
    old_config=OmegaConf.load(prior/'eval-config.yaml')
    assert old_config.actor_rollout_ref.model.path==c.actor_rollout_ref.model.path
    assert old_config.data.val_files==c.data.val_files and old_config.data.val_batch_size==512
    assert old_config.max_turns==c.max_turns and old_config.force_final_answer==c.force_final_answer
    offset=json.loads((prior/'progress.jsonl').read_text().splitlines()[-1])['completed']
    prefix=[json.loads(line) for line in (prior/'qa-rows.jsonl').read_text().splitlines()][:offset]
    assert len(prefix)==offset and offset==12800
    c.eval_resume_completed=offset
    c.eval_resume_from=str(prior)
    with (run/'qa-rows.jsonl').open('x') as f:
        for row in prefix:f.write(json.dumps(row)+'\n')
    (run/'resume-prefix.json').write_text(json.dumps(dict(prior=str(prior),completed=offset,scope='exact saved score prefix;next row12801;no model/optimizer changes')))
    OmegaConf.save(c,run/'eval-config.yaml')
    ray.init(address=os.environ['RAY_ADDRESS'],runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'false','NCCL_DEBUG':'WARN'}})
    print(ray.get(evaluate.remote(c,str(run))),flush=True)
if __name__=='__main__':main()
