"""Hooks installed only inside dedicated evaluator processes, never live trainers."""
import json
import os
import time

if os.environ.get('TREE_SEVEN_EVAL_MODE') != 'frozen_milestone_v1':
    raise RuntimeError('Frozen worker requires the independent evaluation entry')

from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker

_original_init = ActorRolloutRefWorker.init_model


class Hooks:
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result=_original_init(self)
        import torch
        for p in self.actor_module_fsdp.parameters():p.requires_grad_(False)
        def forbidden(*args,**kwargs):raise RuntimeError('Optimizer updates forbidden in frozen evaluation')
        self.actor_optimizer.step=forbidden
        self.actor.update_policy=forbidden
        self._seven_identity=os.environ['TREE_SEVEN_CHECKPOINT_IDENTITY']
        self._seven_calls=0;self._seven_tokens=0
        prepare=self.rollout_sharding_manager.prepare_policy_cache
        self.rollout_sharding_manager.prepare_policy_cache=lambda enabled,version:prepare(True,('seven',self._seven_identity))
        generate=self.rollout.generate_sequences
        def frozen_generate(prompts):
            if (prompts.meta_info.get('validate') is not True or prompts.meta_info.get('do_sample') is not False
                    or prompts.meta_info.get('recompute_log_prob',False)):
                raise RuntimeError('Only greedy native validation is allowed')
            started=time.time();out=generate(prompts)
            mask=out.batch.get('response_mask')
            if mask is None:raise RuntimeError('Raw emitted-token mask missing')
            count=int(mask.sum().item());self._seven_calls+=1;self._seven_tokens+=count
            print('SEVEN_EVAL_GPU='+json.dumps(dict(rank=torch.distributed.get_rank(),
                call=self._seven_calls,raw_emitted_tokens=count,seconds=time.time()-started,
                checkpoint_identity=self._seven_identity,time=time.time())),flush=True)
            return out
        self.rollout.generate_sequences=frozen_generate
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def seven_frozen_evidence(self):
        import torch,subprocess
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        uuids={r.split(',')[1].strip() for r in rows if r.split(',')[0].strip()==str(os.getpid())}
        if len(uuids)!=1:raise RuntimeError('Worker must bind one physical GPU')
        if self.actor_optimizer.state or any(p.requires_grad for p in self.actor_module_fsdp.parameters()):
            raise RuntimeError('Unexpected optimizer state or trainable parameters')
        return dict(rank=torch.distributed.get_rank(),gpu_uuid=next(iter(uuids)),
            communication_transport={k:os.environ[k] for k in ('NCCL_P2P_DISABLE','NCCL_IB_DISABLE') if k in os.environ},
            checkpoint_identity=self._seven_identity,calls=self._seven_calls,
            raw_emitted_tokens=self._seven_tokens,optimizer_states=0,trainable_parameters=0)


ActorRolloutRefWorker.init_model=Hooks.init_model
ActorRolloutRefWorker.seven_frozen_evidence=Hooks.seven_frozen_evidence
FrozenWorker=ActorRolloutRefWorker
