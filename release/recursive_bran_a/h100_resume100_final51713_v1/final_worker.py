"""Final-only worker: frozen policy cache and per-rank live generation timings."""
import json
import os
import time

from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class FrozenFinalWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        import subprocess
        gpu_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        gpu_uuids = [row.split(',')[1].strip() for row in gpu_rows
            if row.split(',')[0].strip() == str(os.getpid())]
        assert len(set(gpu_uuids)) == 1, 'Final DP worker must have one actual GPU UUID'
        self._final_gpu_uuid = gpu_uuids[0]
        manager = self.rollout_sharding_manager
        prepare = manager.prepare_policy_cache
        version = os.environ['RBA_FINAL_CHECKPOINT_SHA']

        def frozen_cache(enabled, ignored_version):
            # This process never updates parameters. Greedy validation can retain
            # exactly the same policy and KV allocation across calls/batches.
            return prepare(True, ('final200', version))

        manager.prepare_policy_cache = frozen_cache
        generate = self.rollout.generate_sequences
        self._final_calls = 0

        def timed_generate(prompts):
            import torch
            assert prompts.meta_info.get('validate') is True
            assert prompts.meta_info.get('do_sample') is False
            assert not prompts.meta_info.get('recompute_log_prob', False)
            started = time.monotonic()
            self._final_calls += 1
            rank = torch.distributed.get_rank()
            print('FINAL51713_GPU=' + json.dumps(dict(event='start', rank=rank,
                pid=os.getpid(), call=self._final_calls, rows=len(prompts),
                cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu_uuid=self._final_gpu_uuid,
                checkpoint_sha=version, time=time.time())), flush=True)
            output = generate(prompts)
            width = output.batch['responses'].shape[-1]
            tokens = int(output.batch['attention_mask'][:, -width:].sum().item())
            seconds = time.monotonic() - started
            print('FINAL51713_GPU=' + json.dumps(dict(event='complete', rank=rank,
                pid=os.getpid(), call=self._final_calls, rows=len(prompts),
                gpu_uuid=self._final_gpu_uuid,
                generated_tokens=tokens, seconds=seconds,
                tokens_per_second=tokens / max(seconds, 1e-9), time=time.time())), flush=True)
            return output

        self.rollout.generate_sequences = timed_generate
        return result
