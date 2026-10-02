"""Isolated two-GPU smoke of actual CUDA/FSDP actors, without model downloads.

Uses tiny random Qwen weights and scripted token trajectories, not a benchmark
run. Parent verifies the physical GPUs before child processes initialize CUDA.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def check_free(indices):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for line in raw.splitlines():
        index, uuid, util, used, total = [x.strip() for x in line.split(',')]
        rows[index] = dict(uuid=uuid, utilization=float(util), used_mib=float(used), total_mib=float(total))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    for index in indices:
        row = rows[index]
        if row['utilization'] > 15 or row['used_mib'] / row['total_mib'] > .15:
            raise RuntimeError(f'GPU {index} no longer passes <=15% utilization and memory')
        if any(line.split(',')[0].strip() == row['uuid'] for line in apps.splitlines()):
            raise RuntimeError(f'GPU {index} has an existing compute process; select another empty GPU')
    return {index: rows[index] for index in indices}


def worker(rank, world, port, root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from transformers import Qwen2ForCausalLM, AutoTokenizer
    from omegaconf import OmegaConf
    import numpy as np
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from search_r1.baselines.actor import DistillationActor
    from search_r1.baselines.advantages import gigpo, tree_grpo, topd, grouped_center
    from search_r1.baselines.objectives import attach_policy_weights
    from search_r1.baselines.rollout import padded
    from search_r1.baselines.bapo import BoundaryStage, boundary_scores
    from search_r1.baselines.cosearch import GenerativeRanker

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(.10, rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=world)
    root = Path(root)
    checkpoint = str(root / 'tiny-model')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = torch.device('cuda', rank)
    result = {'rank': rank, 'device': torch.cuda.get_device_name(rank), 'methods': {}}
    try:
        prompts = padded([tokenizer.encode('Question: capital of France?'), tokenizer.encode('Question: France?')] * 2,
                          tokenizer.pad_token_id, True)
        texts = ['<think>query</think><search>France capital</search>', '<answer>Paris</answer>']
        first, observation, last = [tokenizer.encode(x) for x in (texts[0], '\n<information>Paris</information>\n', texts[1])]
        response_ids = first + observation + last
        response = padded([response_ids] * 4, tokenizer.pad_token_id)
        mask = torch.tensor([[1] * len(first) + [0] * len(observation) + [1] * len(last)] * 4).float()
        step_ids = torch.tensor([[0] * len(first) + [-1] * len(observation) + [1] * len(last)] * 4)
        inputs = torch.cat((prompts, response), 1)
        attention = (inputs != tokenizer.pad_token_id).long()
        positions = (attention.cumsum(-1) - 1).clamp_min(0) * attention
        uid = ['q'] * 4
        scores = torch.zeros_like(mask)
        scores[:, -1] = torch.tensor([1., 0., 1., 0.])
        for method in ('gigpo', 'tree_grpo', 'bapo', 'topd', 'opsd', 'cosearch'):
            # New model/optimizer for every algorithm. No user weights are opened.
            model = Qwen2ForCausalLM.from_pretrained(checkpoint, attn_implementation='flash_attention_2',
                                                   torch_dtype=torch.float32).to(device)
            model = FSDP(model, device_id=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            config = OmegaConf.create(dict(use_remove_padding=False, ulysses_sequence_parallel_size=1,
                ppo_mini_batch_size=4, ppo_micro_batch_size=1, use_dynamic_bsz=False,
                ppo_max_token_len_per_gpu=4096, state_masking=True, use_kl_loss=False, entropy_coeff=0.,
                clip_ratio=.2, ppo_epochs=2, grad_clip=1., baseline_method=method,
                distillation=dict(teacher_path=checkpoint, chunk_size=4, beta=0., temperature=1.,
                                  component_clip=.05, max_student_logits_gib=1.)))
            actor = DistillationActor(config, model, optimizer, tokenizer) if method == 'opsd' else DataParallelPPOActor(config, model, optimizer)
            batch = DataProto.from_dict(dict(prompts=prompts, responses=response, input_ids=inputs,
                            attention_mask=attention, position_ids=positions, loss_mask=mask,
                            info_mask=torch.cat(((prompts != tokenizer.pad_token_id).long(), mask), 1),
                            step_ids=step_ids),
                non_tensors={'uid': np.asarray(uid), 'reward_model': np.array([{'ground_truth': {'target': ['Paris']}}] * 4, dtype=object)},
                meta_info=dict(temperature=1., micro_batch_size=1, use_dynamic_bsz=False, max_token_len=4096)).to(device)
            old = actor.compute_log_prob(batch)
            self_check = actor.compute_log_prob(batch)
            torch.testing.assert_close(old, self_check)
            batch.batch['old_log_probs'] = old
            if method == 'gigpo':
                adv = gigpo(scores.to(device), mask.to(device), step_ids.to(device), [['root', 'obs']] * 4, uid)
            elif method == 'tree_grpo':
                adv = tree_grpo(scores.to(device), mask.to(device), uid, ['t1', 't1', 't2', 't2'])
            elif method == 'bapo':
                outcomes = boundary_scores(['Paris', 'wrong', 'Paris', 'wrong'], [['Paris']] * 4, uid, BoundaryStage())
                adv = grouped_center(torch.tensor(outcomes, device=device), uid)[:, None] * mask.to(device)
            elif method == 'cosearch':
                adv = grouped_center(scores.sum(-1).to(device), uid)[:, None] * mask.to(device)
            elif method == 'topd':
                # Controlled nonconstant teacher logp checks proximal CUDA/PPO path.
                teacher = old + scores.to(device) + torch.arange(old.shape[1], device=device)[None, :] * .01
                adv, _, _ = topd(teacher, old, mask.to(device), uid)
            else:
                adv = torch.zeros_like(old)
            batch.batch['advantages'] = adv
            if method in ('gigpo', 'tree_grpo', 'topd', 'cosearch'):
                attach_policy_weights(batch, method)
            before = [p.detach().clone() for p in model.parameters()]
            metrics = actor.update_policy(batch)
            torch.cuda.synchronize()
            delta = sum(float((p.detach() - b).abs().sum()) for p, b in zip(model.parameters(), before))
            grad_norms = metrics['actor/grad_norm']
            if not all(np.isfinite(g) and g > 0 for g in grad_norms):
                raise AssertionError(f'{method}: nonfinite or zero gradient')
            if not delta > 0 or metrics['actor/optimizer_steps'] != 2:
                raise AssertionError(f'{method}: no real parameter update or wrong epoch count')
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise AssertionError('Nonfinite updated parameter')
            result['methods'][method] = dict(parameter_delta_l1=delta, optimizer_steps=metrics['actor/optimizer_steps'],
                                                    grad_norms=grad_norms, weight_decay=0.)
            # Real FSDP full-state checkpoint collective, separate output per method.
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state = model.state_dict()
            if rank == 0:
                torch.save(state, root / (method + '-fsdp-state.pt'))
            del actor, model, optimizer, batch, before, state
            torch.cuda.empty_cache()
            dist.barrier()
        # Auxiliary ranker is intentionally single process on GPU0; GPU1 waits.
        if rank == 0:
            cfg = OmegaConf.create(dict(model_path=checkpoint, device='cuda:0', lr=.001, max_new_tokens=16,
                keep=3, temperature=1., threshold=.5, similarity=.8, split_availability=False,
                min_group_size=3, epochs=1, batch_size=3, clip_ratio=.2, grad_clip=1.))
            ranker = GenerativeRanker(cfg)
            for group in ranker.optimizer.param_groups:
                group["weight_decay"] = 0.
            ids = tokenizer('Question: capital? <ranking>[1,2,3]</ranking>', return_tensors='pt').input_ids[0]
            length = len(tokenizer.encode('<ranking>[1,2,3]</ranking>'))
            with torch.no_grad():
                logits = ranker.model(ids[None, :].to(device)).logits[:, -length - 1:-1].float()
                old = logits.log_softmax(-1).gather(-1, ids[-length:][None, :, None].to(device)).squeeze(0).squeeze(-1).cpu()
            ranker.events = [dict(question_uid='q', trajectory_uid=str(i), query='capital',
                candidates=['Paris', 'Rome', 'Berlin'], ranking=[1, 2, 3], input_ids=ids,
                response_length=length, old_logp=old) for i in range(3)]
            before = ranker.model.lm_head.weight.detach().clone()
            metrics = ranker.update({str(i): (float(i == 0), ['Paris'], True) for i in range(3)})
            delta = float((ranker.model.lm_head.weight.detach() - before).abs().sum())
            if not delta > 0 or metrics['ranker/training_events'] != 3:
                raise AssertionError('Ranker did not update')
            result['methods']['cosearch_ranker'] = dict(parameter_delta_l1=delta, optimizer_steps=ranker.cost['optimizer_steps'], weight_decay=0.)
            ranker.save(str(root / 'ranker-checkpoint'))
        result['peak_allocated_mib'] = torch.cuda.max_memory_allocated(rank) / 1024 ** 2
        (root / f'rank-{rank}.json').write_text(json.dumps(result, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', required=True, help='Physical nvidia-smi indices, e.g. 0,1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    indices = args.gpus.split(',')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    locks = []
    for index in indices:
        lock = open(root.parent / ('gpu-test-' + index + '.lock'), 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    snapshot = check_free(indices)
    time.sleep(1)
    snapshot = check_free(indices)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['NCCL_NVLS_ENABLE'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    import torch.multiprocessing as mp
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(['Question France Paris wrong capital teacher accepted answers '
         '<think>query</think><search>France capital</search><information>Paris</information>'
         '<answer>Paris</answer><ranking>[1,2,3]</ranking>'],
         trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             special_tokens=['<pad>', '<eos>', '<unk>']))
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    hf.save_pretrained(root / 'tiny-model')
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(hf), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        pad_token_id=hf.pad_token_id, eos_token_id=hf.eos_token_id, attention_dropout=0.))
    model.save_pretrained(root / 'tiny-model')
    del model
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    (root / 'allocation.json').write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),
                                                        selected=snapshot, torch=torch.__version__), indent=2))
    # Verify again immediately before creating GPU processes (model setup took time).
    check_free(indices)
    mp.spawn(worker, args=(len(indices), port, str(root)), nprocs=len(indices), join=True)
    results = [json.loads((root / f'rank-{i}.json').read_text()) for i in range(len(indices))]
    (root / 'result.json').write_text(json.dumps(dict(status='passed', ranks=results,
        scope='tiny CUDA/FSDP actor and ranker optimizer/checkpoint; no vLLM or production benchmark'), indent=2))
    print(json.dumps(dict(status='passed', output=str(root), ranks=results)), flush=True)


if __name__ == '__main__':
    main()
