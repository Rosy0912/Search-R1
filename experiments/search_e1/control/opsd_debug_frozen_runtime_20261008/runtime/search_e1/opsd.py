"""Frozen base teacher, trainable LoRA student; aligned response tokens, Eq. 7 loss."""
import argparse
import random
import os
from datetime import timedelta
import torch.distributed as dist
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from .core import prompt, response_encoding, clipped_forward_kl
from .io import read_jsonl, write_json
from .rollout import SearchAgent, evaluate

def encoded_pair(tokenizer, pair, max_context):
    response, mask = response_encoding(tokenizer, pair['student'], pair.get('student_policy_spans'))
    prefixes = [tokenizer.apply_chat_template(prompt(pair['question'], ref), tokenize=True, add_generation_prompt=True)
                for ref in (None, pair['reference'])]
    if not response or not sum(mask) or any(len(p) + len(response) > max_context for p in prefixes):
        return None  # Never independently truncate teacher/student responses.
    return prefixes, response, mask

def pair_loss(model, encoded, clip, chunk):
    prefixes, response, mask = encoded
    device = next(model.parameters()).device
    base = model.get_base_model()
    transformer, head = base.base_model, base.get_output_embeddings()
    student_ids = torch.tensor([prefixes[0] + response], device=device)
    teacher_ids = torch.tensor([prefixes[1] + response], device=device)
    model.eval()  # Frozen teacher, with dropout disabled.
    with torch.no_grad(), model.disable_adapter():
        teacher = transformer(input_ids=teacher_ids, use_cache=False).last_hidden_state[0]
        teacher = teacher[len(prefixes[1])-1:len(prefixes[1])+len(response)-1].detach()
    model.train()  # Activate transformer gradient checkpointing; LoRA dropout is zero.
    student = transformer(input_ids=student_ids, use_cache=False).last_hidden_state[0]
    student = student[len(prefixes[0])-1:len(prefixes[0])+len(response)-1]
    positions = torch.tensor(mask, device=device, dtype=torch.bool).nonzero().flatten()
    loss = student.sum() * 0.
    for indices in positions.split(chunk):
        sh, th = student[indices], teacher[indices]
        def loss_chunk(s, t):
            return clipped_forward_kl(head(s), head(t), clip).sum()
        # Recompute vocabulary logits during backward to bound activation memory.
        loss = loss + checkpoint(loss_chunk, sh, th, use_reentrant=False)
    return loss / len(positions)

def _opsd_collective_reject(local_failed, world, message):
    failed = torch.tensor(int(local_failed), device='cuda', dtype=torch.int64)
    if world > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise FloatingPointError(message)


def _opsd_global_validation(agent, validation, rank, world):
    local = validation[rank::world]
    local_em = evaluate(agent, local) if local else 0.
    counts = torch.tensor([round(local_em * len(local)), len(local)],
                          device='cuda', dtype=torch.int64)
    if world > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if counts[1].item() == 0:
        raise RuntimeError('Validation must be nonempty.')
    return counts[0].item() / counts[1].item()


def main():
    p = argparse.ArgumentParser()
    for name in ('model', 'pairs', 'validation', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--retriever-url', default='http://127.0.0.1:8000/retrieve')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--clip', type=float, default=10.)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--max-context', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=24))
    def barrier():
        if world > 1:
            dist.barrier()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.,
                                          target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    pairs = read_jsonl(a.pairs)
    encoded = [encoded_pair(tokenizer, pair, a.max_context) for pair in pairs]
    examples = [item for item in encoded if item is not None]
    if not examples:
        raise RuntimeError('No trainable pairs within context budget.')
    validation = read_jsonl(a.validation)
    if not validation:
        raise RuntimeError('Validation must be nonempty and separate from final test splits.')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(out/'config.json', dict(vars(a), usable_pairs=len(examples), skipped_pairs=len(pairs)-len(examples), world_size=world))
    agent = SearchAgent(model, tokenizer, a.retriever_url, max_context=a.max_context)
    best = _opsd_global_validation(agent, validation, rank, world)
    if rank == 0:
        model.save_pretrained(out/'best_adapter')  # Include baseline to avoid selecting a harmful adapter.
    barrier()
    history = [{'epoch': 0, 'validation_em': best}]
    opsd_updates = 0
    tracked = next(p for name,p in model.named_parameters() if "lora_B" in name)
    audit_initial = tracked.detach().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(a.epochs):
        random.shuffle(examples)
        total = 0.
        trainable = [p for p in model.parameters() if p.requires_grad]
        global_batch = a.accumulation * world
        for start in range(0, len(examples), global_batch):
            batch = examples[start:start+global_batch]
            local = batch[rank::world]
            optimizer.zero_grad(set_to_none=True)
            local_failed = False
            for example in local:
                loss = pair_loss(model, example, a.clip, a.chunk)
                # All-reduce SUM yields the exact mean over the global trajectory batch,
                # including unequal last batches and ranks with zero remaining examples.
                if not torch.isfinite(loss).all().item():
                    local_failed = True
                    continue
                (loss / len(batch)).backward()
                total += loss.detach().item()
            _opsd_collective_reject(local_failed, world, 'Nonfinite OPSD loss; all ranks refuse update')
            if world > 1:
                for parameter in trainable:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            invalid_grad = not torch.isfinite(grad_norm).all().item()
            invalid_grad = invalid_grad or any(
                p.grad is not None and not torch.isfinite(p.grad).all().item() for p in trainable)
            _opsd_collective_reject(invalid_grad, world, 'Nonfinite OPSD gradient; all ranks refuse update')
            optimizer.step()
            opsd_updates += 1
            torch.cuda.synchronize()
            audit_finite = all(torch.isfinite(p).all().item() for p in trainable)
            audit_changed = not torch.equal(audit_initial, tracked.detach().cpu())
            audit_steps = sorted(set(float(state['step']) for state in optimizer.state.values() if 'step' in state))
            _opsd_collective_reject(not audit_finite or not audit_changed or audit_steps != [float(opsd_updates)], world, 'OPSD real-update audit failed')
            free_bytes,total_bytes = torch.cuda.mem_get_info()
            evidence = dict(rank=rank, updates=opsd_updates, source_grpo_step=100, display_step=100+opsd_updates,
                finite=audit_finite, changed_from_initial=audit_changed, adam_steps=audit_steps, grad_norm=float(grad_norm),
                max_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes=free_bytes, total_bytes=total_bytes)
            all_evidence=[None]*world
            if world > 1: dist.all_gather_object(all_evidence,evidence)
            else: all_evidence=[evidence]
            if opsd_updates in (1,5):
                if rank == 0: model.save_pretrained(out/f'adapter_update_{opsd_updates}')
                torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),updates=opsd_updates,rank=rank),out/f'update_{opsd_updates}_rank_{rank}_state.pt')
                barrier()
            if rank == 0:
                report=dict(status='RUNNING',world_size=world,opsd_updates=opsd_updates,source_grpo_step=100,display_step=100+opsd_updates,
                    ranks=all_evidence,five_real_updates_accepted=opsd_updates>=5,
                    reduction='Mean over valid response tokens per trajectory, then mean over global trajectory batch',
                    source_model=a.model)
                write_json(out/'live-update.json',report)
                if opsd_updates == 5: write_json(out/'five-update-acceptance.json',report)
                print('REAL_OPSD_UPDATE',report,flush=True)
            if rank == 0:
                print({'epoch': epoch+1, 'pairs': start+len(batch), 'local_loss_sum': total}, flush=True)
        barrier()
        em = _opsd_global_validation(agent, validation, rank, world)
        if world > 1:
            loss_sum = torch.tensor(total, device='cuda', dtype=torch.float64)
            dist.all_reduce(loss_sum)
            total = loss_sum.item()
        if rank == 0:
            model.save_pretrained(out/f'adapter_epoch_{epoch+1}')
            history.append({'epoch': epoch+1, 'validation_em': em, 'loss': total/len(examples)})
            if em > best:
                best = em
                model.save_pretrained(out/'best_adapter')
            write_json(out/'history.json', history)
        barrier()
    # Reload selected checkpoint before merging, even if last epoch was worse.
    from peft import PeftModel
    del model, base, optimizer, agent
    torch.cuda.empty_cache()
    if rank == 0:
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
        selected = PeftModel.from_pretrained(base, out/'best_adapter')
        selected.merge_and_unload().save_pretrained(out/'merged', safe_serialization=True)
        tokenizer.save_pretrained(out/'merged')
        write_json(out/'history.json', history)
    barrier()
    if world > 1:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
