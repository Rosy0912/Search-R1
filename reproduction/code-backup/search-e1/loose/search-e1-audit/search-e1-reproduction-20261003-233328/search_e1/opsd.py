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
        dist.init_process_group('nccl', timeout=timedelta(hours=3))
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
    best = evaluate(agent, validation) if rank == 0 else 0.
    if rank == 0:
        model.save_pretrained(out/'best_adapter')  # Include baseline to avoid selecting a harmful adapter.
    barrier()
    history = [{'epoch': 0, 'validation_em': best}]
    for epoch in range(a.epochs):
        random.shuffle(examples)
        total = 0.
        trainable = [p for p in model.parameters() if p.requires_grad]
        global_batch = a.accumulation * world
        for start in range(0, len(examples), global_batch):
            batch = examples[start:start+global_batch]
            local = batch[rank::world]
            optimizer.zero_grad(set_to_none=True)
            for example in local:
                loss = pair_loss(model, example, a.clip, a.chunk)
                # All-reduce SUM yields the exact mean over the global trajectory batch,
                # including unequal last batches and ranks with zero remaining examples.
                (loss / len(batch)).backward()
                total += loss.detach().item()
            if world > 1:
                for parameter in trainable:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            if rank == 0:
                print({'epoch': epoch+1, 'pairs': start+len(batch), 'local_loss_sum': total}, flush=True)
        barrier()
        em = evaluate(agent, validation) if rank == 0 else 0.
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
