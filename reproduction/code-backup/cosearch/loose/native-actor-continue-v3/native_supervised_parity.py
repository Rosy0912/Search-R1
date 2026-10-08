"""Tiny native CUDA supervised full-gradient/two-Adam parity, not real RL."""
import copy
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from search_r1.baselines.warm_start import SupervisedSequenceLoss
from search_r1.baselines.cosearch import eval_decoder_checkpoint


def run_native_supervised_parity(devices):
    master = torch.device('cuda', devices[0])
    torch.cuda.set_device(master)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for count in (2, 8):
        for rows in (1, 3, 8):
            torch.manual_seed(31)
            initial = Qwen2ForCausalLM(Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                attention_dropout=0., attn_implementation='sdpa')).eval().to(master)
            reference = copy.deepcopy(initial); actual = copy.deepcopy(initial)
            opts = [torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.) for m in (reference, actual)]
            examples = [torch.tensor([2+(i+j)%32 for j in range(4+i)]+[0], device=master) for i in range(rows)]
            lengths = torch.tensor([1+i%3 for i in range(rows)], device=master)
            width = max(map(len, examples))
            ids = torch.stack([F.pad(x, (width-len(x),0)) for x in examples])
            mask = torch.stack([F.pad(torch.ones_like(x),(width-len(x),0)) for x in examples])
            maximum_error = 0.
            for step in range(2):
                for opt in opts: opt.zero_grad(set_to_none=True)
                for x, length in zip(examples, lengths):
                    logits = reference(x[None], use_cache=False).logits[0,-length-1:-1]
                    (F.cross_entropy(logits, x[-length:])/rows).backward()
                objective = torch.nn.DataParallel(SupervisedSequenceLoss(actual),
                    device_ids=devices[:min(count,rows)], output_device=devices[0])
                with eval_decoder_checkpoint(actual, True):
                    objective(ids,mask,lengths,max_response_length=int(lengths.max())).mean().backward()
                for p,q in zip(reference.parameters(),actual.parameters()):
                    maximum_error=max(maximum_error,float((p.grad-q.grad).abs().max()))
                    torch.testing.assert_close(p.grad,q.grad,atol=3e-7,rtol=3e-5)
                norms=[torch.nn.utils.clip_grad_norm_(m.parameters(),1.) for m in (reference,actual)]
                torch.testing.assert_close(norms[0],norms[1],atol=3e-7,rtol=3e-5)
                for opt in opts:opt.step()
                for p,q in zip(reference.parameters(),actual.parameters()):
                    torch.testing.assert_close(p,q,atol=2e-7,rtol=3e-5)
                for a,b in zip(opts[0].state.values(),opts[1].state.values()):
                    for key in ('step','exp_avg','exp_avg_sq'):
                        torch.testing.assert_close(a[key],b[key],atol=3e-7,rtol=3e-5)
            results.append(dict(devices=count,rows=rows,two_adam_steps=True,maximum_gradient_error=maximum_error))
            del initial,reference,actual,opts,objective
    for device in devices:
        with torch.cuda.device(device): torch.cuda.empty_cache()
    return results
