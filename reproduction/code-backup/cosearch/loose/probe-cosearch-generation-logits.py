"""Read-only fixed-HF tiny Qwen logits probe; no production model/job changes."""
import json
import platform
import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig

torch.manual_seed(31)
torch.set_num_threads(1)
weights = None
print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, device='cpu', dtype='float32', threads=1)))
for kernel in ('eager', 'sdpa'):
    config = Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=2, max_position_embeddings=256,
                        pad_token_id=0, eos_token_id=1, attention_dropout=0.,
                        attn_implementation=kernel)
    model = Qwen2ForCausalLM(config).eval()
    if weights is None:
        weights = {n: p.detach().clone() for n, p in model.state_dict().items()}
    else:
        model.load_state_dict(weights)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    for sample in (False, True):
        for eos_enabled in (False, True):
            with torch.no_grad():
                eos = int(model(ids).logits[0, -1].argmax()) if eos_enabled else None
                generation = GenerationConfig(max_new_tokens=8, do_sample=sample,
                    temperature=.7, top_k=0, top_p=1., num_beams=1,
                    pad_token_id=0, eos_token_id=eos,
                    output_logits=True, return_dict_in_generate=True)
                output = model.generate(ids, generation_config=generation)
                response = output.sequences[:, ids.shape[1]:]
                length = response.shape[1]
                cached = torch.stack([step.div(.7).log_softmax(-1).gather(
                    -1, response[:, i, None]).squeeze(-1)
                    for i, step in enumerate(output.logits)], 1)
                full = model(output.sequences, use_cache=False).logits[:, -length-1:-1].div(.7).log_softmax(-1).gather(-1, response[:, :, None]).squeeze(-1)
                delta = float((cached-full).abs().max())
                assert bool(cached.isfinite().all()) and delta < 2e-6
                if eos_enabled and not sample:
                    assert length == 1 and int(response[0, 0]) == eos
                print(json.dumps(dict(kernel=kernel,
                    actual_attention_class=type(model.model.layers[0].self_attn).__name__,
                    sample=sample, eos_enabled=eos_enabled, response_length=length,
                    ended_in_eos=eos is not None and int(response[0,-1]) == eos,
                    max_logp_delta=delta, finite=bool(cached.isfinite().all()))))
