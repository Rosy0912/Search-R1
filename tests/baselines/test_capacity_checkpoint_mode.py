"""Real HF eval/train counterexample: an enabled checkpoint flag alone does not recompute."""
import importlib.util
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('capacity_probe_runtime', ROOT / 'reproduction/baselines/opsd-validation-20261003/runtime/microbatch_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_eval_checkpoint_flag_is_inactive_but_audited_train_recomputes_every_decoder():
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.,
        attn_implementation='eager'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.tensor([[1, 2, 3, 4]])
    # Equivalent to from_pretrained's default eval mode: true checkpoint flag,
    # but no actual decoder backward recomputation.
    model.eval()
    assert model.model.gradient_checkpointing and not model.model.training
    eval_counts = [0, 0]
    hooks = []
    for index, decoder in enumerate(model.model.layers):
        def count(_module, _args, index=index):
            eval_counts[index] += 1
        hooks.append(decoder.register_forward_pre_hook(count))
    model(ids, use_cache=False).logits.square().mean().backward()
    assert eval_counts == [1, 1]
    for hook in hooks:
        hook.remove()
    flags, counts, hooks = probe.audited_checkpoint_training(model, model)
    try:
        assert flags['student_decoder_container_training'] and flags['checkpoint_use_reentrant'] is False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, use_cache=False).logits.square().mean()
            assert set(counts.values()) == {2*step+1}
            loss.backward()
            assert set(counts.values()) == {2*(step+1)}
            optimizer.step()
        assert set(counts.values()) == {4}
    finally:
        for hook in hooks:
            hook.remove()
