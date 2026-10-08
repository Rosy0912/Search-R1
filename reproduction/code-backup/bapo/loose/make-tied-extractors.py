from pathlib import Path
helper='''def resolve_qwen2_tied_alias(base, checkpoint, before, after):
    """Only admit the exact observed redundant Qwen2 tied-head serialization."""
    if set(before) == set(after):
        return {}
    if set(before) - set(after) or set(after) - set(before) != {'lm_head.weight'}:
        raise ValueError('Base/checkpoint tensor key sets differ')
    embed, head = 'model.embed_tokens.weight', 'lm_head.weight'
    if embed not in before or embed not in after or head in before:
        raise ValueError('Invalid Qwen2 tied alias pattern')
    configs = [json.loads((root / 'config.json').read_text()) for root in (base, checkpoint)]
    for config in configs:
        if (config.get('model_type') != 'qwen2' or config.get('tie_word_embeddings') is not True
                or config.get('architectures') != ['Qwen2ForCausalLM']):
            raise ValueError('Extra head requires verified tied Qwen2 configurations')
    shape = [configs[0]['vocab_size'], configs[0]['hidden_size']]
    if shape != [configs[1]['vocab_size'], configs[1]['hidden_size']]:
        raise ValueError('Tied embedding dimensions differ')
    import torch
    from safetensors import safe_open
    checked = 0
    with safe_open(str(after[embed]), framework='pt', device='cpu') as f, safe_open(str(after[head]), framework='pt', device='cpu') as g:
        if (f.get_slice(embed).get_shape() != shape or g.get_slice(head).get_shape() != shape
                or f.get_slice(embed).get_dtype() != g.get_slice(head).get_dtype()):
            raise ValueError('Serialized tied weights shape/dtype differ')
        for start in range(0, shape[0], 1024):
            a = f.get_slice(embed)[start:start+1024]
            b = g.get_slice(head)[start:start+1024]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a, b):
                raise ValueError('Serialized tied weights must be entirely finite and exactly equal')
            checked += a.numel()
    before[head] = before[embed]
    return dict(status='QWEN2_TIED_DUPLICATE_EXACTLY_VERIFIED',
                base_logical_key=head, base_storage_key=embed,
                checkpoint_extra_key=head, checkpoint_equal_key=embed,
                shape=shape, elements_compared=checked, exact_equal=True)


'''
for name in ('bapo-smoke2-evidence-gpu64.py','bapo-five-iter-evidence-gpu64.py'):
    p=Path('outputs')/name;s=p.read_text();s=s.replace('def weight_witness(base, checkpoint):',helper+'def weight_witness(base, checkpoint):')
    s=s.replace("    if set(before) != set(after):\n        raise ValueError('Base/checkpoint tensor key sets differ')\n    # Validate every shape", "    alias = resolve_qwen2_tied_alias(base, checkpoint, before, after)\n    def storage_key(key):\n        return alias['base_storage_key'] if alias and key == alias['base_logical_key'] else key\n    # Validate every shape")
    s=s.replace('f.get_slice(key)', 'f.get_slice(storage_key(key))').replace('f.get_tensor(key)', 'f.get_tensor(storage_key(key))')
    s=s.replace("return dict(changed=True, tensor=key,", "return dict(changed=True, tensor=key, tied_weight_resolution=alias,")
    s=s.replace("return dict(changed=False, tensors_checked=checked,", "return dict(changed=False, tensors_checked=checked, tied_weight_resolution=alias,")
    s=s.replace("            model=AutoModelForCausalLM.from_pretrained", "            model, loading_info=AutoModelForCausalLM.from_pretrained")
    s=s.replace("trust_remote_code=False,torch_dtype='auto',attn_implementation='eager')", "trust_remote_code=False,torch_dtype='auto',attn_implementation='eager',output_loading_info=True)\n            if any(loading_info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):\n                raise ValueError('HF checkpoint loading reported missing/unexpected/mismatched/error weights')\n            tied_storage = (model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr())\n            if model.config.tie_word_embeddings and not tied_storage:\n                raise ValueError('HF model does not preserve declared tied weight storage')")
    s=s.replace("forward=forward_check,device='cpu', parameters=", "forward=forward_check,device='cpu', loading_info=loading_info, tied_input_output_storage=tied_storage, parameters=")
    p.with_name(p.stem+'-tied.py').write_text(s)
