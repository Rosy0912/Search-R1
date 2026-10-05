"""Preserve original Qwen rotary frequencies during parameter dtype conversion."""
import torch


def cast_preserving_qwen_rope(module, dtype):
    saved = []
    for child in module.modules():
        if child.__class__.__name__ not in ('Qwen2RotaryEmbedding', 'Qwen2DynamicNTKScalingRotaryEmbedding',
                                             'Qwen2LinearScalingRotaryEmbedding'):
            continue
        original = child._buffers.get('inv_freq')
        if original is None or original.is_meta or original.dtype != torch.float32:
            raise RuntimeError('Qwen rotary frequencies must be real original FP32 before dtype conversion')
        alias = getattr(child, 'original_inv_freq', None)
        if alias is not None and alias is not original:
            raise RuntimeError('Unexpected Qwen original_inv_freq alias before dtype conversion')
        saved.append((child, original.detach().clone(), alias is not None))
    module.to(dtype)
    for child, original, has_alias in saved:
        child._buffers['inv_freq'] = original.to(device=child._buffers['inv_freq'].device)
        if has_alias:
            child.original_inv_freq = child._buffers['inv_freq']
    return module
