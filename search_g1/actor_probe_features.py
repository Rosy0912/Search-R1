"""Read current actor states without training the actor or its generation head."""

from contextlib import nullcontext


def detached_last_token_features(module, input_ids, attention_mask, layers,
                                 *, micro_batch_size=2, position_ids=None,
                                 autocast_dtype=None, agree=None):
    import torch
    backbone = module.model
    def checked(callback, stage):
        return agree(callback, stage=stage) if agree is not None else callback()
    def validate_inputs():
        if type(micro_batch_size) is not int or micro_batch_size < 1:
            raise ValueError('Invalid feature microbatch')
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError('Feature input shape differs')
        if len(input_ids) == 0 or not attention_mask[:, -1].bool().all():
            raise ValueError('Probe inputs require nonempty left-padded contexts')
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise ValueError('Feature position IDs differ')
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('Duplicate or missing probe layers')
        if any(type(layer) is not int or not 1 <= layer <= len(backbone.layers) for layer in layers):
            raise ValueError('Invalid readout layer')
        return dict(rows=len(input_ids), length=input_ids.shape[1], layers=list(layers), micro=micro_batch_size)
    checked(validate_inputs, 'actor_probe_feature_inputs')
    modes = [(m, m.training) for m in module.modules()]
    captured = {}
    handles = []
    collected = {layer: [] for layer in layers}
    from search_g1.actor_probe_state import rng_state, restore_rng
    saved_rng = rng_state()
    def capture(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[:, -1, :].detach().float().cpu()
        return hook
    try:
        for layer in layers:
            handles.append(backbone.layers[layer - 1].register_forward_hook(capture(layer)))
        module.eval()
        for start in range(0, len(input_ids), micro_batch_size):
            stop = start + micro_batch_size
            captured.clear()
            amp = (torch.autocast(input_ids.device.type, dtype=autocast_dtype)
                   if autocast_dtype is not None else nullcontext())
            with torch.no_grad(), amp:
                kwargs = dict(input_ids=input_ids[start:stop],
                              attention_mask=attention_mask[start:stop], use_cache=False)
                if position_ids is not None:
                    kwargs['position_ids'] = position_ids[start:stop]
                module(**kwargs)
            def validate_capture():
                if set(captured) != set(layers):
                    raise ValueError('Missing current actor layer capture')
                for layer in layers:
                    value = captured[layer]
                    if value.grad_fn is not None or value.requires_grad or not torch.isfinite(value).all():
                        raise ValueError('Probe representation must be detached and finite')
                return dict(rows=next(iter(captured.values())).shape[0], layers=list(layers))
            checked(validate_capture, 'actor_probe_feature_micro_' + str(start // micro_batch_size))
            for layer in layers:
                collected[layer].append(captured[layer])
        return {layer: torch.cat(values) for layer, values in collected.items()}
    finally:
        for handle in handles:
            handle.remove()
        # Preserve mixed eval/train submodule settings as well as the root mode.
        for m, mode in modes:
            m.training = mode
        restore_rng(saved_rng)
