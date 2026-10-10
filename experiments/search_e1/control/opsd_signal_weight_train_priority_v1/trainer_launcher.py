"""Run exact original trainer; repair only named outer-old FP32 storage."""
import importlib
import os
import sys

from recovery_common import bind, checked_json, install_original, receipt, require


def option(name):
    require(sys.argv.count(name) == 1, 'Exactly one original ' + name + ' required')
    index = sys.argv.index(name)
    require(index + 1 < len(sys.argv), 'Missing original option value')
    return sys.argv[index + 1]


def adapter_values(model, name):
    marker = '.' + name + '.'
    values = {}
    for key, parameter in model.named_parameters():
        if marker in key:
            canonical = key.replace(marker, '.')
            require(canonical not in values, 'Named adapter canonical key collision')
            values[canonical] = parameter
    return values


def repaired_setter(original, digest, expected_digest, record):
    """Promote BEFORE setter; post-load BF16 upcasting cannot recover source bits."""
    def setter(model, weights, *args, **kwargs):
        adapter = kwargs.get('adapter_name', args[0] if args else 'default')
        if adapter != 'outer_old':
            return original(model, weights, *args, **kwargs)
        import torch
        old = adapter_values(model, 'outer_old')
        current = adapter_values(model, 'default')
        require(len(old) == len(current) == len(weights) == 504
                and set(old) == set(current) == set(weights),
                'Actual504 named outer-old/current/file tensors differ')
        require(all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in weights.values()),
                'Original outer-start file is not finite FP32')
        require(all(parameter.dtype == torch.float32 for parameter in current.values()),
                'Current optimizer master is not FP32')
        require(all(parameter.shape == weights[name].shape for name, parameter in old.items()),
                'Original named outer-old/file tensor shapes differ')
        require(digest(weights) == expected_digest,
                'Input old tensors differ from immutable original start')
        before_current = digest(current)
        old_ids = {id(parameter) for parameter in old.values()}
        nonold = [(name, parameter, parameter._version, parameter.dtype, parameter.requires_grad)
                  for name, parameter in model.named_parameters() if id(parameter) not in old_ids]
        before_dtypes = sorted({str(parameter.dtype) for parameter in old.values()})
        for parameter in old.values():
            require(parameter.is_floating_point(), 'Outer-old parameter is not floating point')
            parameter.data = parameter.data.to(dtype=torch.float32)
        result = original(model, weights, *args, **kwargs)
        actual = adapter_values(model, 'outer_old')
        require({id(parameter) for parameter in actual.values()} == old_ids,
                'Original setter replaced named adapter parameter objects')
        require(all(parameter.dtype == torch.float32 for parameter in actual.values())
                and digest(actual) == expected_digest,
                'Repaired actual outer-old FP32 tensors still differ from bound start')
        require(digest(weights) == expected_digest, 'Original setter mutated input source tensors')
        require(digest(adapter_values(model, 'default')) == before_current,
                'Repair changed current adapter tensors')
        after_nonold = [(name, parameter, parameter._version, parameter.dtype, parameter.requires_grad)
                       for name, parameter in model.named_parameters() if id(parameter) not in old_ids]
        require(len(after_nonold) == len(nonold) and all(
                left[0] == right[0] and left[1] is right[1] and left[2:] == right[2:]
                for left, right in zip(nonold, after_nonold)),
                'Repair changed non-old parameters, dtype, versions or trainable flags')
        record(dict(adapter_name='outer_old', tensors=504,
                    old_storage_dtypes_before=before_dtypes, old_storage_dtypes_after=['torch.float32'],
                    original_input_tensor_sha256=expected_digest, actual_old_tensor_sha256=digest(actual),
                    current_tensor_sha256=before_current, current_unchanged=True,
                    nonold_parameter_objects_versions_dtypes_flags_unchanged=True,
                    promotion_before_original_setter=True, original_setter_return_preserved=True,
                    original_digest_and_frozen_guard_not_modified=True))
        return result
    return setter


def run(repair_ref, config_ref, outer_ref, run_id, rank, world):
    value, config, root, manifest, _ = bind(repair_ref, config_ref, run_id)
    require(world == 4 and type(rank) is int and 0 <= rank < 4,
            'Only original actual four-rank trainer may use this launcher')
    require(int(os.environ.get('WORLD_SIZE', '0')) == world
            and int(os.environ.get('RANK', '-1')) == rank,
            'Actual distributed rank/world differs')
    outer = checked_json(outer_ref)
    require(outer['run_id'] == run_id and outer['config_sha256'] == config_ref['sha256'],
            'Original outer/run/config differs')
    old_ref = outer.get('previous_complete_checkpoint') or config['common_initial_adapter_manifest']
    require(outer['outer_old_adapter'] == old_ref, 'Original outer-start adapter changed')
    old = checked_json(old_ref)
    expected = old['all_rank_adapter_sha256'][0] if 'adapter' in old else old['adapter_tensor_sha256']
    install_original(root, manifest)
    trainer = importlib.import_module('trainer')
    import peft
    require(peft.__version__ == '0.13.2', 'Original PEFT version differs')
    native_setter = peft.set_peft_model_state_dict
    def record(fields):
        receipt(config, repair_ref, 'trainer-rank-' + str(rank),
                dict(config=config_ref, outer_manifest=outer_ref, original_outer_start=old_ref,
                     rank=rank, world_size=4, original_trainer=str(root / 'trainer.py'),
                     repair_source_manifest=value['source_manifest'], publication=value['publication'], **fields))
    peft.set_peft_model_state_dict = repaired_setter(native_setter, trainer.tensor_digest, expected, record)
    try:
        return trainer.main()
    finally:
        peft.set_peft_model_state_dict = native_setter


def main():
    require(os.environ.get('E1_SIGNAL_RECOVERY_MANIFEST')
            and os.environ.get('E1_SIGNAL_RECOVERY_MANIFEST_SHA256'),
            'Explicit recovery manifest environment required')
    return run(dict(path=os.environ['E1_SIGNAL_RECOVERY_MANIFEST'],
                    sha256=os.environ['E1_SIGNAL_RECOVERY_MANIFEST_SHA256']),
               dict(path=option('--config'), sha256=option('--config-sha256')),
               dict(path=option('--outer-manifest'), sha256=option('--outer-manifest-sha256')),
               os.environ.get('E1_RUN_ID'), int(os.environ.get('RANK', '-1')),
               int(os.environ.get('WORLD_SIZE', '0')))


if __name__ == '__main__':
    main()
