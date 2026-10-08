"""Call original trainer.main; promote only outer-old storage before its setter."""
import importlib
import os
from pathlib import Path
import sys
from recovery_common import bind, checked_json, install_original, receipt, require


def option(name):
    require(sys.argv.count(name) == 1, 'Exactly one original ' + name + ' required')
    index = sys.argv.index(name)
    require(index + 1 < len(sys.argv), 'Missing original option value')
    return sys.argv[index + 1]


def adapter_values(model, name):
    marker = '.' + name + '.'
    return {n.replace(marker, '.'): p for n, p in model.named_parameters() if marker in n}


def repaired_setter(original, digest, expected_digest, record):
    def setter(model, weights, *args, **kwargs):
        adapter = kwargs.get('adapter_name', args[0] if args else 'default')
        if adapter != 'outer_old':
            return original(model, weights, *args, **kwargs)
        import torch
        old = adapter_values(model, 'outer_old')
        current = adapter_values(model, 'default')
        require(len(old) == len(current) == len(weights) == 504 and set(old) == set(weights),
                'Actual 504 named outer-old/current/file tensors differ')
        require(all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in weights.values()),
                'Original outer-start file is not finite FP32')
        require(all(p.dtype == torch.float32 for p in current.values()), 'Current optimizer master is not FP32')
        require(digest(weights) == expected_digest, 'Input old tensors differ from immutable original start')
        before_current = digest(current)
        before_dtypes = sorted({str(p.dtype) for p in old.values()})
        for parameter in old.values():
            require(parameter.is_floating_point(), 'Outer-old parameter is not floating point')
            parameter.data = parameter.data.to(dtype=torch.float32)
        result = original(model, weights, *args, **kwargs)
        actual = adapter_values(model, 'outer_old')
        require(all(p.dtype == torch.float32 for p in actual.values()) and digest(actual) == expected_digest,
                'Repaired actual outer-old FP32 tensors still differ from bound start')
        require(digest(adapter_values(model, 'default')) == before_current, 'Repair changed current adapter tensors')
        record(dict(adapter_name='outer_old', tensors=504, old_storage_dtypes_before=before_dtypes,
                    old_storage_dtypes_after=['torch.float32'], original_input_tensor_sha256=expected_digest,
                    actual_old_tensor_sha256=digest(actual), current_tensor_sha256=before_current,
                    current_unchanged=True, promotion_before_original_setter=True,
                    original_setter_return_preserved=True, original_digest_and_frozen_guard_not_modified=True))
        return result
    return setter


def main():
    require(os.environ.get('E1_DTYPE_REPAIR_MANIFEST') and os.environ.get('E1_DTYPE_REPAIR_MANIFEST_SHA256'),
            'Explicit overlay repair manifest environment required')
    repair_ref = dict(path=os.environ['E1_DTYPE_REPAIR_MANIFEST'], sha256=os.environ['E1_DTYPE_REPAIR_MANIFEST_SHA256'])
    config_ref = dict(path=option('--config'), sha256=option('--config-sha256'))
    run_id = os.environ.get('E1_RUN_ID')
    value, config, root, manifest, _ = bind(repair_ref, config_ref, run_id)
    require(int(os.environ.get('WORLD_SIZE', '0')) == 8 and 0 <= int(os.environ.get('RANK', '-1')) < 8,
            'Only original actual eight-rank trainer may use this launcher')
    outer_ref = dict(path=option('--outer-manifest'), sha256=option('--outer-manifest-sha256'))
    outer = checked_json(outer_ref)
    require(outer['run_id'] == run_id and outer['config_sha256'] == config_ref['sha256'], 'Original outer/run/config differs')
    old_ref = outer['previous_complete_checkpoint'] or config['common_initial_adapter_manifest']
    require(outer['outer_old_adapter'] == old_ref, 'Original outer-start adapter changed')
    old = checked_json(old_ref)
    expected = old['all_rank_adapter_sha256'][0] if 'adapter' in old else old['adapter_tensor_sha256']
    install_original(root, manifest)
    trainer = importlib.import_module('trainer')
    import peft
    require(peft.__version__ == '0.13.2', 'Original PEFT version differs')
    native_setter = peft.set_peft_model_state_dict
    rank = int(os.environ['RANK'])
    def record(fields):
        receipt(config, repair_ref, 'trainer-rank-' + str(rank), dict(config=config_ref, outer_manifest=outer_ref,
                original_outer_start=old_ref, rank=rank, world_size=8, original_trainer=str(root / 'trainer.py'),
                repair_source_manifest=value['source_manifest'], publication=value['publication'], **fields))
    peft.set_peft_model_state_dict = repaired_setter(native_setter, trainer.tensor_digest, expected, record)
    try:
        trainer.main()
    finally:
        peft.set_peft_model_state_dict = native_setter


if __name__ == '__main__':
    main()
