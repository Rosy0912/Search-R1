"""Actual frozen PEFT CPU fixture with a tiny504-LoRA Qwen2, no real model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

from recovery_common import checked_json, checked, require, ORIGINAL_TRAINER_SHA, ORIGINAL_MANIFEST_SHA
from trainer_launcher import adapter_values, repaired_setter


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-helper-root', required=True)
    parser.add_argument('--original-helper-manifest', required=True)
    parser.add_argument('--original-helper-manifest-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Real PEFT fixture requires empty CUDA visibility')
    require(args.original_helper_manifest_sha256 == ORIGINAL_MANIFEST_SHA,
            'Fixture requires the exact immutable original helper manifest')
    root = Path(args.original_helper_root).resolve()
    manifest = checked_json(dict(path=args.original_helper_manifest,
                                 sha256=args.original_helper_manifest_sha256))
    entry = manifest['files']['trainer.py']
    require(entry['sha256'] == ORIGINAL_TRAINER_SHA, 'Fixture original trainer identity differs')
    checked(dict(path=str(root / 'trainer.py'), sha256=ORIGINAL_TRAINER_SHA))
    target_ref = dict(path=str(root / 'targets.py'), sha256=manifest['files']['targets.py']['sha256'])
    target_bytes = checked(target_ref)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader('_actual_targets_for_peft_fixture', loader=None))
    module.__file__ = target_ref['path']
    exec(compile(target_bytes, target_ref['path'], 'exec'), module.__dict__)
    import torch
    import peft
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    require((torch.__version__.split('+')[0], peft.__version__, transformers.__version__)
            == ('2.4.0', '0.13.2', '4.46.3'), 'Actual original scientific dependency versions differ')
    require(not torch.cuda.is_initialized(), 'CPU fixture already initialized CUDA')
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32)).to(dtype=torch.bfloat16, device='cpu')
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    current = adapter_values(model, 'default')
    require(len(current) == 504 and all(parameter.dtype == torch.float32 for parameter in current.values()),
            'Real PEFT default adapter is not504FP32')
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-6)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    model.add_adapter('outer_old', config)
    old = adapter_values(model, 'outer_old')
    require(len(old) == 504 and {parameter.dtype for parameter in old.values()} == {torch.bfloat16},
            'Actual PEFT named-adapter BF16 creation did not reproduce production precondition')
    inputs = {name: torch.full(parameter.shape, 0.12345679 + number / 100000,
                              dtype=torch.float32, device='cpu')
              for number, (name, parameter) in enumerate(old.items())}
    digest = module.tensor_digest(inputs)
    native_return = peft.set_peft_model_state_dict(model, inputs, adapter_name='outer_old')
    native_failed = False
    try:
        module.SharedTargets(model, object(), old_adapter='outer_old',
                             old_identity={'adapter_tensor_sha256': digest})
    except ValueError as error:
        require('actual tensors differ' in str(error), 'Native guard failed for a different cause')
        native_failed = True
    require(native_failed, 'Native BF16 setter did not reproduce bound-tensor mismatch')
    for parameter in adapter_values(model, 'outer_old').values():
        parameter.data = parameter.data.float()
    require(module.tensor_digest(adapter_values(model, 'outer_old')) != digest,
            'Post-load promotion unexpectedly recovered discarded FP32 bits')
    for parameter in adapter_values(model, 'outer_old').values():
        parameter.data = parameter.data.bfloat16()
    before_current = module.tensor_digest(adapter_values(model, 'default'))
    records = []; rng = torch.get_rng_state().clone(); py_rng = random.getstate()
    result = repaired_setter(peft.set_peft_model_state_dict, module.tensor_digest, digest, records.append)(
                    model, inputs, adapter_name='outer_old')
    require(isinstance(result, type(native_return)), 'Actual original setter return type changed')
    require(torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate(), 'Repair consumed RNG')
    targets = module.SharedTargets(model, object(), old_adapter='outer_old',
                    old_identity={'adapter_tensor_sha256': digest})
    targets.check_old()
    require(module.tensor_digest(adapter_values(model, 'default')) == before_current,
            'Actual real-PEFT current adapter changed')
    require(optimizer_ids == [id(parameter) for group in optimizer.param_groups for parameter in group['params']],
            'Actual optimizer parameter objects changed')
    require({id(parameter) for parameter in model.parameters() if parameter.requires_grad} == set(optimizer_ids),
            'Actual native target activation leaked old parameters into optimizer inventory')
    require(not torch.cuda.is_initialized() and all(parameter.device.type == 'cpu' for parameter in model.parameters()),
            'Actual CPU fixture touched CUDA or moved a parameter offCPU')
    print(json.dumps(dict(schema='opsd-signal-real-peft-cpu-storage-fixture-v1', accepted=True,
          versions=dict(torch=torch.__version__, peft=peft.__version__, transformers=transformers.__version__),
          tiny_qwen_hidden_size=8, tiny_qwen_layers=36, actual_lora_tensors=504,
          native_named_old_bf16_mismatch_reproduced=True, postload_upcast_cannot_recover=True,
          exact_input_fp32_digest=digest, actual_old_fp32_digest=module.tensor_digest(adapter_values(model, 'outer_old')),
          original_native_target_guard_passed=True, current_unchanged=True, optimizer_parameter_objects_unchanged=True,
          no_rng_consumed=True, cuda_initialized=False, repair_records=records,
          original_targets=target_ref, scope='Tiny synthetic actual PEFT CPU fixture; no trained checkpoint or GPU acceptance'), sort_keys=True))


if __name__ == '__main__':
    main()
