"""Unexecuted checkpoint controls for separate pre-execution source review.

CPU collectives below are sequential control mocks, never eight real ranks.
Tiny Adam/module/HF fixtures prove only these local controls and loader
semantics; they cannot certify Torch FSDP, a real 3B model, or production IO.
No fixture downloads assets or creates a Ray/CUDA/distributed runtime.
"""
import ast
import copy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from search_r1.baselines.instruct300 import equal_state, hf_export_identity, reload_hf_checkpoint_cpu
from search_r1.baselines.recovery import cpu_snapshot, restore_rng, rng_state, worker_training_state


ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCE = ROOT / 'verl/workers/fsdp_workers.py'


def source_function(name, *, torch_surface=torch):
    """Load the exact reviewed helper/method without importing worker startup.

    This strips only the dispatch decorator. Body code is taken verbatim from
    the source AST and executed only when these tests are later authorized.
    """
    tree = ast.parse(WORKER_SOURCE.read_text())
    functions = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ActorRolloutRefWorker':
            functions.extend(node.body)
    selected = copy.deepcopy(next(node for node in functions
                                  if isinstance(node, ast.FunctionDef) and node.name == name))
    selected.decorator_list = []
    namespace = dict(torch=torch_surface, os=os)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
                 str(WORKER_SOURCE), 'exec'), namespace)
    return namespace[name]


@pytest.fixture(autouse=True)
def preserve_cpu_test_rng(monkeypatch):
    # These are CPU tests even when the later runner has visible CUDA devices.
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    initial = cpu_snapshot(rng_state())
    yield
    restore_rng(initial)


class SequentialCPUCollectives:
    """Fixed global decisions, observed separately by each simulated rank."""

    def __init__(self, rank, failures):
        self.rank, self.failures, self.events = rank, failures, []
        self.ReduceOp = SimpleNamespace(MAX='MAX')

    def all_reduce(self, value, op):
        self.events.append('MAX')
        assert op == self.ReduceOp.MAX
        assert int(value) == int(self.failures[self.rank] is not None)
        value.fill_(int(any(error is not None for error in self.failures)))

    def get_world_size(self):
        return len(self.failures)

    def all_gather(self, rows, local):
        self.events.append('bounded_diagnostics')
        assert local.numel() == 1024 and local.dtype == torch.uint8
        for row, error in zip(rows, self.failures):
            row.zero_()
            if error is not None:
                encoded = (type(error).__name__ + ': ' + str(error)).encode()[:1024]
                row[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)


def cpu_stage_surface(distributed):
    def tensor(value, **kwargs):
        kwargs.pop('device', None)
        return torch.tensor(value, **kwargs)

    def zeros(value, **kwargs):
        kwargs.pop('device', None)
        return torch.zeros(value, **kwargs)

    return SimpleNamespace(device=lambda *_: torch.device('cpu'),
        cuda=SimpleNamespace(current_device=lambda: 0), tensor=tensor, zeros=zeros,
        empty_like=torch.empty_like, int32=torch.int32, uint8=torch.uint8, distributed=distributed)


@pytest.mark.parametrize('phase', ['save/rank0_hf_write_and_cpu_reload',
                                 'save/all_rank_training_state_write', 'load/local_read_and_validate'])
@pytest.mark.parametrize('failed_rank', [0, 3, 7])
def test_cpu_mock_every_rank_refuses_same_matched_local_io(phase, failed_rank):
    failures = [None] * 8
    failures[failed_rank] = OSError('fixture disk failure ' + 'x' * 3000)
    messages = []
    for rank in range(8):
        collective = SequentialCPUCollectives(rank, failures)
        decide = source_function('_checkpoint_matched_stage', torch_surface=cpu_stage_surface(collective))
        with pytest.raises(RuntimeError, match='Matched checkpoint phase rejected') as result:
            decide(phase, failures[rank])
        assert collective.events == ['MAX', 'bounded_diagnostics']
        assert 'automatic_rollback=False' in str(result.value)
        assert 'collective_failure_recovery=False' in str(result.value)
        assert len(str(result.value)) < 1400
        messages.append(str(result.value))
    assert len(set(messages)) == 1  # sequential CPU observation, not real synchronization


def test_cpu_mock_common_success_skips_error_diagnostics():
    for rank in range(8):
        collective = SequentialCPUCollectives(rank, [None] * 8)
        source_function('_checkpoint_matched_stage', torch_surface=cpu_stage_surface(collective))('fixture', None)
        assert collective.events == ['MAX']


def worker_method_ast(name):
    tree = ast.parse(WORKER_SOURCE.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == 'ActorRolloutRefWorker')
    return next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_static_sha_read_slot_validation_precedes_common_decision_and_actual_load():
    """Bad SHA must be caught in the existing call path before any peer load."""
    source = ast.get_source_segment(WORKER_SOURCE.read_text(), worker_method_ast('load_training_state'))
    positions = [source.index(fragment) for fragment in (
        'file_sha256(path)', "torch.load(path, map_location='cpu'", '_validate_checkpoint_state(',
        "_checkpoint_matched_stage('load/local_read_and_validate'",
        'initialize_fsdp_before_full_restore(self.actor_module_fsdp)',
        "_checkpoint_matched_stage('load/initialized_local_validation'", 'step = load_worker_training_state(',
        "_checkpoint_matched_stage('load/local_state_copy'", 'torch.distributed.barrier()')]
    assert positions == sorted(positions)
    first_try = next(node for node in worker_method_ast('load_training_state').body if isinstance(node, ast.Try))
    text = ast.get_source_segment(WORKER_SOURCE.read_text(), first_try)
    assert 'file_sha256(path)' in text and 'Training-state SHA differs' in text
    assert 'except Exception as local_error' in text and 'error = local_error' in text


def test_static_save_collective_is_outside_local_io_error_catches():
    method = worker_method_ast('save_checkpoint')
    export = next(node for node in method.body if isinstance(node, ast.With)
                  and 'FSDP.state_dict_type' in ast.get_source_segment(WORKER_SOURCE.read_text(), node))
    assert 'state_dict()' in ast.get_source_segment(WORKER_SOURCE.read_text(), export)
    for node in method.body:
        if isinstance(node, ast.Try):
            assert 'FSDP.state_dict_type' not in ast.get_source_segment(WORKER_SOURCE.read_text(), node)
    text = ast.get_source_segment(WORKER_SOURCE.read_text(), method)
    assert text.index("_checkpoint_matched_stage('save/rank0_hf_write_and_cpu_reload'") < text.index('torch.distributed.barrier()')
    assert text.index("_checkpoint_matched_stage('save/all_rank_training_state_write'") < text.rindex('torch.distributed.barrier()')


def tiny_worker():
    """Tiny CPU shape/state surrogate; _local_shard is not an actual FSDP handle."""
    module = torch.nn.Linear(3, 2)
    module.register_buffer('fixture_buffer', torch.tensor([.25, .5]))
    module._is_root = True
    for parameter in module.parameters():
        parameter._local_shard = parameter.detach()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., step / 57))
    for _ in range(20):
        optimizer.zero_grad()
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    scheduler.step()  # exactly one completed outer, twenty real tiny Adam calls
    worker = SimpleNamespace(_is_actor=True, rank=0, actor_module_fsdp=module,
        actor_optimizer=optimizer, actor_lr_scheduler=scheduler,
        rollout_sharding_manager=SimpleNamespace(gen_random_states=None, torch_random_states=None))
    metadata = source_function('checkpoint_runtime_metadata')
    worker.checkpoint_runtime_metadata = lambda: metadata(worker)
    saved = worker_training_state(module, optimizer, scheduler, 0, 1, 1,
                                  worker.rollout_sharding_manager, complete_internal=True)
    saved['runtime_metadata'] = worker.checkpoint_runtime_metadata()
    return worker, saved


def live_snapshot(worker):
    return cpu_snapshot(dict(model=worker.actor_module_fsdp.state_dict(),
        local_shards={name: parameter._local_shard for name, parameter in worker.actor_module_fsdp.named_parameters()},
        gradients={name: parameter.grad for name, parameter in worker.actor_module_fsdp.named_parameters()},
        optimizer=worker.actor_optimizer.state_dict(), scheduler=worker.actor_lr_scheduler.state_dict(),
        runtime=worker.checkpoint_runtime_metadata()))


def validate(worker, state):
    return source_function('_validate_checkpoint_state')(state, worker.actor_module_fsdp,
        worker.actor_optimizer, worker.actor_lr_scheduler, 0, 1, worker.rollout_sharding_manager,
        expected_step=1, expected_adam_steps=20, require_complete=True)


def test_nonmutating_validation_accepts_tiny_adam20_scheduler1():
    worker, state = tiny_worker()
    before = live_snapshot(worker)
    validate(worker, state)
    assert all(slot['step'].item() == 20 for slot in state['optimizer']['state'].values())
    assert state['scheduler']['last_epoch'] == 1
    assert equal_state(live_snapshot(worker), before)


def test_constructor_surrogate_missing_local_shards_is_not_a_saved_state_error():
    """CPU constructor-layout surrogate only, not PyTorch FSDP initialization."""
    worker, state = tiny_worker()
    worker.actor_module_fsdp._is_root = None
    for parameter in worker.actor_module_fsdp.parameters():
        del parameter._local_shard
    before = cpu_snapshot(dict(model=worker.actor_module_fsdp.state_dict(),
        optimizer=worker.actor_optimizer.state_dict(), rng=rng_state()))
    validate(worker, state)
    after = cpu_snapshot(dict(model=worker.actor_module_fsdp.state_dict(),
        optimizer=worker.actor_optimizer.state_dict(), rng=rng_state()))
    assert equal_state(before, after)
    assert all(not hasattr(parameter, '_local_shard') for parameter in worker.actor_module_fsdp.parameters())


def test_static_runtime_restore_uses_original_fsdp_helpers_and_actual_modes():
    tree = ast.parse(WORKER_SOURCE.read_text())
    restore = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_restore_checkpoint_runtime_metadata')
    source = ast.get_source_segment(WORKER_SOURCE.read_text(), restore)
    assert 'offload_fsdp_param_and_grad(module' in source and 'load_fsdp_param_and_grad(module' in source
    assert 'offload_fsdp_optimizer(worker.actor_optimizer)' in source
    assert 'load_fsdp_optimizer(worker.actor_optimizer' in source
    assert "child.training = metadata['module_modes'][name]" in source
    assert 'parameter.data =' not in source and 'parameter._local_shard =' not in source
    load = ast.get_source_segment(WORKER_SOURCE.read_text(), worker_method_ast('load_training_state'))
    assert load.index('initialize_fsdp_before_full_restore(self.actor_module_fsdp)') < load.index('_restore_checkpoint_runtime_metadata(self,')


@pytest.mark.parametrize('corruption', ['schema', 'logical_step', 'model_nan', 'buffer_dtype',
    'adam19', 'adam_nan', 'adam_extra_field', 'scheduler_step', 'scheduler_count', 'scheduler_lr',
    'python_rng', 'numpy_rng', 'torch_rng', 'runtime_rng_disagrees', 'runtime_missing_mode',
    'runtime_illegal_device', 'runtime_moment_dtype'])
def test_corrupt_local_state_refuses_before_any_live_mutation(corruption):
    worker, state = tiny_worker()
    before = live_snapshot(worker)
    slot = next(iter(state['optimizer']['state'].values()))
    if corruption == 'schema':
        state['schema'] = 2
    elif corruption == 'logical_step':
        state['logical_step'] = 2
    elif corruption == 'model_nan':
        next(iter(state['model_shards'].values())).fill_(float('nan'))
    elif corruption == 'buffer_dtype':
        state['buffers']['fixture_buffer'] = state['buffers']['fixture_buffer'].double()
    elif corruption == 'adam19':
        slot['step'].fill_(19)
    elif corruption == 'adam_nan':
        slot['exp_avg'].fill_(float('nan'))
    elif corruption == 'adam_extra_field':
        slot['unexpected'] = torch.tensor(0.)
    elif corruption == 'scheduler_step':
        state['scheduler']['last_epoch'] = 2
    elif corruption == 'scheduler_count':
        state['scheduler']['_step_count'] = 1
    elif corruption == 'scheduler_lr':
        state['scheduler']['_last_lr'] = [float('nan')]
    elif corruption == 'python_rng':
        state['rng']['python'] = ('invalid fixture',)
    elif corruption == 'numpy_rng':
        state['rng']['numpy'] = ('invalid fixture',)
    elif corruption == 'torch_rng':
        state['rng']['torch'] = torch.zeros(3, dtype=torch.float32)
    elif corruption == 'runtime_rng_disagrees':
        state['runtime_metadata']['rng']['numpy'][1][0] ^= np.uint32(1)
    elif corruption == 'runtime_missing_mode':
        state['runtime_metadata']['module_modes'].pop('')
    elif corruption == 'runtime_illegal_device':
        next(iter(state['runtime_metadata']['parameters'].values()))['device'] = 'meta'
    else:
        next(iter(state['runtime_metadata']['optimizer_tensors'].values()))['dtype'] = 'torch.float64'
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        validate(worker, state)
    assert equal_state(live_snapshot(worker), before)


@pytest.mark.parametrize('changed', ['python_rng', 'numpy_rng', 'scheduler_full_state',
                                  'module_mode', 'parameter_dtype_shape_placement', 'sharding_rng'])
def test_small_live_runtime_metadata_detects_complete_state_changes(changed):
    worker, _ = tiny_worker()
    before = worker.checkpoint_runtime_metadata()
    if changed == 'python_rng':
        random.random()
    elif changed == 'numpy_rng':
        np.random.random()
    elif changed == 'scheduler_full_state':
        worker.actor_lr_scheduler._step_count += 1
    elif changed == 'module_mode':
        worker.actor_module_fsdp.eval()
    elif changed == 'parameter_dtype_shape_placement':
        worker.actor_module_fsdp.double()
    else:
        worker.rollout_sharding_manager.gen_random_states = torch.zeros(4, dtype=torch.uint8)
    assert not equal_state(worker.checkpoint_runtime_metadata(), before)
    assert {'python', 'numpy', 'torch', 'cuda', 'cuda_device'} == set(before['rng'])
    assert {'module_modes', 'parameters', 'buffers', 'local_shards'} <= set(before)


def test_real_single_process_iterator_reload_keeps_identity_next_data_and_rng():
    from torch.utils.data import DataLoader, TensorDataset
    from search_r1.baselines.fresh200 import driver_flow_state, restore_driver_flow
    from search_r1.baselines.recovery import RecoverableSampler
    sampler = RecoverableSampler(16, shuffle=False, seed=0)
    iterator = iter(DataLoader(TensorDataset(torch.arange(16)), batch_size=4,
                               sampler=sampler, num_workers=0))
    assert next(iterator)[0].tolist() == [0, 1, 2, 3]
    trainer = SimpleNamespace(fresh200_active_train_iterator=iterator,
        recovery_sampler=sampler, validation_trace_serial=3)
    saved = driver_flow_state(trainer)
    identity, parent_rng = id(iterator), cpu_snapshot(rng_state())
    receipt = restore_driver_flow(trainer, saved)
    assert id(trainer.fresh200_active_train_iterator) == identity
    assert driver_flow_state(trainer) == saved and equal_state(rng_state(), parent_rng)
    assert receipt['actual_iterator_metadata_reload'] and receipt['actual_validation_trace_reload']
    assert receipt['same_live_iterator'] and receipt['iterator_num_yielded'] == 1
    assert next(iterator)[0].tolist() == [4, 5, 6, 7]
    assert sampler.cursor == 8
    assert equal_state(rng_state(), parent_rng)  # no second iterator/base-seed draw
    with pytest.raises(ValueError, match='progress changed'):
        restore_driver_flow(trainer, saved)


def test_offline_tiny_actual_hf_subprocess_binds_tensor_config_tokenizer_and_bytes(tmp_path, monkeypatch):
    """Real tiny HF loader only; no inference/FSDP/3B production claim."""
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    monkeypatch.chdir(ROOT)  # ensure the independent child can import this source
    directory = tmp_path / 'global_step_5'
    directory.mkdir()
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        vocab={'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, 'fixture': 4, 'answer': 5}, unk_token='<unk>'))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token='<pad>', bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
        chat_template="{{ messages[0]['content'] }}")
    tokenizer.save_pretrained(directory)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(vocab_size=len(tokenizer),
        hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32, tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, attention_dropout=0.))
    model.save_pretrained(directory, max_shard_size='1KB')
    expected = hf_export_identity(model.state_dict(), model.config, tokenizer)
    assert expected['config']['id2label'] == {'0': 'LABEL_0', '1': 'LABEL_1'}
    assert json.loads(json.dumps(expected, allow_nan=False)) == expected
    parent_rng = cpu_snapshot(rng_state())
    receipt = reload_hf_checkpoint_cpu(directory, expected)
    identity = json.loads(Path(receipt['path']).read_text())
    assert identity == receipt['identity']
    assert identity['status'] == 'ACTUAL_CPU_HF_LOADER_ROUNDTRIP_PASS'
    assert identity['probe_pid'] != os.getpid() and identity['isolated_process'] is True
    assert identity['device'] == 'cpu' and identity['dtype'] == 'torch.float32'
    assert identity['tensors'] == expected['tensors'] and identity['config'] == expected['config']
    assert identity['tokenizer'] == expected['tokenizer']
    assert identity['tokenizer']['chat_template'] == tokenizer.chat_template
    assert any(value['path'].endswith('model.safetensors.index.json') for value in identity['hf_files'])
    assert equal_state(rng_state(), parent_rng)
    wrong = copy.deepcopy(expected)
    next(iter(wrong['tensors'].values()))['sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='CPU HF loader failed'):
        reload_hf_checkpoint_cpu(directory, wrong)
    assert equal_state(rng_state(), parent_rng)
    wrong_config = copy.deepcopy(expected)
    wrong_config['config']['hidden_size'] += 1
    with pytest.raises(RuntimeError, match='Actual HF config loader differs'):
        reload_hf_checkpoint_cpu(directory, wrong_config)
    wrong_dtype = copy.deepcopy(expected)
    wrong_dtype['config']['torch_dtype'] = 'bfloat16'
    with pytest.raises(RuntimeError, match='Actual HF config loader differs'):
        reload_hf_checkpoint_cpu(directory, wrong_dtype)
    wrong_tokenizer = copy.deepcopy(expected)
    wrong_tokenizer['tokenizer']['chat_template'] += ' changed'
    with pytest.raises(RuntimeError, match='Actual HF tokenizer loader differs'):
        reload_hf_checkpoint_cpu(directory, wrong_tokenizer)
    assert equal_state(rng_state(), parent_rng)


def test_hf_semantic_config_json_transport_keeps_values_and_rejects_collisions():
    from search_r1.baselines.instruct300 import _semantic_config
    original = {'hidden_size': 8, 'torch_dtype': 'float32',
                'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
                'nested': {'ids': (1, 2), 'enabled': True}}
    config = SimpleNamespace(to_dict=lambda: copy.deepcopy(original))
    actual = _semantic_config(config)
    assert actual == json.loads(json.dumps(original, allow_nan=False))
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual
    assert original['id2label'] == {0: 'LABEL_0', 1: 'LABEL_1'}
    colliding = SimpleNamespace(to_dict=lambda: {'id2label': {0: 'original', '0': 'replacement'}})
    with pytest.raises(ValueError, match='colliding JSON object keys'):
        _semantic_config(colliding)
