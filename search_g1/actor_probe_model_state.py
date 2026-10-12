"""Canonical FP32 actor weight loading, independent of the fixed reference."""
import json
import os
from pathlib import Path


def validate_roundtrip_receipt(receipt, run, completed_outer, rank):
    """Validate the complete shared receipt before selecting a local rank."""
    identity = {key: run[key] for key in ('run_id', 'execution_id', 'source_commit',
        'execution_backend', 'training_world_size', 'profile', 'slurm_job_id')}
    assert receipt['role'] == 'actor_probe_full_state_checkpoint'
    assert receipt['identity'] == identity
    assert receipt['completed_outer'] == completed_outer
    assert receipt['next_logical_update'] == completed_outer + 1
    workers = receipt['workers']
    assert all(type(row['rank']) is int for row in workers)
    assert sorted(row['rank'] for row in workers) == list(range(run['training_world_size'])), \
        'Full-state receipt has an incomplete or duplicate rank set'
    assert all(row['completed_outer'] == completed_outer and
               row['optimizer_writes'] == receipt['actual_optimizer_writes'] for row in workers)
    return next(row for row in workers if row['rank'] == rank)


def validate_parameter_state(state, shapes, *, tied_alias=False):
    import torch
    alias, embedding = 'lm_head.weight', 'model.embed_tokens.weight'
    assert shapes, 'Canonical actor cannot have an empty parameter domain'
    allowed = {alias} if tied_alias and alias not in shapes else set()
    assert set(state) - allowed == set(shapes), 'Canonical actor parameter keys differ'
    for name, tensor in state.items():
        shape = shapes[embedding] if name in allowed else shapes[name]
        assert isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(shape)
        assert tensor.dtype == torch.float32, 'Canonical restore must preserve FP32 master weights'
        assert torch.isfinite(tensor).all(), 'Nonfinite canonical actor weight'
    if tied_alias:
        assert torch.equal(state[alias], state[embedding]), 'Canonical tied embedding/head differs'
    return len(state)


def read_canonical_actor_state(directory, run, completed_outer, receipt_sha256):
    """Read only the run's own remote checkpoint after checking every bound file."""
    from safetensors import safe_open
    from reproduction.common8.topology import validate_base
    from reproduction.common8.verify_data import digest
    base = Path(os.environ['SEARCH_G1_BASE']).resolve(); validate_base(base, run)
    directory = Path(directory)
    assert directory == directory.resolve() and directory.is_relative_to(base)
    assert directory == Path(run['checkpoint_dir'])/'actor'/f'global_step_{completed_outer}'
    assert digest(directory/'complete.json') == receipt_sha256
    complete = json.loads((directory/'complete.json').read_text())
    assert complete['run_id'] == run['run_id'] and complete['step'] == completed_outer
    assert complete['finite_all_saved_tensors'] is True
    assert Path(complete['checkpoint']).resolve() == directory
    for name, sha in complete['files_sha256'].items():
        assert Path(name).name == name, 'Checkpoint filename escapes its namespace'
        path = directory/name
        assert path.resolve() == path and path.stat().st_uid == os.getuid()
        assert digest(path) == sha, 'Canonical checkpoint bytes changed'
    index_name = 'model.safetensors.index.json'
    assert {index_name, 'config.json'} <= set(complete['files_sha256'])
    saved = json.loads((directory/index_name).read_text())['weight_map']
    official = Path(run['initial_model'])
    assert digest(official/index_name) == run['initial_model_sha256'][index_name]
    assert digest(official/'config.json') == run['initial_model_sha256']['config.json']
    expected = json.loads((official/index_name).read_text())['weight_map']
    configs = [json.loads((path/'config.json').read_text()) for path in (official, directory)]
    tied = all(c.get('model_type') == 'qwen2' and c.get('tie_word_embeddings') is True
               and c.get('architectures') == ['Qwen2ForCausalLM'] for c in configs)
    alias, embedding = 'lm_head.weight', 'model.embed_tokens.weight'
    permitted = {alias} if tied and embedding in expected and alias not in expected else set()
    assert set(saved) - permitted == set(expected)
    assert all(Path(name).name == name and name in complete['files_sha256'] for name in saved.values())
    shapes = {}
    for filename in set(expected.values()):
        with safe_open(str(official/filename), framework='pt', device='cpu') as reader:
            for name, shard in expected.items():
                if shard == filename: shapes[name] = tuple(reader.get_slice(name).get_shape())
    state = {}
    for filename in sorted(set(saved.values())):
        with safe_open(str(directory/filename), framework='pt', device='cpu') as reader:
            keys = {name for name, shard in saved.items() if shard == filename}
            assert set(reader.keys()) == keys
            for name in keys: state[name] = reader.get_tensor(name)
    validate_parameter_state(state, shapes, tied_alias=tied and alias in saved)
    assert len(state) == 435
    return state


def load_fsdp_canonical_state(module, state):
    """Use the same full-state FSDP graph as export; preserve Parameter objects."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        result = module.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
