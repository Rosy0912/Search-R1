"""Actual rollout transport body and explicit healthy-checkpoint config migration."""
import ast
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from search_r1.baselines.quality_resume import migration_diff

ROOT = Path(__file__).resolve().parents[2]


def actual_rollout_body():
    source = ROOT / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
    tree = ast.parse(source.read_text())
    pre = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_pre_process_inputs')
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'vLLMRollout')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'generate_sequences')
    for function in (pre, method):
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
    wrapper = ast.ClassDef(name='Fixture', bases=[], keywords=[], body=[method], decorator_list=[])
    namespace = dict(torch=torch, TensorDict=TensorDict, DataProto=DataProto,
                     get_eos_mask=get_eos_mask, pad_sequence_to_length=pad_sequence_to_length)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pre, wrapper], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['Fixture']()


@pytest.mark.parametrize('raw', [True, False])
def test_actual_transport_preserves_real_sampled_pad_and_ignores_rectangular_padding(raw):
    rollout = actual_rollout_body()
    pad, eos = 151643, 151645
    rollout.config = OmegaConf.create(dict(free_cache_engine=False, response_length=5, n=1,
        actor_token_identity='sampled_action_prefix_v1' if raw else 'legacy_canonical'))
    rollout.pad_token_id = pad
    rollout.sampling_params = SimpleNamespace(temperature=1.)
    @contextmanager
    def params(**kwargs):
        yield
    rollout.update_sampling_params = params
    engine = SimpleNamespace(last_completion_token_lengths=None)
    def generate(**kwargs):
        assert kwargs['prompt_token_ids'] == [[1, 2], [3, 4]]
        engine.last_completion_token_lengths = torch.tensor([2, 3])
        return (torch.tensor([[27, pad, pad], [27, 397, eos]]),
                torch.tensor([[-.1, -.2, float(pad)], [-.1, -.2, -.3]]))
    engine.generate = generate
    rollout.inference_engine = engine
    input_ids = torch.tensor([[pad, 1, 2], [pad, 3, 4]])
    prompts = DataProto.from_dict(dict(input_ids=input_ids,
        attention_mask=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 1], [0, 0, 1]])), meta_info=dict(eos_token_id=eos))
    result = rollout.generate_sequences(prompts)
    assert result.batch['responses'][0].tolist() == [27, pad, pad, pad, pad]
    if raw:
        assert result.batch['response_sampled_lengths'].tolist() == [2, 3]
        assert result.batch['response_sample_mask'].tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]
        assert torch.equal(result.batch['response_sample_mask'], result.batch['attention_mask'][:, 3:])
        assert result.meta_info['generation_actor_token_identity'] == 'sampled_completion_v1'
    else:
        assert 'response_sampled_lengths' not in result.batch
        assert result.batch['attention_mask'][0, 3:].tolist() == [1] * 5
        assert 'generation_actor_token_identity' not in result.meta_info


def raw_migration_pair():
    old = dict(actor_rollout_ref=dict(actor=dict(ppo_mini_batch_size=128), ref={}, rollout={}),
               baselines=dict(method='gigpo'), trainer={})
    new = copy.deepcopy(old)
    new['actor_rollout_ref']['actor']['ppo_mini_batch_size'] = 256
    new['trainer'].update(quality_resume=dict(enabled=True), trajectory_logging=True)
    new['actor_token_identity'] = 'sampled_action_prefix_v1'
    for part in ('actor', 'ref', 'rollout'):
        new['actor_rollout_ref'][part]['actor_token_identity'] = 'sampled_action_prefix_v1'
    return old, new


def test_explicit_raw_migration_preserves_old_config_and_requires_all_four_modes():
    old, new = raw_migration_pair()
    frozen = copy.deepcopy(old)
    diff = migration_diff(old, new)
    assert diff['actor_token_identity'] == dict(old=None, new='sampled_action_prefix_v1')
    assert old == frozen
    for part in ('actor', 'ref', 'rollout'):
        broken = copy.deepcopy(new)
        broken['actor_rollout_ref'][part].pop('actor_token_identity')
        with pytest.raises(ValueError, match='agree across'):
            migration_diff(old, broken)


@pytest.mark.parametrize('mutation', ['mean_joint', 'other_algorithm', 'change_existing_mode'])
def test_raw_migration_refuses_extra_mathematical_or_incompatible_changes(mutation):
    old, new = raw_migration_pair()
    if mutation == 'mean_joint':
        new['actor_rollout_ref']['actor']['joint_ratio_reduction'] = 'mean'
    elif mutation == 'other_algorithm':
        old['baselines']['method'] = new['baselines']['method'] = 'grpo'
    else:
        old['actor_token_identity'] = 'different_mode'
    with pytest.raises(ValueError):
        migration_diff(old, new)
