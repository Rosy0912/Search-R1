import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from search_r1.baselines.baseline_alignment import bind_frozen_hf, validate_aligned_config
from search_r1.baselines.fresh200_contract import outer_contract, validate_rank_updates
from search_r1.baselines.fresh200_driver import Fresh200Progress


def configuration():
    root = Path(__file__).resolve().parents[2] / 'verl/trainer/config'
    base = OmegaConf.load(root / 'ppo_bapo_fresh200.yaml')
    overlay = OmegaConf.load(root / 'ppo_bapo_baseline_aligned.yaml')
    c = OmegaConf.to_container(OmegaConf.merge(base, overlay), resolve=False)
    # Fields supplied by the inherited ppo_baselines config.
    c['algorithm']['adv_estimator'] = 'grpo'
    c['data']['train_data_num'] = c['data']['val_data_num'] = None
    return c


def test_real_profile_configuration_and_overrides():
    c = configuration()
    assert validate_aligned_config(c) == (200, 57)
    for section, key, value in [('data','val_data_num',32), ('data','train_data_num',512),
                                 ('trainer','fresh200_diagnostic_only',True),
                                 ('baseline_aligned',None,False)]:
        broken = copy.deepcopy(c)
        if key is None: broken[section] = value
        else: broken[section][key] = value
        with pytest.raises(ValueError): validate_aligned_config(broken)


def test_hf_denominator_remains_frozen_and_sampling_unmodified():
    old = torch.tensor([[-2., -3.]])
    sampled = torch.tensor([[-9., -8.]])
    data = SimpleNamespace(batch={'old_log_probs':old, 'sampling_log_probs':sampled}, meta_info={})
    bind_frozen_hf(data)
    assert data.batch['old_log_probs'] is old
    assert data.batch['sampling_log_probs'] is sampled
    assert torch.equal(data.batch['frozen_hf_log_probs'], old)
    assert data.batch['frozen_hf_log_probs'].data_ptr() != old.data_ptr()
    with pytest.raises(ValueError): bind_frozen_hf(data)


def events(outer):
    return [dict(rank=i, baseline_aligned=True, status='COMPLETE', actual_outer=outer,
        attempt_id=outer, expected_optimizer_steps=20, optimizer_opportunities_total=20,
        optimizer_steps=20, processed_opportunities=20, skipped_opportunities=0,
        unprocessed_opportunities=0, postabort=False, update_complete=True, attempt_complete=True)
        for i in range(8)]


def test_all200_complete_require4000_real_writes_per_rank():
    p = Fresh200Progress(baseline_aligned=True)
    for outer in range(1, 201):
        assert outer_contract(outer, True)['expected_optimizer_steps'] == 20
        p.begin_attempt(outer)
        assert p.record(events(outer), outer)
    assert p.completed_outers == 200
    assert p.actual_adam_writes_per_rank == [4000]*8
    assert p.state_dict()['no_skip_expected_writes_per_rank'] == 4000
    broken = events(50)
    for e in broken: e.update(optimizer_steps=19, skipped_opportunities=1, status='COMPLETE_WITH_SKIPS')
    with pytest.raises(ValueError): validate_rank_updates(broken, 50)


def test_startup_refuses_prior_logs_and_duplicate_claim(tmp_path):
    from search_r1.baselines.aligned_startup import claim_new_run
    old = tmp_path / 'runs' / 'old'
    (old / 'logs').mkdir(parents=True)
    (old / 'logs' / 'train.log').write_text('partial previous run')
    with pytest.raises(ValueError): claim_new_run(old, 'a'*40)
    new = tmp_path / 'runs' / 'new'
    new.mkdir()
    (new / 'resource-watch.jsonl').write_text('{}\n')
    claim_new_run(new, 'a'*40)
    with pytest.raises((ValueError, FileExistsError)): claim_new_run(new, 'a'*40)
