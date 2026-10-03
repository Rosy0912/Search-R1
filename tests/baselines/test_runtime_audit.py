"""Real actor audit must preserve metrics, weights and RNG state."""
import importlib.util,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from verl import DataProto
from search_r1.baselines.runtime_audit import ActorAudit, runtime_env_vars

def test_real_gigpo_audit_is_observational(tmp_path):
    source=Path(__file__).with_name('test_paper_metric_logging.py')
    spec=importlib.util.spec_from_file_location('audit_actor_fixture',source)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    expected,weights=fixture.run(1)
    original=ActorAudit.__init__
    def initialize(self,model,batch,meta,epochs):
        original(self,model,batch,dict(meta,baseline_outer_step=5),epochs)
    with patch.dict(os.environ,{'SEARCH_R1_BASELINE_AUDIT_DIR':str(tmp_path),
            'SEARCH_R1_BASELINE_RUN_ID':'cpu-audit','SEARCH_R1_BASELINE_COMMIT':'test',
            'SEARCH_R1_BASELINE_CONFIG_SHA256':'config','SEARCH_R1_BASELINE_DATA_IDENTITY':'data'}), \
            patch.object(ActorAudit,'__init__',initialize):
        actual,new_weights=fixture.run(1)
        assert runtime_env_vars()['SEARCH_R1_BASELINE_RUN_ID']=='cpu-audit'
    torch.testing.assert_close(weights,new_weights,rtol=0,atol=0)
    for key in expected: np.testing.assert_equal(expected[key],actual[key])
    record=json.loads((tmp_path/'0.jsonl').read_text())
    assert record['outer_step']==5 and record['optimizer_steps']==1
    assert record['mask_binary'] and record['fixed_unchanged']
    assert record['shard_delta']['nonzero']>0
    assert record['grad_finite']==record['grad_nonzero']==1
    assert all(x['joint_action_ratios']['finite']==x['joint_action_ratios']['count'] for x in record['micro_stats'])

def test_audit_sampling_does_not_consume_rng():
    model=torch.nn.Linear(1,1,bias=False)
    batch=DataProto.from_dict({'old_log_probs':torch.ones(1,1),'advantages':torch.ones(1,1)}).batch
    before=torch.get_rng_state().clone()
    audit=ActorAudit(model,batch,{'baseline_outer_step':1},1)
    audit.epoch(); audit.micro(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1,1),torch.ones(1))
    torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)

def test_actual_config_boundary_rejects_declared_hash_and_split_drift(tmp_path):
    import hashlib
    import pytest
    from omegaconf import OmegaConf
    from search_r1.baselines.runtime_audit import config_boundary
    config=OmegaConf.create({'trainer':{'total_training_steps':6},'data':{'path':'local'}})
    text=OmegaConf.to_yaml(config);identity={'content_identity':{'sha256':'split'}}
    with patch.dict(os.environ,{'SEARCH_R1_BASELINE_AUDIT_DIR':str(tmp_path),
            'SEARCH_R1_BASELINE_CONFIG_SHA256':hashlib.sha256(text.encode()).hexdigest(),
            'SEARCH_R1_BASELINE_DATA_IDENTITY':'split'}):
        config.trainer.total_training_steps=501
        with pytest.raises(ValueError):config_boundary(config,identity)
        assert not list(tmp_path.iterdir())
        config.trainer.total_training_steps=6
        with pytest.raises(ValueError):config_boundary(config,{'content_identity':{'sha256':'other'}})
        config_boundary(config,identity)
    assert (tmp_path/'entry-config.yaml').read_text()==text
    record=json.loads((tmp_path/'entry.jsonl').read_text())
    assert record['actual_data_identity']=='split' and record['entry_total_training_steps']==6
