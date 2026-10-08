from pathlib import Path
from unittest.mock import patch
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.baseline_alignment import verify_goal_common_inputs

SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'

def config(tmp):
    return OmegaConf.create({'data':{'train_files':str(tmp/'train'), 'val_files':str(tmp/'dev'),
        'final_val_files':str(tmp/'final')}, 'baselines':{'alignment':{'training_em_enabled':True,
        'common_manifest_path':str(tmp/'manifest.json'), 'common_manifest_sha256':SHA},
        'holdout':{},'bapo':{'stage_validation_files':str(tmp/'stage')}}})

def test_production_original_manifest_and_actual_loader_bindings(tmp_path):
    c=config(tmp_path); paths={k:str(tmp_path/v) for k,v in [('train','train'),('validation','dev'),('final','final'),('stage','stage')]}
    bound={'train_rows':169028,'effective_paths':paths,'content_identity':{'sha256':SHA}}
    with patch('search_r1.baselines.baseline_alignment.read_bound_json') as read, patch('search_r1.baselines.holdout._validate_prebuilt',return_value=bound) as verify:
        r=verify_goal_common_inputs(c)
        read.assert_called_once_with(str(tmp_path/'manifest.json'),SHA)
        assert verify.call_args[0][0].baselines.holdout.input_mode=='prebuilt_manifest'
        assert r['train_rows']==169028
        c.data.train_files=str(tmp_path/'other-train')
        with pytest.raises(ValueError,match='loader path'): verify_goal_common_inputs(c)
        c.data.train_files=paths['train'];bound['train_rows']=169103
        with pytest.raises(ValueError,match='169028'):verify_goal_common_inputs(c)

def test_missing_or_other_original_manifest_not_production(tmp_path):
    c=config(tmp_path);c.baselines.alignment.common_manifest_sha256='a'*64
    with pytest.raises(ValueError,match='user-pinned'):verify_goal_common_inputs(c)
    c.baselines.alignment.pop('common_manifest_sha256')
    with pytest.raises(ValueError):verify_goal_common_inputs(c)
    c.baselines.alignment={}
    assert verify_goal_common_inputs(c) is None


def test_production_cannot_disable_same_process_five_gate():
    from tests.baselines.test_baseline_alignment_contract import configuration
    from search_r1.baselines.baseline_alignment import validate_aligned_config
    c=configuration();c['baselines']['alignment']['training_em_enabled']=True
    c['trainer']['fresh200_gate_enabled']=True
    c['trainer']['aligned_stage_commit_enabled']=True
    c['actor_rollout_ref']['actor']['aligned_stage_commit_enabled']=True
    c['baselines']['bapo'].update(stage_validation_freq=1,stage_validation_size=64)
    assert validate_aligned_config(c)==(200,57)
    c['trainer']['fresh200_gate_enabled']=False
    with pytest.raises(ValueError):validate_aligned_config(c)
