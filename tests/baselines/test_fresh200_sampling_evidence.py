"""Actual CPU tensor transport/replay tests; no native CUDA acceptance."""
import json
import numpy as np
import pytest
import torch
from unittest.mock import patch
from verl import DataProto
from search_r1.baselines.fresh200_driver import persist_first_sampling_batch
from search_r1.baselines.fresh200_gate import validate_first_sampling_evidence


def batch(gap=0.):
    return DataProto.from_dict(tensors=dict(old_log_probs=torch.zeros(1,1),frozen_hf_log_probs=torch.tensor([[gap]]),
        sampling_log_probs=torch.zeros(1,1),sampling_mask=torch.ones(1,1,dtype=torch.bool),
        loss_mask=torch.ones(1,1,dtype=torch.bool)),non_tensors=dict(fresh200_turns=np.array([
            [dict(effective_stop_provenance=dict(probability_role='stochastic_sampling_policy'))]],dtype=object)),
        meta_info=dict(behavior_policy_role='actual_native_stochastic_sampling_policy',frozen_hf_probability_role='unchanged_hf_recomputation_diagnostic',attempt_id=1,actual_outer=1,policy_snapshot_id='actual-first-policy',sampling_context_audit={'captured':True}))


@pytest.mark.parametrize('gap',[0.,.1,float('nan')])
def test_first_actual_tensor_proof_retained_even_when_verifier_fails(tmp_path,gap):
    directory=tmp_path/'proof';data=batch(gap)
    with patch('search_r1.llm_agent.sampling_tokens.audit_batch',return_value=dict(probability_role='stochastic_sampling_policy',context_verified=True,policy_snapshot_id='actual-first-policy')):
        if gap==gap:
            report=persist_first_sampling_batch(data,directory,{},'a'*40,tmp_path,{'real_input_sha':'fixture'})
            assert report['sampling_verifier_status']=='PASS_ACTUAL_BATCH_BINDING_NOT_RELEASE'
        elif gap!=gap:
            report=persist_first_sampling_batch(data,directory,{},'a'*40,tmp_path,{})
            assert report['sampling_verifier_status']=='NUMERIC_REJECTED_ACTUAL_BATCH_BINDING'
        else:
            with pytest.raises(ValueError,match='immutable raw proof retained'):
                persist_first_sampling_batch(data,directory,{},'a'*40,tmp_path,{})
    payload=torch.load(directory/'formal-first-sampling-batch.pt',weights_only=False)
    assert torch.equal(payload['tensors']['sampling_mask'],data.batch['sampling_mask'])
    assert payload['meta_info']==data.meta_info and payload['attempt_id']==1
    report=json.loads((directory/'formal-first-sampling-batch.json').read_text())
    assert report['max_selected_logprob_abs_diff']==(None if gap!=gap else pytest.approx(gap))
    assert report['sampling_verifier_status'].startswith('PASS' if gap==gap else 'NUMERIC_REJECTED' if gap!=gap else 'FAIL')
    with pytest.raises(FileExistsError):persist_first_sampling_batch(data,directory,{},'a'*40,tmp_path,{})


def test_gate_rejects_missing_formal_raw_evidence(tmp_path):
    with pytest.raises(ValueError,match='immutable first sampling/initial evidence missing'):
        validate_first_sampling_evidence({}, {},'a'*40,tmp_path,123)


def test_gate_replays_actual_saved_probabilities_and_requires_all_initial_files(tmp_path):
    from search_r1.baselines.fresh200_gate import digest
    from search_r1.baselines.fresh200_contract import critical_source_hashes
    data=batch();data.meta_info['fresh200_shared_input_bindings']={'data_sha':'actual-fixture'}
    with patch('search_r1.llm_agent.sampling_tokens.audit_batch',return_value=dict(probability_role='stochastic_sampling_policy',context_verified=True,policy_snapshot_id='actual-first-policy')):
        report=persist_first_sampling_batch(data,tmp_path/'proof',{},'a'*40,tmp_path,data.meta_info['fresh200_shared_input_bindings'])
        initial=tmp_path/'initial';initial.mkdir()
        official=initial/'native-initial-official-hf.pt';torch.save({'CPU_FIXTURE_NOT_NATIVE':True},official)
        alignment=dict(path=str(official),sha256=digest(official));(initial/'native-initial-official-alignment.json').write_text(json.dumps(alignment))
        for rank in range(8):
            path=initial/f'native-initial-rank_{rank}.pt';torch.save({'CPU_FIXTURE_NOT_NATIVE':True},path)
            meta=dict(path=str(path),sha256=digest(path),official_alignment=alignment,source_sha256=critical_source_hashes(),
                optimizer_state_entries=0,scheduler_last_epoch=0,lr=[1e-6/57],
                status='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE',native_process=dict(rank=rank))
            path.with_suffix('.json').write_text(json.dumps(meta))
        members={str(path):digest(path) for path in tmp_path.rglob('*') if path.is_file()}
        config=dict(actor_rollout_ref=dict(actor=dict(policy_safety={})))
        validate_first_sampling_evidence(members,config,'a'*40,tmp_path,report['driver_pid'])
        removed={p:h for p,h in members.items() if not p.endswith('native-initial-rank_7.pt')}
        with pytest.raises(ValueError,match='initial evidence missing'):
            validate_first_sampling_evidence(removed,config,'a'*40,tmp_path,report['driver_pid'])
        raw_path=tmp_path/'proof'/'formal-first-sampling-batch.pt'
        raw=torch.load(raw_path,weights_only=False);raw['tensors']['old_log_probs'][0,0]=.1;torch.save(raw,raw_path)
        members[str(raw_path)]=digest(raw_path)
        meta_path=raw_path.with_suffix('.json');meta=json.loads(meta_path.read_text());meta['sha256']=members[str(raw_path)]
        meta_path.write_text(json.dumps(meta));members[str(meta_path)]=digest(meta_path)
        with pytest.raises(ValueError,match='PPO denominator differs'):
            validate_first_sampling_evidence(members,config,'a'*40,tmp_path,report['driver_pid'])
