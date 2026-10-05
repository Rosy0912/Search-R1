"""Actual actor control-flow tests with CPU forward, never a GPU acceptance claim."""
import json
from types import MethodType
import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


def actor_and_data(monkeypatch, audit_root, target):
    monkeypatch.setattr(TensorDict, 'cuda', lambda self: self)
    monkeypatch.delenv('SEARCH_R1_BASELINE_AUDIT_DIR', raising=False)
    monkeypatch.delenv('SEARCH_R1_BASELINE_COMMIT', raising=False)
    config = OmegaConf.create(dict(ppo_mini_batch_size=256, ppo_micro_batch_size=16, ppo_epochs=1,
        use_dynamic_bsz=False, state_masking=True, use_kl_loss=True, kl_loss_type='low_var_kl',
        kl_loss_coef=.001, clip_ratio=.2, entropy_coeff=0., grad_clip=1., baseline_method='gigpo',
        joint_loss_scaling=True, complete_internal_state=True,
        policy_safety=dict(enabled=True, dual_clip=3., kl_log_bound=30., monitor_first_global_rows=256,
                           target_old_policy_kl=target, negative_tail_limit=.1,
                           calibration_evidence_sha256='c'*64)))
    # Synthetic acceptance fixture only exercises the parser; it is not a GPU proof.
    import hashlib
    proof = dict(status='PASS',scope='GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION',rank_count=8,
        restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
        target_old_policy_kl=target,negative_tail_limit=.1,dual_clip=3.,kl_log_bound=30.)
    raw=json.dumps(proof).encode();path=audit_root/'synthetic-calibration-fixture.json';path.write_bytes(raw)
    config.policy_safety.calibration_evidence_sha256=hashlib.sha256(raw).hexdigest()
    config.policy_safety.calibration_evidence_path=str(path)
    actor = object.__new__(DataParallelPPOActor)
    actor.config=config
    actor.actor_module=torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():actor.actor_module.weight.zero_()
    actor.actor_optimizer=torch.optim.Adam(actor.actor_module.parameters(), lr=.001)
    forward_weights=[]
    def forward(self, micro_batch, temperature):
        forward_weights.append(float(self.actor_module.weight.detach()))
        logp = self.actor_module.weight.reshape(1, 1).expand_as(micro_batch['old_log_probs'])
        return logp * 0, logp
    actor._forward_micro_batch=MethodType(forward, actor)
    tensors={key:torch.ones(2560,1) for key in ('loss_mask','attention_mask','policy_weights','advantages')}
    tensors.update({key:torch.zeros(2560,1,dtype=torch.long) for key in ('responses','input_ids','position_ids','step_ids')})
    tensors.update(old_log_probs=torch.zeros(2560,1),ref_log_prob=torch.zeros(2560,1))
    data=DataProto.from_dict(tensors,meta_info=dict(temperature=1.,baseline_outer_step=26))
    # Diagnostic stream is tested independently of optional legacy ActorAudit.
    import search_r1.baselines.runtime_audit as audit
    monkeypatch.setattr(audit,'enabled',lambda:False)
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR',str(audit_root))
    return actor, data, forward_weights


def test_ten_real_adam_writes_and_post_monitor_reads_new_weights(monkeypatch,tmp_path):
    actor,data,seen=actor_and_data(monkeypatch,tmp_path,.01)
    before=data.batch['old_log_probs'].clone()
    result=actor.update_policy(data)
    assert result['actor/optimizer_steps']==10
    assert int(next(iter(actor.actor_optimizer.state.values()))['step'])==10
    assert torch.equal(before,data.batch['old_log_probs'])
    records=[json.loads(line) for line in (tmp_path/'policy-safety/rank-0.jsonl').read_text().splitlines()]
    assert len(records)==20
    assert records[0]['phase']=='before' and records[0]['statistics']['max_abs_token_delta']==0
    assert records[1]['phase']=='after' and records[1]['actual_optimizer_writes']==1
    assert records[1]['statistics']['max_abs_token_delta']>0
    assert all(r['allowed'] for r in records)
    assert len(seen)==480  # 160 gradient forwards plus160 pre and160 post monitor forwards.


def test_post_brake_keeps_one_write_and_refuses_outer_completion(monkeypatch,tmp_path):
    actor,data,_=actor_and_data(monkeypatch,tmp_path,1e-8)
    with pytest.raises(RuntimeError,match='OLD_POLICY_TOKEN_KL'):
        actor.update_policy(data)
    assert int(next(iter(actor.actor_optimizer.state.values()))['step'])==1
    records=[json.loads(line) for line in (tmp_path/'policy-safety/rank-0.jsonl').read_text().splitlines()]
    assert len(records)==2 and records[-1]['actual_optimizer_writes']==1
    assert not records[-1]['allowed']
    assert len(list((tmp_path/'policy-safety').glob('rejected-*.pt')))==1

@pytest.mark.parametrize('failure',['calibration_missing','input_key_missing'])
def test_entry_errors_reject_before_any_optimizer_write(monkeypatch,tmp_path,failure):
    actor,data,_=actor_and_data(monkeypatch,tmp_path,.01)
    if failure=='calibration_missing':
        (tmp_path/'synthetic-calibration-fixture.json').unlink()
    else:
        data.batch.pop('ref_log_prob')
    with pytest.raises(RuntimeError,match='POLICY_SAFETY_REJECTED'):
        actor.update_policy(data)
    assert not actor.actor_optimizer.state
    assert len(list((tmp_path/'policy-safety').glob('numeric-*.json')))==1
