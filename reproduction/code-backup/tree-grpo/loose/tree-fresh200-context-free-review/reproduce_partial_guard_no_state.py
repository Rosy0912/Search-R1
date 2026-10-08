"""Actual affected CPU actor/Adam fixture; no candidate source changes."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parent.parent / 'tree-behavior-mu-ppo'
sys.path.insert(0, str(SOURCE))
import torch
import test_tree_behavior_mu_actor_cpu as fixture

factory = fixture.collector_fixture.BehaviorMuDataTests()
factory.setUp()
try:
    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'SEARCH_R1_TREE_RANK_AUDIT_DIR': temp}):
        cfg = fixture.config()
        actor = fixture.Actor(cfg)
        data = fixture.make_data(factory, cfg)
        measure = fixture.PolicySafetyRuntime(actor, data.batch, data.meta_info, cfg.policy_safety, source_data=data)
        initial = measure.measure('before_optimizer', publish=False)['sampled_old_policy_kl']
        cfg.policy_safety['target_old_policy_kl'] = initial + 1e-6
        before = actor.actor_module.weight.detach().clone()
        returned_metrics = None
        refusal = None
        try:
            returned_metrics = actor.update_policy(data)
        except fixture.PolicySafetyRefusal as exc:
            refusal = str(exc)
        files = sorted(Path(temp).rglob('*'))
        evidence = [json.loads(path.read_text()) for path in files if path.is_file()]
        parameter_changed = not torch.equal(before, actor.actor_module.weight.detach())
        committed = actor._policy_safety.committed
        assert refusal is not None and committed == 1 and parameter_changed
        assert returned_metrics is None
        assert all(not item['outer_completed'] and not item['scheduler_advanced'] for item in evidence)
        print(json.dumps(dict(refused=True, refusal=refusal, initial_sampling_kl=initial,
            target_sampling_kl=cfg.policy_safety['target_old_policy_kl'],
            parameters_changed_by_real_Adam_step=parameter_changed,
            actual_committed_optimizer_calls=committed,
            in_memory_Adam_state_entries=len(actor.actor_optimizer.state),
            in_memory_Adam_steps=[float(s['step']) for s in actor.actor_optimizer.state.values()],
            actor_update_returned_metrics=returned_metrics,
            written_files=[str(path.relative_to(temp)) for path in files if path.is_file()],
            saved_evidence_top_level_keys=sorted(set().union(*(set(item) for item in evidence))),
            saved_model_or_Adam_state=False,
            phase_decisions=[dict(phase=item['phase'], committed_optimizer_calls=item['committed_optimizer_calls'],
                allowed=item['decision']['allowed'], reasons=item['decision']['reasons']) for item in evidence],
            scope='Affected source update_policy/_optimizer_step/PolicySafetyRuntime on one-parameter CPU Adam fixture; no Ray/FSDP/NCCL or wrapper cleanup execution'), indent=2))
finally:
    factory.tearDown()
