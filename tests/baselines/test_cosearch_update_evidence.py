from types import SimpleNamespace
import numpy as np
import torch

from search_r1.baselines import controller as module


def test_controller_evidence_preserves_outcomes_and_separates_record_legality(monkeypatch, tmp_path):
    data = SimpleNamespace(batch=dict(responses=torch.tensor([[2, 3], [4, 5]]),
        info_mask=torch.ones(2, 2), attention_mask=torch.ones(2, 2),
        token_level_scores=torch.tensor([[0., 1.], [0., -.2]])),
        non_tensor_batch=dict(trajectory_uid=['a', 'b'], index=np.array([17, 17]),
            reward_model=[dict(ground_truth=dict(target=np.array(['x', 'alias'])))] * 2,
            action_records=[[dict(text='<answer>x</answer>')], [dict(text='x')]]))
    monkeypatch.setattr(module, 'decode_answers',
        lambda data, tokenizer, require_legal: ['x', None] if require_legal else ['x', 'x'])
    class Ranker:
        def update(self, outcomes):
            self.outcomes = outcomes
            return {'ranker/events': 1}
    def run(enabled):
        if enabled:
            monkeypatch.setenv('COSEARCH_UPDATE_EVIDENCE_ROOT', str(tmp_path))
        else:
            monkeypatch.delenv('COSEARCH_UPDATE_EVIDENCE_ROOT', raising=False)
        controller = module.Controller.__new__(module.Controller)
        controller.ranker = Ranker()
        controller.trainer = SimpleNamespace(global_steps=4,
            tokenizer=SimpleNamespace(name_or_path=str(tmp_path)))
        metrics = {}
        controller.after_actor(data, metrics)
        return controller.ranker, metrics
    off, baseline = run(False)
    on, actual = run(True)
    assert baseline == actual
    for uid in ('a', 'b'):
        assert off.outcomes[uid][0] == on.outcomes[uid][0]
        np.testing.assert_array_equal(off.outcomes[uid][1], on.outcomes[uid][1])
        assert off.outcomes[uid][2] == on.outcomes[uid][2]
    assert on.diagnostic_provenance['outer_step'] == 4
    assert on.diagnostic_trajectories['a']['records_legal'] is True
    assert on.diagnostic_trajectories['b']['records_legal'] is False
    assert on.diagnostic_trajectories['b']['raw_answer'] == 'x'
    assert on.diagnostic_trajectories['b']['answer'] is None
    assert on.diagnostic_trajectories['a']['question_uid'] == 17
    assert on.diagnostic_trajectories['a']['response_ids'] == [2, 3]
