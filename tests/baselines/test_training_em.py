import hashlib
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from search_r1.baselines.training_em import TrainingEMJournal, verify_training_em_audit
from search_r1.baselines.controller import Controller
from search_r1.baselines.bapo import BoundaryStage


class Batch(SimpleNamespace):
    def __len__(self):
        return self.batch['responses'].shape[0]


class Tokenizer:
    texts = {1: '<answer>Paris</answer>', 2: '<answer>London</answer>',
             3: '<search>Paris</search>', 4: '<answer>I do not know</answer>',
             5: '<answer>the PARIS!</answer>', 6: '<answer>Paris<search>x</search></answer>'}
    def decode(self, ids):
        return ''.join(self.texts[int(i)] for i in ids if int(i))


def output(codes, trajectories=None, pad=0):
    ids = torch.tensor([[i] + [0] * pad for i in codes])
    return Batch(batch=dict(responses=ids, info_mask=(ids != 0).long(), attention_mask=(ids != 0).long()),
                 non_tensor_batch=dict(trajectory_uid=np.array(trajectories or [str(i) for i in range(len(codes))], dtype=object)))


def questions(count=2, siblings=2):
    ids = [f'nq:train:{i+1234}' for i in range(count) for _ in range(siblings)]
    return Batch(batch=dict(responses=torch.zeros(len(ids), 1)),
        non_tensor_batch=dict(index=np.array(ids, dtype=object),
            reward_model=np.array([dict(ground_truth=dict(target=['Paris'])) for _ in ids], dtype=object)))


def journal(tmp_path, qa=None, count=2, siblings=2):
    return TrainingEMJournal(qa or questions(count, siblings), Tokenizer(), actual_outer=50,
        attempt_id=51, run_root=tmp_path, policy_snapshot_id='snapshot-before-outer50-attempt51',
        source_commit='c'*40, bindings={'common_input_binding': {'manifest_sha256': '0b-user-pin'}},
        expected_questions=count, siblings=siblings)


def test_initial_and_retained_em_any_siblings_and_resample_original_provenance(tmp_path):
    j = journal(tmp_path)
    first = output([5, 2, 2, 6])
    j.capture(first, [0, 1, 2, 3], 'initial', 0)
    replacement = output([1, 1], ['replacement-original2', 'replacement-original3'])
    j.capture(replacement, [2, 3], 'resample', 1)
    retained = output([5, 2, 1, 1], ['0', '1', 'replacement-original2', 'replacement-original3'], pad=2)
    metrics, proof = j.finish(retained, [0, 1, 2, 3])
    assert metrics['train_em/initial/em'] == .25
    assert metrics['train_em/initial/any_of_2_em'] == .5
    assert metrics['train_em/initial/correct_response_count'] == 1
    assert metrics['train_em/initial/correct_question_count'] == 1
    assert metrics['train_em/retained/em'] == .75
    assert metrics['train_em/retained/any_of_2_em'] == 1
    assert metrics['train_em/resampled_response_count'] == 2
    assert all(type(v) in (int, float) for v in metrics.values())
    verified = verify_training_em_audit(proof['path'], proof['sha256'], {'actual_outer': 50, 'attempt_id': 51})
    assert verified['metrics'] == metrics
    records = [json.loads(line) for line in open(proof['path'])]
    resampled = [r for r in records if r.get('phase') == 'resample']
    assert [r['question_id'] for r in resampled] == ['nq:train:1235'] * 2
    assert [r['response_ordinal'] for r in resampled] == [0, 1]
    assert [r['source_row_index'] for r in resampled] == [2, 3]
    assert all(r['generation_ordinal'] == 1 for r in resampled)
    with pytest.raises(FileExistsError):
        other = journal(tmp_path)
        other.capture(first, range(4), 'initial', 0)
        other.finish(first, range(4))


def test_missing_terminal_and_observation_answer_do_not_score_idk_can_match_reference(tmp_path):
    qa = questions(2, 2)
    qa.non_tensor_batch['reward_model'][2] = dict(ground_truth=dict(target=['I do not know']))
    qa.non_tensor_batch['reward_model'][3] = dict(ground_truth=dict(target=['I do not know']))
    j = journal(tmp_path, qa)
    actual = output([3, 6, 4, 4])
    j.capture(actual, range(4), 'initial', 0)
    metrics, proof = j.finish(actual, range(4))
    assert metrics['train_em/initial/em'] == .5  # Independent EM, not BAPO's IDK reward0.
    assert metrics['train_em/initial/valid_answer_count'] == 2
    assert verify_training_em_audit(proof['path'], proof['sha256'])['metrics'] == metrics


def test_verifier_recomputes_em_and_immutable_metadata_not_only_sha(tmp_path):
    j = journal(tmp_path); first = output([1, 2, 2, 2])
    j.capture(first, range(4), 'initial', 0)
    _, proof = j.finish(first, range(4))
    with pytest.raises(ValueError, match='SHA256'):
        verify_training_em_audit(proof['path'], '0'*64)
    with pytest.raises(ValueError, match='provenance'):
        verify_training_em_audit(proof['path'], proof['sha256'], {'policy_snapshot_id': 'other'})
    rows = [json.loads(line) for line in open(proof['path'])]
    rows[1]['em'] = 0
    corrupted = tmp_path / 'corrupted.jsonl'
    raw = ''.join(json.dumps(r) + '\n' for r in rows).encode(); corrupted.write_bytes(raw)
    with pytest.raises(ValueError, match='answer/score'):
        verify_training_em_audit(corrupted, hashlib.sha256(raw).hexdigest())


def test_actual_authored_mask_keeps_observation_answer_out_of_em(tmp_path):
    authored = ['<search>x</search>', '<search>y</search>']
    observation = '<information><answer>Paris</answer></information>'
    text = authored[0] + observation + authored[1]
    ids = torch.tensor([[ord(c) for c in text]])
    mask = torch.tensor([[1]*len(authored[0]) + [0]*len(observation) + [1]*len(authored[1])])
    actual = Batch(batch=dict(responses=ids, info_mask=mask, attention_mask=torch.ones_like(mask)),
                   non_tensor_batch={'trajectory_uid':np.array(['authored-boundary'])})
    class CharTokenizer:
        def decode(self, tokens): return ''.join(chr(int(i)) for i in tokens)
    j=TrainingEMJournal(questions(1,1), CharTokenizer(), actual_outer=1, attempt_id=1,
        run_root=tmp_path, policy_snapshot_id='original', source_commit='c'*40, expected_questions=1, siblings=1)
    events=j.capture(actual, [0], 'initial', 0)
    assert events[0]['model_spans'] == authored
    assert events[0]['answer'] is None and events[0]['em'] == 0
    metrics, proof=j.finish(actual, [0])
    assert metrics['train_em/retained/em'] == 0
    assert verify_training_em_audit(proof['path'], proof['sha256'])['metrics'] == metrics


def test_actual_controller_full512x5_before_after_resampling_and_no_generation_changes(tmp_path, monkeypatch):
    import search_r1.baselines.controller as module
    qa = questions(512, 5)
    gen = SimpleNamespace(meta_info={'policy_snapshot_id': 'run:attempt_51:before_outer_50'})
    class Engine:
        calls = []
        def __init__(self, *args): self.turn = 0
        def run(self, batch, tree=None):
            self.turn += 1
            count = 2560 if self.turn == 1 else len(batch)
            Engine.calls.append(count)
            codes = [1 if i // 5 % 2 == 0 and i % 5 == 0 else 2 for i in range(count)] if self.turn == 1 else [1] * count
            self.last_states = [SimpleNamespace(source=i, code=code, trajectory=f'{self.turn}-{i}') for i, code in enumerate(codes)]
            self.cost = {'generated_tokens': count, 'trajectories': count}
            return self.compose(self.last_states)
        def compose(self, states):
            return output([s.code for s in states], [s.trajectory for s in states]), np.array([s.source for s in states])
    monkeypatch.setattr(module, 'SearchRollout', Engine)
    monkeypatch.setattr(module, 'select_rows', lambda batch, rows: list(rows))
    monkeypatch.setenv('BAPO_RUN_ID', str(tmp_path))
    c = Controller.__new__(Controller)
    c.method='bapo'; c.ranker=None; c.stage=BoundaryStage(early=False); c.reward_profile='baseline_aligned'
    c.config=OmegaConf.create(dict(alignment={'training_em_enabled':False, 'order_manifest':'order.json', 'order_sha256':'o'*64}, algorithm_source_commit='c'*40,
        gigpo={'anchor_mode':'context'}, seed=42, context_alignment='report',
        bapo={'resample_attempts':2}, holdout={'manifest_path':'shared.json','manifest_sha256':'a'*64}))
    c.trainer=SimpleNamespace(tokenizer=Tokenizer(), global_steps=50,
        fresh200_progress=SimpleNamespace(attempts=51),
        fresh200_shared_input_bindings={'common_input_binding': {'manifest_sha256': '0b-user-pin'}},
        config=SimpleNamespace(retriever=SimpleNamespace(identity_manifest='retriever.json', identity_manifest_sha256='b'*64)))
    before={}; ordinary, sources=c.generate(None, gen, qa, before)
    ordinary_calls=Engine.calls.copy(); Engine.calls.clear()
    c.config.alignment.training_em_enabled=True
    after={}; instrumented, actual_sources=c.generate(None, gen, qa, after)
    assert Engine.calls == ordinary_calls == [2560, 1280]
    assert torch.equal(instrumented.batch['responses'], ordinary.batch['responses'])
    assert np.array_equal(actual_sources, sources)
    assert {k:v for k,v in after.items() if k.startswith('rollout/')} == before
    assert after['train_em/initial/em'] == .1
    assert after['train_em/initial/any_of_5_em'] == .5
    assert after['train_em/retained/em'] == .6
    assert after['train_em/retained/any_of_5_em'] == 1
    assert after['train_em/retained/response_count'] == 2560
    assert after['train_em/retained/unique_question_count'] == 512
    assert after['train_em/initial/correct_response_count'] == 256
    assert after['train_em/initial/correct_question_count'] == 256
    assert after['train_em/retained/correct_response_count'] == 1536
    assert after['train_em/retained/correct_question_count'] == 512
    proof=c.training_em_last_proof
    verified=verify_training_em_audit(proof['path'], proof['sha256'], {'actual_outer':50,'attempt_id':51,'expected_questions':512,'siblings':5})
    assert verified['metrics'] == {k:v for k,v in after.items() if k.startswith('train_em/')}
    assert verified['header']['bindings']['common_input_binding']['manifest_sha256'] == '0b-user-pin'
