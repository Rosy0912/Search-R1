"""Meaningful CPU synthetic loop/guard tests; no actual final200/GPU claim."""
import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/final200-eval-candidate')
REPO = HERE.parents[1] / 'opsd-unclipped-multitraj200-teacherrecompute'
sys.path.insert(0, str(REPO))
from verl import DataProto
from search_r1.llm_agent import generation as original


def load(name, path):
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rsplit('.', 1)[0]
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


candidate = load('search_r1.llm_agent.generation_final200_candidate', Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/opsd-final200-paired-eval-source/search_r1/llm_agent/generation.py'))
subject = load('search_r1.baselines.final200_evaluation', Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/opsd-final200-paired-eval-source/search_r1/baselines/final200_evaluation.py'))


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = '<pad>'
    def __len__(self):
        return 100000
    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False):
        if isinstance(text, str):
            return {'input_ids': [0 if c == '\x00' else ord(c) + 2 for c in text.replace('[PAD]', '\x00')]}
        rows = [self(t)['input_ids'] for t in text]
        if return_tensors == 'pt':
            width = max(map(len, rows), default=0)
            return {'input_ids': torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.int64)}
        return {'input_ids': rows}
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(int(i) - 2) for i in ids if int(i) > 1)
    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(r, skip_special_tokens=skip_special_tokens) for r in rows]
    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=False):
        return ''.join('[' + turn['role'] + ']' + turn['content'] + '\n' for turn in prompt) + '[assistant]'


TOKENIZER = Tokenizer()
META = dict(eos_token_id=1, pad_token_id=0, recompute_log_prob=False, do_sample=False, validate=True)


class Group:
    def __init__(self):
        self.calls = []
        self.round = 0
    def generate_sequences(self, batch):
        self.calls.append({k: v.clone() for k, v in batch.batch.items()})
        texts = []
        for row in batch.batch['input_ids']:
            q = int(re.search('Q([0-9]+)', TOKENIZER.decode(row)).group(1))
            if q % 4 == 0 or (q % 4 == 1 and self.round > 0):
                text = '<think>yes</think><answer>A</answer>discarded suffix'
            elif q % 4 == 2:
                text = 'invalid action'
            else:
                text = '<think>query</think><search>query ' + str(q) + '</search>discarded suffix'
            texts.append(text)
        self.round += 1
        ids = TOKENIZER(texts, return_tensors='pt', padding=True)['input_ids']
        return DataProto.from_dict(tensors={'responses': ids}, meta_info={})


def input_batch(n):
    prompts = [[dict(role='user', content='original instruction Q' + str(i))] for i in range(n)]
    rows = [TOKENIZER(TOKENIZER.apply_chat_template(p))['input_ids'] for p in prompts]
    width = max(map(len, rows))
    inputs = torch.tensor([[0] * (width - len(r)) + r for r in rows], dtype=torch.int64)
    attention = (inputs != 0).long()
    positions = (attention.cumsum(1) - 1) * attention
    batch = DataProto.from_dict(tensors=dict(input_ids=inputs, attention_mask=attention, position_ids=positions), meta_info=META.copy())
    records = [dict(ordinal=i, file_row=i, file_sha256='a' * 64, question_id='nq:test:' + str(i),
                    question_uid=hashlib.sha256(str(i).encode()).hexdigest(),
                    identity=dict(prompt=p, source='nq', question_prompt_text=p[0]['content'], targets=['A'])) for i, p in enumerate(prompts)]
    return batch, records


def loop(module, n, observe=False, long_obs=True, literal_pad=False):
    group = Group()
    config = module.GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=8, search_url='http://127.0.0.1:18084/retrieve',
        topk=3, force_final_answer=False)
    manager = module.LLMGenerationManager(TOKENIZER, group, config, is_validation=True)
    observer = subject.TurnObserver(TOKENIZER) if observe else None
    if observer is not None:
        manager.final200_observer = observer
    batch, records = input_batch(n)
    requests = []
    def request(url, json):
        requests.append(copy.deepcopy(json))
        body = ('long body ' * 100 if long_obs else 'body') + ('[PAD]body' if literal_pad else '')
        docs = [dict(document=dict(contents='Title\n' + body), score=1.0) for _ in range(3)]
        return types.SimpleNamespace(json=lambda: dict(result=[copy.deepcopy(docs) for _ in json['queries']]))
    with mock.patch.object(module.requests, 'post', side_effect=request):
        output = manager.run_llm_loop(batch, batch.batch['input_ids'][:, -2048:].clone())
    raw = observer.consume(subject.arrays_of(output.batch)) if observer else None
    return output, group.calls, requests, raw, records


def raw_rows(raw, records):
    rows = []
    n = len(records)
    for i, record in enumerate(records):
        turns = []
        for t in raw['turns']:
            if i not in t['active_indices']:
                continue
            active_i = t['active_indices'].index(i)
            turns.append(dict(step=t['step'], active_indices=t['active_indices'],
                generation_context={k: v[active_i] for k, v in t['generation_context'].items()},
                raw_generation={k: v[active_i] for k, v in t['raw_generation'].items()},
                action_token_ids=t['action_token_ids'][i], action_text=t['action_text'][i],
                observation_text=t['observation_text'][i], observation_ids_untruncated=t['observation_ids_untruncated'][i],
                observation_ids_retained=t['observation_ids_retained'][i], done=t['dones'][i],
                valid_action=t['valid_actions'][i], search=t['searches'][i], retrieval_step_ref=t['step'],
                rolling_after={k: v[i] for k, v in t['rolling_after'].items()}, right_after={k: v[i] for k, v in t['right_after'].items()}))
        rows.append(dict(kind='ACTUAL_FINAL200_QUESTION_FULL_TOOL_TRACE', model_role='official_base', input_record=record,
            gold_location='DRIVER_SCORING_ONLY', initial_input={k: v[i] for k, v in raw['initial_input'].items()},
            initial_start=raw['initial_start'][i], generation_meta=raw['generation_meta'], turns=turns,
            final={k: v[i] for k, v in raw['final'].items()},
            final_meta={k: (v[i] if isinstance(v, list) and len(v) == n else v) for k, v in raw['final_meta'].items()}))
    tools = dict(start=records[0]['ordinal'], count=n, turns=[dict(step=t['step'], active_indices=t['active_indices'], retrieval=t['retrieval']) for t in raw['turns']])
    return subject.canonical(rows), subject.canonical(tools)


class LoopTests(unittest.TestCase):
    def assert_equal_outputs(self, left, right):
        self.assertEqual(set(left.batch.keys()), set(right.batch.keys()))
        for name in left.batch.keys():
            self.assertTrue(torch.equal(left.batch[name], right.batch[name]), name)
        self.assertEqual(left.meta_info, right.meta_info)
    def test_original_vs_candidate_with_and_without_hooks(self):
        old, calls, requests, _, _ = loop(original, 9)
        for observed in (False, True):
            new, new_calls, new_requests, _, _ = loop(candidate, 9, observed)
            self.assert_equal_outputs(old, new)
            self.assertEqual(requests, new_requests)
            self.assertEqual(len(calls), len(new_calls))
            for a, b in zip(calls, new_calls):
                for name in a:
                    self.assertTrue(torch.equal(a[name], b[name]))
    def test_last_single_question_padding_does_not_invent_rows(self):
        output, calls, _, raw, records = loop(candidate, 1, True)
        self.assertEqual(len(calls[0]['input_ids']), 8)
        self.assertEqual(len(raw['turns'][0]['raw_generation']['responses']), 1)
        rows, tools = raw_rows(raw, records)
        self.assertEqual(subject.validate_raw_batch(records, rows, tools, TOKENIZER)['questions'], 1)
    def test_full_raw_replay_active_and_inactive_observation_truncation(self):
        _, _, _, raw, records = loop(candidate, 4, True)
        rows, tools = raw_rows(raw, records)
        result = subject.validate_raw_batch(records, rows, tools, TOKENIZER)
        self.assertEqual(result['turns'], 11)
        self.assertGreater(len(rows[1]['turns'][0]['observation_ids_untruncated']), 500)
        self.assertEqual(len(rows[1]['turns'][0]['observation_ids_retained']), 500)
        self.assertEqual(len(rows[0]['turns']), 1)
    def test_observation_literal_pad_id_is_preserved_before_original_masking(self):
        output, _, _, raw, records = loop(candidate, 4, True, long_obs=False, literal_pad=True)
        old, _, _, _, _ = loop(original, 4, False, long_obs=False, literal_pad=True)
        self.assert_equal_outputs(old, output)
        rows, tools = raw_rows(raw, records)
        self.assertIn('[PAD]', rows[1]['turns'][0]['observation_text'])
        self.assertEqual(subject.validate_raw_batch(records, rows, tools, TOKENIZER)['questions'], 4)
    def test_full_raw_replay_rejects_wrong_identity_context_mask_tool_and_clock(self):
        _, _, _, raw, records = loop(candidate, 4, True)
        rows, tools = raw_rows(raw, records)
        mutations = [lambda a, b: a[0]['input_record'].update(question_uid='b' * 64),
            lambda a, b: a[1]['turns'][0]['generation_context']['input_ids'].__setitem__(-1, 99),
            lambda a, b: a[1]['turns'][0]['observation_ids_retained'].__setitem__(0, 99),
            lambda a, b: a[0]['final']['info_mask'].__setitem__(len(a[0]['final']['prompts']), 0),
            lambda a, b: b['turns'][0]['retrieval'][0]['request']['queries'].append('invented query'),
            lambda a, b: b['turns'][0]['retrieval'][0].update(finished_monotonic_ns=0),
            lambda a, b: a[1]['generation_meta'].update(targets=['A']),
            lambda a, b: a[0]['turns'][0].update(done=True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                a, b = copy.deepcopy(rows), copy.deepcopy(tools)
                mutate(a, b)
                with self.assertRaises(ValueError):
                    subject.validate_raw_batch(records, a, b, TOKENIZER)
    def test_gold_non_tensor_and_meta_rejected_before_any_generate(self):
        for tensors, meta in (({'reward_model': np.array([dict(ground_truth=dict(target=['A']))], dtype=object)}, META),
                              ({}, dict(META, targets=['A']))):
            batch, _ = input_batch(1)
            batch.non_tensor_batch = tensors
            batch.meta_info = meta
            observer = subject.TurnObserver(TOKENIZER)
            with self.assertRaises(ValueError):
                observer.event('begin', gen_batch=batch, initial_input_ids=batch.batch['input_ids'])
    def test_real_dataproto_pop_keeps_gold_only_in_driver(self):
        batch, _ = input_batch(1)
        batch.non_tensor_batch = {'reward_model': np.array([dict(ground_truth=dict(target=['A']))], dtype=object),
                                  'raw_prompt': np.array([['question']], dtype=object)}
        gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        self.assertEqual(gen.non_tensor_batch, {})
        self.assertIn('reward_model', batch.non_tensor_batch)
        self.assertEqual(gen.meta_info, {})
    def test_double_begin_unknown_event_and_wrong_final_arrays_fail(self):
        batch, _ = input_batch(1)
        observer = subject.TurnObserver(TOKENIZER)
        observer.event('begin', gen_batch=batch, initial_input_ids=batch.batch['input_ids'])
        with self.assertRaises(ValueError):
            observer.event('begin', gen_batch=batch, initial_input_ids=batch.batch['input_ids'])
        with self.assertRaises(ValueError):
            observer.event('invented')
        observer.event('end', output=batch)
        wrong = subject.arrays_of(batch.batch)
        wrong['input_ids'][0, 0] = 777
        with self.assertRaises(ValueError):
            observer.consume(wrong)


class GuardAndIOTests(unittest.TestCase):
    def test_actual_common_manifest_copy_uses_explicit_data_root_not_manifest_parent(self):
        # Only path routing is synthetic. The actual common manifest bytes are used;
        # the mocked frame checker raises rather than claim this fake parquet valid.
        actual_manifest = HERE.parent / 'job3304-first5-complete-originals-v3/raw/bindings/data_manifest.json'
        original = actual_manifest.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), subject.COMMON_SHA)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = root / 'run' / 'bindings'
            bindings.mkdir(parents=True)
            copied = bindings / 'data_manifest.json'
            copied.write_bytes(original)
            data = root / 'separate-common-data'
            data.mkdir()
            final = data / 'final.parquet'
            final.write_bytes(b'CPU fake placeholder, never accepted as parquet')
            with mock.patch('search_r1.baselines.sharedholdout._prebuilt_frame', side_effect=RuntimeError('frame-sentinel')) as check:
                with self.assertRaisesRegex(RuntimeError, 'frame-sentinel'):
                    subject.prepare_inputs(copied, data, root / 'inputs.json')
                self.assertEqual(check.call_args[0][0], final.resolve())
            self.assertFalse((root / 'inputs.json').exists())
    def test_actual_writer_native_scorer_and_full_raw_files_replay(self):
        from search_r1.baselines.evaluation import batch_answer_metrics
        from verl.utils.reward_score.qa_em import compute_score_generated_em
        from verl.trainer.main_ppo import decode_model_response_spans
        output, _, _, raw, records = loop(candidate, 4, True)
        output.non_tensor_batch = dict(
            data_source=np.array(['nq'] * 4, dtype=object),
            raw_prompt=np.array([r['identity']['prompt'] for r in records], dtype=object),
            reward_model=np.array([dict(ground_truth=dict(target=['A'])) for _ in range(4)], dtype=object))
        rewards = []
        for row in output:
            width = row.batch['responses'].shape[-1]
            spans = decode_model_response_spans(TOKENIZER, row.batch['responses'],
                row.batch['attention_mask'][-width:].bool() & row.batch['info_mask'][-width:].bool())
            rewards.append(compute_score_generated_em(spans, dict(target=['A'])))
        online = batch_answer_metrics(output, TOKENIZER)
        with tempfile.TemporaryDirectory() as directory:
            writer = subject.FullTrajectoryWriter.__new__(subject.FullTrajectoryWriter)
            writer.directory = Path(directory)
            writer.manifest = dict(records=records)
            writer.tokenizer = TOKENIZER
            writer.provenance = dict(model_role='official_base')
            writer.cursor = 0
            writer.files, writer.scores, writer.rewards, writer.raw_files = [], [], [], []
            writer.generation_observer = subject.TurnObserver(TOKENIZER)
            writer.generation_observer.completed = raw
            fsync_events = []
            actual_fsync = subject.os.fsync
            def fsync_spy(fd):
                st = subject.os.fstat(fd)
                fsync_events.append((st.st_dev, st.st_ino))
                return actual_fsync(fd)
            with mock.patch.object(subject.os, 'fsync', side_effect=fsync_spy):
                writer.append(output, online, torch.tensor(rewards).reshape(4, 1))
                fsync_events.append('append-returned')
            before_return = fsync_events[:fsync_events.index('append-returned')]
            for field in ('tensors', 'rows'):
                st = (Path(directory) / writer.files[0][field]).stat()
                self.assertIn((st.st_dev, st.st_ino), before_return)
            st = Path(directory).stat()
            self.assertIn((st.st_dev, st.st_ino), before_return)
            self.assertEqual(writer.cursor, 4)
            saved = writer.raw_files[0]
            content = gzip.decompress((Path(directory) / saved['path']).read_bytes())
            self.assertEqual(hashlib.sha256(content).hexdigest(), saved['raw_sha256'])
            rows = [subject.loads(line) for line in content.splitlines()]
            tools = subject.loads((Path(directory) / saved['tool_batches']).read_bytes())
            self.assertEqual(subject.validate_raw_batch(records, rows, tools, TOKENIZER)['questions'], 4)
            with np.load(Path(directory) / writer.files[0]['tensors'], allow_pickle=False) as arrays:
                replay = subject.legacy.score_arrays(dict(arrays), records, TOKENIZER)
            self.assertEqual(subject.encoded(replay), subject.encoded(online))

    def test_eval_only_profile_budget_and_training_rejection(self):
        keys = {'data':dict(val_batch_size=64,val_data_num=None,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500),
            'max_turns':4,'force_final_answer':False,'do_search':True,
            'retriever':dict(url='http://127.0.0.1:18084/retrieve',topk=3),
            'trainer':dict(val_only=True,val_before_train=True,n_gpus_per_node=8,nnodes=1),
            'baselines':dict(method='opsd',holdout_eval_only=False,opsd_e1=dict(enabled=False,profile=None),
                local_runtime_only=True,local_runtime_host='h100-10',seed=42,final200_evaluation=dict(enabled=True)),
            'algorithm':dict(no_think_rl=False,kl_ctrl=dict(kl_coef=0)),
            'actor_rollout_ref':dict(actor=dict(use_kl_loss=False,kl_loss_coef=0,entropy_coeff=0),rollout=dict(tensor_model_parallel_size=1,
                gpu_memory_utilization=.35,enforce_eager=True,free_cache_engine=True,dtype='bfloat16',temperature=1.,top_k=-1,
                top_p=1.,ignore_eos=False,n=1,max_num_batched_tokens=6144,max_num_seqs=512))}
        config=OmegaConf.create(keys)
        self.assertEqual(subject.validate_eval_config(config)['max_turns'],4)
        mutations = [('trainer.val_only',False),('baselines.opsd_e1.enabled',True),('baselines.opsd_e1.profile','multitraj200'),
                     ('data.val_batch_size',512),('max_turns',5),('force_final_answer',True),('retriever.url','http://foreign/retrieve')]
        for key,value in mutations:
            changed=OmegaConf.create(copy.deepcopy(keys))
            OmegaConf.update(changed,key,value)
            with self.assertRaises(ValueError):
                subject.validate_eval_config(changed)
        with self.assertRaises(RuntimeError):
            subject.reject_training(config)

    def worker(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        return types.SimpleNamespace(rank=0, world_size=8, actor_optimizer=optimizer,
            actor_lr_scheduler=scheduler, actor=types.SimpleNamespace(actor_optimizer=optimizer))
    def invoke(self, worker, phase):
        probe = DataProto.from_dict(tensors={'audit_probe': torch.zeros((1, 1), dtype=torch.int64)}, meta_info={'zero_step_phase': phase})
        process = dict(pid=123, uid=0, process_start_ticks='456')
        with mock.patch('search_r1.baselines.corrected_runtime_capture.process_identity', return_value=process):
            return subject.zero_optimizer_callback(worker, probe)
    def test_true_optimizer_scheduler_guards_and_independent_clock_scope(self):
        worker = self.worker()
        initial = copy.deepcopy(worker.actor_lr_scheduler.state_dict())
        self.invoke(worker, 'install')
        result = self.invoke(worker, 'finish').non_tensor_batch['zero_step_audit'][0]
        self.assertEqual(result['before'], result['after'])
        self.assertIn('NOT_TRAINING_ADAM_RESTORE', result['scope'])
        self.assertEqual(initial, subject.optimizer_clock(worker)['scheduler'])
        for field in ('actor_optimizer', 'actor_lr_scheduler'):
            worker = self.worker()
            self.invoke(worker, 'install')
            with self.assertRaises(RuntimeError):
                getattr(worker, field).step()
            with self.assertRaises(ValueError):
                self.invoke(worker, 'finish')
    def test_clock_change_guard_replacement_and_second_install_rejected(self):
        for action in (lambda w: setattr(w.actor_optimizer, 'step', lambda: None),
                       lambda w: setattr(w.actor_lr_scheduler, 'last_epoch', 50)):
            worker = self.worker()
            self.invoke(worker, 'install')
            action(worker)
            with self.assertRaises(ValueError):
                self.invoke(worker, 'finish')
        worker = self.worker()
        self.invoke(worker, 'install')
        with self.assertRaises(ValueError):
            self.invoke(worker, 'install')
    def test_no_overwrite_sha_mismatch_symlink_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'proof.json'
            subject.write_once(path, b'{"ok":1}')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(subject.pinned_json(path, sha), {'ok': 1})
            with self.assertRaises(FileExistsError):
                subject.write_once(path, b'overwrite')
            with self.assertRaises(ValueError):
                subject.pinned_json(path, '0' * 64)
            alias = Path(directory) / 'alias.json'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                subject.pinned_json(alias, sha)
            for raw in (b'{"ok":1,"ok":2}', b'{"value":NaN}'):
                bad = Path(directory) / ('bad' + str(len(raw)))
                bad.write_bytes(raw)
                with self.assertRaises(ValueError):
                    subject.pinned_json(bad, hashlib.sha256(raw).hexdigest())
    def test_old_roles_training_identity_and_unknown_actual_allocation_rejected(self):
        base = dict(kind='REVIEWED_FINAL200_PAIRED_EVAL_STAGE', model_role='actual_hf200', training_commit=subject.TRAINING_SOURCE,
            training_job_id='3304', actual_outer_updates=200, common_manifest_sha256=subject.COMMON_SHA,
            official_base_revision=subject.OFFICIAL_REVISION, official_base_manifest_sha256=subject.OFFICIAL_MANIFEST_SHA,
            eval_job_id=None)
        for patch in (dict(model_role='hf500'), dict(training_job_id='3007'), dict(actual_outer_updates=500),
                      dict(eval_job_id='3304'), dict(eval_job_id=None)):
            value = dict(base, **patch)
            with self.assertRaises(ValueError):
                subject.verify_stage(value)


if __name__ == '__main__':
    unittest.main()
