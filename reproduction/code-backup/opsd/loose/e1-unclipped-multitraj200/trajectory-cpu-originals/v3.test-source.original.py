"""CPU fixtures validate identity, real generation and state, not GPU acceptance."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from verl import DataProto

from search_r1.baselines import trajectory_archive as a
from search_r1.baselines.rollout import object_array, padded, select_rows


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    chat_template = 'explicit-small-CPU-fixture-not-official-tokenizer'
    def __init__(self):
        self.all_special_ids = list(type(self).all_special_ids)
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(v) - 10) if int(v) >= 10 else f'[{int(v)}]' for v in ids)
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        content = '<user>' + messages[0]['content'] + '</user>' + ('<assistant>' if add_generation_prompt else '')
        return self.encode(content) if tokenize else content


class GeneratingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
        self.register_buffer('inv_freq', torch.tensor([3., 4.]))
        self.original_inv_freq = self.inv_freq.clone()
        self.max_seq_len_cached = 5
        self.config = SimpleNamespace(max_position_embeddings=8192, flag='original')
        self.generation_config = SimpleNamespace(flag='original')
        self.calls = []
        self.eval()
    def generate(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), copy.deepcopy(kwargs)))
        random.random(); np.random.rand(); torch.rand(4)
        self.inv_freq.add_(10)
        self.original_inv_freq.add_(3)
        self.max_seq_len_cached = 200
        self.new_cache = [42]
        self.config.flag = 'changed'
        self.generation_config.flag = 'changed'
        self.training = True
        # Genuine tiny generate output, unrelated to the given gold text.
        tail = torch.tensor([[110, 111, 2]], device=input_ids.device)
        return torch.cat((input_ids, tail), 1)


def questions(count=512):
    return [dict(question_uid=f'nq:train:{i}', question_position=i, data_source='nq',
        question=f'question {i}', accepted_answers=[f'gold {i}'],
        accepted_answers_provenance='common_training_reward_model.ground_truth.target') for i in range(count)]


def fixture_trainer(root):
    tokenizer = Tokenizer()
    qs = questions()
    rows = [dict(prompt=[dict(role='user', content=a.E1_INSTRUCTION + '\nQuestion: ' + q['question'])],
            extra_info=dict(index=q['question_uid']), data_source=q['data_source'],
            reward_model=dict(style='rule', ground_truth=dict(target=q['accepted_answers']))) for q in qs]
    frame = pd.DataFrame(rows)
    ids = padded([tokenizer.apply_chat_template(row['prompt']) for row in rows], 0, left=True)
    non = dict(index=object_array([q['question_uid'] for q in qs]),
               data_source=object_array([q['data_source'] for q in qs]),
               reward_model=object_array([row['reward_model'] for row in rows]))
    batch = DataProto.from_dict(dict(input_ids=ids, attention_mask=(ids != 0).long(),
                                    position_ids=(ids != 0).long().cumsum(-1) - 1), non_tensors=non)
    config = SimpleNamespace(baselines=SimpleNamespace(
        opsd_e1=SimpleNamespace(run_root=str(root)), distillation=SimpleNamespace(teacher_path='/fixture/official-model')),
        data=SimpleNamespace(prompt_key='prompt'))
    trainer = SimpleNamespace(train_dataset=SimpleNamespace(dataframe=frame), tokenizer=tokenizer,
        config=config, e1_gate=SimpleNamespace(job='99999', source='a'*40, config_sha='b'*64))
    return trainer, batch


def completed_student(batch):
    count = len(batch)
    prompt = batch.batch['input_ids']
    response = torch.tensor([[110, 111, 112, 0]]).repeat(count, 1)
    model_mask = torch.tensor([[1, 0, 1, 0]]).repeat(count, 1)
    batch.batch = DataProto.from_dict(dict(prompts=prompt, responses=response,
        input_ids=torch.cat((prompt, response), -1), attention_mask=torch.cat(((prompt != 0).long(), (response != 0).long()), -1),
        info_mask=torch.cat(((prompt != 0).long(), model_mask), -1),
        position_ids=torch.arange(prompt.shape[1] + 4).repeat(count, 1),
        step_ids=torch.tensor([[0, -1, 0, -1]]).repeat(count, 1))).batch
    batch.non_tensor_batch['trajectory_uid'] = object_array([f'actual-trajectory-{i}' for i in range(count)])
    turn = dict(generation_context_token_ids=[100], raw_response_token_ids=[110, 111, 0],
        raw_response_attention_mask=[1, 1, 0], observation_text_before_truncation='full actual observation',
        observation_token_ids_before_truncation=[113, 114, 115], retained_observation_token_ids=[113],
        postprocessed_action_token_ids=[110, 111])
    batch.non_tensor_batch['action_records'] = object_array([[copy.deepcopy(turn)] for _ in range(count)])
    return batch


def publish_teacher_fixtures(archive):
    """Explicitly synthetic rank files for archive-reader CPU tests, never production."""
    context = archive.context
    directory = Path(context['identity']['directory'])
    for rank in range(8):
        qs = context['questions'][rank*64:(rank+1)*64]
        model = GeneratingModel()
        with a.isolated_rng(rank) as rw, a.isolated_generation_state(model) as mw:
            rows = a.short_teacher_records(model, archive.trainer.tokenizer, qs, 'cpu')
        member = a.write_member(directory/f'teacher-rank-{rank}.jsonl', b''.join(a.canonical(row)+b'\n' for row in rows))
        report = dict(version=1, kind='CPU_SYNTHETIC_RANK_REPORT_NOT_GPU_ACCEPTANCE',
            identity=context['identity'], request_sha256=context['request_sha256'], rank=rank, world_size=8,
            pid=os.getpid(), question_count=64, teacher=member,
            generated_tokens=sum(len(r['output_token_ids']) for r in rows),
            input_tokens=sum(len(r['input_token_ids']) for r in rows), host_wall_seconds=0.,
            rng_restored=rw['rng_restored'], rng_witness=rw,
            model_state_restored=mw['model_state_restored'], model_state_witness=mw,
            diagnostic_answers_used_as_targets=False)
        a.write_member(directory/f'teacher-rank-{rank}.manifest.json', a.canonical(report))


class StateIsolationTests(unittest.TestCase):
    def test_all_cpu_rng_streams_restored_after_generation(self):
        before = a.state_digest(a.rng_state())
        model = GeneratingModel()
        with a.isolated_rng(73) as rw, a.isolated_generation_state(model) as mw:
            rows = a.short_teacher_records(model, Tokenizer(), questions(2), 'cpu')
        self.assertEqual(before, a.state_digest(a.rng_state()))
        self.assertTrue(rw['rng_restored']); self.assertTrue(mw['model_state_restored'])
        self.assertEqual(rows[0]['output_token_ids'], [110, 111, 2])
        self.assertNotEqual(rows[0]['output_text'], rows[0]['accepted_answers'][0])
        self.assertEqual(len(model.calls), 2)
    def test_rng_buffer_config_cache_and_flags_restore_on_exception(self):
        model = GeneratingModel(); before = a.state_digest(a.rng_state())
        inv = model.inv_freq; config = model.config; generation = model.generation_config
        with self.assertRaisesRegex(RuntimeError, 'actual failure'):
            with a.isolated_rng(3) as rw, a.isolated_generation_state(model) as mw:
                model.generate(torch.tensor([[50]])); model.register_buffer('added_buffer', torch.ones(2))
                raise RuntimeError('actual failure')
        self.assertEqual(before, a.state_digest(a.rng_state()))
        self.assertIs(model.inv_freq, inv); self.assertTrue(torch.equal(model.inv_freq, torch.tensor([3., 4.])))
        self.assertNotIn('added_buffer', model._buffers); self.assertFalse(hasattr(model, 'new_cache'))
        self.assertIs(model.config, config); self.assertIs(model.generation_config, generation)
        self.assertEqual(model.config.flag, 'original'); self.assertEqual(model.generation_config.flag, 'original')
        self.assertEqual(model.max_seq_len_cached, 5); self.assertFalse(model.training)
        self.assertTrue(rw['rng_restored']); self.assertTrue(mw['model_state_restored'])
    def test_parameter_mutation_refuses_training_continuation(self):
        model = GeneratingModel()
        with self.assertRaisesRegex(ValueError, 'parameter identity/version'):
            with a.isolated_generation_state(model):
                model.weight.add_(1)
    def test_true_generate_prompt_and_output_are_retained(self):
        model = GeneratingModel()
        with a.isolated_generation_state(model):
            rows = a.short_teacher_records(model, Tokenizer(), questions(1), 'cpu')
        row = rows[0]
        self.assertIn('gold 0', row['input_messages'][0]['content'])
        self.assertEqual(model.calls[0][0][0].tolist(), row['input_token_ids'])
        self.assertEqual(row['diagnostic_loss_mask'], [0, 0, 0])
        self.assertFalse(row['is_full_teacher_search_trajectory'])
        self.assertFalse(model.calls[0][1]['use_cache'])
    def test_bad_context_limit_or_chat_control_refused(self):
        model = GeneratingModel(); model.config.max_position_embeddings = 10
        with self.assertRaisesRegex(ValueError, 'context'):
            a.short_teacher_records(model, Tokenizer(), questions(1), 'cpu')
        tokenizer = Tokenizer(); tokenizer.all_special_ids += [ord('Q')+10]
        with self.assertRaisesRegex(ValueError, 'chat control'):
            a.short_teacher_records(GeneratingModel(), tokenizer, questions(1), 'cpu')
    def test_old_profile_noop_and_cpu_production_diagnostic_refused(self):
        old = SimpleNamespace(meta_info={})
        self.assertEqual(a.generate_rank_teacher_diagnostics(object(), old, object()), {})
        data = SimpleNamespace(meta_info={a.META_KEY: dict(identity=dict(profile='multitraj200',common_manifest_sha256=a.COMMON_SHA),questions=questions())})
        with self.assertRaisesRegex(ValueError, 'eight-rank'):
            a.generate_rank_teacher_diagnostics(object(), data, Tokenizer())


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.trainer, self.batch = fixture_trainer(self.root)
        self.archive = a.TrajectoryArchive(self.trainer)
    def tearDown(self):
        for path in self.root.rglob('*'):
            if path.is_dir() and not path.is_symlink(): path.chmod(0o700)
        self.tmp.cleanup()
    def prepared(self):
        self.archive.prepare_outer(self.batch, 1)
        batch = self.batch.repeat(5, interleave=True); self.archive.attach_groups(batch)
        return completed_student(batch)
    def test_before_repeat_questions_and_all_five_student_identities(self):
        batch = self.prepared(); self.archive.archive_student(batch)
        qs = self.archive.context['questions']
        self.assertEqual(len(qs), 512); self.assertEqual(len(set(q['question_uid'] for q in qs)),512)
        self.assertEqual(list(batch.non_tensor_batch['opsd_response_index'][:10]), list(range(5))*2)
        self.assertEqual(batch.meta_info[a.META_KEY]['questions'],qs)
        raw,_=a.read_member(Path(self.archive.context['identity']['directory'])/'student.jsonl')
        rows=[a.parse_json(line) for line in raw.splitlines()]
        self.assertEqual(len(rows),2560)
        self.assertEqual(rows[0]['tensors']['loss_mask'],[1,0,1,0])
        self.assertEqual(rows[0]['tool_turns'][0]['observation_token_ids_before_truncation'],[113,114,115])
    def test_actual_reordered_batch_preserves_all_row_hashes(self):
        batch = self.prepared(); self.archive.archive_student(batch)
        indices=torch.arange(len(batch)-1,-1,-1); batch.reorder(indices)
        batch.batch['loss_mask']=batch.batch['info_mask'][:,-4:].clone()
        self.archive.verify_training_rows(batch)
        batch.batch['responses'][1,0]+=1
        with self.assertRaises(ValueError):self.archive.verify_training_rows(batch)
    def test_loss_mask_group_or_missing_raw_turn_refused(self):
        batch=self.prepared(); batch.batch['loss_mask']=torch.ones_like(batch.batch['responses'])
        with self.assertRaises(ValueError):self.archive.archive_student(batch)
        del batch.batch['loss_mask']; batch.non_tensor_batch['opsd_response_index'][1]=0
        with self.assertRaises(ValueError):self.archive.archive_student(batch)
    def test_duplicate_question_and_changed_runtime_gold_refused(self):
        qs=questions();qs[1]['question_uid']=qs[0]['question_uid']
        with self.assertRaises(ValueError):a.validate_questions(qs)
        self.batch.non_tensor_batch['reward_model'][0]=dict(style='rule',ground_truth=dict(target=['wrong']))
        with self.assertRaises(ValueError):self.archive.prepare_outer(self.batch,1)
    def test_existing_partial_outer_is_not_overwritten(self):
        self.archive.prepare_outer(self.batch,1)
        path=Path(self.archive.context['identity']['directory'])/'questions.json'; raw=path.read_bytes()
        with self.assertRaises(FileExistsError):self.archive.prepare_outer(self.batch,1)
        self.assertEqual(path.read_bytes(),raw)
    def test_exclusive_member_and_symlink_parent_refused(self):
        p=self.root/'record';a.write_member(p,b'first')
        with self.assertRaises(FileExistsError):a.write_member(p,b'new')
        self.assertEqual(p.read_bytes(),b'first')
        d=self.root/'dir';d.mkdir();alias=self.root/'alias';alias.symlink_to(d,target_is_directory=True)
        with self.assertRaises(ValueError):a.write_member(alias/'record',b'forbidden')
    def test_paired_full_raw_validator_and_checkpoint_progress(self):
        batch=self.prepared();self.archive.archive_student(batch);publish_teacher_fixtures(self.archive)
        self.archive.finish_outer({'actor/optimizer_steps':10})
        directory=Path(self.archive.context['identity']['directory'])
        result=a.validate_archive(directory,1,self.archive.context['identity'],tokenizer=self.trainer.tokenizer)
        self.assertEqual((result['student_count'],result['teacher_count']),(2560,512))
        self.assertFalse(result['actual_worker_bindings_verified'])
        inventory=self.archive.checkpoint_inventory(1)
        self.assertEqual(len(inventory),20)
        with self.assertRaises(ValueError):self.archive.checkpoint_inventory(2)
        with self.assertRaises(ValueError):a.validate_archive(directory,1,self.archive.context['identity'],{r:1000+r for r in range(8)},self.trainer.tokenizer)
        with self.assertRaises(ValueError):a.validate_archive(directory,1,self.archive.context['identity'],{r:1000 for r in range(8)},self.trainer.tokenizer)
        state=self.archive.state_dict(1)
        self.assertLess(len(a.canonical(state)),1000)
        self.archive.load_state_dict(state,1)
        self.assertEqual(self.archive.completed_outer,1)
        with self.assertRaises(ValueError):a.validate_archive(directory,2)
    def test_missing_teacher_and_false_optimizer_count_refused(self):
        batch=self.prepared();self.archive.archive_student(batch)
        with self.assertRaises(ValueError):self.archive.finish_outer({'actor/optimizer_steps':9})
        with self.assertRaises(FileNotFoundError):self.archive.finish_outer({'actor/optimizer_steps':10})
    def test_immutable_record_byte_change_refused(self):
        p=self.root/'json';record=a.write_member(p,b'{"a":1}')
        p.chmod(0o600);p.write_bytes(b'{"a":2}')
        with self.assertRaises(ValueError):a.read_member(p,record['sha256'])
    def test_written_inode_replacement_even_identical_bytes_refused(self):
        p=self.root/'original'
        original=a.read_member
        def replace_before_read(path,expected=None):
            temp=self.root/'replacement'
            temp.write_bytes(b'same reviewed bytes');temp.chmod(0o444);temp.replace(path)
            return original(path,expected)
        with mock.patch.object(a,'read_member',side_effect=replace_before_read):
            with self.assertRaisesRegex(ValueError,'inode/parent'):a.write_member(p,b'same reviewed bytes')
    def test_raw_dispatch_metadata_keeps_global_questions_not_local_dedup(self):
        batch=self.prepared();self.archive.archive_student(batch)
        shuffled=select_rows(batch,list(range(30,80)))
        self.assertEqual(len(shuffled.meta_info[a.META_KEY]['questions']),512)
        generated=shuffled.pop(batch_keys=['input_ids','attention_mask','position_ids'])
        self.assertNotIn(a.META_KEY,generated.meta_info)
        self.assertNotIn('reward_model',generated.non_tensor_batch)
    def test_checkpoint_signed_missing_raw_roles_refused(self):
        manifest=dict(global_step=5,actual_outer_updates=5,world_size=8,run_root=str(self.root),files={})
        with self.assertRaisesRegex(ValueError,'extra/missing'):a.validate_checkpoint_archives(manifest,self.root)
    def test_nonfinite_duplicate_json_refused(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):a.parse_json(raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
