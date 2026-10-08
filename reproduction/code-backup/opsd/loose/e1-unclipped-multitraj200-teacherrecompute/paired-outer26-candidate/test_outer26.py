"""Synthetic local boundary fixtures; no remote/GPU/actual26 acceptance."""
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=module('c',HERE/'collect_outer26.py');a=module('a',HERE/'analyze_observed26.py')
k=a.load(HERE.parent/'paired-trajectory-cpu-analysis-candidate/analyze_outer.py',a.KERNEL_SHA,'fixture_exact_kernel')


class Boundaries(unittest.TestCase):
    def pair(self):
        def rec(name):return dict(path=str(c.DIR/name),sha256='0'*64,bytes=10)
        pair=dict(request=rec('questions.json'),student_manifest=rec('student-manifest.json'),teacher_rank_members=[
            dict(rank=i,manifest=rec(f'teacher-rank-{i}.manifest.json'),teacher=rec(f'teacher-rank-{i}.jsonl')) for i in range(8)])
        student={'student':rec('student.jsonl')};return pair,student

    def test_exact20(self):
        p,s=self.pair();self.assertEqual(set(c.expected_members(p,s)),set(c.NAMES))

    def test_path_traversal(self):
        p,s=self.pair();s['student']['path']=str(c.DIR/'../student.jsonl')
        with self.assertRaises(ValueError):c.expected_members(p,s)

    def test_rank_bool(self):
        p,s=self.pair();p['teacher_rank_members'][0]['rank']=False
        with self.assertRaises(ValueError):c.expected_members(p,s)

    def test_byte_bool_and_total_cap(self):
        for bad in (True,c.MAX_RAW):
            p,s=self.pair();s['student']['bytes']=bad
            with self.assertRaises(ValueError):c.expected_members(p,s)

    def test_unknown_record_key(self):
        p,s=self.pair();s['student']['extra']='x'
        with self.assertRaises(ValueError):c.expected_members(p,s)

    def test_exact_authority(self):
        raw=(HERE/'AUTHORITY-v2.original.json').read_bytes();self.assertEqual(c.sha(raw),c.AUTHORITY_SHA)
        value=json.loads(raw);value['outer_step']=10;self.assertNotEqual(c.sha(c.canonical(value)),c.AUTHORITY_SHA)

    def test_full_json_not_lf_prefix(self):
        with self.assertRaises(json.JSONDecodeError):c.parse(b'{"x":1}\n{"unfinished":')
        self.assertEqual(c.parse(b'{"x":1}\n'),{'x':1})

    def test_duplicate_nonfinite(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):c.parse(raw)

    def test_no_first5_inventory_alias(self):
        with self.assertRaises(ValueError):a.validate_inventory(k,c,dict(version=1,kind='TRAINING_CHECKPOINT_MANIFEST'))

    def test_same_kernel_missing_answer_not_agreement(self):
        q=dict(question_uid='fixture',question_position=0,data_source='fixture',dataset_row_position=0)
        answer=dict(answer=None,normalized_answer=None,em=0,f1=0,completed_terminal_answer=False,legal_complete_trajectory=False)
        row=k.question_summary(q,[dict(answer,response_index=i) for i in range(5)],answer)
        self.assertEqual(row['same_completed_pair_count'],0);self.assertEqual(row['student_teacher_agreement_count'],0)

    def envelope(self,name='unknown',truncated=False):
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as tar:
            header=tarfile.TarInfo(name);header.size=1;header.mode=0o444;tar.addfile(header,io.BytesIO(b'x'))
        result=gzip.compress(buf.getvalue());return result[:-4] if truncated else result

    def reject_extract(self,blob,returncode=0):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();payload=root/'payload.gz';payload.write_bytes(blob)
            wait=dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT',waited=True,returncode=returncode,runtime_sha256='fixture',
                authority_sha256=c.AUTHORITY_SHA,remote_host='h100-10',compressed_stdout_sha256=hashlib.sha256(blob).hexdigest())
            path=root/'wait.json';path.write_bytes(k.canonical(wait));output=root/'once'
            with self.assertRaises((ValueError,EOFError)):a.extract_once(k,c,payload,output,path,k.sha(path.read_bytes()),'fixture')
            return output.exists()

    def test_unknown_tar_member(self):self.assertTrue(self.reject_extract(self.envelope()))
    def test_tar_path_escape(self):self.assertTrue(self.reject_extract(self.envelope('../escape')))
    def test_partial_gzip_footer(self):self.assertTrue(self.reject_extract(self.envelope(truncated=True)))
    def test_failed_wait_is_not_acceptance(self):self.assertFalse(self.reject_extract(self.envelope(),returncode=1))
    def test_bool_wait_is_not_zero(self):self.assertFalse(self.reject_extract(self.envelope(),returncode=False))

    def test_exclusive_extract_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();output=root/'once';output.mkdir();payload=root/'a.gz';blob=self.envelope();payload.write_bytes(blob)
            wait=dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT',waited=True,returncode=0,runtime_sha256='fixture',authority_sha256=c.AUTHORITY_SHA,
                      remote_host='h100-10',compressed_stdout_sha256=k.sha(blob));record=root/'wait';record.write_bytes(k.canonical(wait))
            with self.assertRaises(FileExistsError):a.extract_once(k,c,payload,output,record,k.sha(record.read_bytes()),'fixture')

    def test_actual_raw_reward_tolerance_and_macro(self):
        authority=json.loads((HERE/'AUTHORITY-v2.original.json').read_bytes());m=authority['outer26_log_metrics']
        student=dict(em=0.14140625,recorded_executor_legal_action_macro_mean=m['env/ratio_of_valid_action'],
            search_calls_mean=m['env/number_of_valid_search'],recorded_executor_done_rate=m['env/finish_ratio'])
        a.validate_reward_and_macro(k,{'all':{'student':student}},authority)
        student['em']+=1/2560
        with self.assertRaises(ValueError):a.validate_reward_and_macro(k,{'all':{'student':student}},authority)

    def test_macro_not_micro(self):
        rows=[dict(data_source='nq',raw_sampled_token_count=1,recorded_legal_actions=1,tool_turn_count=1),
              dict(data_source='nq',raw_sampled_token_count=3,recorded_legal_actions=0,tool_turn_count=3)]
        cohorts={'all':{'student':{}},'nq':{'student':{}}};a.add_length_and_macro_metrics(k,cohorts,rows)
        self.assertEqual(cohorts['all']['student']['recorded_executor_legal_action_macro_mean'],.5)
        self.assertEqual(cohorts['nq']['student']['raw_sampled_tokens_mean'],2)


if __name__=='__main__':unittest.main()
