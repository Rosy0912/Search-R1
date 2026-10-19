"""CPU fault tests for observation handling and exact repeat-2 admission."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from reproduction.trained_tree7b_v2.test_contract import fixture
from reproduction.trained_tree7b_v2 import contract as old, submit as old_submit
from . import contract as c
from .progress import ProgressObserver
from .submit import command, params


def plans():
    reference, _, _, _ = fixture()
    reference.update(repeat=2, job=c.OLD_JOB, root=str(old.ROOT / 'r2'),
                     source_commit=c.SCIENCE_SOURCE, source=str(c.BASE / 'sources' / c.SCIENCE_SOURCE))
    plan = copy.deepcopy(reference)
    plan.update(job=c.JOB, root=str(c.ROOT / 'r2'))
    plan['recovery'] = dict(superseded_uid=c.OLD_UID, original_plan_sha256=c.OLD_PLAN_SHA,
                           failed_receipt_sha256=c.FAILED_RECEIPT_SHA, entry_source_commit='a' * 40,
                           entry_source=str(c.BASE / 'sources' / ('a' * 40)),
                           entry_source_files=copy.deepcopy(reference['source_files']))
    return plan, reference


class Observation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 0
        self.observer = ProgressObserver(lambda: self.now)
    def tearDown(self): self.temp.cleanup()
    def put(self, n=128):
        value = dict(rank=2, completed=n, total=len(range(2, 51713, 8)), at=0)
        (self.root / 'rank-2.progress.json').write_text(json.dumps(value))
        return value
    def test_actual_glob_then_read_missing_reproduced(self):
        self.put()
        paths = list(self.root.glob('rank-*.progress.json'))
        paths[0].unlink()
        with self.assertRaises(FileNotFoundError): [json.loads(p.read_text()) for p in paths]
        self.assertEqual(self.observer.read(self.root), [])
    def test_no_progress_during_worker_initialization(self):
        self.now = 2000
        self.assertEqual(self.observer.read(self.root), [])
    def test_missing_observation_keeps_last_then_recovers(self):
        self.put(); self.observer.read(self.root)
        (self.root / 'rank-2.progress.json').unlink()
        self.now = 10
        self.assertEqual(self.observer.read(self.root)[0]['completed'], 128)
        self.now = 65
        self.put(256)
        self.assertEqual(self.observer.read(self.root)[0]['completed'], 256)
    def test_persistent_missing_bounded(self):
        self.put(); self.observer.read(self.root)
        (self.root / 'rank-2.progress.json').unlink()
        self.observer.read(self.root); self.now = 60
        with self.assertRaises(TimeoutError): self.observer.read(self.root)
    def test_truncated_json_bounded_then_valid(self):
        (self.root / 'rank-2.progress.json').write_text('{')
        self.assertEqual(self.observer.read(self.root), [])
        self.now = 59; self.put()
        self.assertEqual(len(self.observer.read(self.root)), 1)
    def test_persistent_invalid_json_bounded(self):
        (self.root / 'rank-2.progress.json').write_text('{')
        self.observer.read(self.root); self.now = 60
        with self.assertRaises(TimeoutError): self.observer.read(self.root)
    def test_other_io_error_not_swallowed(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError('fixture')):
            with self.assertRaises(PermissionError): self.observer.read(self.root)
    def test_schema_rank_total_and_regression_rejected(self):
        for field, value in [('rank', 3), ('total', 51713), ('completed', -1), ('completed', True)]:
            data=self.put();data[field]=value
            (self.root/'rank-2.progress.json').write_text(json.dumps(data))
            with self.assertRaises(AssertionError):self.observer.read(self.root)
        self.put(256);self.observer.read(self.root);self.put(128)
        with self.assertRaises(AssertionError):self.observer.read(self.root)
    def test_no_prediction_read_or_rng_change(self):
        state=random.getstate();self.put();self.observer.read(self.root)
        self.assertEqual(state,random.getstate())
        self.assertFalse(list(self.root.glob('*predictions*')))


class Recovery(unittest.TestCase):
    def test_exact_reference_only_namespace_differs(self):
        plan, reference=plans();self.assertEqual(c.validate_recovery(plan,reference),c.ROOT/'r2')
    def test_any_scientific_or_extra_field_rejected(self):
        p,r=plans()
        for key,value in [('repeat',3),('model','other'),('kind','base'),('data_sha256','0'*64),
                          ('source_commit','b'*40),('repeat_count',4),('gpus',16),('actual_training_iterations',150),
                          ('new_unreviewed_field',True)]:
            plan=copy.deepcopy(p);plan[key]=value
            with self.assertRaises(AssertionError):c.validate_recovery(plan,r)
        p['science']['engine_seed']=1
        with self.assertRaises(AssertionError):c.validate_recovery(p,r)
    def test_wrong_failure_or_wrapper_source_rejected(self):
        p,r=plans()
        for key,value in [('superseded_uid','other'),('original_plan_sha256','0'*64),
                          ('failed_receipt_sha256','0'*64),('entry_source','/other')]:
            plan=copy.deepcopy(p);plan['recovery'][key]=value
            with self.assertRaises(AssertionError):c.validate_recovery(plan,r)
        p['recovery']['entry_source_files'][next(iter(r['source_files']))]='0'*64
        with self.assertRaises(AssertionError):c.validate_recovery(p,r)
    def test_gate_uid_resources_and_adapter_binding(self):
        p,_=plans();sub=dict(job=p['job'],uid='new',plan_sha256='b'*64,
                            superseded_uid=c.OLD_UID,entry_source_commit='a'*40)
        gate=dict(source=p['source_commit'],plan_sha256='b'*64,superseded_uid=c.OLD_UID,
                  entry_source_commit='a'*40,actual=dict(name=p['job'],uid='new',state='RUNNING',resources=c.ALLOCATION))
        c.validate_gate(p,sub,gate)
        for key,value in [('superseded_uid','other'),('entry_source_commit','b'*40)]:
            g=copy.deepcopy(gate);g[key]=value
            with self.assertRaises(AssertionError):c.validate_gate(p,sub,g)
        gate['actual']['uid']='old'
        with self.assertRaises(AssertionError):c.validate_gate(p,sub,gate)
    def test_only_observation_entry_ast_changed(self):
        here=Path(__file__).parent
        before=here.with_name('trained_tree7b_v2').joinpath('entry.py').read_text()
        after=(here/'entry.py').read_text()
        after=after.replace('from .progress import ProgressObserver\n','').replace(' observer=ProgressObserver()\n','')
        after=after.replace('   progress=observer.read(out)',"   progress=[json.loads(q.read_text()) for q in out.glob('rank-*.progress.json')]")
        self.assertEqual(ast.dump(ast.parse(before)),ast.dump(ast.parse(after)))
        for name,h in old.FROZEN.items():self.assertEqual(old.sha(here.parents[1]/name),h)
    def test_command_science_and_gpu_resources_identical(self):
        p,_=plans();path=c.ROOT/'plans/r2.json'
        a=command(p,path).replace(p['recovery']['entry_source'],p['source']).replace(
            'reproduction.trained_tree7b_r2_recovery_v1.launch','reproduction.trained_tree7b_v2.launch')
        self.assertEqual(a,old_submit.command(p,path))
        before=old_submit.params(p,path);after=params(p,path)
        before.pop('cmd');after.pop('cmd');before.pop('tmpdir');after.pop('tmpdir')
        self.assertEqual(before,after)


if __name__ == '__main__': unittest.main()
