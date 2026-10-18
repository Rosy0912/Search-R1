import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import test_searchr1_recovery as fixtures
from .contracts import sha
from .searchr1_contracts import validate_searchr1_plan


class NodeRecoveryTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.SearchR1RecoveryTests(); f.setUp(); self.addCleanup(f.doCleanups)
        self.previous = f.plan; self.template = f.template
        self.root = Path(self.previous['output']).parent; self.root.mkdir()
        self.pp = self.root/'plan.json'; self.pp.write_text(json.dumps(self.previous))
        sub = {k:self.previous[k] for k in ('job', 'output', 'checkpoint_identity', 'group', 'outer')}
        sub.update(job_id='139033', plan=str(self.pp), plan_sha256=sha(self.pp), repeat_index=1, repeat_count=3)
        self.sp = self.root/'submission.json'; self.sp.write_text(json.dumps(sub))
        self.out = self.root/'output'; self.out.mkdir()
        (self.out/'predictions.jsonl').write_text('preserved partial\n')
        (self.out/'progress.json').write_text(json.dumps(dict(completed=7168,total=51713,
            job=self.previous['job'],checkpoint_identity=self.previous['checkpoint_identity'])))
        files = {n:dict(bytes=(self.out/n).stat().st_size, sha256=sha(self.out/n))
                 for n in ('predictions.jsonl', 'progress.json')}
        self.inc = self.root/'incident.json'
        self.inc.write_text(json.dumps(dict(output=str(self.out),files=files,scheduler={
            'node':{'stdout':'BootTime=2026-10-18T16:35:12 Prolog error'},
            'accounting':{'stdout':'139033|REQUEUED|2657|'}})))
        self.back = self.root/'backup.json'
        self.back.write_text(json.dumps(dict(source=str(self.out),preserved_only=True,
            accepted_complete=False,files=files)))
        self.ret = self.root/'retirement.json'
        self.ret.write_text(json.dumps(dict(previous_job_id='139033',
            prior_state='PENDING_HELD_NO_ALLOCATION',retired_state='CANCELLED',
            incident_sha256=sha(self.inc),backup_sha256=sha(self.back))))
        proof = dict(reason='node4_reboot_139033_partial_preserved')
        for k, v in dict(previous_plan=self.pp,previous_submission=self.sp,
                         incident=self.inc,backup=self.back,retirement=self.ret).items():
            proof[k] = str(v); proof[k+'_sha256'] = sha(v)
        self.plan = dict(self.previous); del self.plan['repeat_recovery']
        self.plan.update(evaluation_code_commit='a'*40,evaluation_run_id='b'*32,
            job='zhangjunming-ttt-sr1-r1-nodefix',
            output=str(self.root/('node-recovery-'+'a'*12)/'output'),node_recovery=proof)
        self.state = '139033|jmzhang|CANCELLED by 1066|\n'
        def accounting(cmd, **kwargs):
            return '139030|jmzhang|FAILED\n' if cmd[cmd.index('-j')+1]=='139030' else self.state
        self.check = patch('reproduction.seven_dataset.searchr1_node_recovery.subprocess.check_output',side_effect=accounting).start()
        self.addCleanup(patch.stopall)

    def test_preserved_same_repeat_new_full_execution(self):
        self.assertEqual(validate_searchr1_plan(self.plan,self.template),Path(self.plan['output']).parent)
        self.assertEqual(self.plan['repeat_index'],1)
        self.assertEqual(self.plan['repeat_count'],3)

    def test_scientific_or_namespace_changes_rejected(self):
        for k,v in [('model_revision','different'),('repeat_index',2),('repeat_count',4),
                    ('protocol_identity','f'*64),('data_root','other'),('job',self.previous['job']),
                    ('output',self.previous['output']),('evaluation_code_commit',self.previous['evaluation_code_commit']),
                    ('evaluation_run_id',self.previous['evaluation_run_id']),('sampling_override',True)]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):validate_searchr1_plan(p,self.template)

    def test_live_pending_completed_or_unknown_predecessor_rejected(self):
        for state in ['RUNNING','PENDING','COMPLETED','FAILED','CANCELLED by 0']:
            self.state='139033|jmzhang|'+state+'|\n'
            with self.subTest(state=state),self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)
        self.state=''
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_changed_partial_or_completed_repeat_rejected(self):
        f=self.out/'predictions.jsonl';old=f.read_bytes();f.write_bytes(old+b'changed')
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)
        f.write_bytes(old);(self.out/'summary.json').write_text('{}')
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_missing_or_tampered_backup_rejected(self):
        self.back.write_text(self.back.read_text()+' ')
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_unbound_retirement_rejected(self):
        v=json.loads(self.ret.read_text());v['prior_state']='RUNNING';self.ret.write_text(json.dumps(v))
        self.plan['node_recovery']['retirement_sha256']=sha(self.ret)
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_live_queue_or_scheduler_error_rejected(self):
        with patch('reproduction.seven_dataset.searchr1_node_recovery.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='139033|RUNNING',stderr='')):
            with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)

    def test_recursive_node_retry_rejected(self):
        v=json.loads(self.pp.read_text());v['node_recovery']={};self.pp.write_text(json.dumps(v))
        self.plan['node_recovery']['previous_plan_sha256']=sha(self.pp)
        with self.assertRaises(ValueError):validate_searchr1_plan(self.plan,self.template)


if __name__ == '__main__':unittest.main()
