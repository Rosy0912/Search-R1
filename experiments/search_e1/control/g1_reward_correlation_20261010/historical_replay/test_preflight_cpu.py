import hashlib, importlib.util, json, os, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

SOURCE=Path(__file__).with_name('preflight_cpu.py')
spec=importlib.util.spec_from_file_location('preflight_cpu_fixture',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PreflightTests(unittest.TestCase):
    def run_binding(self):
        return dict(auxiliary_physical_indices=[5,6],training_physical_indices=[0,1,2,3],excluded_physical_indices=[4,7],probe_physical_index=6,run_id='db474b6abf654ad9892b02ae67ab6f86')
    def test_candidate5_allowed_but_not_allocation_permission(self):self.assertTrue(m.candidate_allowed(self.run_binding(),5))
    def test_production_probe6_rejected(self):self.assertFalse(m.candidate_allowed(self.run_binding(),6))
    def test_training_gpu_rejected(self):self.assertFalse(m.candidate_allowed(self.run_binding(),0))
    def test_excluded4_rejected(self):self.assertFalse(m.candidate_allowed(self.run_binding(),4))
    def test_excluded7_rejected(self):self.assertFalse(m.candidate_allowed(self.run_binding(),7))
    def test_unlisted_device_rejected(self):self.assertFalse(m.candidate_allowed(self.run_binding(),8))
    def test_process_exact_tuple(self):
        r=dict(pid=10,start_ticks=20,sid=30,uid=40);self.assertTrue(m.same_process(dict(r,state='S'),r))
    def test_reused_pid_rejected(self):
        r=dict(pid=10,start_ticks=20,sid=30,uid=40);self.assertFalse(m.same_process(dict(r,state='S',start_ticks=21),r))
    def test_foreign_uid_rejected(self):
        r=dict(pid=10,start_ticks=20,sid=30,uid=40);self.assertFalse(m.same_process(dict(r,state='S',uid=41),r))
    def test_zombie_rejected(self):
        r=dict(pid=10,start_ticks=20,sid=30,uid=40);self.assertFalse(m.same_process(dict(r,state='Z'),r))
    def test_csv_whitespace_and_empty(self):
        self.assertEqual(m.parse_csv('GPU-X, 10, 20\n')[0][0],'GPU-X');self.assertEqual(m.parse_csv(''),[])
    def test_full_byte_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture';data=b'abcd'*101;p.write_bytes(data);self.assertEqual(m.digest(p,7),hashlib.sha256(data).hexdigest())
    def snapshot(self,known=True):
        td=tempfile.TemporaryDirectory();base=Path(td.name);(base/'logs').mkdir();r=self.run_binding();record=dict(pid=10,start_ticks=20,sid=30,uid=40,command='fake-retriever ')
        (base/'logs/process-ledger.json').write_text(json.dumps(dict(run_id=r['run_id'],jobs=dict(retriever=dict(ledger={'10':record})))))
        results=[subprocess.CompletedProcess([],0,'5, GPU-X, 81920, 20000, 61920, A800\n',''),subprocess.CompletedProcess([],0,('GPU-X, 10, 20000\n' if known else 'GPU-X, 11, 20000\n'),'')]
        actual=dict(record,state='S',command_sha256='hash');actual.pop('command');actual['pid']=10 if known else 11
        with patch.object(m.subprocess,'run',side_effect=results) as tool,patch.object(m,'proc_identity',return_value=actual):
            o=m.resource_snapshot(base,r,5);self.assertTrue(all('-i' in call.args[0] and '5' in call.args[0] for call in tool.call_args_list))
        td.cleanup();return o
    def test_snapshot_shared_original_retriever(self):
        o=self.snapshot();self.assertTrue(o['only_original_retriever_shares_candidate']);self.assertFalse(o['dedicated_gpu']);self.assertFalse(o['resource_authorization_to_allocate']);self.assertEqual(o['candidate_gpu']['actual_free_mib'],61920)
    def test_snapshot_foreign_process_holds(self):
        o=self.snapshot(False);self.assertEqual(o['foreign_or_unbound_pids'],[11]);self.assertFalse(o['only_original_retriever_shares_candidate'])

if __name__=='__main__':unittest.main()
