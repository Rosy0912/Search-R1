import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent

def load(name):
    s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pf=load('preflight_cpu');core=load('replay_core');runner=load('replay_gpu')
RUN={'run_id':'fixture','training_physical_indices':[0,1,2,3],'auxiliary_physical_indices':[5,6],'excluded_physical_indices':[4,7]}

class SharedResources(unittest.TestCase):
    def test_all_authorized_devices_only(self):
        for i in [0,1,2,3,5,6]:self.assertTrue(pf.candidate_allowed(RUN,i))
        for i in [-1,4,7,8]:self.assertFalse(pf.candidate_allowed(RUN,i))
    def test_declared_run_assignment_required(self):
        r=dict(RUN,auxiliary_physical_indices=[5]);self.assertFalse(pf.candidate_allowed(r,6))
    def fixture(self,base,candidate,role,pid=11,command='original',actual_command='original'):
        p=base/'logs';p.mkdir()
        rec=dict(pid=pid,start_ticks=9,sid=pid,uid=0,state='S',command=command)
        jobs={name:{'ledger':{str(pid):rec} if name==role else {}} for name in ['trainer','retriever','probe-server']}
        (p/'process-ledger.json').write_text(json.dumps({'run_id':'fixture','jobs':jobs}))
        actual={k:v for k,v in rec.items() if k!='command'};actual['command_sha256']=hashlib.sha256(actual_command.encode()).hexdigest()
        def query(args,**kwargs):
            self.assertEqual(args[1:3],['-i',str(candidate)])
            text=f'{candidate},GPUfixture,81920,30000,51920,A800\n' if 'query-gpu' in args[3] else f'GPUfixture,{pid},29000\n'
            return types.SimpleNamespace(stdout=text)
        return actual,query
    def snap(self,candidate,role,**extra):
        with tempfile.TemporaryDirectory() as d:
            actual,query=self.fixture(Path(d),candidate,role,**extra)
            with patch.object(pf,'proc_identity',return_value=actual),patch.object(pf.subprocess,'run',side_effect=query):
                return pf.resource_snapshot(Path(d),RUN,candidate)
    def test_training_gpu_matches_only_original_trainer(self):
        self.assertEqual(self.snap(0,'trainer')['foreign_or_unbound_pids'],[])
    def test_training_gpu_does_not_allow_other_original_role(self):
        self.assertEqual(self.snap(0,'retriever')['foreign_or_unbound_pids'],[11])
    def test_aux5_allows_only_original_retriever(self):
        self.assertEqual(self.snap(5,'retriever')['foreign_or_unbound_pids'],[])
        self.assertEqual(self.snap(5,'probe-server')['foreign_or_unbound_pids'],[11])
    def test_aux6_allows_original_probe_or_retriever(self):
        for role in ['probe-server','retriever']:self.assertEqual(self.snap(6,role)['foreign_or_unbound_pids'],[])
    def test_bound_PID_with_changed_argv_rejected(self):
        self.assertEqual(self.snap(0,'trainer',actual_command='different')['foreign_or_unbound_pids'],[11])
    def test_reused_PID_birth_rejected(self):
        a=dict(pid=11,start_ticks=10,sid=11,uid=0,state='S');b=dict(a,start_ticks=9)
        self.assertFalse(pf.same_process(a,b))
    def test_sampled_peak_not_directfree_claim(self):
        c=dict(expected_node='node',expected_run_id='fixture',gpu_index=3,gpu_uuid='g3',max_replay_memory_mib=12032,running_min_free_mib=6144)
        p=dict(node='node',samples=10,partial_lines=0,read_prefix_sha256='a'*64,direct_memory_free_history_available=False,
            path='/data/fixture/gpu-samples-fixture.jsonl',max_used_by_gpu={'3':dict(uuid='g3',used_mib=62763,total_mib=81920)})
        v=runner.validate_shared_peak_contract(c,p,core)
        self.assertFalse(v['sampled_peak_plus_own_budget_would_cross_requested_floor'])
        self.assertTrue(v['crossing_actual_floor_aborts_only_own_replay_at_next_micro'])
    def test_nominal_budget_over_total_HOLD(self):
        c=dict(expected_node='node',expected_run_id='fixture',gpu_index=0,gpu_uuid='g0',max_replay_memory_mib=12288,running_min_free_mib=8192)
        p=dict(node='node',samples=10,partial_lines=0,read_prefix_sha256='a'*64,direct_memory_free_history_available=False,
            path='/data/fixture/gpu-samples-fixture.jsonl',max_used_by_gpu={'0':dict(uuid='g0',used_mib=75000,total_mib=81920)})
        with self.assertRaises(core.Hold):runner.validate_shared_peak_contract(c,p,core)
    def test_gpu4_and7_forbidden_even_in_fake_aux(self):
        r=dict(RUN,auxiliary_physical_indices=[4,5,6,7],excluded_physical_indices=[])
        for i in [4,7]:self.assertFalse(pf.candidate_allowed(r,i))
    def test_core_not_changed_from_dense(self):
        dense=HERE.parent/'replay_core.py'
        self.assertEqual(hashlib.sha256(dense.read_bytes()).hexdigest(),hashlib.sha256((HERE/'replay_core.py').read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
