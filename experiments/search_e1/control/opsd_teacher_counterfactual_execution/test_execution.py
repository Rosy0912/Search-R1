"""CPU doubles; actual frozen native classes; no torch/models/HTTP/import."""
import copy
from contextlib import nullcontext, contextmanager
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
p = types.ModuleType('m3exec'); p.__path__ = [str(HERE)]; sys.modules['m3exec'] = p
sys.path.insert(0, str(HERE.parent / 'opsd_teacher_counterfactual'))
import test_cpu as native
import state
from m3exec.continuation import run, score_tap
from m3exec.admission import select_phase, validate_ticket, validate_config, sha
from m3exec.engine import execute_case

class Vector(list):
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Vector(self)
    def tolist(self): return list(self)

class Tensor(native.Tensor):
    def __getitem__(self, key):
        v = super().__getitem__(key)
        return Vector(v) if isinstance(v, list) else v

class Tokenizer(native.Tokenizer):
    def __call__(self, text, **kwargs): return types.SimpleNamespace(input_ids=Tensor([self.encode(text)]))

class Model(native.Model):
    def generate(self, ids, **kwargs):
        self.inputs.append(self.tokenizer.decode(ids.rows[0])); self.options.append(kwargs)
        if not self.segments: raise AssertionError('extra opportunity')
        return Tensor([ids.rows[0] + self.tokenizer.encode(self.segments.pop(0))])

class ExecutionFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls): native.CPUFixtures.setUpClass(); cls.core = native.CPUFixtures.core
    def compare(self, segments, resume=0):
        tok = Tokenizer(); allmodel = Model(tok, segments); requests = native.Requests()
        native.CPUFixtures.globals['requests'] = requests
        expected = native.CPUFixtures.Agent(allmodel, tok, 'fixture://no-http').run('fixture question', temperature=0)
        cur = state.initial()
        for i in range(resume):
            a,b = expected['policy_spans'][i]; end = expected['policy_spans'][i+1][0]
            obs = expected['response'][b:end]
            if i+1 == 4: obs = obs[:-len(state.FORCED)]
            cur = state.advance(cur, expected['response'][a:b], obs)
        model = Model(tok, segments[resume:]); newrequests = native.Requests()
        def retrieve(q): return newrequests.post('fixture://no-http', json={'queries':[q], 'topk':3,'return_scores':True}, timeout=120).json()
        rollout = types.SimpleNamespace(StoppingCriteriaList=list, StopTags=native.CPUFixtures.globals['StopTags'])
        result = run(model,tok,self.core,rollout,state,cur,'fixture question',['x'],None,retrieve,lambda e:nullcontext())
        for key in ('response','search_calls','termination'): self.assertEqual(result[key],expected[key])
        self.assertEqual(result['policy_spans'],[list(x) for x in expected['policy_spans']])
        self.assertEqual(model.inputs,allmodel.inputs[resume:])
        for actual,want in zip(model.options,allmodel.options[resume:]):
            self.assertEqual(set(actual),set(want))
            for k in actual:
                if k != 'stopping_criteria': self.assertEqual(actual[k],want[k])
            self.assertEqual(actual['stopping_criteria'][0].start,want['stopping_criteria'][0].start)
        self.assertEqual(len(newrequests.calls),expected['search_calls']-resume)
        return result
    def test_empty_direct_answer(self): self.compare(['<answer>x</answer>'])
    def test_active_after_search(self): self.compare(['<search>a</search>','<answer>x</answer>'],1)
    def test_foursearch_forced_once(self):
        r=self.compare(['<search>a</search>']*4+['<answer>x</answer>'],4)
        self.assertEqual(r['response'].count(state.FORCED),1);self.assertEqual(len(r['turns']),1)
    def test_budget_fifth_search(self): self.compare(['<search>a</search>']*5,4)
    def test_invalid_malformed_final(self): self.assertEqual(self.compare(['<think>bad<answer>x</answer>'])['protocol_em'],0)
    def test_native_cut_retains_actual_overrun_ids(self):
        r=self.compare(['<answer>x</answer>extra'])
        self.assertTrue(r['turns'][0]['decoded_segment_before_structural_cut'].endswith('extra'))
        self.assertFalse(r['response'].endswith('extra'))
    def test_invalid_no_retrieval(self): self.compare(['bad'])
    def test_search_badthink_still_native(self): self.compare(['<think>bad<search>a</search>','<answer>x</answer>'],1)
    def test_terminal_refused(self):
        tok=Tokenizer();m=Model(tok,[])
        with self.assertRaises(ValueError): run(m,tok,self.core,None,state,state.advance(state.initial(),'<answer>x</answer>'),'q',['x'],None,None,None)
    def test_context_limit_no_generate_or_force_append(self):
        cur=state.initial();cur['response_before_turn']='a'*8000
        tok=Tokenizer(); m=Model(tok,[])
        r=run(m,tok,self.core,None,state,cur,'q',['x'],None,None,None)
        self.assertEqual(r['termination'],'context_limit');self.assertEqual(r['response'],'a'*8000);self.assertFalse(m.inputs)
    def test_processor_order_inplace_identity_restore(self):
        class Processors(list):
            def __call__(self,ids,scores):
                for f in self: f(scores)
                return scores
        calls=[];p=Processors([lambda x:(calls.append(1),x.__setitem__(0,x[0]+1)),lambda x:(calls.append(2),x.__setitem__(0,x[0]*2))])
        score=Vector([3]); captures=[]; original=Processors.__call__
        with score_tap(Processors,lambda ids,raw,processed,ps:captures.append((raw[:],processed[:],ps is p))):
            self.assertIs(p([],score),score)
        self.assertEqual(calls,[1,2]);self.assertEqual(captures,[([3],[8],True)]);self.assertIs(Processors.__call__,original)
    def test_processor_restores_on_failure(self):
        class Processors(list):
            def __call__(self,ids,scores): raise ValueError('fixture')
        original=Processors.__call__
        with self.assertRaises(ValueError):
            with score_tap(Processors,lambda *a:None): Processors()([],Vector([1]))
        self.assertIs(Processors.__call__,original)
    def records(self): return [dict(case_id=str(i),pilot=i<16) for i in range(64)]
    def test_same_run_unique_pilot_no_repeat(self):
        rs=self.records();pilot=dict(accepted=True,run_id='r',completed_case_ids=[str(i) for i in range(16)],state_reconstruction_accepted=True,null_effect_accepted=True)
        self.assertEqual(len(select_phase(rs,'remaining',pilot,'r')),48)
        self.assertFalse(any(r['pilot'] for r in select_phase(rs,'remaining',pilot,'r')))
        pilot['run_id']='other'
        with self.assertRaises(ValueError):select_phase(rs,'remaining',pilot,'r')
    def test_duplicate_or_missing_pilot_refused(self):
        rs=self.records();rs[-1]=rs[0]
        with self.assertRaises(ValueError):select_phase(rs,'pilot')
        with self.assertRaises(ValueError):select_phase(self.records(),'remaining',None,'r')
    def test_four_teacher_refs_two_students_crossforced_common_branch(self):
        cur=state.boundary(state.initial(),'A',False);cur['aliases']=['A','B'];record=dict(case_id='c',pilot=True,source_train_row={'row':{'question':'q','answers':['x']}},original_student={'executable_states':[cur],'skipped_boundaries':[]},actual_hf_states={},reference_interventions={'conditions':[dict(name=n,reference=None if n=='none' else n,status='ready') for n in ('original','none','remove_final_answer','mismatched')]})
        class M:
            disabled=False
            def set_adapter(self,v):self.version=v
            @contextmanager
            def disable_adapter(self):
                self.disabled=True
                try:yield
                finally:self.disabled=False
        m=M();seen=[];forced=[];saved=[]
        def fake(*args,**kw):
            seen.append((m.disabled,m.version,args[8]));return dict(response='<answer>x</answer>',policy_spans=[[0,18]],protocol_em=1)
        def forcing(response,spans,q,ref,v,key):forced.append((response,spans,ref,v));return {}
        with patch('m3exec.engine.run',fake):out=execute_case(record,m,None,None,None,state,None,None,forcing,lambda n,v:saved.append(n))
        self.assertEqual(len(out['slots']),6);self.assertEqual([x[0] for x in seen],[True]*4+[False]*2)
        self.assertEqual([x[2] for x in seen],['original',None,'remove_final_answer','mismatched',None,None]);self.assertEqual(len(forced),24);self.assertTrue(all(x[2]=='original' for x in forced));self.assertFalse(m.disabled)
    def ticket_fixture(self, directory):
        import os,time
        config=dict(run_id='r',approved_gpu_uuids=['gpu'],_config_sha256='config')
        ticket=dict(schema='m3-external-resource-ticket-v1',accepted=True,run_id='r',phase='pilot',worker_index=0,
                    owner=dict(pid=123,uid=os.getuid(),starttick=42),m4_fully_released=True,root_lock_held=True,
                    gpu_uuid='gpu',config_sha256='config',expiry_unix=time.time()+60)
        for key in ('m4_release_proof','lock_proof','gpu_inventory_proof'):
            path=Path(directory)/key;path.write_text(json.dumps(dict(accepted=True,run_id='r',sample_unix=time.time())))
            ticket[key]=dict(path=str(path),sha256=sha(path))
        return ticket,config
    def test_ticket_foreign_gpu_notready_expired_and_pid_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            ticket,config=self.ticket_fixture(d)
            live=lambda p:ticket['owner'].copy()
            validate_ticket(ticket,config,'pilot',0,live)
            for key,value in [('gpu_uuid','foreign'),('m4_fully_released',False),('root_lock_held',False),('expiry_unix',0),('config_sha256','wrong')]:
                bad=copy.deepcopy(ticket);bad[key]=value
                with self.assertRaises(ValueError):validate_ticket(bad,config,'pilot',0,live)
            with self.assertRaises(ValueError):validate_ticket(ticket,config,'pilot',0,lambda p:dict(ticket['owner'],starttick=43))
    def test_bound_resource_proof_changed_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ticket,config=self.ticket_fixture(d)
            Path(ticket['lock_proof']['path']).write_text('{}')
            with self.assertRaises(ValueError):validate_ticket(ticket,config,'pilot',0,lambda p:ticket['owner'].copy())
    def test_exclusive_output_directory_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'worker';out.mkdir(exist_ok=False)
            (out/'kept').write_text('preserve')
            with self.assertRaises(FileExistsError):out.mkdir(exist_ok=False)
            self.assertEqual((out/'kept').read_text(),'preserve')
    def test_preserved_m4_pane_identity_real_exit_not_absence(self):
        from m3exec.admission import validate_m4_pane
        import hashlib
        server=dict(pid=1,uid=1005,starttick=2);binding=dict(pane_id='%1',pane_pid=3,pane_start_command_sha256=hashlib.sha256(b'fixture command').hexdigest(),tmux_server=server)
        raw='1\t0\t%1\t3\tfixture command'
        self.assertEqual(validate_m4_pane(binding,raw,server)['pane_dead_status'],0)
        for bad in ('',raw.replace('1\t0','0\t0'),raw.replace('1\t0','1\t'),raw.replace('\t3\t','\t4\t')):
            with self.assertRaises(ValueError):validate_m4_pane(binding,bad,server)
        with self.assertRaises(ValueError):validate_m4_pane(binding,raw,dict(server,starttick=4))
    def test_controller_and_acceptance_import_cpu_only(self):
        import m3exec.controller,m3exec.acceptance
        self.assertNotIn('torch',sys.modules)
    def test_owned_orphan_filter_exact_identity_not_list_order(self):
        from m3exec.acceptance import owned_leftovers
        owner=dict(pid=5,uid=1005,starttick=10);child=dict(pid=6,uid=1005,starttick=11)
        self.assertEqual(owned_leftovers([child,owner],owner),[child])
        self.assertEqual(owned_leftovers([dict(owner,starttick=12)],owner),[dict(owner,starttick=12)])
    def test_fresh_retriever_observer_stale_foreign_pid_reuse(self):
        from m3exec.admission import fresh_retriever
        import time
        expected=dict(pid=1,uid=1005,starttick=2,source_identity_sha256='src');observer=dict(pid=3,uid=1005,starttick=4)
        adm=types.SimpleNamespace(expected=expected,check_live=lambda:dict(ready=True))
        live=lambda pid:{1:{k:expected[k] for k in ('pid','uid','starttick')},3:observer}[pid]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'memory.json';now=time.time();sample=dict(sample_end_unix=now,candidate_pid=1,candidate_uid=1005,candidate_starttick=2,
                source_identity_sha256='src',physical_gpus=[0,1,2,3],foreign_compute_apps_seen=[],current={'foreign_compute_apps':[]},observer={'pid':3,'source_sha256':'script'})
            config=dict(retriever_observer_path=str(path),retriever_observer_identity=observer,retriever_observer_source_sha256='script')
            path.write_text(json.dumps(sample));fresh_retriever(config,adm,live,now)
            for key,value in [('sample_end_unix',now-6),('foreign_compute_apps_seen',[{'pid':9}]),('candidate_starttick',99)]:
                path.write_text(json.dumps(dict(sample,**{key:value})))
                with self.assertRaises(ValueError):fresh_retriever(config,adm,live,now)
    def test_missing_actual_terminal_artifacts_cannot_accept(self):
        from m3exec.acceptance import phase_acceptance
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):phase_acceptance(d,'pilot',{'all':[],'pilot':[]},'r',[],sha,None,state,self.core)
    def test_base12_actual_content_hash_rejects_changed_bytes(self):
        from m3exec.admission import fresh_base_weights
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for i in range(12):
                path=root/str(i);path.write_text('fixture'+str(i));entries.append(dict(relative_path=str(i),sha256=sha(path)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(source100=dict(path=d,files=entries))))
            self.assertEqual(len(fresh_base_weights(manifest,d)),12)
            (root/'0').write_text('changed')
            with self.assertRaises(ValueError):fresh_base_weights(manifest,d)
    def test_actual_m4_nested_process_status_binding(self):
        from m3exec.admission import validate_m4_worker_identity
        identity=dict(pid=1,uid=1005,starttick=2);status=dict(status='COMPLETE',process=dict(identity,argv=['fixture']))
        validate_m4_worker_identity(status,identity)
        with self.assertRaises(ValueError):validate_m4_worker_identity(status,dict(identity,starttick=3))
        with self.assertRaises(ValueError):validate_m4_worker_identity(dict(status,status='RUNNING'),identity)
    def test_kernel_uid_not_proc_directory_owner(self):
        from m3exec.admission import proc_identity
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);proc=root/'101';proc.mkdir()
            fields=['S','1','101']+['0']*16+['1001']
            (proc/'stat').write_text('101 (python) '+' '.join(fields));(proc/'status').write_text('Uid: 1005 1005 1005 1005\n')
            self.assertEqual(proc_identity(101,d),dict(pid=101,uid=1005,starttick=1001))
    def test_changed_reference_real_frozen_region_math_honest_status(self):
        from m3exec.numeric_regions import probe
        region_dir=HERE.parent/'opsd_token_region_analysis';sys.path.insert(0,str(region_dir))
        import summarize_regions as region
        import test_summarize_regions as factory
        with tempfile.TemporaryDirectory() as d:
            response='<think>a</think><answer>x</answer>'
            summary,path,spans=factory.make_probe(Path(d),'c','source100','changed',response,'counterfactual_reference_fixed_original_y',.5)
            summary['reference_status']='counterfactual_reference_not_original_success_claim'
            before=copy.deepcopy(summary)
            with self.assertRaises(ValueError):region.probe(summary,path,response,spans)
            proof=probe(region,summary,path,response,spans,True)
            self.assertTrue(proof['reconstruction_passed']);self.assertFalse(proof['M3_math_overlay']['success_inferred'])
            self.assertEqual(summary,before)
            with self.assertRaises(ValueError):probe(region,summary,path,response,spans,False)
    def test_actual_v2_scanner_preserves_self_owned_opaque_zombie(self):
        from m3exec.ownership import classify,strict_scan
        source=HERE.parent/'opsd_native512_eval_v2'/'owned_process_audit.py'
        spec=importlib.util.spec_from_file_location('_m3_actual_v2_audit',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        def proc(pid,tick,uid=1005,state_value='S',ppid=1):
            import hashlib
            return dict(pid=pid,uid=uid,starttick=tick,ppid=ppid,pgid=pid,state=state_value,comm='python',argv=['fixture'],kernel_uids=[uid]*4,argv_sha256=hashlib.sha256(b'fixture').hexdigest())
        owner=proc(100,1000);child=proc(101,1001);init=proc(1,1,0,ppid=0)
        def scan(extra=None,opaque=False):
            table={1:init,100:owner};table.update({x['pid']:x for x in (extra or [])})
            def env(pid):
                if opaque and pid==209:raise PermissionError('fixture')
                return {'E1_RUN_ID':'r'} if pid==100 else {}
            def read(pid):
                if pid not in table:raise FileNotFoundError()
                return copy.deepcopy(table[pid])
            return audit.audit_same_uid('r',1005,1000,[child],['/owned/worker.py'],ignored_exact=[owner],ancestry_anchors=[owner],snapshot_reader=read,environment_reader=env,pid_reader=lambda:list(table))
        raw=scan();result=classify(raw,owner,owner)
        self.assertFalse(raw['accepted_no_owned_or_opaque']);self.assertTrue(result['accepted_no_unexpected_owned_or_opaque'])
        self.assertEqual(len(result['expected_still_running_controller']),1);self.assertFalse(result['all_owned_absent_claimed'])
        for extra,opaque in [([child],False),([dict(child,state='Z')],False),([proc(209,1500)],True)]:
            self.assertFalse(classify(scan(extra,opaque),owner,owner)['accepted_no_unexpected_owned_or_opaque'])
        self.assertFalse(classify(raw,owner,dict(owner,starttick=1002))['accepted_no_unexpected_owned_or_opaque'])
        legacy=types.SimpleNamespace(owned_run_processes=lambda run,uid:[])
        with self.assertRaises(ValueError):strict_scan(legacy,'r',1005,owner,[],[])
    def test_second_popen_failure_latches_no_renew_wait_hold(self):
        from m3exec.supervision import supervise
        class Child:
            returncode=None;waited=0
            def poll(self):return self.returncode
            def wait(self):self.waited+=1;return self.returncode
        child=Child();attempts=[];children=[]
        def popen():
            attempts.append(1)
            if len(attempts)==2:raise OSError('second Popen fixture')
            return child
        error=None
        try:
            for index in range(3):children.append(dict(process=popen(),worker_index=index))
        except OSError as e:error=repr(e)
        renew=[];observe=[]
        failure=supervise(children,lambda:observe.append(1),lambda x:renew.append(1),lambda e:None,lambda _:setattr(child,'returncode',3),error)
        self.assertTrue(failure);self.assertFalse(renew);self.assertFalse(observe);self.assertEqual(child.waited,1)
    def test_no_search_multi_branch_expired_guard_no_next_gpu_call(self):
        tok=Tokenizer()
        class M(Model):
            def set_adapter(self,v):self.version=v
            def disable_adapter(self):return nullcontext()
        model=M(tok,['<answer>x</answer>']*6);cur=state.boundary(state.initial(),'A',False);cur['aliases']=['A']
        record=dict(case_id='c',pilot=True,source_train_row={'row':{'question':'q','answers':['x']}},original_student={'executable_states':[cur],'skipped_boundaries':[]},actual_hf_states={},reference_interventions={'conditions':[dict(name=n,reference=None if n=='none' else n,status='ready') for n in ('original','none','remove_final_answer','mismatched')]})
        checks=[];calls=[]
        def guard():
            checks.append(1)
            if len(checks)>1:raise ValueError('expired lease fixture')
        @contextmanager
        def capture(event):guard();yield
        def forcing(*args):guard();calls.append('probe');return {}
        rollout=types.SimpleNamespace(StoppingCriteriaList=list,StopTags=native.CPUFixtures.globals['StopTags'])
        with self.assertRaises(ValueError):execute_case(record,model,tok,self.core,rollout,state,lambda q:None,capture,forcing,lambda n,v:None)
        self.assertEqual(len(model.inputs),1);self.assertFalse(calls)
    def test_release_read_error_recovery_keeps_locks_until_proof(self):
        from m3exec.release_wait import await_release
        events=[];records=[];clock=[0];locks={'held':True}
        pending=[OSError('driver fixture'),dict(released=False,known_identities_absent_or_reused=False,own_cuda_clear=False),dict(released=True)]
        def read():
            self.assertTrue(locks['held']);value=pending.pop(0);events.append('read')
            if isinstance(value,Exception):raise value
            return value
        def sleep(n):self.assertTrue(locks['held']);clock[0]+=n;events.append('wait')
        result=await_release(read,records.append,sleep,lambda:clock[0])
        self.assertTrue(result['released']);self.assertEqual(events,['read','wait','read','wait','read'])
        self.assertEqual(records[0]['status'],'UNCERTAIN_READ_RETRY_LOCKS_HELD');self.assertFalse(records[0]['released'])
    def test_release_evidence_write_failure_does_not_grant_release(self):
        from m3exec.release_wait import await_release
        saved=[];waits=[]
        def record(value):
            saved.append(value)
            if len(saved)==1:raise OSError('receipt fixture')
        result=await_release(lambda:dict(released=True),record,waits.append)
        self.assertTrue(result['released']);self.assertEqual(waits,[5]);self.assertEqual(len(saved),2)
    def test_release_opaque_hold_requires_known_and_cuda_clear(self):
        from m3exec.release_wait import await_release
        clock=[0];proofs=[dict(released=False,known_identities_absent_or_reused=False,own_cuda_clear=True,only_independent_opaque_blockers=True),dict(released=False,known_identities_absent_or_reused=True,own_cuda_clear=False,only_independent_opaque_blockers=True),dict(released=True)]
        def sleep(n):clock[0]+=100
        result=await_release(lambda:proofs.pop(0),lambda x:None,sleep,lambda:clock[0],1)
        self.assertTrue(result['released'])
        with self.assertRaisesRegex(RuntimeError,'known owned/CUDA release verified'):
            await_release(lambda:dict(released=False,known_identities_absent_or_reused=True,own_cuda_clear=True,only_independent_opaque_blockers=True),lambda x:None,sleep,lambda:clock[0],0)
    def test_import_workers_cpu_only(self):
        import m3exec.worker
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
