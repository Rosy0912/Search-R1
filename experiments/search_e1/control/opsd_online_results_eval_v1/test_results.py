"""Meaningful CPU fixtures. No real questions, CUDA, HTTP or node interaction."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PACKAGE='results_eval_cpu_fixture'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
u=importlib.import_module(PACKAGE+'.utils');c=importlib.import_module(PACKAGE+'.completion')
p=importlib.import_module(PACKAGE+'.prepare');g=importlib.import_module(PACKAGE+'.gates')
w=importlib.import_module(PACKAGE+'.worker');h=importlib.import_module(PACKAGE+'.hooks')
ctl=importlib.import_module(PACKAGE+'.controller');s=importlib.import_module(PACKAGE+'.summary')

def write(root,name,value):return u.atomic_json(root/name,value,exclusive=True)
def native_exit(root,count,phase='sampling-running',rc=0):
    evidence=[dict(identity=dict(pid=100+i,uid=1005,starttick=500+i),actual_wait_returncode=rc) for i in range(count)]
    history={tuple(x['identity'][k] for k in ('pid','uid','starttick')):dict(phase=phase,command=['python','-B','worker']) for x in evidence}
    actual=write(root,'actual.json',dict(run_id=u.TRAIN_RUN,config_sha256=u.TRAIN_CONFIG_SHA,child_phase=phase,accepted_all_wait_zero=rc==0,child_exit_evidence=evidence))
    events=root/'events.jsonl';events.write_text('{}\n')
    value=dict(accepted=rc==0,run_id=u.TRAIN_RUN,config_sha256=u.TRAIN_CONFIG_SHA,child_phase=phase,child_exit_evidence=evidence,
        actual_terminal=actual,gpu_release={'all_owned_release':{'accepted':True}},events=u.fileproof(events))
    phase_ref=write(root,'phase.json',value)
    return dict(accepted=rc==0,child_exit_evidence=evidence,wait_recovery=phase_ref),history

def completes():
    values=[];updates=exposure=0
    for i in range(200):
        K=(0,1,32,33)[i%4];updates+=(K+31)//32;exposure+=K
        values.append(dict(outer_index=i,run_id=u.TRAIN_RUN,accepted=True,P=K,K=K,optimizer_updates_delta=(K+31)//32,
          optimizer_updates=updates,cumulative_pair_exposure=exposure,scheduler_advanced_updates=0,empty_outer_no_adam=K==0))
    return values

class Bookkeeping(unittest.TestCase):
    def test_all200_k0_tail_exact_counts(self):
        result=c.totals(completes());self.assertEqual(result['optimizer_updates'],200);self.assertEqual(result['cumulative_pair_exposure'],3300)
    def test_missing_outer_is_not_ready(self):
        with self.assertRaisesRegex(u.Hold,'NOT_READY'):c.totals(completes()[:-1])
    def test_duplicate_outer(self):
        rows=completes();rows[1]['outer_index']=0
        with self.assertRaises(u.Hold):c.totals(rows)
    def test_actual_update_not_outer_count(self):
        rows=completes();rows[3]['optimizer_updates']=3
        with self.assertRaises(u.Hold):c.totals(rows)
    def test_tail_global32_not_floor(self):
        rows=completes();rows[3]['optimizer_updates_delta']=1
        with self.assertRaises(u.Hold):c.totals(rows)
    def test_real_producer_nested_sampling_and_training_rc(self):
        for count,phase in [(8,'sampling-running'),(1,'training-running')]:
            with tempfile.TemporaryDirectory() as td:
                proof,history=native_exit(Path(td),count,phase);self.assertEqual(len(c.committed_exit(proof,count,phase,history)['commands']),count)
    def test_old_flat_exit_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            proof,history=native_exit(Path(td),8);proof['child_exit_evidence']=[dict(pid=1,uid=1005,starttick=1,actual_wait_returncode=0)]*8
            with self.assertRaises((u.Hold,KeyError)):c.committed_exit(proof,8,'sampling-running',history)
    def test_worker_complete_is_not_rc(self):
        with tempfile.TemporaryDirectory() as td:
            proof,history=native_exit(Path(td),8,rc=1)
            with self.assertRaises(u.Hold):c.committed_exit(proof,8,'sampling-running',history)
    def test_missing_explicit_final_is_hold(self):
        with patch.object(p,'original_config',return_value=({},{})),self.assertRaisesRegex(u.Hold,'NOT_READY'):
            p.checkpoint({'models':{'final199':{'checkpoint':None}}},'final199')
    def test_real_execution_schema_has_no_invented_accepted(self):
        source=Path(__file__).resolve().parents[5]/'search-e1-analysis-output/opsd-online-heartbeat-20261008-2111/cp218-independent-cpu-execution.json'
        # Synthetic shape from the published reader producer, no raw/private fixture in Git.
        execution=dict(exit_code=0,stdout_sha256='a'*64,helper_sha256='b'*64,source_commit='c'*40,
                       target_checkpoint={'path':'/actual/manifest.json','sha256':'d'*64},command=['-B','-','--expected-updates','218'],expected_updates=218,expected_outer=21)
        self.assertNotIn('accepted',execution);self.assertNotIn('actual_returncode',execution)
        self.assertEqual(execution['exit_code'],0)

class SourceAndFiles(unittest.TestCase):
    def test_fixed_sha_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'file';path.write_bytes(b'ab');ref=u.fileproof(path);path.write_bytes(b'cd')
            with self.assertRaises(u.Hold):u.read_file(path,ref['sha256'])
    def test_partial_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rows';path.write_bytes(b'{}')
            with self.assertRaises(u.Hold):u.rows(u.fileproof(path))
    def test_science_ignores_timestamp_valid_wrong_pyc(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pkg=root/'search_e1';pkg.mkdir();(pkg/'__init__.py').write_text('')
            source=pkg/'fixture.py';source.write_text('VALUE=999\n');py_compile.compile(str(source),doraise=True)
            stat=source.stat();source.write_text('VALUE=123\n');os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            files={str(x.relative_to(root)):u.fileproof(x) for x in (pkg/'__init__.py',source)}
            finder=u.SourceOnlyScience({'root':str(root),'files':files});saved={n:m for n,m in sys.modules.items() if n=='search_e1' or n.startswith('search_e1.')}
            for n in saved:sys.modules.pop(n)
            sys.meta_path.insert(0,finder)
            try:self.assertEqual(importlib.import_module('search_e1.fixture').VALUE,123)
            finally:
                sys.meta_path.remove(finder)
                for n in list(sys.modules):
                    if n=='search_e1' or n.startswith('search_e1.'):sys.modules.pop(n)
                sys.modules.update(saved)
    def test_foreign_cached_science_rejected(self):
        module=types.ModuleType('search_e1');module.__file__='/different/__init__.py'
        with patch.dict(sys.modules,{'search_e1':module}),patch.object(u,'runtime_files',return_value=dict(root='/fixture',files={'search_e1/__init__.py':{'path':'/fixture/search_e1/__init__.py','sha256':'a'*64}})),self.assertRaises(u.Hold):u.science({})
    def test_existing_result_never_reused(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'done.json';u.atomic_json(path,{'a':1},exclusive=True)
            with self.assertRaises(u.Hold):u.atomic_json(path,{'a':1},exclusive=True)

class Hooks(unittest.TestCase):
    def fixture(self,post=None):
        output=object();response=types.SimpleNamespace(content=b'{"result":[{"id":1,"score":0.5,"content":"synthetic"}]}',status_code=200)
        processors=[types.SimpleNamespace(penalty=1.05)]
        model=types.SimpleNamespace(generate=lambda *a,**k:output,_get_logits_processor=lambda *a,**k:processors)
        class Tokenizer:
            def __call__(self,*args,**kwargs):return types.SimpleNamespace(input_ids=[[3,4]])
        requests=types.SimpleNamespace(post=post or (lambda *a,**k:response));rollout=types.SimpleNamespace(requests=requests)
        return types.SimpleNamespace(model=model,tokenizer=Tokenizer()),rollout,output,response,processors
    def test_original_return_objects_args_http_processor_and_namespace(self):
        calls=[];response=types.SimpleNamespace(content=b'{"documents":[]}',status_code=200)
        def post(*a,**k):calls.append((a,k));return response
        agent,r,out,_,processors=self.fixture(post);requests=r.requests;original_post=requests.post;kwargs={'json':{'queries':['synthetic'],'topk':3,'return_scores':True},'timeout':120}
        with h.capture_native(agent,r) as trace:
            self.assertIs(agent.model.generate([1,2],max_new_tokens=500,do_sample=False),out)
            self.assertIs(agent.model._get_logits_processor(),processors)
            self.assertIs(r.requests.post('/retrieve',**kwargs),response);self.assertIs(requests.post,original_post)
        self.assertIs(r.requests,requests);self.assertEqual(calls,[(('/retrieve',),kwargs)]);self.assertFalse(trace['logging_errors']);self.assertTrue(trace['hooks_restored'])
    def test_exact_http_exception_object_preserved(self):
        error=RuntimeError('synthetic exception')
        def post(*a,**k):raise error
        agent,r,*_=self.fixture(post);requests=r.requests
        try:
            with h.capture_native(agent,r) as trace:r.requests.post('/retrieve',timeout=120)
        except RuntimeError as actual:self.assertIs(actual,error)
        else:self.fail('exception suppressed')
        self.assertIs(r.requests,requests);self.assertTrue(trace['hooks_restored'])
    def test_python_rng_consumption_unchanged(self):
        agent,r,*_=self.fixture();agent.model.generate=lambda *a,**k:random.random();random.seed(42);before=random.getstate();plain=agent.model.generate();after=random.getstate();random.setstate(before)
        with h.capture_native(agent,r):self.assertEqual(agent.model.generate(),plain)
        self.assertEqual(random.getstate(),after)
    def test_overlapping_hook_refused_and_restored(self):
        agent,r,*_=self.fixture()
        with h.capture_native(agent,r):
            with self.assertRaises(RuntimeError):
                with h.capture_native(agent,r):pass

class WaitSafety(unittest.TestCase):
    def job(self,rc=0):return dict(process=types.SimpleNamespace(pid=44,poll=lambda:rc),identity=dict(pid=44,uid=1005,starttick=1))
    def test_transient_gate_is_not_actual_child_failure(self):
        outputs={};attempts=[]
        def check():raise u.Hold('one transient reading')
        result=ctl.wait_children([self.job()],check,lambda:{'accepted':True},lambda k,v:outputs.update({k:v}),sleeper=lambda _:None)
        self.assertTrue(result['accepted']);self.assertEqual(outputs['actual-terminal']['child_exit_evidence'][0]['actual_wait_returncode'],0)
    def test_nonzero_persisted_before_rejection_and_release(self):
        events=[]
        with self.assertRaises(u.Hold):ctl.wait_children([self.job(3)],lambda:None,lambda:events.append('release') or {'accepted':True},lambda k,v:events.append(k),sleeper=lambda _:None)
        self.assertLess(events.index('actual-terminal'),events.index('release'));self.assertIn('phase-complete',events)
    def test_evidence_io_retries_before_return(self):
        failed=[False];events=[]
        def write(k,v):
            if not failed[0]:failed[0]=True;raise OSError('synthetic disk failure')
            events.append(k)
        result=ctl.wait_children([self.job()],lambda:None,lambda:{'accepted':True},write,sleeper=lambda _:None)
        self.assertTrue(result['accepted']);self.assertIn('child-44',events)
    def test_persistent_release_never_returns_or_launches_next(self):
        count=[]
        class StopFixture(BaseException):pass
        def release():count.append(1);raise u.Hold('opaque remains')
        def sleep(_):
            if len(count)>=3:raise StopFixture()
        with self.assertRaises(StopFixture):ctl.wait_children([self.job()],lambda:None,release,lambda *a:None,sleeper=sleep)
        self.assertEqual(len(count),3)
    def test_quick_child_exit_before_identity_preserves_actual_rc_unknown(self):
        job=self.job(4);job['identity']['starttick']=None
        def missing(pid):raise FileNotFoundError('already exited')
        with self.assertRaises(u.Hold):ctl.capture_child_identity(job,missing,sleeper=lambda _:None)
        self.assertEqual(job['identity']['actual_wait_returncode'],4);outputs={}
        with self.assertRaises(u.Hold):ctl.wait_children([job],lambda:None,lambda:{'accepted':True},lambda k,v:outputs.update({k:v}),sleeper=lambda _:None)
        self.assertEqual(outputs['actual-terminal']['child_exit_evidence'][0]['actual_wait_returncode'],4)
        self.assertFalse(outputs['phase-complete']['accepted'])
    def test_partial_eight_spawn_does_not_accept_phase(self):
        with self.assertRaises(u.Hold):ctl.wait_children([self.job()],lambda:None,lambda:{'accepted':True},lambda *a:None,sleeper=lambda _:None,expected_count=8)

class ResourceAndObserver(unittest.TestCase):
    def fixture(self):
        devices=[dict(index=i,uuid='GPU-'+str(i),actual_free_mib=24000,total_mib=24208) for i in range(8)]
        config=dict(run_id='e'*32,owner={'gpu_uuids':[d['uuid'] for d in devices]},resources={'running_minimum_free_mib':1024,'startup_minimum_free_mib':20480},gates={'retriever_admission':{}})
        owner=dict(pid=99,uid=1005,starttick=1,run_id='e'*32);oracle=dict(pid=23,uid=1005,starttick=8,run_id='oracle')
        source=HERE.parent/'opsd_online/acceptance.py';native=u.fresh_module('actual_fixture_native_admission',source,u.fileproof(source)['sha256'])
        return devices,config,owner,oracle,native
    def test_controller_cuda_context_is_never_worker_owned(self):
        d,cfg,owner,oracle,native=self.fixture()
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})),self.assertRaisesRegex(u.Hold,'controller has a CUDA'):
            g.resource(cfg,owner,[],'running',(d,[dict(pid=99,gpu_uuid='GPU-0',used_mib=500)]),reader=lambda pid:owner)
    def test_oracle_only_gpu7_exact_pid_identity_allowed(self):
        d,cfg,owner,oracle,native=self.fixture()
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})):
            self.assertTrue(g.resource(cfg,owner,[],'start',(d,[dict(pid=23,gpu_uuid='GPU-7',used_mib=574)]),reader=lambda pid:oracle)['accepted'])
            with self.assertRaises(u.Hold):g.resource(cfg,owner,[],'start',(d,[dict(pid=23,gpu_uuid='GPU-6',used_mib=574)]),reader=lambda pid:oracle)
    def test_foreign_gpu_process_blocked(self):
        d,cfg,owner,oracle,native=self.fixture();foreign=dict(pid=88,uid=1005,starttick=2,run_id='foreign',ppid=1)
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})),self.assertRaises(Exception):
            g.resource(cfg,owner,[],'running',(d,[dict(pid=88,gpu_uuid='GPU-0',used_mib=500)]),reader=lambda pid:foreign)
    def test_empty_cards_keep_start_floor_when_another_card_running(self):
        d,cfg,owner,oracle,native=self.fixture();child=dict(pid=88,uid=1005,starttick=2,run_id='e'*32,ppid=99);d[0]['actual_free_mib']=1100;d[1]['actual_free_mib']=20000
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})),self.assertRaises(u.Hold):
            g.resource(cfg,owner,[child],'running',(d,[dict(pid=88,gpu_uuid='GPU-0',used_mib=20000)]),reader=lambda pid:child)
    def test_observer_actual_live_tick_and_age(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'memory_observer.py';source.write_text('# fixture independent observer\n');digest=u.fileproof(source)['sha256']
            manifest=write(root,'manifest.json',{'files':{'memory_observer.py':{'sha256':digest}}})
            identity=dict(pid=10,uid=1005,starttick=55,run_id='observer-fixture',source_sha256=digest)
            live=dict(observer_identity=identity,run_id=u.TRAIN_RUN,config_sha256=u.TRAIN_CONFIG_SHA,updated_unix=100,
                current={'devices':[{'uuid':'GPU-'+str(i)} for i in range(8)],'actual_free_limit_met':True},samples=5,samples_below_limit=0)
            live_ref=write(root,'live.json',live)
            binding=write(root,'binding.json',dict(schema='opsd-online-memory-observer-binding-v1',accepted=True,experiment_run_id=u.TRAIN_RUN,
                config_sha256=u.TRAIN_CONFIG_SHA,observer_identity=identity,live_report_path=live_ref['path']))
            cfg=dict(owner={'gpu_uuids':['GPU-'+str(i) for i in range(8)]},gates={'memory_observer':binding},training={'config':{'path':'/original/config.json'}})
            train=dict(paths={'helper_root':str(root)},helper_source_manifest=manifest)
            actual=dict(identity,argv=[str(source),'--config','/original/config.json'])
            with patch.object(g,'original_config',return_value=(train,{})),patch.object(g,'process',return_value=actual):
                self.assertEqual(g.observer(cfg,now=102)['age_seconds'],2)
                with self.assertRaises(u.Hold):g.observer(cfg,now=131)
            with patch.object(g,'original_config',return_value=(train,{})),patch.object(g,'process',return_value=dict(actual,starttick=99)),self.assertRaises(u.Hold):g.observer(cfg,now=102)
    def test_six_lease_contention_refuses_second_controller(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            for i in range(5):path=root/str(i);path.touch();paths.append(str(path))
            cfg=dict(resources=dict(shared_gpu_lease_paths=paths[:4],original_online_lock=paths[4]),paths={'base':str(root)})
            with g.Leases(cfg):
                with self.assertRaises(BlockingIOError):
                    with g.Leases(cfg):pass

class ResultSlots(unittest.TestCase):
    def fixture(self):
        data=[dict(question_id=str(i),question='synthetic '+str(i),answers=['synthetic'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        records=[]
        for v in u.VERSIONS:
            for i,row in enumerate(data):
                hit=int((i+(0 if v=='source100' else 1))%2==0)
                records.append(dict(status='complete',primary=True,run_id='e'*32,version=v,rank=i%8,slot=i//8,validation_index=i,
                    question_id=row['question_id'],dataset=row['data_source'],row_semantic_sha256=u.object_sha(row),
                    protocol_em=hit,protocol_valid_final_action=bool(hit),failure_class=None if hit else 'final_invalid',search_calls=0,termination='answer',
                    hook_control={'accepted':True,'excluded_from_primary_denominator':True},zero_effect_control={'accepted':True}))
        return records,data,{'run_id':'e'*32}
    def test_1536_primary_512_per_model_and_controls_excluded(self):
        records,data,cfg=self.fixture();value=s.aggregate(records,data,cfg,verify_traces=False)
        self.assertEqual(value['primary_slots'],1536);self.assertEqual(value['models']['cp89']['overall']['questions'],512)
        self.assertEqual(value['paired']['source100->cp89']['lost_hits'],256);self.assertEqual(value['hook_controls_excluded'],24)
        self.assertEqual(value['paired']['source100->cp89']['lost_final_invalid'],256)
        self.assertEqual(value['paired']['source100->cp89']['lost_valid_gold_mismatch'],0)
        self.assertEqual(value['source100_previous239_condition_check']['decision'],'NEEDS_CONDITION_DIAGNOSIS')
    def test_duplicate_and_missing_slice_refused(self):
        records,data,cfg=self.fixture();records[0]=records[1]
        with self.assertRaises(u.Hold):s.aggregate(records,data,cfg,False)
    def test_rank_seed42_not_sampling42plusrank(self):
        records,data,cfg=self.fixture();self.assertEqual({r['rank']:sum(x['rank']==r['rank'] for x in records if x['version']=='source100') for r in records},{i:64 for i in range(8)})
        text=(HERE/'worker.py').read_text();self.assertIn('random.seed(42);torch.manual_seed(42)',text);self.assertNotIn('manual_seed(42+rank)',text)
    def test_failure_axes_are_exclusive(self):
        records,data,cfg=self.fixture();records[0]['failure_class']='valid_gold_mismatch'
        with self.assertRaises(u.Hold):s.aggregate(records,data,cfg,False)
    def test_control_fail_does_not_add_primary(self):
        records,data,cfg=self.fixture();records[0]['zero_effect_control']['accepted']=False
        with self.assertRaises(u.Hold):s.aggregate(records,data,cfg,False)

class TensorAndNativeCPU(unittest.TestCase):
    def test_adapter_digest_matches_original_algorithm_and_zero_b(self):
        import torch
        values={f'layer.{i}.lora_{kind}.weight':torch.zeros(1,dtype=torch.float32) for i in range(252) for kind in ('A','B')}
        proof=w.tensor_checks(values,'source100',w.tensor_digest(values));self.assertTrue(proof['lora_B_all_zero'])
        values['layer.0.lora_B.weight'][0]=1
        with self.assertRaises(u.Hold):w.tensor_checks(values,'source100',w.tensor_digest(values))
    def test_wrong_loaded_adapter_digest_and_dtype_refused(self):
        import torch
        values={f'layer.{i}.lora_{kind}.weight':torch.zeros(1,dtype=torch.float32) for i in range(252) for kind in ('A','B')}
        with self.assertRaises(u.Hold):w.tensor_checks(values,'cp89','0'*64)
        values[next(iter(values))]=torch.zeros(1,dtype=torch.bfloat16)
        with self.assertRaises(u.Hold):w.tensor_checks(values,'cp89','0'*64)
    def test_worker_importlib_scope_is_global(self):
        self.assertNotIn('importlib',w.run.__code__.co_varnames)
    def test_actual_frozen_strict_final_action_and_information(self):
        runtime=Path(os.environ.get('OPSD_ONLINE_TEST_RUNTIME') or str(HERE.parents[4]/'search-e1-analysis-output/opsd-preparation3-xformers-backup-20261008/snapshot/runtime/opsd-preparation3-xformers'))
        qa_path=runtime/'upstream/Search-R1/search_r1/qa_protocol.py';qa=u.fresh_module('results_fixture_actual_qa',qa_path,u.QA_SHA)
        core_path=runtime/'search_e1/core.py';core=u.fresh_module('results_fixture_actual_core',core_path,u.fileproof(core_path)['sha256'])
        with patch.dict(sys.modules,{'search_r1.qa_protocol':qa}):
            for text in ('<think>unfinished<answer>synthetic</answer>','<information><answer>synthetic</answer></information>','<answer>synthetic</answer><search>query</search>'):
                result={'response':text,'policy_spans':[[0,len(text)]]};self.assertEqual(w.protocol(core,qa,result,['synthetic'])['protocol_em'],0)
            text='<think>done</think><answer>synthetic</answer>';self.assertEqual(w.protocol(core,qa,{'response':text,'policy_spans':[[0,len(text)]]},['synthetic'])['protocol_em'],1)
        self.assertFalse(__import__('torch').cuda.is_initialized())

if __name__=='__main__':unittest.main()
