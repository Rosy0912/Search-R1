"""Necessary CPU contracts only; synthetic fixtures, no models/CUDA/HTTP/node."""
import copy
import importlib
import os
from pathlib import Path
import py_compile
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;PACKAGE='ablation_eval_cpu_fixture'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
u=importlib.import_module(PACKAGE+'.utils');p=importlib.import_module(PACKAGE+'.prepare')
c=importlib.import_module(PACKAGE+'.completion');g=importlib.import_module(PACKAGE+'.gates')
h=importlib.import_module(PACKAGE+'.hooks');w=importlib.import_module(PACKAGE+'.worker')
ctl=importlib.import_module(PACKAGE+'.controller');s=importlib.import_module(PACKAGE+'.summary');merge=importlib.import_module(PACKAGE+'.cpu_merge')
driver=importlib.import_module(PACKAGE+'.study_runner')


def completes():
    values=[];updates=exposure=0
    for i in range(10):
        K=(0,1,32,33)[i%4];delta=(K+31)//32;updates+=delta;exposure+=K
        values.append(dict(outer_index=i,run_id='r'*32,accepted=True,P=K,K=K,optimizer_updates_delta=delta,
            optimizer_updates=updates,cumulative_pair_exposure=exposure,scheduler_advanced_updates=0,empty_outer_no_adam=K==0))
    return values


def data_records(versions=('raw10_bf16',)):
    data=[dict(question_id=f'q{i}',question=f'Synthetic item {i}',answers=['synthetic'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
    records=[]
    for version in versions:
        for i,row in enumerate(data):
            score=i%2
            records.append(dict(primary=True,status='complete',run_id='e'*32,version=version,validation_index=i,rank=i%8,slot=i//8,
                question_id=row['question_id'],dataset=row['data_source'],row_semantic_sha256=u.object_sha(row),protocol_em=score,
                protocol_valid_final_action=True,failure_class=None if score else 'valid_gold_mismatch',search_calls=0,termination='answer',
                hook_control=dict(accepted=True,excluded_from_primary_denominator=True),zero_effect_control=dict(accepted=True)))
    config=dict(run_id='e'*32,study_id='study',variant='C0',evaluation_role='group',scientific=dict(versions=list(versions)))
    return data,records,config


class Bookkeeping(unittest.TestCase):
    def test_study_final_binding_only_evidence_no_early_or_overwritten_inputs(self):
        template=dict(training=dict(run_id='r'*32,config={'path':'config','sha256':'a'*64},terminal=None,raw10_checkpoint=None),
            scientific={'temperature':0},state='NOT_READY')
        status=dict(phase='TRAIN_COMPLETE',outer_iterations=10,run_id='r'*32,config_sha256='a'*64,latest_checkpoint={'path':'raw10','sha256':'b'*64})
        final=driver.bind_final_evaluation(template,{'path':'terminal','sha256':'c'*64},status)
        self.assertEqual(final['scientific'],template['scientific']);self.assertIsNone(template['training']['terminal'])
        self.assertEqual(final['training']['raw10_checkpoint'],status['latest_checkpoint'])
        for bad in (dict(status,phase='TRAINING'),dict(status,outer_iterations=9),dict(status,run_id='wrong')):
            with self.assertRaises(u.Hold):driver.bind_final_evaluation(template,{},bad)
        with self.assertRaises(u.Hold):driver.bind_final_evaluation(final,{},status)
        with tempfile.TemporaryDirectory() as td:
            binding=u.atomic_json(Path(td)/'binding.json',{'accepted':True},exclusive=True)
            planned=copy.deepcopy(template);planned['gates']={'memory_observer':dict(path=binding['path'],sha256=None)}
            bound=driver.bind_final_evaluation(planned,{},status)
            self.assertEqual(bound['gates']['memory_observer'],binding)
            self.assertIsNone(planned['gates']['memory_observer']['sha256'])
    def test_autonomous_observer_bootstrap_exact_roles_and_no_restart(self):
        # No process is started: a Popen-shaped CPU double proves the driver-only
        # -> actual observer binding/sample -> exact two-role order and reuse.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);train_source=root/'train-source';train_source.mkdir()
            (train_source/'memory_observer.py').write_text('# synthetic published observer\n')
            receipts=root/'group-receipts';receipts.mkdir()
            config_ref={'path':str(root/'group-config.json'),'sha256':'a'*64}
            train=dict(run_id='c'*32,variant='C0',paths={'receipts':str(receipts),'helper_root':str(train_source),'python':'python'},
                gates={'scoped_cpu_processes_path':str(receipts/'scoped-cpu-processes.json'),
                       'memory_binding_path':str(receipts/'binding.json'),'publication':{'path':'train-pub','sha256':'b'*64}},
                helper_source_manifest={'path':'train-manifest','sha256':'d'*64})
            value=dict(output_root=str(root/'driver-output'),helper_root=str(HERE),
                helper_source_manifest={'path':'eval-manifest','sha256':'e'*64},publication={'path':'eval-pub','sha256':'f'*64})
            instance=driver.Driver(value,{'path':'study-manifest','sha256':'1'*64})
            instance.owner=dict(pid=1,uid=1005,starttick=10,ppid=0,pgid=1,argv=['python','-B',str(HERE/'study_runner.py')],run_id='s'*32)
            actual=dict(pid=2,uid=1005,starttick=11,ppid=1,pgid=2,
                argv=['python','-B',str(train_source/'memory_observer.py'),'--config',config_ref['path']],run_id='observer-'+train['run_id'])
            events=[]
            def roles(_):
                body,_=u.checked_json(u.fileproof(receipts/'scoped-cpu-processes.json'))
                for role in body['roles']:
                    self.assertEqual(role['argv_sha256'],u.object_sha(role['identity']['argv']))
                return dict(accepted=True,identities=[r['identity'] for r in body['roles']])
            def bootstrap(*args,**kwargs):
                events.append('bootstrap');self.assertEqual([r['pid'] for r in roles(train)['identities']],[1])
            def spawn(command,run_id,python,scope,reader):
                events.append('spawn');self.assertEqual(run_id,actual['run_id']);self.assertIn('--binding-output',command)
                scope.mkdir(parents=True);handle=(scope/'foreground.log').open('w')
                u.atomic_json(receipts/'binding.json',dict(observer_identity=actual),exclusive=True)
                return dict(identity=actual,process=types.SimpleNamespace(poll=lambda:None),handle=handle)
            admit=types.SimpleNamespace(require_launch_without_memory=bootstrap,process=lambda _:actual,
                same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')),
                memory_observer_admission=lambda _:dict(observer_identity=actual,current={'sample_unix':1}),scoped_cpu_processes=roles)
            with patch.object(instance,'spawn',side_effect=spawn):
                instance.observer(train,config_ref,{},0,admit)
            self.assertEqual(events,['bootstrap','spawn']);self.assertEqual(len(roles(train)['identities']),2)
            for job in instance.observers:job['handle'].close()
            # A second independent fixture scope reuses live binding, never Popen.
            original_bootstrap=admit.require_launch_without_memory
            admit.require_launch_without_memory=lambda *a,**k:None
            with patch.object(instance,'spawn',side_effect=AssertionError('Must not restart bound observer')):
                instance.observer(train,config_ref,{},1,admit)
            admit.require_launch_without_memory=original_bootstrap
    def test_ten_is_not_adam_ten_including_tail_empty(self):
        result=c.totals(completes(),'r'*32)
        self.assertEqual((result['outer_iterations'],result['optimizer_updates'],result['cumulative_pair_exposure']),(10,9,133))
    def test_no_early_or_duplicate_outer_acceptance(self):
        for values in (completes()[:-1],completes()+completes()[:1]):
            with self.assertRaises(u.Hold):c.totals(values,'r'*32)
        values=completes();values[3]['outer_index']=2
        with self.assertRaises(u.Hold):c.totals(values,'r'*32)
    def test_tail_not_floor_and_foreign_run_blocked(self):
        values=completes();values[3]['optimizer_updates_delta']=1
        with self.assertRaises(u.Hold):c.totals(values,'r'*32)
        with self.assertRaises(u.Hold):c.totals(completes(),'wrong')
    def test_seven_group_plus_common_zeros_exact5632(self):
        total=sum(p.scientific(v,'group')['total_primary_slots'] for v in u.VARIANTS)+p.scientific('C0','common_zero')['total_primary_slots']
        self.assertEqual(total,5632)
        self.assertEqual(p.expected_versions('P','group'),['raw10_bf16','raw10_fp16'])
        self.assertEqual(p.expected_versions('R','group'),['raw10_bf16'])
    def test_actual_producer_ten_complete_schema_and_native_miner_binding(self):
        # File-level fixture follows the current new producer: TRAIN_COMPLETE,
        # nested actual phase terminals, rank320 rows, and raw outer9 cursor.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run='r'*32;sha='a'*64;contract={'direction':'student_to_target_reverse'}
            def save(path,value):return u.atomic_json(root/path,value,exclusive=True)
            def save_rows(path,value):
                q=root/path;q.parent.mkdir(parents=True,exist_ok=True)
                q.write_bytes(b''.join(u.canonical(x)+b'\n' for x in value));return u.fileproof(q)
            train=dict(run_id=run,variant='C0',precision={'mode':'bf16'},paths={'output_root':str(root)},
                runtime_manifest={'sha256':'b'*64},source_model_manifest={'sha256':'c'*64},
                common_initial_adapter_manifest={'path':'initial','sha256':'d'*64},train_identity={},validation_identity={})
            shared={'cohorts':[]};previous=None;native_final=None
            for index in range(10):
                pre=f'outers/outer-{index:05d}';questions=[dict(question_id=f'q{index}-{j}',question='synthetic',answers=['synthetic']) for j in range(512)]
                shared['cohorts'].append({'questions':questions})
                qp=save_rows(pre+'/questions.jsonl',questions);workers=[];combined=[]
                identities=[dict(identity=dict(pid=100+r,uid=1005,starttick=1000+r),actual_wait_returncode=0) for r in range(8)]
                for rank in range(8):
                    sampled=[dict(question_id=x['question_id'],sibling=k) for x in questions[rank::8] for k in range(5)]
                    combined.extend(sampled);rp=save_rows(pre+f'/sampling-attempt-00000/rollouts-{rank}.jsonl',sampled)
                    workers.append(save(pre+f'/worker-{rank}.json',dict(accepted=True,run_id=run,config_sha256=sha,
                        variant='C0',precision={'mode':'bf16'},rows=320,rank=rank,rollouts=rp)))
                rp=save_rows(pre+'/rollouts.jsonl',combined);pairp=save_rows(pre+'/pairs.jsonl',[{'synthetic_pair':index}]);stats={'P':1}
                sampling_terminal=save(pre+'/sampling-terminal.json',dict(run_id=run,config_sha256=sha,phase='sampling-running',child_exit_evidence=identities))
                save(pre+'/sampling-complete.json',dict(accepted=True,run_id=run,workers=workers,rollouts=rp,pairs=pairp,stats=stats,
                    sampling_model_identity=dict(checkpoint=previous,source_model_identity='c'*64,optimizer_updates=index),
                    exit_proof=dict(accepted=True,child_exit_evidence=identities,actual_terminal=sampling_terminal)))
                encrows=[{'native_sha':'synthetic'}];ep=save(pre+'/encoding.json',dict(rows=encrows,sha256=u.object_sha(encrows),P=1,K=1))
                op=save(pre+'/order.json',dict(pair_indices=[0],encoded_sha256=u.object_sha(encrows)))
                native_final=dict(contract=contract,common_initial_adapter=train['common_initial_adapter_manifest'],run_id=run,config_sha256=sha,
                    optimizer_updates=index+1,cumulative_pair_exposure=index+1,cursor=dict(outer_index=index,next_pair_cursor=1,usable_pair_count=1,
                    encoded_sha256=u.object_sha(encrows),order_sha256=op['sha256']),training_input=dict(encoding=ep,order=op))
                cp=save(pre+'/checkpoint.json',native_final)
                save(pre+'/outer-manifest.json',dict(run_id=run,config_sha256=sha,outer_index=index,runtime_identity='b'*64,source_model_identity='c'*64,
                    previous_complete_checkpoint=previous,P=1,questions=qp,rollouts=rp,pairs=pairp,sampling_stats=stats))
                save(pre+'/training/outer-complete.json',dict(accepted=True,run_id=run,outer_index=index,P=1,K=1,optimizer_updates_delta=1,
                    optimizer_updates=index+1,cumulative_pair_exposure=index+1,scheduler_advanced_updates=0,empty_outer_no_adam=False,contract=contract,checkpoint=cp))
                evidence=[dict(identity=dict(pid=500+index,uid=1005,starttick=5000+index),actual_wait_returncode=0)]
                tp=save(pre+'/trainer-terminal.json',dict(run_id=run,config_sha256=sha,phase='training-running',child_exit_evidence=evidence))
                save(pre+'/trainer-logs/attempt-00000-exit.json',dict(accepted=True,child_exit_evidence=evidence,actual_terminal=tp));previous=cp
            train['common_cohort_manifest']=save('common-cohorts.json',shared)
            tp=save('terminal.json',dict(accepted=True,run_id=run,config_sha256=sha,actual_exit_code=0,controller=dict(pid=50,uid=1005,starttick=500)))
            status=dict(phase='TRAIN_COMPLETE',run_id=run,config_sha256=sha,outer_iterations=10,actual_optimizer_updates=10,latest_checkpoint=previous)
            u.atomic_json(root/'status.json',status)
            cfg=dict(training=dict(terminal=tp,raw10_checkpoint=previous),study_id='study',variant='C0')
            miner=types.SimpleNamespace(verify_rollouts=lambda rows,q,v,i,core:([{'synthetic_pair':i}],{'P':1}))
            modules=dict(prepare=types.SimpleNamespace(validate_config=lambda *a,**k:None),
                checkpoint=types.SimpleNamespace(science_contract=lambda _:contract,check_checkpoint=lambda *a:(native_final,{})),
                cohort_v1=types.SimpleNamespace(verify_from_files=lambda *a:{'accepted':True}),miner_v1=miner)
            with patch.object(c,'original_config',return_value=(train,{'sha256':sha})),patch.object(c,'frozen_helpers',return_value=(modules,{})),patch.object(c,'science',return_value=(object(),object(),{})):
                self.assertEqual(c.validate(cfg)['totals']['optimizer_updates'],10)
                bad=dict(status,phase='COMPLETE');u.atomic_json(root/'status.json',bad)
                with self.assertRaises(u.Hold):c.validate(cfg)
                u.atomic_json(root/'status.json',status);miner.verify_rollouts=lambda *args:([],{'P':0})
                with self.assertRaises(u.Hold):c.validate(cfg)


class SourceAndHooks(unittest.TestCase):
    def test_sequential_byte_identical_runtime_clones_cold_bind_not_foreign_cache(self):
        runtime=Path(os.environ.get('OPSD_ABLATION_TEST_RUNTIME') or str(HERE.parent/'opsd_debug_frozen_runtime_20261008/runtime'))
        names=['search_e1/__init__.py','search_e1/core.py','upstream/Search-R1/search_r1/__init__.py','upstream/Search-R1/search_r1/qa_protocol.py']
        saved={n:m for n,m in sys.modules.items() if n=='search_e1' or n.startswith('search_e1.') or n=='search_r1' or n.startswith('search_r1.')}
        old_meta=list(sys.meta_path);old_path=list(sys.path)
        for n in saved:sys.modules.pop(n)
        try:
            with tempfile.TemporaryDirectory() as td:
                for group in ('C0','C1'):
                    root=Path(td)/group;files={}
                    for relative in names:
                        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((runtime/relative).read_bytes());files[relative]=u.fileproof(path)
                    with patch.object(u,'runtime_files',return_value=dict(root=str(root),files=files)):
                        core,qa,_=u.science({})
                        self.assertEqual(Path(core.__file__).parent.parent,root.resolve())
                        self.assertEqual(qa.extract_final_answer(['<answer>synthetic</answer>']),'synthetic')
                core._opsd_results_source_sha256='foreign'
                with patch.object(u,'runtime_files',return_value=dict(root=str(root),files=files)),self.assertRaises(u.Hold):u.science({})
        finally:
            sys.meta_path[:]=old_meta;sys.path[:]=old_path
            for n in list(sys.modules):
                if n=='search_e1' or n.startswith('search_e1.') or n=='search_r1' or n.startswith('search_r1.'):sys.modules.pop(n)
            sys.modules.update(saved)
    def test_source_loader_ignores_timestamp_valid_wrong_pyc(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pkg=root/'search_e1';pkg.mkdir();(pkg/'__init__.py').write_text('')
            source=pkg/'fixture.py';source.write_text('VALUE=999\n');py_compile.compile(str(source),doraise=True)
            old=source.stat();source.write_text('VALUE=123\n');os.utime(source,ns=(old.st_atime_ns,old.st_mtime_ns))
            finder=u.SourceOnlyScience(dict(root=str(root),files={str(x.relative_to(root)):u.fileproof(x) for x in (pkg/'__init__.py',source)}))
            saved={n:m for n,m in sys.modules.items() if n=='search_e1' or n.startswith('search_e1.')}
            for n in saved:sys.modules.pop(n)
            sys.meta_path.insert(0,finder)
            try:self.assertEqual(importlib.import_module('search_e1.fixture').VALUE,123)
            finally:
                sys.meta_path.remove(finder)
                for n in list(sys.modules):
                    if n=='search_e1' or n.startswith('search_e1.'):sys.modules.pop(n)
                sys.modules.update(saved)
    def test_sha_mutation_and_partial_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rows';path.write_bytes(b'{}\n');ref=u.fileproof(path);path.write_bytes(b'{}')
            with self.assertRaises(u.Hold):u.rows(ref)
            with self.assertRaises(u.Hold):u.rows(u.fileproof(path))
    def test_native_return_kwargs_response_and_hooks_restored(self):
        output=object();response=types.SimpleNamespace(content=b'{"documents":[]}',status_code=200);calls=[];processors=[object()]
        def post(*args,**kwargs):calls.append((args,kwargs));return response
        model=types.SimpleNamespace(generate=lambda *a,**k:output,_get_logits_processor=lambda *a,**k:processors)
        tokenizer=types.SimpleNamespace();requests=types.SimpleNamespace(post=post);rollout=types.SimpleNamespace(requests=requests)
        agent=types.SimpleNamespace(model=model,tokenizer=tokenizer)
        with h.capture_native(agent,rollout) as trace:
            self.assertIs(model.generate([1],do_sample=False,max_new_tokens=500),output)
            self.assertIs(model._get_logits_processor(),processors)
            self.assertIs(rollout.requests.post('/retrieve',json={'queries':['synthetic']},timeout=120),response)
            self.assertIs(requests.post,post)
        self.assertIs(rollout.requests,requests);self.assertIs(agent.tokenizer,tokenizer);self.assertTrue(trace['hooks_restored'])
        self.assertEqual(calls,[ (('/retrieve',),dict(json={'queries':['synthetic']},timeout=120)) ])
    def test_hook_exception_and_rng_unchanged(self):
        error=RuntimeError('synthetic');request=types.SimpleNamespace(post=lambda *a,**k:(_ for _ in ()).throw(error))
        model=types.SimpleNamespace(generate=lambda *a,**k:random.random());agent=types.SimpleNamespace(model=model,tokenizer=object());rollout=types.SimpleNamespace(requests=request)
        random.seed(42);before=random.getstate();plain=model.generate();after=random.getstate();random.setstate(before)
        with h.capture_native(agent,rollout):self.assertEqual(model.generate(),plain)
        self.assertEqual(random.getstate(),after)
        try:
            with h.capture_native(agent,rollout) as trace:rollout.requests.post('/retrieve')
        except RuntimeError as actual:self.assertIs(actual,error)
        else:self.fail('Original exception suppressed')
        self.assertTrue(trace['hooks_restored']);self.assertIs(rollout.requests,request)
    def test_actual_frozen_qa_legal_final_and_information_exclusion(self):
        runtime=Path(os.environ.get('OPSD_ABLATION_TEST_RUNTIME') or str(HERE.parent/'opsd_debug_frozen_runtime_20261008/runtime'))
        qa=u.fresh_module('ablation_eval_actual_qa',runtime/'upstream/Search-R1/search_r1/qa_protocol.py',u.QA_SHA)
        core_path=runtime/'search_e1/core.py';core=u.fresh_module('ablation_eval_actual_core',core_path,u.fileproof(core_path)['sha256'])
        with patch.dict(sys.modules,{'search_r1.qa_protocol':qa}):
            for text in ('<think>unfinished<answer>synthetic</answer>','<information><answer>synthetic</answer></information>','<answer>synthetic</answer><search>query</search>'):
                actual=w.protocol(core,qa,dict(response=text,policy_spans=[[0,len(text)]]),['synthetic'])
                self.assertEqual((actual['protocol_em'],actual['failure_class']),(0,'final_invalid'))
            text='<think>done</think><answer>synthetic</answer>'
            self.assertEqual(w.protocol(core,qa,dict(response=text,policy_spans=[[0,len(text)]]),['synthetic'])['protocol_em'],1)
            text='<answer>synthetic</answer><information>external</information><answer>wrong</answer>'
            first=len('<answer>synthetic</answer>');last=text.index('<answer>wrong')
            self.assertEqual(w.protocol(core,qa,dict(response=text,policy_spans=[[0,first],[last,len(text)]]),['synthetic'])['failure_class'],'valid_gold_mismatch')


class WaitAndResource(unittest.TestCase):
    def job(self,rc=0):return dict(process=types.SimpleNamespace(pid=44,poll=lambda:rc),identity=dict(pid=44,uid=1005,starttick=1))
    def test_transient_admission_not_child_failure(self):
        events={}
        def check():raise u.Hold('live read uncertain')
        result=ctl.wait_children([self.job()],check,lambda:{'accepted':True},lambda k,v:events.update({k:v}),sleeper=lambda _:None)
        self.assertTrue(result['accepted']);self.assertEqual(events['actual-terminal']['child_exit_evidence'][0]['actual_wait_returncode'],0)
    def test_actual_nonzero_preserved_before_release(self):
        events=[]
        with self.assertRaises(u.Hold):ctl.wait_children([self.job(4)],lambda:None,lambda:events.append('release') or {'accepted':True},lambda k,v:events.append(k),sleeper=lambda _:None)
        self.assertLess(events.index('actual-terminal'),events.index('release'))
    def test_quick_exit_not_infinite_identity_retry_or_rc0(self):
        job=self.job(3);job['identity']['starttick']=None
        with self.assertRaises(u.Hold):ctl.capture_child_identity(job,lambda pid:(_ for _ in ()).throw(FileNotFoundError()),sleeper=lambda _:None)
        self.assertEqual(job['identity']['actual_wait_returncode'],3);self.assertEqual(job['identity']['capture_status'],'UNKNOWN_EXITED_BEFORE_FULL_IDENTITY')
    def test_partial_spawn_count_never_accepts_eight_phase(self):
        with self.assertRaises(u.Hold):ctl.wait_children([self.job()],lambda:None,lambda:{'accepted':True},lambda *a:None,sleeper=lambda _:None,expected_count=8)
    def test_read_error_retains_release_loop(self):
        attempts=[]
        class StopFixture(BaseException):pass
        def release():attempts.append(1);raise OSError('uncertain proc read')
        def sleep(_):
            if len(attempts)==3:raise StopFixture()
        with self.assertRaises(StopFixture):ctl.wait_children([self.job()],lambda:None,release,lambda *a:None,sleeper=sleep)
        self.assertEqual(len(attempts),3)
    def fixture(self):
        native_path=Path(os.environ.get('OPSD_ABLATION_TEST_TRAINING_HELPER') or str(HERE.parent/'opsd_ablation_v1'))/'gates.py'
        native=u.fresh_module('eval_native_resource_fixture',native_path,u.fileproof(native_path)['sha256'])
        devices=[dict(index=i,uuid=f'GPU-{i}',actual_free_mib=24000,total_mib=24576) for i in range(8)]
        config=dict(run_id='e'*32,owner=dict(gpu_uuids=[d['uuid'] for d in devices]),resources=dict(startup_minimum_free_mib=20480,running_minimum_free_mib=1024),gates={'retriever_admission':{}})
        owner=dict(pid=99,uid=1005,starttick=1);oracle=dict(pid=77,uid=1005,starttick=2,run_id='oracle',ppid=1)
        return devices,config,owner,oracle,native
    def test_parent_cuda_never_worker_owned(self):
        devices,config,owner,oracle,native=self.fixture()
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})),self.assertRaises(u.Hold):
            g.resource(config,owner,[],'running',inventory=(devices,[dict(pid=99,gpu_uuid='GPU-0',used_mib=500)]),reader=lambda _:owner)
    def test_unused_lowfree_foreign_context_and_oracle_identity_rejected(self):
        devices,config,owner,oracle,native=self.fixture()
        with patch.object(g,'native_admission',return_value=native),patch.object(g,'checked_json',return_value=(oracle,{})):
            devices[0]['actual_free_mib']=2000
            with self.assertRaises(u.Hold):g.resource(config,owner,[],'running',inventory=(devices,[]))
            devices[0]['actual_free_mib']=24000
            with self.assertRaises(Exception):g.resource(config,owner,[],'running',inventory=(devices,[dict(pid=8,gpu_uuid='GPU-0',used_mib=1000)]),reader=lambda _:dict(pid=8,uid=1005,starttick=1,run_id='foreign',ppid=1))
            with self.assertRaises(u.Hold):g.resource(config,owner,[],'released',inventory=(devices,[dict(pid=77,gpu_uuid='GPU-7',used_mib=1025)]),reader=lambda _:oracle)


class ScoringAndPrecision(unittest.TestCase):
    def test_complete_single512_and_controls_excluded(self):
        data,records,config=data_records();result=s.aggregate(records,data,config,verify_traces=False)
        self.assertEqual(result['primary_slots'],512);self.assertEqual(result['models']['raw10_bf16']['overall']['strict_hits'],256)
        self.assertEqual(result['excluded_extra_actual_generations']['hook_null_effect_controls'],8)
        self.assertEqual(result['models']['raw10_bf16']['datasets']['nq']['questions'],256)
    def test_duplicate_slot_and_rank_and_nonexclusive_failures_rejected(self):
        data,records,config=data_records()
        for mutation in ('duplicate','rank','failure','hook'):
            changed=copy.deepcopy(records)
            if mutation=='duplicate':changed[-1]=changed[0]
            elif mutation=='rank':changed[0]['rank']=1
            elif mutation=='failure':changed[0]['failure_class']='final_invalid'
            else:changed[0]['hook_control']['accepted']=False
            with self.assertRaises(u.Hold):s.aggregate(changed,data,config,verify_traces=False)
    def test_paired_lost_axes_and_question_bootstrap(self):
        _,baseline,_=data_records();after=copy.deepcopy(baseline)
        after[1].update(protocol_em=0,protocol_valid_final_action=False,failure_class='final_invalid')
        after[3].update(protocol_em=0,failure_class='valid_gold_mismatch')
        after[0].update(protocol_em=1,failure_class=None)
        result=s.paired(baseline,after,replicates=1000)
        self.assertEqual((result['lost_hits'],result['new_hits'],result['net_hits']),(2,1,-1))
        self.assertEqual((result['lost_final_invalid'],result['lost_valid_gold_mismatch']),(1,1))
        self.assertEqual(result['bootstrap']['question_units'],512)
        self.assertEqual(result['bootstrap'],s.paired(baseline,after,replicates=1000)['bootstrap'])
    def test_actual504_zero_dtype_digest_and_nonfinite(self):
        import torch
        values={f'layer.{i}.lora_{kind}.weight':torch.zeros(1,dtype=torch.float32) for i in range(252) for kind in ('A','B')}
        self.assertTrue(w.tensor_checks(values,'bf16_zero',w.tensor_digest(values))['lora_B_all_zero'])
        values['layer.0.lora_B.weight'][0]=1
        with self.assertRaises(u.Hold):w.tensor_checks(values,'fp16_zero',w.tensor_digest(values))
        with self.assertRaises(u.Hold):w.tensor_checks(values,'raw10_bf16','0'*64)
        values['layer.0.lora_B.weight'][0]=float('nan')
        with self.assertRaises(u.Hold):w.tensor_checks(values,'raw10_bf16',w.tensor_digest(values))
        self.assertFalse(torch.cuda.is_initialized())
    def test_once_cast_fp32_cpu_and_fp16_overflow_rejection(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self,value=1,dtype=torch.float32):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([value],dtype=dtype));self.calls=[]
            def to(self,*args,**kwargs):self.calls.append((args,kwargs));return super().to(*args,**kwargs)
        model=Model();merge.to_fp16_once(model,torch);self.assertEqual(len(model.calls),1);self.assertEqual(model.weight.dtype,torch.float16)
        wrong=Model(dtype=torch.bfloat16)
        with self.assertRaises(u.Hold):merge.to_fp16_once(wrong,torch)
        self.assertEqual(wrong.calls,[])
        with self.assertRaises(u.Hold):merge.to_fp16_once(Model(value=70000),torch)


if __name__=='__main__':unittest.main()
