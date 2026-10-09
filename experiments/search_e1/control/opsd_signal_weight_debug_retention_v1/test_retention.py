"""Small synthetic native-format fixtures; actual frozen stdlib verifier calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
try:
    from . import audit as a
except ImportError:
    import audit as a

HERE = Path(__file__).resolve().parent
DEFAULT_HELPER = HERE.parent/'opsd_signal_weight_v1'
ORIGINAL_COMMIT = '00fec71f6e2481f9e1bc4576d956748bdda77aaa'


def native_modules():
    root = Path(os.environ.get('OPSD_RETENTION_TEST_ORIGINAL_HELPER_ROOT', DEFAULT_HELPER))
    manifest = dict(path=str(root/'source-manifest.json'), sha256=a.ORIGINAL_SHA)
    return a.bind_helper(root, manifest, ('checkpoint', 'sampling', 'controller', 'teacher_diag'))


def science_bytes(relative):
    root = os.environ.get('OPSD_RETENTION_TEST_RUNTIME_ROOT')
    if root:
        return (Path(root)/relative).read_bytes()
    project = HERE
    while project.parent != project and not (project/'.git').exists():
        project = project.parent
    path = 'experiments/search_e1/control/opsd_debug_frozen_runtime_20261008/runtime/'+relative
    return subprocess.check_output(['git', '-C', str(project), 'show', ORIGINAL_COMMIT+':'+path])


def http_event(url='http://fixture.invalid/retrieve',queries=None,timeout=300):
    queries=['search query'] if queries is None else queries
    result=dict(result=[[dict(document=dict(id='fixture-doc',contents='fixture retained text'),score=.5)] for _ in queries])
    return dict(schema='opsd-online-private-http-request-v1',request_id=0,rank=0,sampling_model_identity='c'*64,
                generation_call=0,started_at_unix=1.,ended_at_unix=2.,positional_args=[url],
                keyword_args=dict(json=dict(queries=queries,topk=3,return_scores=True),timeout=timeout),
                status_code=200,response_body_text=json.dumps(result),response_json=result)


class Fixture:
    def __init__(self, root, native, teacher_na=False):
        self.root = Path(root); self.native = native; self.output = self.root/'group-G_POS'/'runs'
        self.output.mkdir(parents=True); self.run = 'a'*32
        self.runtime = self.root/'runtime'; self.runtime.mkdir()
        files = {}
        for name in ('search_e1/__init__.py', 'search_e1/core.py', 'upstream/Search-R1/search_r1/__init__.py', 'upstream/Search-R1/search_r1/qa_protocol.py'):
            path = self.runtime/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(science_bytes(name))
            files[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # Production checks its exact305 SHA; the small fixture only exercises
        # native scorer dependencies and is explicitly not a frozen305 proof.
        for index in range(301): files['unused-fixture-%03d.py'%index] = dict(sha256='0'*64)
        runtime = self.put(self.root/'runtime.json', dict(files=files))
        source = self.put(self.root/'source.json', {'fixture':True})
        initial = self.put(self.root/'initial.json', {'fixture':True})
        self.config = dict(schema='opsd-signal-weight-config-v1',study_id='fixture-study',
                           variant='G_POS', run_id=self.run, owner=dict(uid=1001,physical_gpus=[4,5,6,7]), precision=dict(mode='bf16'),
                           paths=dict(output_root=str(self.output),runtime=str(self.runtime),model=str(self.root/'model'),study_base=str(self.root),
                                      retriever_url='http://fixture.invalid/retrieve'),
                           scientific=dict(outer_iterations=25,global_batch=32,micro_batch=1,world_size=4,gradient_accumulation_full_batch_per_rank=8),
                           sampling=dict(dtype='bfloat16'), source_model_manifest=source,runtime_manifest=runtime,
                           common_initial_adapter_manifest=initial, common_initial_adapter_tensor_sha256='b'*64,
                           common_cohort_manifest=source,validation_identity=source,train_identity=source)
        self.cfg = self.put(self.root/'config.json',self.config)
        self.reader = a.controller_reader(native,self.config,self.cfg)
        self.outer = self.output/'outers'/'outer-00000'; self.outer.mkdir(parents=True)
        self.identity = dict(sha256='c'*64,checkpoint=None,optimizer_updates=0,
                             source_model_identity=source['sha256'],adapter_tensor_sha256='b'*64)
        self.questions = [dict(question_id='q%03d'%i, question='question %d'%i,answers=['gold'],data_source='nq') for i in range(512)]
        self.questions_ref = self.lines(self.outer/'questions.jsonl',self.questions)
        self.terminal = self.exits(8,100)
        waves = [self.exits(4,100),self.exits(4,104)]
        self.terminal['child_exit_evidence'] = [e for wave in waves for e in wave['child_exit_evidence']]
        terminal,_=a.read(self.terminal['actual_terminal']);terminal['child_exit_evidence']=self.terminal['child_exit_evidence']
        self.terminal['actual_terminal']=self.put(self.terminal['actual_terminal']['path'],terminal)
        self.terminal['waves'] = waves
        self.worker_refs = []; self.http_refs = []; combined = []
        for rank in range(8):
            rollouts = [dict(question_id=q['question_id'],question=q['question'],answers=q['answers'],sibling=s,
                             response='x',policy_spans=[[0,1]],search_calls=0,reward=0,termination='turn_limit')
                        for q in self.questions[rank::8] for s in range(5)]
            if not teacher_na:
                answer = '<think>x</think><answer>gold</answer>'
                for row in rollouts:
                    if row['question_id'] == 'q000':
                        row.update(response=answer,policy_spans=[[0,len(answer)]],reward=1,termination='answer')
            combined.extend(rollouts)
            rr = self.lines(self.outer/('rollouts-%d.jsonl'%rank),rollouts)
            tr = self.lines(self.outer/('generation-%d.jsonl'%rank),[dict(call=0,index=i,rank=rank,started=1.,ended=2.,
                            input_text_sha256='d'*64,prompt_token_ids=[1],generated_token_ids=[2],native_output_text=rollouts[i]['response'],
                            finish_reason='length',stop_reason=None,sampling_model_identity=self.identity['sha256']) for i in range(320)])
            hp = self.lines(self.outer/('http-%d.jsonl'%rank),[]); self.http_refs.append(hp)
            self.worker_refs.append(self.put(self.outer/('worker-%d.json'%rank),dict(
                schema='opsd-online-sampling-worker-complete-v1',accepted=True,rank=rank,pid=100+rank,uid=1001,
                run_id=self.run,config_sha256=self.cfg['sha256'],sampling_model_identity=self.identity['sha256'],
                rows=320,rollouts=rr,generation_trace=tr,http_trace=hp,http_requests=0,variant='G_POS',
                precision=self.config['precision'],actual_llm_load=dict(actual_dtype='torch.bfloat16'))))
        self.rollouts = self.lines(self.outer/'rollouts.jsonl',combined)
        valid = '<think>x</think><answer>gold</answer>'
        pair = dict(question_id='q000',question='question 0',answers=['gold'],reference_reward=1,
                    reference=valid,student=valid,reference_sibling=0,student_sibling=0,
                    mining=dict(self_pair_by_sibling=True,pair_class='CC'))
        self.pairs = self.lines(self.outer/'pairs.jsonl',[] if teacher_na else [pair])
        self.manifest = dict(run_id=self.run,config_sha256=self.cfg['sha256'],outer_index=0,
            previous_complete_checkpoint=None,cumulative_optimizer_updates_before=0,runtime_identity=runtime['sha256'],
            source_model_identity=source['sha256'],common_initial_adapter_manifest=initial,
            outer_old_adapter=initial,
            contract=native['checkpoint'].science_contract(self.config),sampling_model_identity=self.identity,
            rollouts=self.rollouts,pairs=self.pairs,questions=self.questions_ref,P=0 if teacher_na else 1)
        self.sample = dict(schema='opsd-online-sampling-complete-v1',accepted=True,run_id=self.run,
                           sampling_model_identity=self.identity,rollouts=self.rollouts,pairs=self.pairs,
                           exit_proof=self.terminal,workers=self.worker_refs,private_http_traces=self.http_refs)
        self.parent = self.teacher(pair,teacher_na)

    def put(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
        return self.native['checkpoint'].fileproof(path)

    def lines(self,path,values):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
        return self.native['checkpoint'].fileproof(path)

    def exits(self,count,pid,phase='sampling-running'):
        entries=[dict(identity=dict(pid=pid+i,starttick=1000+i,uid=1001,run_id=self.run),actual_wait_returncode=0) for i in range(count)]
        ref=self.put(self.outer/('%s-terminal-%s-%s.json'%(phase,pid,count)),dict(schema='opsd-ablation-actual-phase-terminal-v1',
            run_id=self.run,variant='G_POS',config_sha256=self.cfg['sha256'],phase=phase,child_exit_evidence=entries))
        return dict(accepted=True,child_exit_evidence=entries,actual_terminal=ref,gpu_release=dict(all_owned_release=dict(accepted=True)))

    def teacher(self,pair,na):
        selection=self.native['teacher_diag'].select_cases([] if na else [pair],0)
        selection_ref=self.put(self.outer/'teacher-selection.json',selection)
        parent=dict(schema='opsd-ablation-teacher-diagnostic-parent-v1',accepted=True,run_id=self.run,variant='G_POS',
            outer_index=0,config_sha256=self.cfg['sha256'],pairs=self.pairs,selection=selection_ref,count=selection['count'],
            not_applicable=na,request=None,child=None,exit_proof=None,diagnostic_teacher_autoregressive=True,
            original_training_objective=False,diagnostic_precision_common_bf16=True,training_or_validation_denominator_contribution=0)
        if na:return self.put(self.outer/'teacher-parent.json',parent)
        model=self.root/'model';model.mkdir(); file=model/'small-source-metadata.txt';file.write_text('fixture source bytes')
        source=dict(kind='raw-source100',path=str(model),source_model_identity=self.config['source_model_manifest']['sha256'],
                    files=[self.native['checkpoint'].fileproof(file)])
        request=dict(config=self.cfg,run_id=self.run,variant='G_POS',outer_index=0,selection=selection_ref,pairs=self.pairs,source_identity=source)
        req=self.put(self.outer/'teacher-request.json',request)
        http=self.lines(self.outer/'teacher-http.jsonl',[])
        text=pair['reference'];result=dict(question_id=pair['question_id'],answers=pair['answers'],response=text,
                                         policy_spans=[[0,len(text)]],reward=1,search_calls=0,termination='answer')
        trace=dict(schema='opsd-ablation-privileged-teacher-autoregressive-v1',logging_errors=[],return_objects_modified=False,
            generation_options_modified=False,teacher_forced=False,original_training_objective=False,diagnostic_teacher_autoregressive=True,
            diagnostic_precision_common_bf16=True,events=[dict(kind='native_generate',call=0,started_unix=1.,ended_unix=2.,
                                                             args=[[[1,2]]],kwargs=dict(max_new_tokens=500),raw_output_ids=[[1,2,3]])])
        case=dict(selected=selection['selected'][0],run_id=self.run,variant='G_POS',outer_index=0,
            source_model_identity=self.config['source_model_manifest']['sha256'],original_pairs=self.pairs,
            original_B=pair['reference'],original_D=pair['student'],result=result,trace=trace,http=http,
            diagnostic_teacher_autoregressive=True,original_training_objective=False,diagnostic_precision_common_bf16=True)
        case_ref=self.put(self.outer/'teacher-case.json',case)
        child=dict(schema='opsd-ablation-teacher-diagnostic-child-v1',accepted=True,request=req,selection=selection_ref,pairs=self.pairs,
            run_id=self.run,variant='G_POS',config_sha256=self.cfg['sha256'],pid=200,uid=1001,count=1,cases=[case_ref],source_identity=source,
            diagnostic_teacher_autoregressive=True,original_training_objective=False,diagnostic_precision_common_bf16=True,
            training_or_validation_denominator_contribution=0)
        parent.update(request=req,child=self.put(self.outer/'teacher-child.json',child),exit_proof=self.exits(1,200))
        return self.put(self.outer/'teacher-parent.json',parent)

    def entry(self):
        K=self.manifest['P'];manifest=self.put(self.outer/'outer-manifest.json',self.manifest)
        cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=self.run,config_sha256=self.cfg['sha256'],
            optimizer_updates=K,cursor=dict(outer_index=0,next_pair_cursor=K,usable_pair_count=K),
            training_input=dict(outer_manifest=manifest),sampling_model_identity=self.identity,cumulative_pair_exposure=K,
            runtime_identity=self.config['runtime_manifest']['sha256'],source_model_identity=self.config['source_model_manifest']['sha256'],
            contract=self.native['checkpoint'].science_contract(self.config),common_initial_adapter=self.config['common_initial_adapter_manifest'],
            outer_old_adapter=self.config['common_initial_adapter_manifest'],
            all_rank_adapter_sha256=['e'*64]*4)
        checkpoint=self.put(self.outer/'checkpoint.json',cp)
        complete=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=self.run,outer_index=0,
            checkpoint=checkpoint,P=K,K=K,optimizer_updates_delta=K,optimizer_updates=K,cumulative_pair_exposure=K)
        return dict(variant='G_POS',outer_index=0,outer_manifest=manifest,sampling_complete=self.put(self.outer/'sampling-complete.json',self.sample),
                    outer_complete=self.put(self.outer/'outer-complete.json',complete),end_checkpoint=checkpoint,
                    trainer_exit=self.put(self.outer/'trainer-exit.json',self.exits(1,300,'training-running')),teacher_parent=self.parent)


class Retention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native=native_modules()

    def setUp(self):
        # These names belong only to this standalone synthetic fixture process.
        for name in list(sys.modules):
            if name=='search_e1' or name.startswith('search_e1.') or name=='search_r1' or name.startswith('search_r1.'):
                del sys.modules[name]
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name,self.native)

    def test_actual_native_first_outer_debug_reader(self):
        value,*_=a.outer_debug(a.Inventory(),self.native,self.f.config,self.f.cfg,self.f.entry(),None,0,0)
        self.assertEqual(value['sampling']['workers'],8);self.assertEqual(value['teacher']['cases'],1)
        self.assertEqual(value['sampling']['generated_turns'],2560);self.assertNotIn('torch',sys.modules)

    def test_missing_generation_file_rejected(self):
        worker,_=a.read(self.f.worker_refs[0]);Path(worker['generation_trace']['path']).unlink()
        with self.assertRaises(FileNotFoundError):a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_changed_generation_bytes_rejected_at_same_length(self):
        worker,_=a.read(self.f.worker_refs[0]);path=Path(worker['generation_trace']['path'])
        raw=path.read_bytes();path.write_bytes(raw.replace(b'"native_output_text": "x"',b'"native_output_text": "y"',1))
        with self.assertRaises(ValueError):a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_changed_http_bytes_rejected(self):
        worker,_=a.read(self.f.worker_refs[0]);Path(worker['http_trace']['path']).write_text('{}\n')
        with self.assertRaises(ValueError):a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_partial_actual_wait_rejected(self):
        self.f.sample['exit_proof']['child_exit_evidence'][0]['actual_wait_returncode']=None
        with self.assertRaises(ValueError):a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_false_release_rejected(self):
        self.f.sample['exit_proof']['gpu_release']['all_owned_release']['accepted']=False
        with self.assertRaises(ValueError):a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_http_native_provenance_shape(self):
        event=http_event()
        ref=self.f.lines(self.f.outer/'one-http.jsonl',[event])
        self.native['sampling'].verify_http_trace(ref,0,'c'*64,1)
        event['rank']=1;ref=self.f.lines(self.f.outer/'one-http.jsonl',[event])
        with self.assertRaises(ValueError):self.native['sampling'].verify_http_trace(ref,0,'c'*64,1)

    def test_http_payload_accepts_real_batch_empty_query_and_exact_body(self):
        event=http_event(queries=['','second query'])
        result=a.http_payload([event],self.f.config['paths']['retriever_url'],300)
        self.assertEqual(result['queries'],2);self.assertEqual(result['requests'],1)
        self.assertTrue(result['query_count_not_inferred_from_rollout_search_calls'])

    def test_http_payload_missing_actual_request_fields_rejected(self):
        for field in ('positional_args','keyword_args','status_code','response_body_text','response_json'):
            with self.subTest(field=field):
                event=http_event();del event[field]
                with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_payload_changed_url_rejected(self):
        event=http_event(url='http://different.invalid/retrieve')
        with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_payload_missing_or_mutated_json_budget_rejected(self):
        for key,value in (('queries',None),('queries',[42]),('queries',[]),('topk',4),('return_scores',False),('return_scores',1)):
            with self.subTest(key=key,value=value):
                event=http_event()
                if value is None:del event['keyword_args']['json'][key]
                else:event['keyword_args']['json'][key]=value
                with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_payload_non2xx_or_missing_timeout_rejected(self):
        for status in (199,300,404,None,True):
            with self.subTest(status=status):
                event=http_event();event['status_code']=status
                with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)
        event=http_event();del event['keyword_args']['timeout']
        with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_payload_body_and_json_disagreement_rejected(self):
        event=http_event();event['response_json']['result'][0][0]['document']['contents']='changed retained text'
        with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)
        event=http_event();event['response_body_text']='{"result":'
        with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_payload_query_result_count_and_shape_rejected(self):
        for value in ([],{},[{}]):
            with self.subTest(result=value):
                event=http_event();event['response_json']['result']=value;event['response_body_text']=json.dumps(event['response_json'])
                with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],300)

    def test_http_teacher_singleton_timeout120_and_empty_query(self):
        event=http_event(queries=[''],timeout=120)
        self.assertEqual(a.http_payload([event],self.f.config['paths']['retriever_url'],120,singleton=True)['queries'],1)
        event=http_event(queries=['a','b'],timeout=120)
        with self.assertRaises(ValueError):a.http_payload([event],self.f.config['paths']['retriever_url'],120,singleton=True)
        with self.assertRaises(ValueError):a.http_payload([http_event()],self.f.config['paths']['retriever_url'],120,singleton=True)

    def test_sampling_reader_rejects_mutated_payload_even_with_updated_file_sha(self):
        worker,_=a.read(self.f.worker_refs[0]);event=http_event(url='http://changed.invalid/retrieve')
        worker['http_trace']=self.f.lines(worker['http_trace']['path'],[event]);worker['http_requests']=1
        self.f.worker_refs[0]=self.f.put(self.f.worker_refs[0]['path'],worker)
        self.f.sample['workers']=self.f.worker_refs;self.f.sample['private_http_traces'][0]=worker['http_trace']
        with self.assertRaisesRegex(ValueError,'HTTP URL'):
            a.sampling_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,self.f.sample)

    def test_teacher_reader_rejects_non2xx_payload_even_with_updated_file_sha(self):
        parent,_=a.read(self.f.parent);child,_=a.read(parent['child']);case,_=a.read(child['cases'][0])
        event=http_event(timeout=120);event['sampling_model_identity']='source100-teacher-diagnostic';event['status_code']=500
        case['http']=self.f.lines(case['http']['path'],[event]);case['result']['search_calls']=1
        child['cases']=[self.f.put(child['cases'][0]['path'],case)];parent['child']=self.f.put(parent['child']['path'],child)
        self.f.parent=self.f.put(self.f.parent['path'],parent)
        with self.assertRaisesRegex(ValueError,'successful2xx'):
            a.teacher_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,0,self.f.parent)

    def test_context_zero_span_and_empty_token_generation_accepted(self):
        rows=[dict(response='',policy_spans=[])]+[dict(response='',policy_spans=[[0,0]]) for _ in range(319)]
        events=[dict(call=0,index=i,rank=0,started=1.,ended=2.,input_text_sha256='a'*64,prompt_token_ids=[1],generated_token_ids=[],
                     native_output_text='',finish_reason='stop',stop_reason=1,sampling_model_identity='c'*64) for i in range(319)]
        measured=a.generation_trace(events,rows,0,'c'*64)
        self.assertEqual(measured['rows'],319);self.assertEqual(measured['zero_span_rollouts'],1)

    def test_raw_generation_turn_mismatch_rejected(self):
        worker,_=a.read(self.f.worker_refs[0]);events,_=a.rows(worker['generation_trace']);data,_=a.rows(worker['rollouts'])
        events[0]['native_output_text']='y'
        with self.assertRaises(ValueError):a.generation_trace(events,data,0,'c'*64)

    def test_generation_count_is_not_fixed320_for_multiple_turns(self):
        rows=[dict(response='xy',policy_spans=[[0,1],[1,2]]) for _ in range(320)]
        events=[dict(call=c,index=i,rank=0,started=1.,ended=2.,input_text_sha256='a'*64,prompt_token_ids=[1],generated_token_ids=[2],
                    native_output_text='xy'[c],finish_reason='stop',stop_reason=1,sampling_model_identity='c'*64) for c in range(2) for i in range(320)]
        self.assertEqual(a.generation_trace(events,rows,0,'c'*64)['rows'],640)
        with self.assertRaises(ValueError):a.generation_trace(events[:-1],rows,0,'c'*64)

    def test_teacher_na_requires_no_child_no_fabricated8(self):
        f=Fixture(Path(self.tmp.name)/'NA',self.native,teacher_na=True)
        value=a.teacher_debug(a.Inventory(),self.native,f.reader,f.manifest,0,f.parent)
        self.assertTrue(value['not_applicable']);self.assertEqual(value['cases'],0)
        parent,_=a.read(f.parent);parent['count']=8;ref=f.put(f.parent['path'],parent)
        with self.assertRaises(ValueError):a.teacher_debug(a.Inventory(),self.native,f.reader,f.manifest,0,ref)

    def test_teacher_raw_ids_full_prefix_must_match(self):
        parent,_=a.read(self.f.parent);child,_=a.read(parent['child']);case,_=a.read(child['cases'][0])
        case['trace']['events'][0]['raw_output_ids']=[[9,2,3]]
        with self.assertRaises(ValueError):a.teacher_events(case['trace'],case['result'])

    def test_changed_teacher_http_reference_rejected(self):
        parent,_=a.read(self.f.parent);child,_=a.read(parent['child']);case,_=a.read(child['cases'][0])
        Path(case['http']['path']).write_text('{}\n')
        with self.assertRaises(ValueError):a.teacher_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,0,self.f.parent)

    def test_teacher_nonzero_wait_rejected(self):
        parent,_=a.read(self.f.parent);parent['exit_proof']['child_exit_evidence'][0]['actual_wait_returncode']=1
        ref=self.f.put(self.f.parent['path'],parent)
        with self.assertRaises(ValueError):a.teacher_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,0,ref)

    def test_teacher_context_limit_zero_generate_allowed(self):
        trace=dict(schema='opsd-ablation-privileged-teacher-autoregressive-v1',teacher_forced=False,original_training_objective=False,
            return_objects_modified=False,generation_options_modified=False,logging_errors=[],events=[dict(kind='tokenize_context',ids=[[1]])])
        self.assertEqual(a.teacher_events(trace,dict(policy_spans=[]))['calls'],0)

    def test_teacher_decode_events_not_generate_count(self):
        parent,_=a.read(self.f.parent);child,_=a.read(parent['child']);case,_=a.read(child['cases'][0])
        case['trace']['events'].extend([dict(kind='decode_actual_selected_ids',ids=[1],text='x') for _ in range(10)])
        self.assertEqual(a.teacher_events(case['trace'],case['result'])['calls'],1)

    def test_real_teacher_wrong_or_protocol_invalid_remains_retained(self):
        for text in ('<think>x</think><answer>wrong</answer>','<think><answer>gold</answer>'):
            with self.subTest(text=text):
                parent,_=a.read(self.f.parent);child,_=a.read(parent['child']);case,_=a.read(child['cases'][0])
                case['result'].update(response=text,policy_spans=[[0,len(text)]],reward=0)
                child['cases']=[self.f.put(child['cases'][0]['path'],case)]
                parent['child']=self.f.put(parent['child']['path'],child)
                self.f.parent=self.f.put(self.f.parent['path'],parent)
                self.assertEqual(a.teacher_debug(a.Inventory(),self.native,self.f.reader,self.f.manifest,0,self.f.parent)['cases'],1)

    def test_uncertain_wait_identity_rejected(self):
        exits=self.f.exits(1,500);exits['child_exit_evidence'][0]['identity']['identity_uncertain']=True
        terminal,_=a.read(exits['actual_terminal']);terminal['child_exit_evidence']=exits['child_exit_evidence']
        exits['actual_terminal']=self.f.put(exits['actual_terminal']['path'],terminal)
        with self.assertRaises(ValueError):a.exit_evidence(a.Inventory(),self.f.reader,exits,'sampling-running',1)

    def test_inventory_fresh_end_sha_not_only_metadata(self):
        inventory=a.Inventory();ref=self.f.put(self.f.outer/'mutable.json',dict(x=1));inventory.file(ref)
        path=Path(ref['path']);path.write_bytes(path.read_bytes().replace(b'1',b'2'))
        with self.assertRaises(ValueError):inventory.finish()

    def test_outer_incomplete_cursor_rejected(self):
        entry=self.f.entry();cp,_=a.read(entry['end_checkpoint']);cp['cursor']['next_pair_cursor']=0
        entry['end_checkpoint']=self.f.put(entry['end_checkpoint']['path'],cp)
        completion,_=a.read(entry['outer_complete']);completion['checkpoint']=entry['end_checkpoint']
        entry['outer_complete']=self.f.put(entry['outer_complete']['path'],completion)
        with self.assertRaises(ValueError):a.outer_debug(a.Inventory(),self.native,self.f.config,self.f.cfg,entry,None,0,0)

    def test_final_not_100_refuses_before_read(self):
        request=dict(schema=a.REQUEST,mode='final',currentN=1,configs={v:{} for v in a.VARIANTS},outers=[])
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):a.audit(request)

    def test_prefix_missing_or_wrong_serial_order_rejected(self):
        request=dict(schema=a.REQUEST,mode='prefix',currentN=2,configs={v:{} for v in a.VARIANTS},
                     outers=[dict(variant='G_POS',outer_index=0),dict(variant='G_NEG',outer_index=0)])
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):a.audit(request)

    def test_original_config_sha_not_interchangeable(self):
        request=dict(schema=a.REQUEST,mode='prefix',currentN=1,configs={v:dict(sha256='f'*64) for v in a.VARIANTS},
            study_manifest=dict(sha256=a.STUDY_SHA),outers=[dict(variant='G_POS',outer_index=0)])
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ValueError,'immutable configs'):a.audit(request)

    def test_prefix_stdout_scope_and_actual_teacher_counts(self):
        configs={'G_POS':self.f.cfg}
        for variant in a.VARIANTS[1:]:
            body=copy.deepcopy(self.f.config);body['variant']=variant;body['run_id']=variant.lower()
            configs[variant]=self.f.put(self.f.root/(variant+'-config.json'),body)
        study=self.f.put(self.f.root/'study.json',dict(study_base=str(self.f.root),study_id='fixture-study',
                            groups=[dict(variant=v,config=configs[v]) for v in a.VARIANTS]))
        request=dict(schema=a.REQUEST,mode='prefix',currentN=1,configs=configs,study_manifest=study,
                     retention_source={},outers=[self.f.entry()])
        # Only authorization/publication fixtures are substituted. All actual
        # first-outer native parsing, SHA, wait, trace/scorer and inventory run.
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(a.os,'getuid',return_value=1001),\
             patch.object(a,'STUDY_ROOT',str(self.f.root)),patch.object(a,'STUDY_SHA',study['sha256']),\
             patch.dict(a.CONFIG_SHAS,{v:r['sha256'] for v,r in configs.items()}),\
             patch.object(a,'published_reader',return_value={'fixture':True}),patch.object(a,'verify_source',return_value=self.native):
            result=a.audit(request)
        self.assertTrue(result['accepted']);self.assertFalse(result['full_study_complete'])
        self.assertEqual((result['complete_outers'],result['sampling_workers'],result['teacher_parents'],result['teacher_cases']),(1,8,1,1))
        self.assertTrue(result['referenced_inventory']['fresh_end_sha_recheck'])
        self.assertNotIn('gold',json.dumps(result));self.assertNotIn('torch',sys.modules)

    def test_own_published_source_manifest_and_readback_bound(self):
        manifest=HERE/'source-manifest.json'
        mp=self.native['checkpoint'].fileproof(manifest)
        publication=dict(accepted=True,branch='search-e1',source_commit='f'*40,remote_readback_sha='f'*40,
                         source_manifest_sha256=mp['sha256'])
        ref=self.f.put(self.f.root/'reader-publication.json',publication)
        spec=dict(helper_root=str(HERE),source_manifest=mp,publication=ref)
        self.assertEqual(a.published_reader(a.Inventory(),spec)['source_commit'],'f'*40)
        publication['remote_readback_sha']='e'*40
        spec['publication']=self.f.put(ref['path'],publication)
        with self.assertRaises(ValueError):a.published_reader(a.Inventory(),spec)


if __name__=='__main__':unittest.main()
