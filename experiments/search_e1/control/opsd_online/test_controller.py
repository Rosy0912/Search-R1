"""Small CPU fixtures only. No CUDA, HTTP, remote deployment or production data."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import checkpoint as cp
import prepare
import sampling
import controller


def frozen_runtime():
    return Path(os.environ.get('OPSD_ONLINE_TEST_RUNTIME') or
        str(HERE.parents[4]/'search-e1-analysis-output/opsd-preparation3-xformers-backup-20261008/snapshot/runtime/opsd-preparation3-xformers'))


def diagnostic_row(text,reward=0,termination='answer',searches=0,spans=None):
    sys.path.insert(0,str(frozen_runtime()/'upstream'/'Search-R1'))
    return dict(response=text,reward=reward,termination=termination,search_calls=searches,
                policy_spans=spans if spans is not None else [[0,len(text)]])


def config_for(path):
    config=prepare.template(path);config['run_id']='a'*32;config['scientific']['component_clip']=None
    config['scientific']['outer_iterations']=2;config['source_commit']='b'*40
    config['authorization']={key:'explicit human fixture only' for key in config['authorization']}
    config['retention']['authorized']=True
    config['gates']={key:dict(path='/fixture/'+key,sha256='c'*64) for key in config['gates']}
    for key in ('source_model_manifest','runtime_manifest','helper_source_manifest','train_identity','validation_identity'):
        config[key]=dict(path='/fixture/'+key,sha256='d'*64)
    return config


def checkpoint_fixture(root,config,updates=9,outer=0,K=33,cursor=33,protection=()):
    directory=Path(root)/'checkpoints'/f'update-{updates:08d}-outer-{outer:05d}';directory.mkdir(parents=True)
    adapter=directory/'adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{}');(adapter/'adapter_model.safetensors').write_bytes(b'fixture adapter')
    ranks=[]
    for rank in range(8):
        path=directory/f'rank-{rank}.pt';path.write_text('CPU byte fixture rank '+str(rank));ranks.append(dict(rank=rank,file=cp.fileproof(path)))
    inputs={}
    for name in ('outer_manifest','pairs','order','encoding'):
        path=Path(root)/(name+'.json');path.write_text('{}');inputs[name]=cp.fileproof(path)
    value=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=config['run_id'],
        config_sha256=config['_config_sha256'],runtime_identity=config['runtime_manifest']['sha256'],
        source_model_identity=config['source_model_manifest']['sha256'],optimizer_updates=updates,cumulative_pair_exposure=K,
        complete_rank_count=8,equal_optimizer_steps=True,ranks=ranks,adapter=dict(path=str(adapter),files=[cp.fileproof(p) for p in sorted(adapter.iterdir())]),
        all_rank_adapter_sha256=['e'*64]*8,all_rank_optimizer_tensor_sha256=['f'*64]*8,
        all_rng=['python','numpy','torch_cpu','cuda_all'],scheduler=dict(kind='none',warmup_updates=0),
        cursor=dict(outer_index=outer,next_pair_cursor=cursor,usable_pair_count=K,order_sha256='1'*64,encoded_sha256='2'*64),
        training_input=inputs,sampling_model_identity=dict(fixture=True),protection_reason=list(protection))
    return value,cp.exclusive_json(directory/'manifest.json',value)


class OnlineCPUFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.config=config_for(self.root)
        self.config['_config_sha256']='3'*64

    def tearDown(self):self.tmp.cleanup()

    def test_pending_answers_hold(self):
        config=prepare.template(self.root)
        reasons=prepare.validate_config(config)
        self.assertTrue(any('component_clip' in s for s in reasons));self.assertTrue(any('outer_iterations' in s for s in reasons))
        self.assertTrue(any('storage' in s for s in reasons))
        with self.assertRaisesRegex(ValueError,'HOLD'):prepare.validate_config(config,True)

    def test_null_clip_is_valid_explicit_choice(self):
        self.assertEqual(prepare.validate_config(self.config,True),[])
        changed=copy.deepcopy(self.config);changed['scientific']['component_clip']=True
        with self.assertRaises(ValueError):prepare.validate_config(changed)

    def test_scientific_changes_rejected(self):
        for key,value in (('micro_batch',2),('learning_rate',1e-5),('loss_reduction','token_mean')):
            config=copy.deepcopy(self.config);config['scientific'][key]=value
            with self.assertRaises(ValueError):prepare.validate_config(config)

    def test_complete_checkpoint_requires_eight_actual_files(self):
        value,ref=checkpoint_fixture(self.root,self.config)
        cp.check_checkpoint(ref,self.config,self.config['run_id'],9,0)
        Path(value['ranks'][4]['file']['path']).unlink()
        with self.assertRaises(FileNotFoundError):cp.check_checkpoint(ref,self.config,self.config['run_id'])

    def test_inode_alias_rejected(self):
        value,ref=checkpoint_fixture(self.root,self.config)
        one=Path(value['ranks'][1]['file']['path']);one.unlink();os.link(value['ranks'][0]['file']['path'],one)
        value['ranks'][1]['file']=cp.fileproof(one);cp.atomic_json(ref['path'],value)
        with self.assertRaisesRegex(ValueError,'inode'):cp.check_checkpoint(cp.fileproof(ref['path']),self.config,self.config['run_id'])

    def test_k33_tail_has_two_updates(self):
        _,checkpoint=checkpoint_fixture(self.root,self.config,9,K=33)
        complete=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=self.config['run_id'],outer_index=0,
            P=40,K=33,optimizer_updates_delta=2,optimizer_updates=9,scheduler_advanced_updates=0,checkpoint=checkpoint)
        ref=cp.exclusive_json(self.root/'outer-complete.json',complete)
        cp.check_outer_complete(ref,self.config,self.config['run_id'],0,7,40)
        with self.assertRaises(ValueError):cp.check_outer_complete(ref,self.config,self.config['run_id'],0,8,40)

    def test_k0_no_optimizer_or_scheduler(self):
        _,checkpoint=checkpoint_fixture(self.root,self.config,0,K=0,cursor=0)
        complete=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=self.config['run_id'],outer_index=0,
            P=2,K=0,optimizer_updates_delta=0,optimizer_updates=0,scheduler_advanced_updates=0,checkpoint=checkpoint)
        ref=cp.exclusive_json(self.root/'outer-complete.json',complete)
        cp.check_outer_complete(ref,self.config,self.config['run_id'],0,0,2)

    def test_partial_outer_resume_selects_actual_latest_cursor(self):
        _,ref=checkpoint_fixture(self.root,self.config,8,K=33,cursor=32)
        selected=controller.latest_resume_checkpoint(self.root,self.config,self.config['run_id'],0)
        self.assertEqual(selected['sha256'],ref['sha256'])
        value,_=cp.check_checkpoint(selected,self.config,self.config['run_id'])
        self.assertEqual(value['cursor']['next_pair_cursor'],32)
        self.assertEqual(value['optimizer_updates'],8)  # outer start7, only last pair remains
        pending=self.root/'checkpoints'/'pending-incomplete';pending.mkdir();(pending/'state.pt').write_bytes(b'partial')
        self.assertEqual(controller.latest_resume_checkpoint(self.root,self.config,self.config['run_id'],0)['sha256'],ref['sha256'])

    def test_model_metadata_detects_post_hash_mutation(self):
        model=self.root/'model';model.mkdir();path=model/'weights';path.write_bytes(b'initial')
        identity=dict(path=str(model),files=[cp.fileproof(path)])
        self.assertTrue(sampling.verify_model_metadata(identity));path.write_bytes(b'changed')
        with self.assertRaises(ValueError):sampling.verify_model_metadata(identity)

    def test_merge_rejects_changed_source_before_torch_import(self):
        model=self.root/'source100';model.mkdir();files=[]
        for i in range(12):
            p=model/str(i);p.write_bytes(b'fixture source');files.append(cp.fileproof(p))
        self.config['paths']['model']=str(model)
        self.config['source_model_manifest']=cp.exclusive_json(self.root/'source-model.json',dict(files=files))
        (model/'0').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'SHA'):
            sampling.merge_latest(self.config,dict(adapter={}),self.root/'merge',dict(path='none',sha256='0'*64))

    def test_merge_rejects_changed_adapter_before_torch_import(self):
        model=self.root/'source100';model.mkdir();files=[]
        for i in range(12):
            p=model/str(i);p.write_bytes(b'fixture source');files.append(cp.fileproof(p))
        self.config['paths']['model']=str(model)
        self.config['source_model_manifest']=cp.exclusive_json(self.root/'source-model.json',dict(files=files))
        adapter=self.root/'adapter';adapter.mkdir();p=adapter/'adapter_model.safetensors';p.write_bytes(b'original');proof=cp.fileproof(p);p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):
            sampling.merge_latest(self.config,dict(adapter=dict(path=str(adapter),files=[proof])),self.root/'merge',dict(path='none',sha256='0'*64))

    def test_rolling_prune_preserves_initial_first_five_and_latest_two(self):
        config=copy.deepcopy(self.config);config.pop('_config_sha256')
        ref=cp.exclusive_json(self.root/'config.json',config);config['_config_sha256']=ref['sha256']
        cp.exclusive_json(self.root/'run-identity.json',dict(run_id=config['run_id'],config=ref,config_sha256=ref['sha256'],
            retention=config['retention'],initial_gate=dict(accepted=True)))
        refs={}
        for updates in range(7):
            _,refs[updates]=checkpoint_fixture(self.root,config,updates,K=512,cursor=updates*32)
        result=cp.prune_checkpoints(self.root,config['run_id'],ref['sha256'],refs[6])
        self.assertEqual(len(result['removed']),3)  # 2,3,4; 0,1,5 plus latest6 remain
        for updates in (0,1,5,6):self.assertTrue(Path(refs[updates]['path']).exists())
        for updates in (2,3,4):self.assertFalse(Path(refs[updates]['path']).exists())
        self.assertEqual(len(list((self.root/'retention-receipts').glob('*.json'))),3)

    def test_prune_never_accepts_foreign_run(self):
        ref=cp.exclusive_json(self.root/'config.json',{k:v for k,v in self.config.items() if not k.startswith('_')})
        cp.exclusive_json(self.root/'run-identity.json',dict(run_id='f'*32,config=ref,config_sha256=ref['sha256'],retention=self.config['retention'],initial_gate=dict(accepted=True)))
        with self.assertRaises(ValueError):cp.prune_checkpoints(self.root,self.config['run_id'],ref['sha256'],dict(path='unused',sha256='0'*64))

    def test_wrong_uid_wrong_sampling_model_rejected_without_gpu(self):
        # Config/identity schemas are checked before importing Torch/vLLM.
        config_ref=cp.exclusive_json(self.root/'config.json',self.config)
        model_ref=cp.exclusive_json(self.root/'model.json',dict(path='unused',sha256='0'*64))
        with patch('sampling.os.getuid',return_value=99):
            with self.assertRaises(ValueError):sampling.worker(config_ref,'unused',model_ref,0,'unused','unused','unused')

    def test_private_http_proxy_preserves_objects_args_and_full_json(self):
        payload=dict(queries=['检索 fixture'],topk=3,return_scores=True)
        body=dict(result=[[dict(document=dict(id='fixture-id',contents='private fixture正文'),score=0.125)]])
        response=SimpleNamespace(status_code=200,text=json.dumps(body,ensure_ascii=False),json=lambda:body)
        calls=[]
        def post(*args,**kwargs):calls.append((args,kwargs));return response
        module=SimpleNamespace(post=post,get=lambda:'unmodified-get');handle=io.StringIO()
        proxy=sampling.RequestsTraceProxy(module,handle,2,'e'*64,lambda:3)
        self.assertIs(proxy.post('http://fixture/retrieve',json=payload,timeout=300),response)
        self.assertIs(calls[0][1]['json'],payload);self.assertEqual(calls[0][0],('http://fixture/retrieve',))
        self.assertEqual(calls[0][1]['timeout'],300);self.assertIs(module.post,post);self.assertIs(proxy.get,module.get)
        record=json.loads(handle.getvalue());self.assertEqual(record['keyword_args']['json'],payload)
        self.assertEqual(record['response_json'],body);self.assertEqual(record['generation_call'],3)
        self.assertEqual(proxy.logging_errors,[])

    def test_http_original_exception_survives_log_write_failure(self):
        original=RuntimeError('fixture original HTTP error')
        def post(*args,**kwargs):raise original
        class BrokenLog:
            def write(self,text):raise OSError('fixture disk unavailable')
        proxy=sampling.RequestsTraceProxy(SimpleNamespace(post=post),BrokenLog(),0,'e'*64)
        try:proxy.post('http://fixture',timeout=300)
        except RuntimeError as actual:self.assertIs(actual,original)
        else:self.fail('Original HTTP exception disappeared')
        self.assertEqual(proxy.next_id,1);self.assertEqual(proxy.logging_errors[0]['type'],'OSError')

    def test_invalid_http_json_returns_original_response_and_records_error(self):
        original=ValueError('fixture invalid JSON')
        def parse():raise original
        response=SimpleNamespace(status_code=200,text='not JSON',json=parse);handle=io.StringIO()
        proxy=sampling.RequestsTraceProxy(SimpleNamespace(post=lambda *a,**k:response),handle,0,'e'*64)
        self.assertIs(proxy.post('http://fixture',json=dict(queries=[])),response)
        record=json.loads(handle.getvalue());self.assertEqual(record['response_json_parse_error']['type'],'ValueError')
        try:response.json()
        except ValueError as actual:self.assertIs(actual,original)
        else:self.fail('Native response JSON exception disappeared')

    def test_http_trace_bound_count_and_provenance_reject_missing_records(self):
        body=dict(result=[]);response=SimpleNamespace(status_code=200,text='{"result":[]}',json=lambda:body)
        handle=io.StringIO();proxy=sampling.RequestsTraceProxy(SimpleNamespace(post=lambda *a,**k:response),handle,1,'e'*64)
        proxy.post('http://fixture',json=dict(queries=[]),timeout=300)
        path=self.root/'http.jsonl';path.write_text(handle.getvalue());proof=cp.fileproof(path)
        self.assertEqual(sampling.verify_http_trace(proof,1,'e'*64,1)['sha256'],proof['sha256'])
        for rank,identity,count in ((2,'e'*64,1),(1,'f'*64,1),(1,'e'*64,0)):
            with self.assertRaises(ValueError):sampling.verify_http_trace(proof,rank,identity,count)
        path.write_text('');
        with self.assertRaises(ValueError):sampling.verify_http_trace(proof,1,'e'*64,1)

    def test_actual_frozen_rollout_http_instrumentation_is_null_effect(self):
        # Execute the unchanged native control flow with CPU-only fake inference/HTTP.
        path=frozen_runtime()/'search_e1'/'vllm_rollout.py';name='http_null_effect_native'
        package=ModuleType(name);package.__path__=[str(path.parent)]
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'fixture-prefix'
            def encode(self,text,**kwargs):return list(text.encode())
            def decode(self,ids):return bytes(ids).decode()
        class LLM:
            def __init__(self,*args,**kwargs):self.calls=0
            def get_tokenizer(self):return Tokenizer()
            def generate(self,prompts,*args,**kwargs):
                text='<think>fixture</think><search>fixture query</search>' if self.calls==0 else '<think>fixture</think><answer>gold</answer>'
                self.calls+=1
                return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for p in prompts]
        vllm=ModuleType('vllm');vllm.LLM=LLM;vllm.SamplingParams=lambda **kwargs:kwargs
        body=dict(result=[[dict(document=dict(id='fixture-id',contents='observation fixture'),score=0.25)]])
        response=SimpleNamespace(status_code=200,text=json.dumps(body),json=lambda:body,raise_for_status=lambda:None)
        requests=ModuleType('requests');calls=[]
        def post(*args,**kwargs):calls.append((args,kwargs));return response
        requests.post=post
        with patch.dict(sys.modules,{name:package,'vllm':vllm,'requests':requests}):
            spec=importlib.util.spec_from_file_location(name+'.vllm_rollout',path)
            native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
            questions=[dict(question_id=1,question='fixture question',answers=['gold'])]
            original=list(native.batched_rollouts('fixture-model',questions,1,'http://fixture',batch_size=64))
            baseline_calls=copy.deepcopy(calls);calls.clear();handle=io.StringIO()
            native.requests=sampling.RequestsTraceProxy(requests,handle,0,'e'*64)
            recorded=list(native.batched_rollouts('fixture-model',questions,1,'http://fixture',batch_size=64))
        self.assertEqual(recorded,original);self.assertEqual(calls,baseline_calls)
        self.assertEqual(recorded[0]['search_calls'],1);self.assertEqual(recorded[0]['termination'],'answer')
        self.assertEqual(json.loads(handle.getvalue())['response_json'],body)
        sys.modules.pop(name+'.core',None)

    def test_diagnostics_separate_final_protocol_reward_and_execution(self):
        correct=diagnostic_row('<think>fixture</think><answer>gold</answer>',1,searches=0)
        wrong=diagnostic_row('<think>fixture</think><answer>wrong</answer>',0,searches=1)
        nested=diagnostic_row('<search>fixture <answer>gold</answer></search>',0,'invalid_or_budget')
        unfinished=diagnostic_row('<think>fixture<answer>gold</answer>',0)
        self.assertTrue(sampling.rollout_diagnostics(correct)['strict_final_answer_valid'])
        self.assertTrue(sampling.rollout_diagnostics(correct)['no_executed_search'])
        self.assertTrue(sampling.rollout_diagnostics(wrong)['strict_final_answer_valid'])
        self.assertFalse(sampling.rollout_diagnostics(wrong)['no_executed_search'])
        self.assertFalse(sampling.rollout_diagnostics(nested)['strict_last_action_valid'])
        self.assertFalse(sampling.rollout_diagnostics(unfinished)['strict_final_answer_valid'])
        self.assertFalse(sampling.rollout_diagnostics(unfinished)['all_policy_spans_tag_balanced'])
        self.assertEqual(sampling.rollout_diagnostics(unfinished)['termination'],'answer')
        counts=sampling.diagnostic_counts([correct,wrong,nested,unfinished])
        self.assertEqual(counts['denominator'],4);self.assertEqual(counts['strict_final_answer_valid'],2)
        self.assertEqual(counts['strict_reward'],{'1':1,'0':3});self.assertTrue(counts['balance_is_diagnostic_only'])

    def test_tag_balance_across_turns_is_diagnostic_not_reward(self):
        first='<think>fixture <search>query</search>';second='</think><answer>gold</answer>'
        row=diagnostic_row(first+second,0,spans=[[0,len(first)],[len(first),len(first+second)]],searches=1)
        before=copy.deepcopy(row);value=sampling.rollout_diagnostics(row)
        self.assertEqual(row,before);self.assertTrue(value['joined_policy_text_tag_balance']['balanced'])
        self.assertFalse(value['all_policy_spans_tag_balanced']);self.assertEqual(value['strict_reward'],0)

    def test_spawn_registers_child_before_identity_failure(self):
        parent=object.__new__(controller.Controller);parent.config=self.config;parent.owner=dict(uid=1005)
        parent.run_id=self.config['run_id'];parent.known=[];parent.active_jobs=[];parent.output=self.root
        parent.gate=lambda phase:dict(accepted=True)
        process=type('Process',(),{'pid':123})()
        with patch('controller.subprocess.Popen',return_value=process) as popen,patch('controller.process_identity',return_value=None):
            with self.assertRaises(ValueError):parent.spawn(['fixture'],self.root/'child.log','sampling-running')
        self.assertEqual(parent.active_jobs[0][0],process);self.assertEqual(parent.known[0]['pid'],123)
        runtime=Path(self.config['paths']['runtime'])
        self.assertEqual(popen.call_args.kwargs['env']['PYTHONPATH'],str(runtime/'upstream'/'Search-R1')+os.pathsep+str(runtime))
        parent.active_jobs[0][1].close()

    def test_gate_error_waits_actual_child_and_never_signals(self):
        parent=object.__new__(controller.Controller);parent.config=self.config;parent.output=self.root;parent.known=[]
        events=[];parent.stage=lambda phase,**fields:events.append(phase)
        parent.gate=lambda phase:(_ for _ in ()).throw(ValueError('fixture gate failed'))
        parent.wait_release=lambda children:dict(accepted=True)
        class Process:
            def __init__(self):self.polls=0;self.waited=False
            def poll(self):self.polls+=1;return None if self.polls==1 else 0
            def wait(self):self.waited=True;return 0
        process=Process();handle=io.StringIO()
        with patch('controller.time.sleep'):
            with self.assertRaises(ValueError):parent.wait([(process,handle,dict(pid=123))],'sampling-running')
        self.assertTrue(process.waited);self.assertTrue(handle.closed);self.assertIn('UNCERTAIN_ADMISSION_WAITING_OWNED_TERMINATION',events)

    def test_json_partial_and_hash_mismatch(self):
        path=self.root/'partial.jsonl';path.write_text('{"a":1}')
        with self.assertRaises(ValueError):sampling.read_rows(path)
        with self.assertRaises(ValueError):cp.checked_file(dict(path=str(path),sha256='0'*64))

    def test_train_only_deterministic_cohort(self):
        train=self.root/'train';validation=self.root/'validation';train.write_text('fixture\n');validation.write_text('fixture\n')
        self.config['paths']['train']=str(train);self.config['paths']['validation']=str(validation)
        self.config['train_identity']=cp.fileproof(train);self.config['validation_identity']=cp.fileproof(validation)
        rows=[dict(question_id=i,question='training fixture '+str(i),answers=['gold']) for i in range(169103)]
        heldout=[dict(question_id=200000+i,question='heldout fixture '+str(i),answers=['gold']) for i in range(512)]
        with patch('sampling.read_rows',side_effect=[rows,heldout,rows,heldout]):
            one,p1=sampling.cohort(self.config,dict(cycle=0,offset=0));two,p2=sampling.cohort(self.config,dict(cycle=0,offset=0))
        self.assertEqual(one,two);self.assertEqual(p1,p2);self.assertEqual(len(one),512)
        heldout[0]['question']=one[0]['question']
        with patch('sampling.read_rows',side_effect=[rows,heldout]):
            with self.assertRaisesRegex(ValueError,'heldout'):sampling.cohort(self.config,dict(cycle=0,offset=0))

    def test_native_min_success_ties_and_allcorrect_fallback(self):
        runtime=str(frozen_runtime())
        core_path=Path(runtime)/'search_e1/core.py'
        self.assertEqual(hashlib.sha256(core_path.read_bytes()).hexdigest(),'0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc')
        spec=importlib.util.spec_from_file_location('fixture_exact_frozen_core',core_path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        sys.path.insert(0,str(Path(runtime)/'upstream'/'Search-R1'))  # exact deployed frozen305 dependency
        # Production uses installed rapidfuzz; this CPU test dependency is an earlier retained local copy.
        deps=HERE.parents[4]/'search-e1-analysis-output/opsd-selection-audit-local-testdeps'
        sys.path.insert(0,str(deps))
        questions=[dict(question_id=i,question='fixture '+str(i),answers=['gold']) for i in range(512)]
        rows=[]
        for q in questions:
            for sibling in range(5):
                text='<think>fixture</think><answer>gold</answer>'
                row=dict(q,sibling=sibling,response=text,search_calls=0,policy_spans=[[0,len(text)]],token_count=20,reward=1,termination='answer')
                rows.append(row)
        pairs,stats=sampling.verify_rollouts(rows,questions,core)
        self.assertEqual(len(pairs),512);self.assertEqual(stats['classes'],{'all_correct':512})
        rows[0]['search_calls']=1;rows[1]['token_count']=19;rows[2]['token_count']=19
        rows[3]['response']='<think>fixture</think><answer>wrong</answer>';rows[3]['reward']=0;rows[3]['policy_spans']=[[0,len(rows[3]['response'])]]
        pairs,_=sampling.verify_rollouts(rows,questions,core)
        self.assertEqual(pairs[0]['reference'],rows[1]['response']);self.assertEqual(pairs[0]['student'],rows[3]['response'])
        with self.assertRaises(ValueError):sampling.verify_rollouts(rows[:-1],questions,core)


if __name__=='__main__':unittest.main()
