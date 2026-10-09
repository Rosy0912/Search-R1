"""Small CPU-only lifecycle and scope fixtures; no model/CUDA/HTTP calls."""
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PACKAGE='_opsd_signal_control_fixtures'
package=types.ModuleType(PACKAGE);package.__path__=[str(HERE)];sys.modules[PACKAGE]=package
for name in ('checkpoint','targets','prepare','scoring','sampling','gates','controller','teacher_diag'):
    path=HERE/(name+'.py');module=types.ModuleType(PACKAGE+'.'+name)
    module.__file__=str(path);module.__package__=PACKAGE
    sys.modules[module.__name__]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
prepare=sys.modules[PACKAGE+'.prepare'];gates=sys.modules[PACKAGE+'.gates'];controller=sys.modules[PACKAGE+'.controller']
sampling=sys.modules[PACKAGE+'.sampling']
teacher=sys.modules[PACKAGE+'.teacher_diag']

class Paused(Exception):pass
class Child:
    def __init__(self,polls,rc=0):self.values=list(polls);self.rc=rc;self.pid=9001
    def poll(self):return self.values.pop(0) if self.values else self.rc
    def wait(self):return self.rc

class Control(unittest.TestCase):
    def build(self,root):
        obj=object.__new__(controller.Controller);obj.output=Path(root);obj.run_id='a'*32
        obj.config=prepare.template(str(root),'G_POS','study',str(Path(root).parent))
        obj.config_ref={'path':'/config.json','sha256':'b'*64};obj.owner=dict(pid=8,uid=1001,starttick=12,run_id=obj.run_id)
        obj.known=[];obj.active_jobs=[];obj.stage=lambda *a,**k:None;obj.admit=None
        obj.wait_release=lambda children:{'accepted':True};obj.gate=lambda *a,**k:{'accepted':True}
        return obj
    def job(self,root,child,known=True):
        handle=(Path(root)/'child.log').open('x');handle.write('real log\n');handle.flush()
        return child,handle,dict(pid=child.pid,uid=1001,starttick=23 if known else None,run_id='a'*32,identity_uncertain=not known)
    def test_matrix_and_missing_new_scope_hold(self):
        for variant in ('G_POS','G_NEG','H_POS','H_NEG'):
            cfg=prepare.template('/group/'+variant,variant,'new-study','/study')
            self.assertEqual(cfg['scientific']['learning_rate'],1e-6)
            self.assertEqual(cfg['scientific']['outer_iterations'],25);self.assertEqual(cfg['scientific']['component_clip'],10)
            self.assertEqual(cfg['precision']['mode'],'bf16')
            self.assertIn('storage receipt missing',prepare.validate_config(cfg))
            cfg['scientific']['learning_rate']=9e-6
            with self.assertRaises(ValueError):prepare.validate_config(cfg)
    def test_old_200_scope_rejected_exact_new_group_accepted(self):
        cfg=prepare.template('/group','G_POS','ablation-new','/study');cfg['run_id']='a'*32
        with self.assertRaises(gates.Hold):gates.scoped(cfg,dict(scope={'study_id':'online200'}))
        value=dict(scope=dict(study_id='ablation-new',variant='G_POS',run_id='a'*32,output_root='/group/runs'))
        self.assertIs(gates.scoped(cfg,value),value)
        value['scope']['variant']='H_NEG'
        with self.assertRaises(gates.Hold):gates.scoped(cfg,value)
    def test_transient_running_gate_does_not_turn_real_zero_into_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.build(tmp);calls=[]
            def gate(*a,**k):
                calls.append(a)
                if len(calls)==1:raise OSError('one observation race')
                return {'accepted':True}
            obj.gate=gate
            job=self.job(tmp,Child([None,0]))
            with patch.object(controller.time,'sleep',lambda _:None):result=obj.wait([job],'sampling-running')
            self.assertTrue(result['accepted']);self.assertEqual(result['child_exit_evidence'][0]['actual_wait_returncode'],0)
            self.assertTrue(Path(result['actual_terminal']['path']).is_file())
    def test_real_nonzero_is_durable_before_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.build(tmp);released=[];obj.wait_release=lambda c:released.append(c) or {'accepted':True}
            with self.assertRaises(ValueError):obj.wait([self.job(tmp,Child([2],2))],'training-running')
            saved=list((Path(tmp)/'phase-exits').glob('*.json'))
            self.assertEqual(len(saved),1);self.assertEqual(json.loads(saved[0].read_text())['child_exit_evidence'][0]['actual_wait_returncode'],2)
            self.assertEqual(len(released),1)
    def test_terminal_persistent_gate_remains_hold_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.build(tmp);obj.gate=lambda *a,**k:(_ for _ in ()).throw(gates.Hold('not ready'))
            with patch.object(controller.time,'sleep',side_effect=Paused):
                with self.assertRaises(Paused):obj.wait([self.job(tmp,Child([0]))],'sampling-running')
            self.assertEqual(len(list((Path(tmp)/'phase-exits').glob('*.json'))),1)
    def test_evidence_io_failure_retries_or_holds(self):
        calls=[]
        def once():
            calls.append(1)
            if len(calls)==1:raise OSError('disk transient')
            return 'durable'
        self.assertEqual(controller.hold_evidence(once,pause=lambda _:None),'durable')
        self.assertEqual(len(calls),2)
        with self.assertRaises(Paused):controller.hold_evidence(lambda:(_ for _ in ()).throw(OSError()),pause=lambda _:(_ for _ in ()).throw(Paused()))
    def test_quick_exit_before_proc_identity_registered_and_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.build(tmp);child=Child([3],3)
            with patch.object(controller.subprocess,'Popen',return_value=child),patch.object(controller,'process_identity',return_value=None):
                job=obj.spawn(['python','sampling.py'],Path(tmp)/'quick.log','sampling-running')
            self.assertEqual(obj.known[0]['pid'],child.pid);self.assertIsNone(obj.known[0]['starttick'])
            with self.assertRaises(ValueError):obj.wait([job],'sampling-running')
    def test_phase_memory_floor_does_not_lower_unstarted_card(self):
        cfg=prepare.template('/group','G_POS','study','/study')
        devices=[dict(uuid='owned',actual_free_mib=6144),dict(uuid='not-started',actual_free_mib=5000)]
        with self.assertRaises(gates.Hold):gates.resource_limits(cfg,'sampling-running',devices,[{}],{'owned'})
        devices[1]['actual_free_mib']=20480;gates.resource_limits(cfg,'sampling-running',devices,[{}],{'owned'})
    def test_cpu_parent_cuda_compute_app_rejected(self):
        cfg=prepare.template('/group','G_POS','study','/study');cfg.update(run_id='a'*32,_config_sha256='b'*64)
        owner=dict(pid=9,uid=1001,starttick=3)
        cards=[dict(index=i+4,uuid=u,actual_free_mib=23000) for i,u in enumerate(cfg['owner']['gpu_uuids'])]
        transition=dict(accepted=True,cluster='HKUST',gpu_uuids=cfg['owner']['gpu_uuids'])
        with patch.object(gates.os,'getuid',return_value=1001),patch.object(gates,'process',return_value=owner),\
             patch.object(gates,'bound',return_value=transition),patch.object(gates,'scoped',side_effect=lambda c,r:r),\
             patch.object(gates,'publication_admission',return_value={}),patch.object(gates,'storage_admission',return_value={}),\
             patch.object(gates,'retriever_admission',return_value={'owner':{'pid':77}}),\
             patch.object(gates,'scoped_cpu_processes',return_value={'accepted':True,'identities':[]}),\
             patch.object(gates,'memory_observer_admission',return_value={}),\
             patch.object(gates,'inventory',return_value=(cards,[dict(pid=9,gpu_uuid=cards[0]['uuid'],used_mib=500)])):
            with self.assertRaisesRegex(gates.Hold,'controller'):gates.require_launch(cfg,'sampling-running','a'*32,owner,[])
    def test_observer_bootstrap_only_missing_memory_not_gpu_phase_bypass(self):
        cfg=prepare.template('/group','G_POS','study','/study');cfg.update(run_id='a'*32,_config_sha256='b'*64)
        owner=dict(pid=9,uid=1001,starttick=3)
        cards=[dict(index=i+4,uuid=u,actual_free_mib=23000) for i,u in enumerate(cfg['owner']['gpu_uuids'])]
        transition=dict(accepted=True,cluster='HKUST',gpu_uuids=cfg['owner']['gpu_uuids'])
        with patch.object(gates.os,'getuid',return_value=1001),patch.object(gates,'process',return_value=owner),\
             patch.object(gates,'bound',return_value=transition),patch.object(gates,'scoped',side_effect=lambda c,r:r),\
             patch.object(gates,'publication_admission',return_value={}),patch.object(gates,'storage_admission',return_value={}),\
             patch.object(gates,'retriever_admission',return_value={'owner':{'pid':77}}),\
             patch.object(gates,'scoped_cpu_processes',return_value={'accepted':True,'identities':[]}),\
             patch.object(gates,'memory_observer_admission',side_effect=gates.Hold('observer not created')) as memory,\
             patch.object(gates,'inventory',return_value=(cards,[])):
            self.assertEqual(gates.require_launch_without_memory(cfg,'a'*32,owner,[])['phase'],'observer-startup')
            memory.assert_not_called()
            with self.assertRaisesRegex(gates.Hold,'observer not created'):gates.require_launch(cfg,'sampling-start','a'*32,owner,[])
            with self.assertRaisesRegex(gates.Hold,'running child'):gates.require_launch_without_memory(cfg,'a'*32,owner,[owner])

    def test_dynamic_cpu_roles_actual_published_sources_argv_and_gpu_refusal(self):
        cp=sys.modules[PACKAGE+'.checkpoint']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=prepare.template(str(root/'group'),'G_POS','study',str(root))
            cfg.update(run_id='a'*32,source_commit='c'*40);cfg['paths']['python']='/python'
            train_root=Path(cfg['paths']['helper_root']);train_root.mkdir(parents=True)
            eval_root=root/'opsd_signal_weight_v1';eval_root.mkdir()
            def published(helper,name,commit):
                source=helper/name;source.write_text('# synthetic CPU fixture; never executed\n')
                ref=cp.fileproof(source);manifest=cp.exclusive_json(helper/'manifest.json',dict(files={name:ref}))
                publication=cp.exclusive_json(helper/'publication.json',dict(accepted=True,branch='search-e1',
                    source_commit=commit,remote_readback_sha=commit,source_manifest_sha256=manifest['sha256']))
                return ref,manifest,publication
            driver_source,driver_manifest,driver_pub=published(eval_root,'study_runner.py','d'*40)
            observer_source,observer_manifest,observer_pub=published(train_root,'memory_observer.py','c'*40)
            cfg['helper_source_manifest']=observer_manifest;cfg['gates']['publication']=observer_pub
            group_config=cp.exclusive_json(root/'group-config.json',cfg);cfg['_config_sha256']=group_config['sha256']
            study=cp.exclusive_json(root/'study.json',dict(study_id='study',helper_root=str(eval_root),
                helper_source_manifest=driver_manifest,publication=driver_pub,source_commit='d'*40))
            roles=[];actuals={}
            for i,(name,source,manifest,pub,input_ref,option) in enumerate([
                ('study_driver',driver_source,driver_manifest,driver_pub,study,'--study-manifest'),
                ('memory_observer',observer_source,observer_manifest,observer_pub,group_config,'--config')]):
                identity=dict(pid=100+i,uid=1001,starttick=20+i,ppid=1,pgid=100+i,run_id=name+'-independent',
                    argv=['/python',source['path'],option,input_ref['path']])
                actuals[identity['pid']]=identity
                roles.append(dict(role=name,identity=identity,source=source,helper_source_manifest=manifest,
                    publication=pub,input_config=input_ref,no_gpu_or_training=True,argv_sha256=cp.object_sha(identity['argv'])))
            registry=Path(cfg['gates']['scoped_cpu_processes_path']);registry.parent.mkdir(parents=True)
            cp.exclusive_json(registry,dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,
                run_id=cfg['run_id'],config_sha256=cfg['_config_sha256'],roles=roles))
            with patch.object(gates,'process',side_effect=lambda pid:actuals[pid]),patch.object(gates,'inventory',return_value=([],[])):
                proof=gates.scoped_cpu_processes(cfg);self.assertEqual(len(proof['identities']),2)
                actuals[100]=dict(actuals[100],argv=['/python','foreign.py'])
                with self.assertRaises(gates.Hold):gates.scoped_cpu_processes(cfg)
                actuals[100]=roles[0]['identity']
            with patch.object(gates,'process',side_effect=lambda pid:actuals[pid]),\
                 patch.object(gates,'inventory',return_value=([],[dict(pid=100)])):
                with self.assertRaisesRegex(gates.Hold,'CPU role argv/GPU'):gates.scoped_cpu_processes(cfg)

    def test_http_proxy_original_object_args_exception_and_private_logging(self):
        response=types.SimpleNamespace(status_code=200,text='{"result":[]}',json=lambda:{'result':[]})
        original=types.SimpleNamespace(post=lambda *a,**k:response);handle=io.StringIO()
        proxy=sampling.RequestsTraceProxy(original,handle,0,'model')
        args=('http://oracle/retrieve',);kw=dict(json={'queries':['synthetic fixture'], 'topk':3,'return_scores':True},timeout=300)
        with patch.object(original,'post',return_value=response) as call:
            self.assertIs(proxy.post(*args,**kw),response);call.assert_called_once_with(*args,**kw)
        record=json.loads(handle.getvalue());self.assertEqual(record['response_json'],{'result':[]})
        exc=RuntimeError('native failure');original.post=lambda *a,**k:(_ for _ in ()).throw(exc)
        with self.assertRaises(RuntimeError) as caught:proxy.post(*args,**kw)
        self.assertIs(caught.exception,exc)

    def test_cpu_merge_original_bf16_method_has_no_parent_cuda_or_extra_cast(self):
        import torch
        target=sys.modules[PACKAGE+'.targets'];cp=sys.modules[PACKAGE+'.checkpoint']
        weights={f'lora_{kind}.layer{i}.weight':torch.ones(1,1) if kind=='A' else torch.zeros(1,1)
                 for i in range(252) for kind in ('A','B')}
        digest=target.tensor_digest(weights)
        for variant in ('G_POS','H_NEG'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);adapter=root/'adapter';adapter.mkdir()
                (adapter/'adapter_model.safetensors').write_bytes(b'fixture-only-payload')
                (adapter/'adapter_config.json').write_text('{}')
                files=[cp.fileproof(p) for p in sorted(adapter.iterdir())]
                initial=dict(payload_files=files,adapter_model=next(f for f in files if f['path'].endswith('.safetensors')),
                             adapter_tensor_sha256=digest)
                initial_ref=cp.exclusive_json(root/'initial.json',initial)
                cfg=prepare.template(str(root),variant,'study',str(root))
                cfg.update(common_initial_adapter_manifest=initial_ref,source_model_manifest={'sha256':'a'*64})
                calls=[]
                class Base:
                    def __init__(self,dtype):self.parameter=torch.ones(2,2,dtype=dtype)
                    def parameters(self):return [self.parameter]
                    def to(self,dtype):calls.append(('cast',dtype));self.parameter=self.parameter.to(dtype);return self
                    def save_pretrained(self,path,**kw):
                        calls.append(('save',kw));Path(path).mkdir();(Path(path)/'model.safetensors').write_bytes(b'fixture-only-merged')
                def load_base(*a,**kw):calls.append(('base',a,kw));return Base(kw['torch_dtype'])
                class Peft:
                    def __init__(self,base):self.base=base
                    def parameters(self):return self.base.parameters()+list(weights.values())
                    def merge_and_unload(self):calls.append(('merge',));return self.base
                def load_peft(*a,**kw):calls.append(('peft',a[1:],kw));return Peft(a[0])
                tokenizer=types.SimpleNamespace(save_pretrained=lambda path:(Path(path)/'tokenizer.json').write_text('{}'))
                transformers=types.ModuleType('transformers')
                transformers.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_base)
                transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:tokenizer)
                peft=types.ModuleType('peft');peft.PeftModel=types.SimpleNamespace(from_pretrained=load_peft)
                peft.get_peft_model_state_dict=lambda model:weights
                with patch.dict(sys.modules,{'transformers':transformers,'peft':peft}),\
                     patch.dict(sampling.os.environ,{'CUDA_VISIBLE_DEVICES':''}),\
                     patch.object(sampling,'bind_model_files',return_value={'path':'/source','files':files}),\
                     patch.object(sampling,'verify_model_metadata',return_value=True):
                    result=sampling.merge_latest(cfg,None,root/'merged',None)
                self.assertEqual(sum(c[0]=='merge' for c in calls),1)
                self.assertEqual(sum(c[0]=='cast' for c in calls),0)
                self.assertEqual(next(c for c in calls if c[0]=='peft')[2],{})
                self.assertEqual(result['device_proof']['before']['visible_cuda_devices'],0)
                self.assertFalse(result['device_proof']['after']['cuda_initialized'])
                self.assertEqual(result['adapter_tensor_sha256'],digest)

    def test_cpu_merge_rejects_nonempty_visibility_before_model_import(self):
        with patch.dict(sampling.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'empty CUDA'):
                sampling.merge_latest({},None,'/unused',None)

    def test_teacher_selection_mutually_exclusive_stable_bounded_and_no_success_na(self):
        pairs=[]
        for i in range(20):
            pairs.append(dict(question_id='synthetic-%02d'%i,reference_reward=1,reference='successful B',student='D',
                mining=dict(self_pair_by_sibling=i%3==0,pair_class='CW' if i%3==1 else 'CC')))
        before=random.getstate();selection=teacher.select_cases(pairs,0)
        self.assertEqual(random.getstate(),before);self.assertEqual(selection['count'],8)
        self.assertEqual(set(selection['selected_strata']),{'CC','CW','self_pair'})
        self.assertEqual(sum(selection['available_strata'].values()),20)
        self.assertEqual(len({x['question_id'] for x in selection['selected']}),8)
        reordered=teacher.select_cases(list(reversed(pairs)),0)
        self.assertEqual([x['question_id'] for x in selection['selected']],[x['question_id'] for x in reordered['selected']])
        selected_index=selection['selected'][0]['pair_index'];pairs[selected_index]['reference']='changed B'
        self.assertNotEqual(teacher.select_cases(pairs,0),selection)
        na=teacher.select_cases([],0);self.assertTrue(na['not_applicable']);self.assertEqual(na['count'],0)
        pairs[0]['reference_reward']=0
        with self.assertRaises(ValueError):teacher.select_cases(pairs,0)

    def test_teacher_scoped_privileged_prompt_transparent_objects_and_exception(self):
        generated=object();encoded=[1,2];tokenized=types.SimpleNamespace(input_ids=[3,4]);response=types.SimpleNamespace(
            status_code=200,text='{"result":[]}',json=lambda:{'result':[]})
        calls=[]
        class Model:
            def generate(self,*a,**kw):calls.append((a,kw));return generated
        model=Model();tokenizer=types.SimpleNamespace(__unused=True,encode=lambda *a,**kw:encoded,decode=lambda *a,**kw:'decoded')
        class Tokenizer:
            def __call__(self,*a,**kw):return tokenized
            def encode(self,*a,**kw):return encoded
            def decode(self,*a,**kw):return 'decoded'
        tokenizer=Tokenizer();agent=types.SimpleNamespace(model=model,tokenizer=tokenizer)
        prompt=lambda question,reference=None:[question,reference]
        requests=types.SimpleNamespace(post=lambda *a,**kw:response);native=types.SimpleNamespace(prompt=prompt,requests=requests)
        before=random.getstate();handle=io.StringIO()
        with teacher.capture(agent,native,dict(question='synthetic Q',reference='full actual B'),handle) as trace:
            self.assertEqual(native.prompt('synthetic Q'),['synthetic Q','full actual B'])
            self.assertIs(agent.tokenizer('context',return_tensors='pt'),tokenized)
            self.assertIs(agent.tokenizer.encode('obs'),encoded)
            args=(tokenized,);kw=dict(max_new_tokens=500,do_sample=False)
            self.assertIs(model.generate(*args,**kw),generated);self.assertEqual(calls[-1],(args,kw))
            self.assertIs(native.requests.post('oracle',json={'queries':['fixture'],'topk':3},timeout=120),response)
        self.assertIs(agent.tokenizer,tokenizer);self.assertIs(native.prompt,prompt);self.assertIs(native.requests,requests)
        self.assertNotIn('generate',vars(model));self.assertEqual(random.getstate(),before)
        self.assertFalse(trace['logging_errors']);self.assertEqual(json.loads(handle.getvalue())['response_json'],{'result':[]})
        original_exception=RuntimeError('original generate failure')
        model.generate=lambda *a,**kw:(_ for _ in ()).throw(original_exception)
        with self.assertRaises(RuntimeError) as caught:
            with teacher.capture(agent,native,dict(question='synthetic Q',reference='full actual B'),io.StringIO()):model.generate()
        self.assertIs(caught.exception,original_exception);self.assertIs(native.prompt,prompt)
        # Snapshot errors are rejected as incomplete evidence after execution, not substituted for its return object.
        model.generate=lambda *a,**kw:generated
        class BrokenSnapshot:
            def detach(self):raise OSError('fixture logging failure')
        with teacher.capture(agent,native,dict(question='synthetic Q',reference='full actual B'),io.StringIO()) as incomplete:
            self.assertIs(model.generate(BrokenSnapshot()),generated)
        self.assertEqual(incomplete['logging_errors'],['OSError'])

if __name__=='__main__':unittest.main()
