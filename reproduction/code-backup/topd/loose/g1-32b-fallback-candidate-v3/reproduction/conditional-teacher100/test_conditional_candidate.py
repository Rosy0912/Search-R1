"""Execute actual isolated source methods with CPU adapters, no Ray/CUDA imports."""
import ast, hashlib, importlib.util, json, pathlib, re, subprocess, threading, time, types
import numpy as np
import pytest
import requests
import torch
ROOT=pathlib.Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
contracts=module('contracts','search_r1/g1_runtime_contracts.py');qa=module('qa','search_r1/qa_protocol.py');em=module('em','verl/utils/reward_score/qa_em.py')
def definitions(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text());nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),'exec'),namespace);return namespace

def reward_class():
    ns={'DataProto':object,'torch':torch,'np':np,'re':re,'requests':requests,'qa_em':em,'decode_model_spans':qa.decode_model_spans,'extract_final_answer':qa.extract_final_answer,'parse_model_action':qa.parse_model_action,'probe_payload':contracts.probe_payload,'validate_probe_response':contracts.validate_probe_response,'_SEARCH_RE':re.compile('<search>'),'_INFO_END_RE':re.compile('</information>'),'_INFO_BLOCK_RE':re.compile('<information>(.*?)</information>',re.S),'_QUESTION_RE':re.compile('Question:\\s*(.+?)\\s*(?:<\\|im_end\\|>|\\n|$)',re.S)}
    definitions('verl/trainer/main_ppo_eco.py',['EcoRewardManager','_causal_trust','_extract_info_raw','_extract_question'],ns);return ns['EcoRewardManager']
class Tok:
    def decode(self,ids):return ''.join(chr(int(x)) for x in ids)
class Data:
    def __init__(self,response,mask=None):
        prompt='Question: q\n<answer>gold</answer> prompt example';ids=torch.tensor([ord(x) for x in response]);plen=len(prompt)
        self.sample=types.SimpleNamespace(batch={'prompts':torch.tensor([ord(x) for x in prompt]),'responses':ids,'attention_mask':torch.ones(plen+len(ids),dtype=torch.long),'info_mask':torch.tensor([1]*plen+(mask or [1]*len(ids)))},non_tensor_batch={'reward_model':{'ground_truth':{'target':['gold']}},'data_source':'fixture'})
        self.batch={'responses':ids[None]}
    def __len__(self):return 1
    def __getitem__(self,i):return self.sample
@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv('ECO_USE_CF','0');monkeypatch.setenv('ECO_TRUST_SIGNAL','probe');monkeypatch.setenv('ECO_TRUST_VARIANT','balanced');monkeypatch.setenv('ECO_WRONG_GROUND_SCALE','0')
    instance=reward_class()(Tok(),0,reward_mode='eco');instance._probe_call=lambda rows:[{'flip':.8,'boundary':.4} for row in rows];return instance
@pytest.mark.parametrize('mini,micro,world',[(32,64,8),(255,64,8),(256,65,8)])
def test_batch_wrong_actual_guard(mini,micro,world):
    with pytest.raises(ValueError):contracts.validate_batch_layout(mini,micro,world)
def test_global64_rank8():assert contracts.validate_batch_layout(256,64,8)==(32,8)
def test_observation_answer_never_rewarded(manager):
    search='<search>x</search>';obs='<information><answer>gold</answer></information>'
    output=manager(Data(search+obs,[1]*len(search)+[0]*len(obs)));assert output.sum().item()==pytest.approx(-.1)
def test_search_markup_observation_excluded_and_features_unchanged(manager):
    search='<search>x</search>';obs='<information><search>fake</search> gold</information>';answer='<answer>gold</answer>';seen=[]
    manager._probe_call=lambda rows:(seen.extend(rows) or [{'flip':.8,'boundary':.4}])
    output=manager(Data(search+obs+answer,[1]*len(search)+[0]*len(obs)+[1]*len(answer)))
    assert output.sum().item()==pytest.approx(1+.6*(1-.4)*.8);assert manager.last_metrics['cost/search_per_query']==1;assert seen[0]['response']==search+obs+answer
@pytest.mark.parametrize('text',['<answer>gold</answer> junk','<think><answer>gold</answer></think>','<answer>wrong</answer><answer>gold</answer>'])
def test_invalid_terminal_real_reward(manager,text):assert manager(Data(text)).sum().item()==pytest.approx(-.1)
def test_cross_observation_tag_join_rejected(manager):
    before='<answer>';obs='gold';after='</answer>';assert manager(Data(before+obs+after,[1]*len(before)+[0]*len(obs)+[1]*len(after))).sum().item()==pytest.approx(-.1)
def test_direct_source_formula(manager):assert manager(Data('<answer>gold</answer>')).sum().item()==pytest.approx(1+.5*.4)
def test_score_probe_core_ast_unchanged():
    original=(ROOT/'reproduction/conditional-teacher100/original-main_ppo_eco.py').read_text()
    def method(s):return next(node for cls in ast.parse(s).body if isinstance(cls,ast.ClassDef) and cls.name=='EcoRewardManager' for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_score_probe')
    assert ast.dump(method(original))==ast.dump(method((ROOT/'verl/trainer/main_ppo_eco.py').read_text()))
def payload(tmp_path):
    p=tmp_path/'manifest';p.write_text(json.dumps({'round':25,'encoder':'frozen25','commit':'c','boundary':'b'}));return contracts.probe_payload([{'prompt':'p','response':'r'},{'prompt':'q','response':'s'}],p)
@pytest.mark.parametrize('failure',['short','order','nan','range','identity'])
def test_invalid_readout_actual_validator(tmp_path,failure):
    p=payload(tmp_path);r={'identity':dict(p['identity']),'results':[{'row_id':x['row_id'],'flip':.8,'boundary':.4} for x in p['items']]}
    if failure=='short':r['results'].pop()
    elif failure=='order':r['results'].reverse()
    elif failure=='nan':r['results'][0]['flip']=float('nan')
    elif failure=='range':r['results'][0]['boundary']=1.1
    else:r['identity']['encoder']='wrong'
    with pytest.raises(ValueError):contracts.validate_probe_response(r,p)
def test_actual_probe_call_rejects_malformed_without_retry(manager,tmp_path,monkeypatch):
    p=tmp_path/'manifest';p.write_text(json.dumps({'round':0,'encoder':'e'}));monkeypatch.setenv('PROBE_MANIFEST_PATH',str(p))
    class Resp:
        def raise_for_status(self):pass
        def json(self):return {'identity':{},'results':[]}
    manager._probe_session=types.SimpleNamespace(post=lambda *a,**k:Resp())
    # Restore actual source method instead of the test's benign-readout adapter.
    with pytest.raises(ValueError):type(manager)._probe_call(manager,[{'prompt':'p','response':'r'}])
def optimizer_method():
    tree=ast.parse((ROOT/'verl/workers/actor/dp_actor.py').read_text());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_optimizer_step');return node
@pytest.mark.parametrize('fail',[False,True])
def test_real_optimizer_step_order_and_finally_cpu(fail):
    events=[];model=torch.nn.Linear(2,1,bias=False);reference=torch.nn.Linear(2,1,bias=False);reference.load_state_dict(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.01);ref=torch.optim.AdamW(reference.parameters(),lr=.01)
    for m in (model,reference):m.weight.grad=torch.ones_like(m.weight)
    funcs=definitions('verl/utils/fsdp_utils.py',['load_fsdp_optimizer','offload_fsdp_optimizer'],{'torch':types.SimpleNamespace(Tensor=torch.Tensor,cuda=types.SimpleNamespace(empty_cache=lambda:None))})
    def load(opt,device_id):events.append('load');funcs['load_fsdp_optimizer'](opt,device_id)
    def offload(opt):events.append('offload');funcs['offload_fsdp_optimizer'](opt)
    actual_step=optimizer.step
    def step():
        events.append('step')
        if fail:raise RuntimeError('injected optimizer failure')
        return actual_step()
    optimizer.step=step
    ns={'torch':types.SimpleNamespace(nn=torch.nn,cuda=types.SimpleNamespace(synchronize=lambda:events.append('sync'),current_device=lambda:'cpu')),'FSDP':type('UnusedFSDP',(),{}),'load_fsdp_optimizer':load,'offload_fsdp_optimizer':offload}
    exec(compile(ast.Module(body=[optimizer_method()],type_ignores=[]),'actual-optimizer-source','exec'),ns)
    self=types.SimpleNamespace(actor_module=model,actor_optimizer=optimizer,config=types.SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}))
    if fail:
        with pytest.raises(RuntimeError):ns['_optimizer_step'](self)
    else:
        ns['_optimizer_step'](self);torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);ref.step();assert torch.equal(model.weight,reference.weight)
    assert events==['sync','load','step','offload'];assert all(v.device.type=='cpu' for st in optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor))
def test_actual_worker_does_not_stage_moments_before_forward():
    tree=ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_text());method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_actor')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_fsdp_optimizer' for n in ast.walk(method))
def test_actual_update_releases_graph_references_before_step():
    tree=ast.parse((ROOT/'verl/workers/actor/dp_actor.py').read_text());method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_policy');lines=(ROOT/'verl/workers/actor/dp_actor.py').read_text();assert lines.index('del entropy, log_prob, policy_loss, loss')<lines.index('grad_norm = self._optimizer_step()')
def test_outer_barrier_failure_blocks_advance(monkeypatch):
    calls=[]
    def run(cmd,check):calls.append(cmd);raise subprocess.CalledProcessError(1,cmd)
    monkeypatch.setattr(contracts.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):contracts.refresh_at_outer_barrier(['python','refresh.py'],'actor/global_step_25',25)
    assert calls==[['python','refresh.py','--checkpoint','actor/global_step_25','--outer-step','25']]
def test_trainer_terminal_checkpoint_and_refresh_are_before_advance():
    source=(ROOT/'verl/trainer/ppo/ray_trainer.py').read_text();tail=source[source.index('terminal_outer='):];assert tail.index('self._save_checkpoint()')<tail.index('self.global_steps += 1');assert tail.index('refresh_at_outer_barrier(refresh_command')<tail.index('self.global_steps += 1');assert 'drop_last=False' in source

def probe_namespace(tmp_path):
    import gc, os
    class App:
        def post(self,*a):return lambda fn:fn
    class HttpError(Exception):
        def __init__(self,*args):super().__init__(*args)
    manifest=tmp_path/'live.json';manifest.write_text(json.dumps({'round':25,'encoder':'new','commit':'c','boundary':'b'}))
    state={'round':0,'encoder':'old','model':None,'tok':None,'commit':None,'boundary':None}
    ns={'STATE':state,'LOCK':threading.Lock(),'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:False)),'gc':gc,'json':json,'os':os,'hashlib':hashlib,'app':App(),'HTTPException':HttpError,'OUTER_BARRIER':True,'MANIFEST':str(manifest)}
    definitions('probe_server.py',['_release_encoder_locked','_apply','barrier_pause','barrier_resume'],ns);return ns,manifest

def test_actual_probe_release_before_reload_no_double_encoder(tmp_path):
    import weakref
    ns,manifest=probe_namespace(tmp_path)
    class Model:pass
    old=Model();ref=weakref.ref(old);ns['STATE']['model']=old;del old
    def load(*args):assert ref() is None;return 'tok',Model(),'commit','boundary'
    ns['_load']=load;ns['_apply'](str(manifest),force=True)
    assert ns['STATE']['encoder']=='new' and ns['STATE']['model'] is not None

def test_actual_probe_reload_waits_for_inflight_lock(tmp_path):
    ns,manifest=probe_namespace(tmp_path);events=[];ns['_load']=lambda *a:(events.append('load') or ('tok',object(),'c','b'))
    ns['LOCK'].acquire();worker=threading.Thread(target=lambda:ns['_apply'](str(manifest),True));worker.start();time.sleep(.02);assert not events;ns['LOCK'].release();worker.join(2);assert not worker.is_alive() and events==['load']

def test_actual_pause_removes_encoder_and_resume_ready(tmp_path):
    ns,manifest=probe_namespace(tmp_path);ns['STATE']['model']=object();ns['_load']=lambda *a:('tok',object(),'c','b')
    import io,builtins
    ns['open']=lambda path,*a,**kw:io.StringIO('0::/cpu-fixture') if path=='/proc/self/cgroup' else builtins.open(path,*a,**kw)
    assert ns['barrier_pause']()['status']=='PAUSED_ENCODER_RELEASED';assert ns['STATE']['model'] is None
    receipt=ns['barrier_resume']();assert receipt['status']=='READY' and receipt['round']==25

def test_actual_refresh_failure_restores_prior_manifest(tmp_path):
    import os
    path=ROOT/'reproduction/conditional-teacher100/refresh_outer_barrier.py'
    ns={'pathlib':pathlib,'hashlib':hashlib,'json':json,'types':types,'os':os,'requests':requests,'refit':None}
    definitions('reproduction/conditional-teacher100/refresh_outer_barrier.py',['refresh'],ns)
    manifest=tmp_path/'live';old=b'{"round":0,"encoder":"old"}';manifest.write_bytes(old);source=ROOT/'ts_probe_dump.py'
    config={'staged_dump_binding':{'scope':'INDEPENDENTLY_REVIEWED_G1_32B_STAGED_DUMP','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'manifest':str(manifest),'probe_url':'http://127.0.0.1:8002/judge_probe','experiment':'fixture','refit_args':{}}
    calls=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'status':'PAUSED_ENCODER_RELEASED'}
    class Session:
        def post(self,url,**kwargs):calls.append(url);return Response()
    def fail(*args):manifest.write_bytes(b'corrupt-new');return False
    original=pathlib.Path.cwd()
    try:
        os.chdir(ROOT)
        with pytest.raises(RuntimeError,match='refit failed'):ns['refresh'](config,'new',25,Session(),fail)
    finally:os.chdir(original)
    assert manifest.read_bytes()==old and calls==['http://127.0.0.1:8002/barrier_pause','http://127.0.0.1:8002/barrier_resume']

def test_actual_executor_uses_terminal_parser():
    tree=ast.parse((ROOT/'search_r1/llm_agent/generation.py').read_text());method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_predictions')
    ns={'List':list,'Any':object,'Tuple':__import__('typing').Tuple,'parse_model_action':qa.parse_model_action}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'actual-generation-method','exec'),ns)
    kinds,content=ns['postprocess_predictions'](None,['<answer>gold</answer>','<think><answer>gold</answer></think>','<search>q</search> trailing'])
    assert kinds==['answer',None,None] and content==['gold','','']

@pytest.mark.parametrize('original_foreach',[None,True])
def test_single_tensor_adamw_matches_original_fp32_multistep(original_foreach):
    # Source optimizer kwargs plus explicit single-tensor candidate; no CUDA claim.
    torch.manual_seed(17)
    original=[torch.nn.Parameter(torch.randn(3,5)),torch.nn.Parameter(torch.randn(7))]
    candidate=[torch.nn.Parameter(x.detach().clone()) for x in original]
    a=torch.optim.AdamW(original,lr=1e-6,betas=(.9,.999),weight_decay=1e-2,foreach=original_foreach)
    b=torch.optim.AdamW(candidate,lr=1e-6,betas=(.9,.999),weight_decay=1e-2,foreach=False)
    for iteration in range(4):
        for x,y in zip(original,candidate):
            grad=torch.randn_like(x);x.grad=grad.clone();y.grad=grad.clone()
        a.step();b.step()
        for x,y in zip(original,candidate):
            torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-7)
            for name in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(a.state[x][name],b.state[y][name],atol=1e-7,rtol=1e-7)

@pytest.mark.parametrize('body_fail',[False,True])
def test_actual_probe_actor_phase_unloads_then_restores_same_encoder(tmp_path,body_fail):
    manifest=tmp_path/'manifest';manifest.write_text(json.dumps({'round':25,'encoder':'frozen25'}));raw=manifest.read_bytes();calls=[]
    class Response:
        def __init__(self,body):self.body=body
        def raise_for_status(self):pass
        def json(self):return self.body
    class Session:
        def post(self,url,**kwargs):
            calls.append(url)
            if url.endswith('pause'):return Response({'status':'PAUSED_ENCODER_RELEASED'})
            return Response({'status':'READY','round':25,'encoder':'frozen25','manifest_sha256':hashlib.sha256(raw).hexdigest()})
    def execute():
        with contracts.probe_training_phase('http://127.0.0.1:8002/judge_probe',manifest,Session()):
            assert calls==['http://127.0.0.1:8002/barrier_pause']
            if body_fail:raise RuntimeError('actor failure')
    if body_fail:
        with pytest.raises(RuntimeError,match='actor failure'):execute()
    else:execute()
    assert calls==['http://127.0.0.1:8002/barrier_pause','http://127.0.0.1:8002/barrier_resume']

def test_critic_initializer_unchanged_from_original():
    source=(ROOT/'verl/workers/fsdp_workers.py').read_text()
    original=(ROOT/'reproduction/conditional-teacher100/original-fsdp_workers.py').read_text()
    def method(text):return next(n for c in ast.parse(text).body if isinstance(c,ast.ClassDef) and c.name=='CriticWorker' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert ast.dump(method(source))==ast.dump(method(original))

def test_actual_update_policy_tail_mini_has_full_mean_gradient():
    # Execute the actual production method with actual source PPO loss and CPU batches.
    masked=definitions('verl/utils/torch_functional.py',['masked_mean'],{'torch':torch})['masked_mean']
    core=definitions('verl/trainer/ppo/core_algos.py',['compute_policy_loss'],{'torch':torch,'verl_F':types.SimpleNamespace(masked_mean=masked)})['compute_policy_loss']
    tree=ast.parse((ROOT/'verl/workers/actor/dp_actor.py').read_text());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='update_policy')
    class Batch(dict):
        def __len__(self):return len(self['responses'])
        def split(self,n):return [Batch({k:v[i:i+n] for k,v in self.items()}) for i in range(0,len(self),n)]
        def cuda(self):return self
    def append(metrics,value):
        for key,x in value.items():metrics.setdefault(key,[]).append(x)
    ns={'DataProto':object,'core_algos':types.SimpleNamespace(compute_policy_loss=core),'verl_F':types.SimpleNamespace(masked_mean=masked),'append_to_dict':append,'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-update-policy','exec'),ns)
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.zero_();opt=torch.optim.SGD(model.parameters(),lr=0.)
    grads=[]
    def forward(micro_batch,temperature):
        logp=model(torch.ones((len(micro_batch),1))*.1);return logp*0,logp
    def step():grads.append(model.weight.grad.clone());return torch.tensor(0.)
    self=types.SimpleNamespace(actor_module=model,actor_optimizer=opt,_forward_micro_batch=forward,_optimizer_step=step,config=types.SimpleNamespace(ppo_mini_batch_size=32,ppo_micro_batch_size=8,use_dynamic_bsz=False,state_masking=False,use_kl_loss=False,clip_ratio=.2,entropy_coeff=0.))
    batch=Batch({'responses':torch.zeros(80,1,dtype=torch.long),'input_ids':torch.zeros(80,2,dtype=torch.long),'attention_mask':torch.ones(80,2),'position_ids':torch.zeros(80,2),'old_log_probs':torch.zeros(80,1),'advantages':torch.ones(80,1)})
    data=types.SimpleNamespace(meta_info={'temperature':1.},select=lambda batch_keys:types.SimpleNamespace(batch=Batch({k:batch[k] for k in batch_keys})))
    ns['update_policy'](self,data)
    assert len(grads)==3
    for grad in grads:torch.testing.assert_close(grad,torch.tensor([[-.1]]),atol=1e-7,rtol=1e-7)
