"""Whole original dump vs staged blocks on CPU adapters; never native/GPU acceptance."""
import argparse,ast,hashlib,importlib.util,json,os,pathlib,re,sys,time,types
import numpy as np
import pandas as pd
import pytest
import torch
HERE=pathlib.Path(__file__).parent
sys.path.insert(0,str(HERE))
from original_blocks import execute,restore,trace,source,DUMP_SHA,blocks,block_hashes
from gates import conditional_base64
from publication import complete_dump,publish
class Batch(dict):
    def to(self,device):return self
class Tokenizer:
    pad_token=None;eos_token='EOS'
    def __call__(self,texts,return_tensors=None,**kwargs):
        if isinstance(texts,str):return {'input_ids':[ord(x)%128 for x in texts][:4096]}
        rows=[[ord(x)%128 for x in text][:4096] for text in texts];width=max(map(len,rows))
        return Batch(input_ids=torch.tensor([[0]*(width-len(row))+row for row in rows]),attention_mask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in rows]))
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(17);self.embed=torch.nn.Embedding(128,4);self.layers=torch.nn.ModuleList([torch.nn.Linear(4,4) for _ in range(28)])
    def forward(self,input_ids,attention_mask,use_cache):
        h=self.embed(input_ids)
        for layer in self.layers:h=layer(h)
        return h
class Model:
    def __init__(self):self.model=Base()
    def eval(self):self.model.eval();return self
class Params:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)
class Engine:
    def __init__(self,**kwargs):self.calls=[]
    def generate(self,prompts,params,use_tqdm):
        self.calls.append((prompts,dict(params.__dict__)));out=[]
        for i,p in enumerate(prompts):
            if params.max_tokens==24:text='gold' if i%2 else 'wrong'
            elif params.max_tokens==32:text=('wrong' if i%2 else 'gold')+'</answer>'
            elif '<information>' not in p:text='<search>fixture</search>'
            else:text='<answer>gold</answer>'
            out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]))
        return out
class Session:
    trust_env=False
    def post(self,url,json,timeout):return types.SimpleNamespace(status_code=200,json=lambda:{'result':[['retrieved gold'] for _ in json['queries']]})
def original_namespace():
    rows=pd.DataFrame([{'prompt':[{'content':f'Question: q{i}'}],'reward_model':{'ground_truth':{'target':['gold']}}} for i in range(6)])
    proxy=types.SimpleNamespace(no_grad=torch.no_grad,device=lambda name:torch.device('cpu'),bfloat16=torch.bfloat16)
    ns={'__name__':'fixture_original','argparse':argparse,'json':json,'os':os,'re':re,'time':time,'Path':pathlib.Path,'np':np,'pd':types.SimpleNamespace(read_parquet=lambda path:rows),'torch':proxy,'requests':types.SimpleNamespace(Session=Session),'LLM':Engine,'SamplingParams':Params,'AutoTokenizer':types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),'AutoModelForCausalLM':types.SimpleNamespace(from_pretrained=lambda *a,**k:Model())}
    tree=source(HERE/'original/ts_probe_dump.py');tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.If))]
    exec(compile(tree,'actual-original-dump-source','exec'),ns);return ns

def test_whole_original_dump_equals_two_stages_full_schema(tmp_path,monkeypatch):
    original=tmp_path/'original';staged=tmp_path/'staged';original.mkdir();staged.mkdir()
    monkeypatch.setattr(sys,'argv',['dump','--model','fixture','--data','fixture','--tag','nq','--cap','6','--G','5','--out',str(original)])
    raw=original_namespace();raw['main']()
    args=types.SimpleNamespace(model='fixture',data='fixture',tag='nq',cap=6,offset=0,G=5,layers='21,24,27',max_turns=4,gpu_mem=.45,out=str(staged))
    A=original_namespace();A.update(args=args,layers=[21,24,27],llm=Engine(),t0=time.time());execute(HERE/'original/ts_probe_dump.py','prepare',A);execute(HERE/'original/ts_probe_dump.py','generation',A)
    public=trace(A);encoded=json.dumps(public,sort_keys=True);decoded=json.loads(encoded)
    B=original_namespace();restore(decoded,args,B);B.update(tok=Tokenizer(),hf=Model().eval(),device=torch.device('cpu'),t0=time.time());execute(HERE/'original/ts_probe_dump.py','hidden',B);execute(HERE/'original/ts_probe_dump.py','assembly',B)
    with np.load(original/'nq_0.npz',allow_pickle=False) as one,np.load(staged/'nq_0.npz',allow_pickle=False) as two:
        assert set(one.files)==set(two.files)=={'y_em','y_flip','y_flip2','q_cb','gid','ds'}|{f'{kind}_{L}' for kind in ('Xc','Xa','Xp') for L in (21,24,27)}
        for key in one.files:np.testing.assert_array_equal(one[key],two[key]);assert one[key].dtype==two[key].dtype
    assert (original/'nq_0.jsonl').read_bytes()==(staged/'nq_0.jsonl').read_bytes()
    assert len(decoded['gens'])==30 and len(decoded['flips'])==30;assert set(decoded['flips'])=={0,1};assert set(decoded['q_cb'])=={0,1}

def test_all_blocks_are_exact_original_ast_and_hash_bound():
    assert set(block_hashes(HERE/'original/ts_probe_dump.py'))=={'prepare','generation','hidden','assembly'}
    assert blocks(HERE/'original/ts_probe_dump.py')['generation'][-1].lineno<blocks(HERE/'original/ts_probe_dump.py')['hidden'][0].lineno

def test_wrong_original_source_rejected(tmp_path):
    p=tmp_path/'dump';p.write_bytes((HERE/'original/ts_probe_dump.py').read_bytes()+b'\n')
    with pytest.raises(ValueError):source(p)

@pytest.mark.parametrize('case',['missing','badcount','badlabel','badorder'])
def test_trace_corruption_blocks_hidden(case,tmp_path):
    ns=original_namespace();args=types.SimpleNamespace(data='fixture',offset=0,cap=6,G=5,max_turns=4,out=str(tmp_path),model='fixture',tag='nq');ns.update(args=args,layers=[21,24,27],llm=Engine(),t0=time.time());execute(HERE/'original/ts_probe_dump.py','prepare',ns);execute(HERE/'original/ts_probe_dump.py','generation',ns);t=trace(ns)
    if case=='missing':t.pop('flips')
    elif case=='badcount':t['cb_ans'].pop()
    elif case=='badlabel':t['flips'][0]=.2
    else:t['valid'].reverse()
    with pytest.raises(ValueError):restore(t,args,{})

def quality(tmp_path,correct):
    rows=[{'uid':str(i),'sample_position':i,'em':int(i<correct)} for i in range(64)];qa=tmp_path/'qa';qa.write_text('\n'.join(json.dumps(row) for row in rows));qh=hashlib.sha256(qa.read_bytes()).hexdigest();r=tmp_path/'result';r.write_text(json.dumps({'status':'COMPLETE64_UNREVIEWED_NO_TRAINING','qa_sha256':qh,'quality':{'correct':correct,'next_path':'SEARCH_G1_GRPO100' if correct<26 else 'TOPD_TEACHER_BACKEND_VALIDATION'}}));return {'result_path':str(r),'qa_path':str(qa),'result_sha256':hashlib.sha256(r.read_bytes()).hexdigest(),'qa_sha256':qh}
@pytest.mark.parametrize('correct',[0,25])
def test_actual_base64_lower_branch(correct,tmp_path):assert conditional_base64(quality(tmp_path,correct))['correct']==correct
@pytest.mark.parametrize('correct',[26,64])
def test_actual_base64_approved_teacher_no_g1(correct,tmp_path):
    with pytest.raises(ValueError):conditional_base64(quality(tmp_path,correct))
def test_changed_base64_bytes_rejected(tmp_path):
    b=quality(tmp_path,25);pathlib.Path(b['qa_path']).write_text('forged')
    with pytest.raises(ValueError):conditional_base64(b)

def test_failed_stage_cannot_publish_old_manifest(tmp_path):
    live=tmp_path/'manifest';live.write_bytes(b'old-complete');(tmp_path/'result.json').write_text(json.dumps({'status':'FAILED_NO_PUBLICATION','complete':False}))
    with pytest.raises(ValueError):publish(tmp_path,live,tmp_path/'heads',{'returncode':0})
    assert live.read_bytes()==b'old-complete'

def test_original_train_and_coevolve_bytes_unchanged():
    bindings=json.loads((HERE/'original-source-bindings.json').read_text())
    for name,digest in bindings.items():assert hashlib.sha256((HERE/'original'/name).read_bytes()).hexdigest()==digest

def complete_fixture(tmp_path):
    from publication import sha
    (tmp_path/'dump').mkdir();dump=tmp_path/'dump/nq_0.npz';dump.write_bytes(b'fixture-array');dump.with_suffix('.jsonl').write_bytes(b'fixture-jsonl')
    config={'dump_args':{'tag':'nq','offset':0,'model':'original-checkpoint'},'outer_step':2}
    p=tmp_path/'frozen-input-config.json';p.write_text(json.dumps(config));ch=sha(p)
    result={'complete':True,'status':'DUMP_COMPLETE_HARDWARE_NOT_INDEPENDENTLY_ACCEPTED','config_sha256':ch,'npz_sha256':sha(dump),'jsonl_sha256':sha(dump.with_suffix('.jsonl'))}
    (tmp_path/'result.json').write_text(json.dumps(result))
    exit_={'returncode':0,'own_processes_remaining':[],'own_gpu_records_remaining':[],'baseline_preserved':True,'original_global_gpu_baseline_preserved':True,'config_sha256':ch}
    for phase in 'AB':(tmp_path/f'phase-{phase}-owner-barrier.json').write_text(json.dumps(exit_))
    heads=tmp_path/'heads';heads.mkdir()
    for name in ('commit_probe.pkl','boundary_probe.pkl'):(heads/name).write_bytes(b'fixture-original-head')
    live=tmp_path/'manifest';live.write_bytes(b'old-complete')
    return live,heads,exit_

def test_success_publication_receipt_precedes_live_manifest(tmp_path,monkeypatch):
    import publication
    live,heads,exit_=complete_fixture(tmp_path);replace=publication.os.replace
    def checked(source,dest):
        assert (tmp_path/'calibration-publication-receipt.json').is_file()
        replace(source,dest)
    monkeypatch.setattr(publication.os,'replace',checked)
    receipt=publish(tmp_path,live,heads,exit_)
    assert publication.sha(live)==receipt['manifest_file_sha256']
    assert json.loads(live.read_text())['round']==2

@pytest.mark.parametrize('failure',['evidence','after_rename'])
def test_publication_fsync_failure_keeps_last_complete_manifest(failure,tmp_path,monkeypatch):
    import publication
    live,heads,exit_=complete_fixture(tmp_path);fsync=publication.os.fsync;calls=[]
    def injected(fd):
        calls.append(fd)
        if len(calls)==(1 if failure=='evidence' else 3):raise OSError('injected fsync failure')
        fsync(fd)
    monkeypatch.setattr(publication.os,'fsync',injected)
    with pytest.raises(OSError):publish(tmp_path,live,heads,exit_)
    assert live.read_bytes()==b'old-complete'

@pytest.mark.parametrize('failure',['heads','barrier','dump'])
def test_incomplete_actual_inputs_never_publish(failure,tmp_path):
    live,heads,exit_=complete_fixture(tmp_path)
    if failure=='heads':(heads/'commit_probe.pkl').unlink()
    elif failure=='barrier':
        exit_['own_processes_remaining']=[123];(tmp_path/'phase-B-owner-barrier.json').write_text(json.dumps(exit_))
    else:(tmp_path/'dump/nq_0.npz').write_bytes(b'changed')
    with pytest.raises((ValueError,KeyError)):publish(tmp_path,live,heads,exit_)
    assert live.read_bytes()==b'old-complete'

def test_original_refit_failure_does_not_call_fit_or_publish(tmp_path):
    from refit_with_owner import refit_with_owner
    args=types.SimpleNamespace(work_dir=str(tmp_path/'work'),manifest_root=str(tmp_path/'manifests'),data='fixture',tag='nq',nq=6,holdout_offset=0,group_size=5,layers='21,24,27',gpu_mem=.45,commit_position='Xa',commit_label='flip')
    context={'evidence_root':str(tmp_path/'evidence'),'support_sha256':'fixture','config':{'leases':[]}}
    calls=[]
    def failed(command,**kwargs):calls.append(command);return types.SimpleNamespace(returncode=9)
    ok,records=refit_with_owner('search_g1',2,'fixturecheckpoint',args,{},context,runner=failed)
    assert not ok and len(calls)==1 and '--dump-only' in calls[0]
    assert not (tmp_path/'manifests/live_search_g1/manifest.json').exists()
    assert records[0]['returncode']==9

def test_mapping_is_all64_whole_layers_and_main_components():
    from stage_worker import mapping
    m=mapping();assert len([k for k in m if k.startswith('model.layers.')])==64
    assert all(m[k]=='cuda:0' for k in ('model.embed_tokens','model.rotary_emb','model.norm','lm_head'))
    assert [sum(v==f'cuda:{i}' for k,v in m.items() if k.startswith('model.layers.')) for i in range(8)]==[6,9,9,8,8,8,8,8]

def test_original_refit_completed_dump_then_missing_heads_does_not_publish(tmp_path,monkeypatch):
    import publication
    from refit_with_owner import refit_with_owner
    args=types.SimpleNamespace(work_dir=str(tmp_path/'work'),manifest_root=str(tmp_path/'manifests'),data='fixture',tag='nq',nq=6,holdout_offset=0,group_size=5,layers='21,24,27',gpu_mem=.45,commit_position='Xa',commit_label='flip')
    context={'evidence_root':str(tmp_path/'evidence'),'support_sha256':'fixture','config':{'leases':[]}}
    artifact=tmp_path/'completed.npz';artifact.write_bytes(b'fixture');artifact.with_suffix('.jsonl').write_bytes(b'fixture-jsonl')
    monkeypatch.setattr(publication,'complete_dump',lambda directory:({}, {},artifact))
    calls=[]
    def successful(command,**kwargs):calls.append(command);return types.SimpleNamespace(returncode=0)
    with pytest.raises(ValueError,match='missing original fitted heads'):refit_with_owner('search_g1',2,'fixturecheckpoint',args,{},context,runner=successful)
    assert len(calls)==2 and calls[1][1]==str(HERE/'original/ts_probe_train.py')
    assert '--commit-position' in calls[1] and calls[1][calls[1].index('--commit-position')+1]=='Xa'
    assert not (tmp_path/'manifests/live_search_g1/manifest.json').exists()

def test_failed_original_refit_after_manifest_write_restores_old_bytes(tmp_path,monkeypatch):
    import refit_with_owner as router
    manifest=tmp_path/'manifests/live_search_g1/manifest.json';manifest.parent.mkdir(parents=True);manifest.write_bytes(b'old-live')
    def failed(*args):manifest.write_bytes(b'new-incomplete');raise OSError('original migration log failure')
    monkeypatch.setattr(router,'load_original_coevolve',lambda:types.SimpleNamespace(refit=failed))
    args=types.SimpleNamespace(manifest_root=str(tmp_path/'manifests'))
    with pytest.raises(OSError):router.refit_with_owner('search_g1',2,'checkpoint',args,{}, {},runner=lambda *a,**k:None)
    assert manifest.read_bytes()==b'old-live'

@pytest.mark.parametrize('timeout',[False,True])
def test_refit_owner_reaps_real_cpu_child_and_keeps_partial(timeout,tmp_path,monkeypatch):
    import subprocess
    import owner,refit_with_owner
    children=[]
    class OwnedCPUFixture:
        baseline={os.getpid()}
        cleanup_record=None
        def __init__(self,*args):self.stage_scope=types.SimpleNamespace(pids=lambda scope:{p.pid for p in children if p.poll() is None},ledger={})
        def spawn(self,command,log,env,pass_fds):
            child=subprocess.Popen(command,env=env,pass_fds=pass_fds);children.append(child);self.stage_scope.ledger[child.pid]={'pid':child.pid,'starttime_ticks':child.pid}
            limit=time.monotonic()+2
            while not (tmp_path/'partial').exists() and time.monotonic()<limit:time.sleep(.01)
            return child
        def cleanup(self):
            for child in children:
                if child.poll() is None:child.terminate()
                child.wait(timeout=3)
            self.cleanup_record={'fixture_cpu_only':True,'reaped':True,'actual_cgroup_empty':True,'remaining':[]}
    tools=types.SimpleNamespace(OwnedProcesses=OwnedCPUFixture,cgroup_scope=lambda job:(tmp_path,'fixture'),cgroup_pids=lambda scope:{os.getpid()}|{p.pid for p in children if p.poll() is None},process_identity=lambda pid:{'pid':pid,'starttime_ticks':pid},gpu_sample=lambda:{'processes':[]})
    monkeypatch.setattr(owner,'support',lambda *args:tools);monkeypatch.setenv('SLURM_JOB_ID','fixture')
    monkeypatch.setattr(refit_with_owner,'owned_stage',lambda *args,**kwargs:OwnedCPUFixture())
    monkeypatch.setattr(refit_with_owner,'protected_stage_baseline',lambda *args,**kwargs:{os.getpid():dict(tools.process_identity(os.getpid()),uid=os.getuid())})
    from stage_baseline import continuous_baseline_missing
    monkeypatch.setattr(refit_with_owner,'continuous_baseline_missing',lambda t,b:continuous_baseline_missing(t,b,read=lambda pid:dict(t.process_identity(pid),uid=os.getuid())))
    context={'support_sha256':'fixture','evidence_root':str(tmp_path/'evidence'),'config':{'job':'fixture','cgroup':'fixture','stage_timeout_seconds':-1 if timeout else 3}}
    script="from pathlib import Path; import time; Path(%r).write_bytes(b'partial-kept'); time.sleep(%s)"%(str(tmp_path/'partial'),'30' if timeout else '0')
    if timeout:
        with pytest.raises(TimeoutError):refit_with_owner.owned_command([sys.executable,'-c',script],dict(os.environ),context)
    else:assert refit_with_owner.owned_command([sys.executable,'-c',script],dict(os.environ),context).returncode==0
    assert children[0].poll() is not None and (tmp_path/'partial').read_bytes()==b'partial-kept'
    receipt=json.loads(next((tmp_path/'evidence').glob('*.json')).read_text())
    assert receipt['own_processes_remaining']==[] and receipt['baseline_preserved']

def test_input_data_and_alias_drift_fail_before_original_prepare(tmp_path,monkeypatch):
    import stage_worker
    monkeypatch.setattr(stage_worker,'ROOT',tmp_path);model=tmp_path/'assets/model';model.mkdir(parents=True);data=tmp_path/'data';data.write_bytes(b'original');out=tmp_path/'logs/run';out.mkdir(parents=True)
    cfg={'dump_args':{'model':str(model),'data':str(data)},'canonical_model':str(model.resolve()),'canonical_data':str(data.resolve()),'data_sha256':stage_worker.sha(data),'model_files':{}}
    stage_worker.validate_inputs(cfg,out);data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='data bytes'):stage_worker.validate_inputs(cfg,out)
    data.write_bytes(b'original');alias=tmp_path/'alias';alias.symlink_to(data);cfg['dump_args']['data']=str(alias)
    with pytest.raises(ValueError):stage_worker.validate_inputs(cfg,out)

def test_actual_global_gpu_baseline_pid_reuse_or_uuid_loss_rejected():
    import owner
    tools=types.SimpleNamespace(process_identity=lambda pid:{'pid':pid,'starttime_ticks':42})
    sample={'processes':['GPU-fixture, 5, 100']};records=owner.gpu_baseline(tools,sample)
    assert owner.baseline_gpu_missing(tools,sample,records)==[]
    assert owner.baseline_gpu_missing(tools,{'processes':['GPU-other, 5, 100']},records)==records
    tools.process_identity=lambda pid:{'pid':pid,'starttime_ticks':43}
    assert owner.baseline_gpu_missing(tools,sample,records)==records
