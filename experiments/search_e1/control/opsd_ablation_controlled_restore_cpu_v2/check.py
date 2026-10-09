#!/usr/bin/env python3
"""CPU controlled actual PEFT restore; CUDA RNG transport is explicitly simulated."""
import argparse, ast, copy, hashlib, json, os, random, stat, sys, time, types
from pathlib import Path

ORIGINAL_COMMIT='871dc46f9ae436e5b4115dd33e51715b858ee57d'
MANIFEST_SHA='8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41'
REPAIR_MANIFEST_SHA='eee5adf6621f85274584412c19be642d8a7e4a8f6f005c84b8a90a972f9803b9'
LIMITS=['CUDA_STATE_TRANSPORT_SIMULATED using eight independent CPU generators, not actual saved CUDA Philox state.',
        'NCCL_NOT_EXECUTED: ranks are checked serially; no distributed liveness/collective acceptance.',
        'SOURCE100_NUMERICS_NOT_COVERED: tiny random actual Qwen2/PEFT, CPU FP32 forward, no production weights.',
        'CPU FP32 arithmetic checks restore/scaler state only, not BF16/FP16 GPU compute or overflow transport.',
        'Actual nested restoration/capture bodies extracted unchanged; trainer.main and full checkpoint writer/atomic commit are not executed.',
        'Shared payload equality proofs are fixture records of one captured actual optimizer; actual eight independently executing pre-save ranks are not represented. Eight disk rank states are subsequently restored separately.']

def require(ok,msg):
    if not ok: raise ValueError(msg)

def read(path,sha):
    p=Path(path); require(p.is_absolute() and '..' not in p.parts and not any(q.is_symlink() for q in (p,*p.parents)),'Canonical nonsymlink input')
    def meta(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
    before=meta(p.stat()); require(stat.S_ISREG(before[-1]),'Regular file required')
    with p.open('rb') as f:
        require(meta(os.fstat(f.fileno()))==before,'Source open changed');raw=f.read();require(meta(os.fstat(f.fileno()))==before,'Source read changed')
    require(meta(p.stat())==before and hashlib.sha256(raw).hexdigest()==sha,'Source identity/SHA differs')
    return raw,dict(path=str(p),sha256=sha,bytes=len(raw),metadata=list(before))

def module(name,raw,path):
    require(name not in sys.modules,'Cached module refused: '+name)
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(raw,str(path),'exec'),m.__dict__);return m

def select(tree,names,namespace,path,proof):
    nodes=[]
    for name in names:
        matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(matches)==1,'Unique actual function required: '+name);n=matches[0];nodes.append(n)
        proof.append(dict(name=name,line=n.lineno,end_line=n.end_lineno,AST_SHA256=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

class Transport:
    """Does not patch real torch.cuda or initialize CUDA."""
    def __init__(self,torch):
        self.generators=[torch.Generator(device='cpu').manual_seed(900+i) for i in range(8)]
    def get_rng_state_all(self):return [g.get_state().clone() for g in self.generators]
    def set_rng_state_all(self,states):
        require(len(states)==8,'Eight explicit simulated states required')
        for g,s in zip(self.generators,states):g.set_state(s)

class TorchFacade:
    def __init__(self,torch):self.actual=torch;self.cuda=Transport(torch)
    def __getattr__(self,name):return getattr(self.actual,name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','source-manifest','publication','publication-sha256','repair-root','repair-publication','repair-publication-sha256','output'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit empty CVD required before Torch import')
    require('torch' not in sys.modules,'Fresh process required')
    out=Path(a.output);require(out.is_absolute() and '..' not in out.parts and not out.exists() and not any(q.is_symlink() for q in out.parents),'New exclusive scratch directory required')
    out.mkdir(mode=0o700);proofs={};extracted=[];started=time.time();rows=[];active_rank=None
    try:
        root=Path(a.source_root);require(root.name=='opsd_ablation_v1','Original namespace required')
        raw,proofs['manifest']=read(a.source_manifest,MANIFEST_SHA);require(Path(a.source_manifest)==root/'source-manifest.json','Root manifest mismatch');entries=json.loads(raw)['files']
        raw,proofs['publication']=read(a.publication,a.publication_sha256);pub=json.loads(raw)
        require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha']==ORIGINAL_COMMIT,'Original actual publication required')
        published={x['path']:x for x in pub['files']};sources={}
        for name in ('checkpoint.py','state_storage.py','precision.py','targets.py','objective.py','trainer.py'):
            entry=entries[name];r=published['experiments/search_e1/control/opsd_ablation_v1/'+name]
            require(r['equal'] is True and r['sha256']==entry['sha256'],'Original publication file differs')
            sources[name],proofs[name]=read(root/name,entry['sha256'])
        repair=Path(a.repair_root);require(repair.name=='opsd_ablation_dtype_recovery_v1','Actual repair namespace required')
        raw,proofs['repair_manifest']=read(repair/'source-manifest.json',REPAIR_MANIFEST_SHA);repair_entries=json.loads(raw)['files']
        raw,proofs['repair_publication']=read(a.repair_publication,a.repair_publication_sha256);rpub=json.loads(raw)
        require(rpub['accepted'] and rpub['source_commit']==rpub['remote_readback_sha'],'Repair published/readback required')
        repaired_public={x['path']:x for x in rpub['files']}
        rname='trainer_launcher.py';rentry=repair_entries[rname]
        require(repaired_public['experiments/search_e1/control/opsd_ablation_dtype_recovery_v1/'+rname]['sha256']==rentry['sha256'] and repaired_public['experiments/search_e1/control/opsd_ablation_dtype_recovery_v1/'+rname]['equal'] is True,'Repair publication differs')
        repair_raw,proofs['repair_setter']=read(repair/rname,rentry['sha256'])
        import torch, numpy as np, peft, transformers
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        require(peft.__version__=='0.13.2','Actual production PEFT0.13.2 required')
        require(not torch.cuda.is_initialized() and torch.cuda.device_count()==0,'No GPU allowed')
        for name in ('checkpoint','state_storage','precision','targets','objective'):module(name,sources[name+'.py'],root/(name+'.py'))
        cp=sys.modules['checkpoint'];storage=sys.modules['state_storage'];precision=sys.modules['precision'];target_module=sys.modules['targets'];objective=sys.modules['objective']
        facade=TorchFacade(torch)
        ns=dict(torch=facade,np=np,random=random,hashlib=hashlib,json=json,Path=Path,os=os,
            RNG_NAMES=['python','numpy','torch_cpu','cuda_all'],science_contract=cp.science_contract)
        tree=ast.parse(sources['trainer.py']);select(tree,['actual_adam_state','verify_rank_state','validate_rng','batches','verify_resume_cursor','capture_rng','restore_rng','tensor_digest','optimizer_tensor_digest','state_to_cpu','save_torch','capture_actual_state','restore_accepted_rank','restore_after_targets'],ns,root/'trainer.py',extracted)
        rns={'require':require};select(ast.parse(repair_raw),['adapter_values','repaired_setter'],rns,repair/rname,extracted)
        modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=modules,task_type='CAUSAL_LM')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        qcfg=Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=36,num_attention_heads=1,num_key_value_heads=1,max_position_embeddings=64,attention_dropout=0.,tie_word_embeddings=False)
        qcfg._attn_implementation='eager';base=Qwen2ForCausalLM(qcfg).float();base_state=copy.deepcopy(base.state_dict())
        model=get_peft_model(base,lora);initial=copy.deepcopy(get_peft_model_state_dict(model));require(len(initial)==504,'Actual504 default tensors required')
        old_digest=ns['tensor_digest'](initial)
        fixture_config={'_config_sha256':'f'*64,'variant':'M','precision':{'mode':'fp32_master_fp16'},'common_initial_adapter_manifest':None,'source_model_manifest':{'sha256':'a'*64},'runtime_manifest':{'sha256':'b'*64}}
        opt={'betas':[.9,.999],'eps':1e-8,'weight_decay':.01};science={'learning_rate':1e-6};run='controlled-cpu-restore-not-production'
        old_payload=out/'outer-old-504-fp32.pt';ns['save_torch'](old_payload,initial,torch);old_payload_ref=cp.fileproof(old_payload)
        old_file=out/'outer-old.json';write(old_file,{'schema':'tiny-actual504-outer-old-cpu-fixture','adapter_tensor_sha256':old_digest,'adapter_payload':old_payload_ref});old_ref=cp.fileproof(old_file);fixture_config['common_initial_adapter_manifest']=old_ref
        encoding={'prefixes':[[1,2],[1,3,4]],'response':[5,6,7],'mask':[1,1,0],'offsets':[(0,1),(1,2),(2,3)]}
        order=list(range(33));order_file=out/'order.json';write(order_file,order);order_ref=cp.fileproof(order_file)
        input_refs={}
        for key in ('outer_manifest','pairs','encoding'):
            f=out/(key+'.json');write(f,{'schema':'tiny-controlled-fixture','name':key});input_refs[key]=cp.fileproof(f)
        input_refs['order']=order_ref
        ns.update(config=fixture_config,science=science,opt=opt,set_peft_model_state_dict=set_peft_model_state_dict,get_peft_model_state_dict=get_peft_model_state_dict)
        def setup(m,old):
            m.add_adapter('outer_old',lora)
            rns['repaired_setter'](set_peft_model_state_dict,ns['tensor_digest'],old_digest,lambda _:None)(m,old,adapter_name='outer_old')
            policy=precision.PrecisionContract(device_type='cpu')
            t=target_module.SharedTargets(m,policy,old_adapter='outer_old',old_identity={'adapter_tensor_sha256':old_digest})
            params=[p for p in m.parameters() if p.requires_grad];require(len(params)==504,'Actual504 optimizer masters required')
            return t,params,torch.optim.AdamW(params,lr=science['learning_rate'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
        targets,trainable,optimizer=setup(model,initial);scaler=precision.SyncedGradScaler(enabled=True,scale=128.)
        def step(m,t,params,optim,scale):
            optim.zero_grad(set_to_none=True);distributions=[]
            with t.current_graph():
                loss,summary=objective.pair_objective(m,encoding,10.,32,1,targets=t,precision=t.precision,use_mixture=True,vocabulary_record=distributions.append,vocabulary_positions=(0,1))
                t.mark_backward();scale.scale_loss(loss).backward()
            grads=[p.grad.detach().clone() for p in params]
            result=scale.finish_step(params,optim,lambda _:None,lambda x:x,lambda _:None)
            require(result['accepted'],'Actual small step rejected');require(len(distributions)==2,'Full fixed target/current rows missing or rematerialization duplicated callback');return float(loss.detach()),grads,distributions
        step(model,targets,trainable,optimizer,scaler)
        ns.update(model=model,trainable=trainable,optimizer=optimizer,updates=1)
        adapter,digest,finite,adam,odigest,steps,_=ns['capture_actual_state']();require(finite and len(adapter)==504,'Finite504 captured')
        # CPU detach().cpu() / state_dict are aliases: freeze fixture reference values
        # before either live optimizer/model can advance. Native capture is unchanged.
        adapter={k:v.detach().cpu().clone() for k,v in adapter.items()};adam=copy.deepcopy(adam)
        require(ns['tensor_digest'](adapter)==digest and ns['optimizer_tensor_digest'](adam)==odigest,'Independent captured snapshot differs')
        model.save_pretrained(out/'adapter',safe_serialization=True,selected_adapters=['default'])
        all_proofs=[dict(rank=r,optimizer_digest=odigest) for r in range(8)]
        shared=out/'optimizer-shared.pt';ns['save_torch'](shared,storage.optimizer_payload(adam,all_proofs,run,fixture_config['_config_sha256'],1,odigest),torch);shared_ref=cp.fileproof(shared)
        cursor=dict(outer_index=0,next_pair_cursor=32,usable_pair_count=33,order_sha256=order_ref['sha256'],encoded_sha256=input_refs['encoding']['sha256'])
        states=[]
        for rank in range(8):
            random.seed(100+rank);np.random.seed(100+rank);torch.manual_seed(100+rank)
            for index,g in enumerate(facade.cuda.generators):g.manual_seed(1000+rank*8+index)
            rng=ns['capture_rng'](facade,np)
            state=dict(schema='opsd-online-rank-state-v1',rank=rank,run_id=run,config_sha256=fixture_config['_config_sha256'],optimizer_storage=storage.STORAGE,optimizer_ref=shared_ref,optimizer_tensor_sha256=odigest,rng=rng,scaler=scaler.state_dict(),contract=cp.science_contract(fixture_config),outer_old_adapter=old_ref,optimizer_updates=1,cumulative_pair_exposure=32,cursor=cursor,source_model_identity='a'*64,runtime_identity='b'*64,adapter_tensor_sha256=digest,training_input=input_refs)
            f=out/('rank-%d-state.pt'%rank);ns['save_torch'](f,state,torch);states.append(dict(rank=rank,file=cp.fileproof(f)))
        manifest=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=run,config_sha256=fixture_config['_config_sha256'],runtime_identity='b'*64,source_model_identity='a'*64,contract=cp.science_contract(fixture_config),common_initial_adapter=old_ref,outer_old_adapter=old_ref,all_rank_scaler_state=[scaler.state_dict() for _ in range(8)],optimizer_storage=storage.STORAGE,shared_optimizer=shared_ref,optimizer_updates=1,cumulative_pair_exposure=32,complete_rank_count=8,equal_optimizer_steps=True,adapter={'path':str(out/'adapter'),'files':[cp.fileproof(f) for f in sorted((out/'adapter').iterdir()) if f.is_file()]},ranks=states,all_rank_adapter_sha256=[digest]*8,all_rank_optimizer_tensor_sha256=[odigest]*8,all_rng=ns['RNG_NAMES'],scheduler={'kind':'none','warmup_updates':0},cursor=cursor,training_input=input_refs,sampling_model_identity='tiny-actual-common-initial')
        mf=out/'manifest.json';write(mf,manifest);resumed,_=cp.check_checkpoint(cp.fileproof(mf),fixture_config,run,1,0);ns['resumed']=resumed
        rows=[]
        def draws():return {'python':random.random(),'numpy':float(np.random.rand()),'torch':torch.rand(4),'simulated_cuda':[torch.rand(4,generator=g) for g in facade.cuda.generators]}
        for rank in range(8):
            active_rank=rank
            require(ns['tensor_digest'](adapter)==digest and ns['optimizer_tensor_digest'](adam)==odigest,'Baseline captured values drifted between ranks')
            saved=torch.load(states[rank]['file']['path'],map_location='cpu',weights_only=False)
            ns['restore_rng'](facade,np,saved['rng']);reference_draws=draws()
            # Baseline branch uses the in-memory captured actual state, not the on-disk restore body.
            model.set_adapter('default');set_peft_model_state_dict(model,adapter);optimizer.load_state_dict(copy.deepcopy(adam));scaler.load_state_dict(saved['scaler'])
            baseline_before=dict(current_digest=ns['tensor_digest'](get_peft_model_state_dict(model)),optimizer_digest=ns['optimizer_tensor_digest'](optimizer.state_dict()),old_digest=target_module.tensor_digest({n.replace('.outer_old.','.'):p for n,p in targets.old_parameters}),scaler=scaler.state_dict())
            require(baseline_before['current_digest']==digest and baseline_before['optimizer_digest']==odigest and baseline_before['old_digest']==old_digest,'Continuous branch is not captured checkpoint state')
            ns['restore_rng'](facade,np,saved['rng']);draws();ref_loss,ref_grad,ref_distributions=step(model,targets,trainable,optimizer,scaler)
            ref_current=copy.deepcopy(get_peft_model_state_dict(model));ref_adam=copy.deepcopy(optimizer.state_dict());ref_scaler=scaler.state_dict();ref_rng=ns['capture_rng'](facade,np)
            newbase=Qwen2ForCausalLM(qcfg).float();newbase.load_state_dict(base_state);restored_model=get_peft_model(newbase,lora)
            restored_params=[p for p in restored_model.parameters() if p.requires_grad];restored_optim=torch.optim.AdamW(restored_params,lr=science['learning_rate'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
            ns.update(model=restored_model,optimizer=restored_optim,rank=rank)
            counters=ns['restore_accepted_rank']();require(counters==(1,32),'Actual restored counters changed')
            old_manifest,_=cp.checked_json(old_ref);require(old_manifest['adapter_payload']==old_payload_ref and old_manifest['adapter_tensor_sha256']==old_digest,'Bound old disk identity differs')
            cp.checked_file(old_manifest['adapter_payload'])
            disk_old=torch.load(old_manifest['adapter_payload']['path'],map_location='cpu',weights_only=False)
            require(len(disk_old)==504 and set(disk_old)==set(initial) and all(t.dtype==torch.float32 and t.device.type=='cpu' and torch.isfinite(t).all().item() for t in disk_old.values()) and ns['tensor_digest'](disk_old)==old_digest,'Fresh disk old504 dtype/keys/finite/digest differs')
            cp.checked_file(old_manifest['adapter_payload'])
            rt,rp,_unused=setup(restored_model,disk_old);del disk_old
            require([id(p) for p in rp]==[id(p) for p in restored_params],'Old setup changed optimizer inventory')
            restored_scale=precision.SyncedGradScaler(enabled=True,scale=128.);ns['scaler']=restored_scale;ns['restore_after_targets']()
            require(ns['verify_resume_cursor'](saved,0,order_ref['sha256'],input_refs['encoding']['sha256'],33)==32 and ns['batches'](order,32)==[[32]],'Exact order/cursor/tail restoration differs')
            got_draws=draws();require(reference_draws['python']==got_draws['python'] and reference_draws['numpy']==got_draws['numpy'] and torch.equal(reference_draws['torch'],got_draws['torch']) and all(torch.equal(x,y) for x,y in zip(reference_draws['simulated_cuda'],got_draws['simulated_cuda'])),'Post-restore RNG sequence differs')
            restored_before=dict(current_digest=ns['tensor_digest'](get_peft_model_state_dict(restored_model)),optimizer_digest=ns['optimizer_tensor_digest'](restored_optim.state_dict()),old_digest=target_module.tensor_digest({n.replace('.outer_old.','.'):p for n,p in rt.old_parameters}),scaler=restored_scale.state_dict())
            require(restored_before==baseline_before,'Restored pre-forward state differs from frozen reference')
            got_loss,got_grad,got_distributions=step(restored_model,rt,rp,restored_optim,restored_scale)
            diagnostics=[]
            for x,y in zip(ref_distributions,got_distributions):
                for key in ('student_logp','teacher_logp','outer_old_logp','target_logp'):
                    diagnostics.append(dict(token_index=x['token_index'],distribution=key,shape=list(x[key].shape),reference_sha256=ns['tensor_digest']({key:x[key]}),restored_sha256=ns['tensor_digest']({key:y[key]}),exact_equal=torch.equal(x[key],y[key]),max_abs_error=float((x[key]-y[key]).abs().max())))
            write(out/('rank-%d-comparison.json'%rank),dict(rank=rank,baseline_before=baseline_before,restored_before=restored_before,reference_loss=ref_loss,restored_loss=got_loss,distributions=diagnostics,reference_snapshot_current_digest=digest,reference_snapshot_optimizer_digest=odigest))
            require(all(x['exact_equal'] for x in diagnostics),'Actual fixed target/current full distributions differ')
            require(ref_loss==got_loss and all(torch.equal(x,y) for x,y in zip(ref_grad,got_grad)),'Next actual objective loss/gradient differs')
            require(ns['tensor_digest'](ref_current)==ns['tensor_digest'](get_peft_model_state_dict(restored_model)) and ns['optimizer_tensor_digest'](ref_adam)==ns['optimizer_tensor_digest'](restored_optim.state_dict()) and ref_scaler==restored_scale.state_dict(),'Next current/Adam/scaler differs')
            got_rng=ns['capture_rng'](facade,np);require(ref_rng['python']==got_rng['python'] and np.array_equal(ref_rng['numpy'][1],got_rng['numpy'][1]) and ref_rng['numpy'][0]==got_rng['numpy'][0] and ref_rng['numpy'][2:]==got_rng['numpy'][2:] and torch.equal(ref_rng['torch_cpu'],got_rng['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(ref_rng['cuda_all'],got_rng['cuda_all'])),'Next complete RNG differs')
            rt.check_old();rows.append(dict(rank=rank,actual_nested_restore=True,actual504_PEFT=True,outer_old_actual_disk_roundtrip=True,counters_equal=True,current_old_Adam_scaler_cursor_order_equal=True,next_loss_gradient_step_RNG_equal=True))
        require(not torch.cuda.is_initialized(),'CUDA initialized')
        result=dict(schema='controlled-actual-peft-cpu-restore-v2',accepted=True,original_source_commit=ORIGINAL_COMMIT,source_proofs=proofs,extracted_functions=extracted,argv=sys.argv,python=sys.executable,torch_version=torch.__version__,peft_version=peft.__version__,transformers_version=transformers.__version__,started_unix=started,ended_unix=time.time(),fixture_checkpoint=cp.fileproof(mf),outer_old_payload=old_payload_ref,outer_old_manifest=old_ref,rows=rows,cuda_initialized=False,CUDA_VISIBLE_DEVICES='',full_GPU_resume_checked=False,limitations=LIMITS)
        write(out/'result.json',result);print(json.dumps(result,allow_nan=False,sort_keys=True))
    except BaseException as e:
        failure=dict(accepted=False,error=type(e).__name__+': '+str(e),source_proofs=proofs,extracted_functions=extracted,argv=sys.argv,limitations=LIMITS,failed_artifacts_preserved=True,active_rank=active_rank,completed_rank_rows=rows)
        write(out/'failed.json',failure);raise

if __name__=='__main__':main()
