#!/usr/bin/env python3
"""One offline, source-bound reverse/tail check; no GPU or production resume."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types

SOURCE_COMMIT = '871dc46f9ae436e5b4115dd33e51715b858ee57d'
MANIFEST_SHA = '8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41'
PIN = {
    'objective.py':'b54324db304a34a3be02d8d5269d1f6bb6968207ede386d0e86138e0fa787892',
    'precision.py':'7dd23d6378c91cb33c730241fe5cd63618768aa95eea6136bf571a7cb64358bc',
    'targets.py':'f0a6e4d7d08d443c01e553ade88d38a8f5a6bb8d1f1f0be5d687b38df95afb74',
    'trainer.py':'4dc352f5091594ab5bcae71ecc17516afd8607601b5932216cbda66befc5409c',
    'test_objective.py':'acb0faaba1f2cf512de9b42f3dc8c06e6532c7219992f20f64daf2597bb46060',
    'test_targets.py':'d177aaaec017acb8c6adb01148d92539f22b5404fde0c614288d6ebcbf447821',
    'test_precision.py':'c2825e497b2fb6d866a69b8b4d9288f45f40e9a83e9ba61314a090152a1e6f59',
}

def require(ok, message):
    if not ok: raise ValueError(message)

def stable_bytes(path, wanted):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and all(not q.is_symlink() for q in (p,*p.parents)), 'Canonical nonsymlink input required')
    s=p.stat(); require(stat.S_ISREG(s.st_mode), 'Regular source required')
    def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
    before=identity(s)
    with p.open('rb') as f:
        require(identity(os.fstat(f.fileno()))==before,'Opened source changed')
        raw=f.read(); require(identity(os.fstat(f.fileno()))==before,'Source changed during read')
    require(identity(p.stat())==before,'Source path changed')
    sha=hashlib.sha256(raw).hexdigest(); require(sha==wanted,'SHA differs: '+str(p))
    return raw,dict(path=str(p),sha256=sha,bytes=len(raw),device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def source_module(name, raw, path):
    require(name not in sys.modules,'Refuse cached/shadow module '+name)
    m=types.ModuleType(name); m.__file__=str(path);sys.modules[name]=m
    exec(compile(raw,str(path),'exec'),m.__dict__)
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--source-manifest',required=True)
    p.add_argument('--publication',required=True)
    p.add_argument('--publication-sha256',required=True)
    a=p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA_VISIBLE_DEVICES must be explicitly empty before import')
    require('torch' not in sys.modules,'Fresh CPU-only process required')
    root=Path(a.source_root);require(root.is_absolute() and root.name=='opsd_ablation_v1','Exact frozen namespace required')
    manifest_raw,manifest_proof=stable_bytes(a.source_manifest,MANIFEST_SHA)
    require(Path(a.source_manifest)==root/'source-manifest.json','Manifest/root differ')
    manifest=json.loads(manifest_raw);entries=manifest.get('files',manifest)
    publication_raw,publication_proof=stable_bytes(a.publication,a.publication_sha256)
    publication=json.loads(publication_raw)
    require(publication.get('accepted') is True and publication.get('source_commit')==SOURCE_COMMIT and publication.get('remote_readback_sha')==SOURCE_COMMIT,'Original source publication/readback required')
    published={r['path']:r for r in publication['files']}
    raw={};proofs={}
    for name,sha in PIN.items():
        require(entries[name]['sha256']==sha,'Frozen manifest entry differs')
        pub=published['experiments/search_e1/control/opsd_ablation_v1/'+name]
        require(pub['equal'] is True and pub['sha256']==sha,'Published file binding differs')
        raw[name],proofs[name]=stable_bytes(root/name,sha)
    trainer_tree=ast.parse(raw['trainer.py'],filename=str(root/'trainer.py'))
    defs=[n for n in trainer_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rank_batch','batches')]
    require({n.name for n in defs}=={'rank_batch','batches'},'Actual batch functions absent')
    # Verify the actual producer divides each trajectory loss once by actualB.
    divisions=[n for n in ast.walk(trainer_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='scale_loss'
               and n.args and ast.dump(n.args[0])==ast.dump(ast.parse('loss/actual',mode='eval').body)]
    require(len(divisions)==1,'Actual trainer loss/B call contract changed')
    ns={};exec(compile(ast.Module(body=defs,type_ignores=[]),str(root/'trainer.py'),'exec'),ns)
    import torch
    require(not torch.cuda.is_initialized() and torch.cuda.device_count()==0,'No visible GPU allowed')
    for name in ('precision','targets','objective','test_targets','test_objective','test_precision'):
        source_module(name,raw[name+'.py'],root/(name+'.py'))
    objective=sys.modules['objective'];precision=sys.modules['precision'];targets=sys.modules['targets'];fixture=sys.modules['test_targets']
    started=time.time();torch.manual_seed(42)
    reused=[]
    checks={
        'test_objective':('ReverseTests',[
            'test_three_asymmetric_full_gradient_and_difference_from_forward',
            'test_equal_and_mixture_endpoints_and_initial_bound','test_clip_only_upper_preserves_negative',
            'test_detaching_student_coefficient_is_wrong','test_full_vocabulary_callback_once_through_checkpoint_recompute']),
        'test_targets':('TargetTests',['test_causal_shift_excludes_last_D_hidden','test_same_D_targets_frozen_and_old_stable_after_current_updates']),
        'test_precision':('PrecisionTests',['test_scaled_sum_no_world_divide_then_unscale_clip_step','test_remote_overflow_skips_adam_restores_capsule_scale'])}
    for module,(class_name,names) in checks.items():
        for name in names:
            getattr(getattr(sys.modules[module],class_name)(methodName=name),name)();reused.append(module+'.'+name)
    base=fixture.Toy();base_state=copy.deepcopy(base.state_dict());rows=[]
    for i in range(32):
        length=3+i%7
        rows.append(dict(prefixes=[[1,2],[3,4,5]],response=[6+(i+j)%60 for j in range(length)],
                         mask=[1]*(length-1)+[0],offsets=[(j,j+1) for j in range(length)]))
    weights=[.25+(i%5)*.375 for i in range(32)]
    def model_and_targets():
        m=fixture.Toy();m.load_state_dict(base_state)
        t=targets.SharedTargets(m,precision.PrecisionContract(device_type='cpu'),old_adapter='outer_old',old_identity=fixture.old_identity(m))
        return m,t
    def backward(m,t,indices,B,mix):
        total=0.
        for i in indices:
            with t.current_graph():
                loss,_=objective.pair_objective(m,rows[i],10,32,B,precision=precision.PrecisionContract(device_type='cpu'),targets=t,pair_weight=weights[i],use_mixture=mix)
                total+=float(loss.detach())/B;t.mark_backward();(loss/B).backward()
        return total
    tails=[]
    for mix in (False,True):
        for B in range(1,33):
            one,ot=model_and_targets();reference_loss=backward(one,ot,list(range(B)),B,mix)
            ref=[p.grad.detach().clone() for p in one.parameters() if p.requires_grad]
            total_loss=0.;summed=[torch.zeros_like(g) for g in ref];counts=[]
            for rank in range(8):
                indices=ns['rank_batch'](list(range(B)),rank,8);counts.append(len(indices))
                m,t=model_and_targets();total_loss+=backward(m,t,indices,B,mix)
                for accum,param in zip(summed,[p for p in m.parameters() if p.requires_grad]):
                    if param.grad is not None:accum.add_(param.grad)
            require(sum(counts)==B and max(counts)-min(counts)<=1,'Native rank partition differs')
            errors=[float((x-y).abs().max()) for x,y in zip(summed,ref)]
            require(all(torch.allclose(x,y,atol=2e-6,rtol=2e-5) for x,y in zip(summed,ref)),'Preclip global gradient differs')
            require(abs(total_loss-reference_loss)<=2e-6+2e-5*abs(reference_loss),'Global loss differs')
            tails.append(dict(B=B,mixture=mix,rank_counts=counts,empty_ranks=counts.count(0),weighted=True,
                              reference_loss=reference_loss,rank_sum_loss=total_loss,max_abs_gradient_error=max(errors)))
    require(ns['batches']([],0,32)==[] and ns['batches'](list(range(33)),32,32)==[[32]] and ns['batches'](list(range(33)),33,32)==[],'K0/tail/endcursor changed')
    for invalid in (1,31):
        try:ns['batches'](list(range(33)),invalid,32)
        except ValueError:pass
        else:raise ValueError('Noncommitted cursor accepted')
    require(not torch.cuda.is_initialized(),'CPU check initialized CUDA')
    result=dict(schema='reverse-ablation-required-cpu-check-v1',accepted=True,started_unix=started,ended_unix=time.time(),
        argv=sys.argv,python=sys.executable,torch_version=torch.__version__,CUDA_VISIBLE_DEVICES='',cuda_initialized=False,
        original_source_commit=SOURCE_COMMIT,source_manifest=manifest_proof,publication=publication_proof,source_files=proofs,
        reused_focused_checks=reused,tail_checks=tails,tail_cases=64,world_size=8,actual_producer_AST_functions=['rank_batch','batches'],
        actual_objective=True,actual_trainer_loss_divisor_AST_verified=True,no_extra_world_divisor=True,
        resume_scope=dict(full_production_resume_checked=False,actual_nested_restore_executed=False,actual_PEFT_Adam_RNG_roundtrip=False,
                          prior_scalar_scaler_restore_only=True,status='MISSING_CONTROLLED_FULL_RESTORE_EVIDENCE'),
        limitations=['Small published Toy adapter model; actual pair_objective/SharedTargets and batch functions, not source100 or actual PEFT model.',
                     'Eight serial CPU replicas and SUM; no NCCL, CUDA RNG transport, GPU precision/memory or distributed deadlock acceptance.',
                     'Focused original math/target/scaler methods reused; no unrelated unittest suite.',
                     'No files/checkpoints/training outputs written; full production Adam/RNG/cursor restore remains separately required.'])
    print(json.dumps(result,allow_nan=False,sort_keys=True))

if __name__=='__main__':main()
