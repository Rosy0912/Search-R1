#!/usr/bin/env python3
"""Candidate c5bf CPU preparation/entry adapter, not a model algorithm change.

Uses exact existing74ce source inventory and e00 physical reader APIs. New full
config construction is explicitly separate from prior partial2b CPU preparation.
No main is executed by author tests. In production eval mode the external ready
approval must already have been consumed by the owning supervisor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
import sys
import types

SUPER_FILES = ('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def supervisor(pin):
    # First executed helper has a bootstrap independent of imported candidate code.
    import stat
    p=Path(__file__).resolve().parent/'paired_supervisor.py'
    if p.resolve(strict=True)!=p:raise ValueError('Supervisor physical path differs')
    a=p.lstat()
    if not stat.S_ISREG(a.st_mode) or a.st_uid!=os.getuid() or a.st_nlink!=1:raise ValueError('Supervisor ownership differs')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        b=os.fstat(fd);parts=[]
        while True:
            x=os.read(fd,1024**2)
            if not x:break
            parts.append(x)
        c=os.fstat(fd)
    finally:os.close(fd)
    d=p.lstat();raw=b''.join(parts)
    if any(getattr(a,k)!=getattr(v,k) for v in (b,c,d) for k in SUPER_FILES) or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('Supervisor exact same bytes differ')
    mod=types.ModuleType('paired_exact_cpu_supervisor');mod.__file__=str(p);sys.modules[mod.__name__]=mod
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),mod.__dict__)
    return mod


def native_context(s,q):
    checker,_=s.exact_module(Path(__file__).parent/'inspect_eval_native_cpu.py',s.CHECKER_SHA,'paired_existing_source_checker')
    common,_=s.exact_module(Path(__file__).parent/'common.py',s.COMMON_SHA,'paired_existing_source_common')
    owner=s.allocation()
    _,runtime=checker.native_runtime(common,owner['job_id'],owner['step_id'])
    repo,inventory,invrec,members=checker.source_inventory(common,q['source_inventory']['path'],q['source_inventory']['sha256'])
    s.same(str(repo),str(s.REPO),'Fixed actual c5bf repository differs')
    bootpath=repo/'reproduction/baselines/e1_import_bootstrap.py'
    boot,_=s.exact_module(bootpath,members['reproduction/baselines/e1_import_bootstrap.py']['sha256'],'reproduction.baselines.e1_import_bootstrap')
    boot.repo_first(repo)
    closure,closerec=s.reference(q['native_import_closure'],expected=s.CLOSURE_SHA)
    required=tuple(dict.fromkeys(tuple(boot.REQUIRED)+checker.ADDED_REQUIRED))
    actual,count=boot.verify_imports(repo,closure,required=required)
    s.same(actual,closure,'Actual c5bf94 import closure differs')
    s.require(len(actual['modules'])==94,'Actual c5bf94 module count differs')
    import torch
    s.require(torch.cuda.is_initialized() is False,'CPU source/import preparation initialized CUDA')
    return dict(owner=owner,runtime=runtime,source_inventory=invrec,source_members=members,native_import_closure=closerec,
                verified_modules=len(actual['modules']),verified_live_function_code_count=count),boot,closure


def role_environment(s,q,run,role,input_record):
    env=s.clean_env()
    env.update(OPSD_E1_OFFICIAL_MODEL=str(s.BASE),OPSD_E1_COMMON_DATA_ROOT=str(s.DATA_ROOT),
        OPSD_E1_COMMON_MANIFEST=q['common_manifest']['path'],OPSD_E1_RETRIEVER_URL='http://127.0.0.1:18084/retrieve',
        SEARCH_R1_BASELINE_RUN_ROOT=str(run/role),OPSD_FINAL200_ACTUAL_MODEL_DIRECTORY=str(s.MODELS[role]),
        OPSD_FINAL200_ACTUAL_REVIEWED_STAGE=str(run/(role+'-stage.json')),
        OPSD_FINAL200_INPUT_MANIFEST=input_record['path'],OPSD_FINAL200_INPUT_MANIFEST_SHA256=input_record['sha256'],
        OPSD_FINAL200_ACTUAL_FRESH_OUTPUT_DIRECTORY=str(run/role/'raw'))
    return env


def compose_role(s,q,run,role,input_record,contract):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    old=dict(os.environ)
    try:
        os.environ.clear();os.environ.update(role_environment(s,q,run,role,input_record))
        with initialize_config_dir(version_base=None,config_dir=str(s.REPO/'verl/trainer/config')):
            cfg=compose(config_name='ppo_opsd_final200_eval',overrides=contract.inactive_overrides()+['trainer.logger=[console]'])
        resolved=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)
    finally:os.environ.clear();os.environ.update(old)
    contract.validate_resolved_config(resolved)
    from search_r1.baselines import final200_evaluation as evaluation
    s.same(evaluation.validate_eval_config(OmegaConf.create(resolved)),contract.BUDGET,'Frozen c5bf config budget differs')
    s.same(resolved['actor_rollout_ref']['model']['path'],str(s.MODELS[role]),'Role model path differs')
    return resolved


def complete_models(s,contract):
    maps={};records={}
    for role,directory in s.MODELS.items():
        s.require(directory.resolve(strict=True)==directory and directory.is_dir() and directory.stat().st_uid==os.getuid(),'Fixed actual model directory differs')
        all_entries=list(directory.iterdir());first=set();directories=set();files={};actual={}
        for entry in all_entries:
            status=entry.lstat()
            s.require(entry.resolve(strict=True)==entry and status.st_uid==os.getuid(),'Model immediate entry symlink/ownership differs')
            if stat.S_ISREG(status.st_mode):first.add(entry.name)
            elif stat.S_ISDIR(status.st_mode):directories.add(entry.name)
            else:raise ValueError('Unsupported immediate model entry type')
        s.same(directories,{'training_state'} if role=='actual_hf200' else set(),'Unexpected model directory; training_state is guarded by original31 map')
        s.require(first,'Model immediate files absent')
        for name in sorted(first):
            s.require(Path(name).name==name,'Model member name differs')
            _,rec=s.capture(directory/name);files[name]=rec['sha256'];actual[name]=rec
        s.same(first,set(p.name for p in directory.iterdir() if p.is_file()),'Model membership changed')
        s.require({'config.json','tokenizer.json'}<=first and any(n.endswith('.safetensors') for n in first),'Actual model assets incomplete')
        if role=='official_base':s.same(files.get('VERIFIED-MODEL-MANIFEST.json'),contract.MODEL_MANIFEST,'Official model manifest differs')
        maps[role]=files;records[role]=actual
    return maps,records


def prepare(s,q,request_ref,run):
    ctx,boot,closure=native_context(s,q)
    contract,_=s.exact_module(Path(__file__).parent/'paired_eval_contract.py',s.CONTRACT_SHA,'paired_prepare_contract')
    from search_r1.baselines import final200_evaluation as evaluation
    pin=evaluation.prepare_inputs(q['common_manifest']['path'],str(s.DATA_ROOT),str(run/'ordered-inputs.json'))
    _,inputrec=s.capture(run/'ordered-inputs.json',pin)
    inputs=evaluation.load_inputs(inputrec['path'],inputrec['sha256'])
    s.same(inputs['source_counts'],contract.COUNTS,'Actual final source counts differ');s.same(inputs['question_ids_sha256'],contract.IDS,'Actual ordered IDs differ')
    configs={}
    for role in s.ROLES:configs[role]=s.write_once(run/(role+'-resolved-config.json'),compose_role(s,q,run,role,inputrec,contract))
    model_files,model_records=complete_models(s,contract)
    assets,assetrec=s.reference(q['tool_asset_manifest'],expected=s.TOOL_ASSET_SHA)
    s.same(assets['status'],'FULL_TOOL_ASSET_CONTENT_VERIFIED','Tools not fully verified')
    tool_records=[]
    for item in assets['results']['e5']+[assets['results'][k] for k in ('wiki_gzip','index','corpus')]:
        record=s.capture(item['path'],item['sha256'],root=s.ASSETS)[1];s.same(record['bytes'],item['bytes'],'Tool byte count differs');tool_records.append(record)
    faiss_records=[s.capture(item['path'],item['sha256'],root=s.GPU_LIB)[1] for item in s.FAISS_MODULE_BINDINGS.values()]
    s.require(contract.REQUIRED_SOURCES<=set(ctx['source_members']),'Required actual generation source pins incomplete')
    sources={n:ctx['source_members'][n]['sha256'] for n in sorted(contract.REQUIRED_SOURCES)}
    after,count=boot.verify_imports(s.REPO,closure,required=tuple(dict.fromkeys(tuple(boot.REQUIRED)+sys.modules['paired_existing_source_checker'].ADDED_REQUIRED)))
    s.same(after,closure,'Config/import closure changed');s.same(count,ctx['verified_live_function_code_count'],'Live function code changed')
    import torch
    s.require(torch.cuda.is_initialized() is False,'Model metadata/input preparation initialized CUDA')
    for rec in ctx['source_members'].values():s.capture(rec['physical_path'],rec['sha256'])
    value=dict(version=1,kind='ACTUAL_NEW_ALLOCATION_C5BF_FULL_CPU_PREPARATION_NOT_GPU_OR_EVAL_PASS',request=request_ref,
        execution=ctx['owner'],runtime=ctx['runtime'],source_inventory=ctx['source_inventory'],native_import_closure=ctx['native_import_closure'],
        verified_modules=ctx['verified_modules'],verified_live_function_code_count=ctx['verified_live_function_code_count'],
        input_manifest=inputrec,common_manifest=q['common_manifest'],configurations=configs,source_files=sources,source_members=ctx['source_members'],
        model_files=model_files,model_physical_records=model_records,tool_asset_manifest=assetrec,tool_asset_records=tool_records,
        faiss_gpu_library_records=faiss_records,cuda_is_initialized=False)
    s.write_once(run/'cpu-prepared.json',value)


def evaluate(s,q,run,role):
    s.require(role in s.ROLES,'Unsupported eval role')
    prepared,_=s.reference(dict(path=str(run/'cpu-prepared.json'),sha256=s.capture(run/'cpu-prepared.json')[1]['sha256']),root=run)
    # Presence and exact linkage are checked again; this is not the approval issuer.
    consumed,_=s.reference(dict(path=str(run/'actual-approval-consumed.json'),sha256=s.capture(run/'actual-approval-consumed.json')[1]['sha256']),root=run)
    ready,ready_rec=s.reference(dict(path=str(run/'pre-gpu-ready.json'),sha256=consumed['ready']['sha256']),root=run)
    s.require(s.same_process(ready['parent_identity']),'Approval owner no longer alive')
    public,_=s.reference(dict(path=str(s.PUBLIC),sha256=s.PUBLIC_SHA))
    approval,_=s.reference(dict(path=str(run/'pre-gpu-approval.json'),sha256=consumed['approval']['sha256']),root=run)
    gate,_=s.exact_module(s.REPO/'search_r1/baselines/checkpoint_gate.py',s.GATE_SHA,'paired_child_approval_gate')
    gate.verify_rsa_signature(approval['payload'],approval['signature_hex'],public)
    s.same(approval['payload']['ready_sha256'],ready_rec['sha256'],'Child approval ready differs')
    stage=s.capture(run/(role+'-stage.json'),ready['stages'][role]['sha256'])[1]
    s.same(os.environ.get('OPSD_FINAL200_EVAL_STAGE_SHA256'),stage['sha256'],'Actual role stage env differs')
    ctx,boot,closure=native_context(s,q)
    # Actual eval child starts CPU-only; the frozen main owns subsequent GPU execution.
    args=['--config-name','ppo_opsd_final200_eval']
    contract,_=s.exact_module(Path(__file__).parent/'paired_eval_contract.py',s.CONTRACT_SHA,'paired_eval_contract_child')
    args+=contract.inactive_overrides()+['trainer.logger=[console]']
    assigned=os.environ.get('PAIRED_EVAL_ASSIGNED_CUDA_DEVICES')
    s.require(type(assigned) is str and len(assigned.split(','))==8,'Actual parent GPU visibility assignment missing')
    os.environ['CUDA_VISIBLE_DEVICES']=assigned
    sys.argv=['search_r1.baselines.main']+args
    runpy.run_module('search_r1.baselines.main',run_name='__main__')


def main():
    p=argparse.ArgumentParser()
    for name in ('mode','request','request-sha256','run','supervisor-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('official_base','actual_hf200'));a=p.parse_args()
    s=supervisor(a.supervisor_sha256);q,rec=s.load_request(Path(a.request),a.request_sha256);run=Path(a.run)
    owner=s.allocation();s.same(run,s.ROOT/'paired-final200-runs'/('eval-'+owner['job_id']),'Actual own run path differs')
    s.require(run.resolve(strict=True)==run and run.stat().st_uid==os.getuid(),'Run physical ownership differs')
    if a.mode=='prepare':prepare(s,q,dict(path=str(a.request),sha256=a.request_sha256),run)
    elif a.mode=='eval':evaluate(s,q,run,a.role)
    else:raise ValueError('Unsupported fixed adapter mode')


if __name__=='__main__':main()
