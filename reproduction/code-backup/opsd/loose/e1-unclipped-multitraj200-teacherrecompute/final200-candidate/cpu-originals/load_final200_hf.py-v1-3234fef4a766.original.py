"""Whole nativeTorch2.4 CPU HF200 load, separate from training recovery."""
import argparse
from pathlib import Path
import time

SCOPE='HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE'


def load_common(expected_sha):
    import hashlib,os,re,stat,types
    p=Path(__file__).resolve().parent/'common.py';before=p.stat()
    if not(re.fullmatch('[0-9a-f]{64}',expected_sha) and p.resolve(strict=True)==p and before.st_uid==os.getuid()):raise ValueError('Exact owned common source required')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    names=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    if not stat.S_ISREG(opened.st_mode) or any(tuple(getattr(x,k) for k in names)!=tuple(getattr(before,k) for k in names) for x in (opened,after,p.stat())) or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Common changed before exact-byte compilation')
    module=types.ModuleType('final200_common');module.__file__=str(p);exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    module.read_file(p,expected_sha);return module


def validate_native(report,inventory,sha,inspector_sha,c):
    for key,target in dict(version=1,scope='NATIVE_CPU_FINAL200_STORED_CHECKPOINT_CONTENT_INSPECTION_NOT_LIVE_RESTORE',passed=True,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),checkpoint_step=200,inventory_sha256=sha,inspector_source_sha256=inspector_sha,cuda_is_initialized=False,hf_all_rank_slices_exact=True).items():c.typed(report.get(key),target,'Native200 result header differs: '+key)
    c.require(c.canonical(report['clock'])==c.canonical(c.clock()) and report['accepted5']==inventory['accepted5'] and report['original_raw_statuses']==inventory['raw_statuses'],'Native200 clocks/original statuses/context differ')
    c.require(type(report['ranks']) is list and len(report['ranks'])==8,'Native200 all8 content missing')
    for rank,r in enumerate(report['ranks']):
        c.typed(r.get('rank'),rank,'Native200 ranked result type differs')
        c.typed(r['pid'],inventory['rank_reports'][rank]['pid'],'Native200 worker PID type differs')
        c.require(r['object_ids']==inventory['rank_reports'][rank]['object_ids'],'Native200 worker object identity differs')
        c.object_ids(r['object_ids'])
        for key,target in dict(raw_saved_status='FULL_STATE_SERIALIZED_OK',model_dtensor_entries=435,model_hf_slice_exact=True,optimizer_state_entries=37).items():c.typed(r.get(key),target,'Native200 content scope differs: '+key)
        c.require(type(r['optimizer_step_values']) is list and r['optimizer_step_values']==[2000] and all(type(v) is int for v in r['optimizer_step_values']),'Native200 actual Adam clock type differs')
        c.require(len(r['model_layouts'])==435 and set(r['model_layouts'])==set(r['model_hf_slice_keys']),'Native200 complete435 layouts missing')
        for layout in r['model_layouts'].values():
            c.require(type(layout['mesh_coordinate']) is list and layout['mesh_coordinate']==[rank] and all(type(v) is int for v in layout['mesh_coordinate']) and layout['local_dtype']==layout['global_dtype']=='torch.float32','Native200 saved local layout differs')


def load(a,c):
    execution=c.execution_identity(a.execution_job_id,a.execution_step_id);started=time.monotonic()
    inventory,irec=c.read_json(a.inventory,a.inventory_sha256);c.validate_inventory(inventory)
    context=c.accepted_context(a.accepted5_receipt_sha256);c.require(inventory['accepted5']==context['accepted5'],'Observed historical context changed')
    native,nrec=c.read_json(a.native_report,a.native_report_sha256);validate_native(native,inventory,a.inventory_sha256,a.native_inspector_sha256,c)
    _,isource=c.read_file(a.native_inspector,a.native_inspector_sha256)
    c.require(native['common_source_sha256']==a.common_sha256 and native['math_helper_sha256']==c.MATH_HELPER,'Native200 helpers differ')
    modules,records,source_map,import_audit=c.cpu_modules(context);records.update({r['physical_path']:r for r in (irec,nrec,isource)})
    import torch
    from omegaconf import OmegaConf
    c.require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(),'NativeTorch2.4 CPU only')
    modules['e1_profile'].validate_profile(OmegaConf.create(context['config']))
    paths=c.verified_members(inventory,False);records.update({rec['physical_path']:rec for rec in inventory['checkpoint_files'].values()})
    for rec in inventory['checkpoint_files'].values():c.require(native['verified_input_records'][rec['physical_path']]==rec,'Native200 inspected different physical checkpoint files')
    helper,hrec=c.exact_module(a.hf_helper,c.HF_HELPER,'final200_unchanged_whole_hf_loader');records[hrec['physical_path']]=hrec
    base=Path(context['config']['actor_rollout_ref']['model']['path']);hf=paths['actor/model.safetensors.index.json'].parent
    original,orec=c.read_json(base/'VERIFIED-MODEL-MANIFEST.json',c.MODEL);records[orec['physical_path']]=orec
    modules['e1_profile'].verify_official_assets(base,base/'VERIFIED-MODEL-MANIFEST.json')
    for p in sorted(base.iterdir()):
        if p.is_file():_,rec=c.read_file(p);records[rec['physical_path']]=rec
    value=helper.load_compare(base,hf)
    c.require(value['cuda_is_initialized'] is False and not torch.cuda.is_initialized(),'Whole HF200 CPU load initialized CUDA')
    c.require(value['actual_unique_parameters']==c.NUMEL and value['all_loaded_parameters_finite'] is True and value['actual_cpu_forward_finite'] is True and value['tied_embeddings'] is True and len(value['hf_map'])==435 and len(value['canonical_state_keys'])==434 and value['base_named_parameter_digest']==native['base_named_parameter_digest'],'Whole loaded HF200 official geometry/content differs')
    hffiles={str(p):inventory['checkpoint_files'][role]['sha256'] for role,p in paths.items() if role.startswith('actor/') and '/training_state/' not in role}
    c.require(hffiles==native['hf_files'],'Whole HF200 load and8 native slice files differ')
    own=c.read_file(Path(__file__).resolve())[1];records[own['physical_path']]=own;common=c.read_file(Path(c.__file__),a.common_sha256)[1];records[common['physical_path']]=common
    c.recheck(records);c.source_files(c.REPO,context['closure'])
    value.update(version=1,scope=SCOPE,passed=True,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),checkpoint_step=200,inventory_sha256=a.inventory_sha256,native_report_sha256=a.native_report_sha256,native_inspector_source_sha256=a.native_inspector_sha256,inspector_source_sha256=own['sha256'],common_source_sha256=common['sha256'],hf_helper_sha256=c.HF_HELPER,execution_identity=execution,cpu_runtime=dict(torch_version=torch.__version__,torch_file=torch.__file__),import_audit=import_audit,accepted5=context['accepted5'],original_raw_statuses=inventory['raw_statuses'],checkpoint=str(hf),official_base=str(base),hf_files=hffiles,verified_input_records=records,seconds=time.monotonic()-started,cuda_is_initialized=False,limitations=['Whole CPU model and finite forward only; no training/Adam/FSDP recovery','Historical5 context is not200 receipt','No final evaluation/terminal/resource release/effect qualification'])
    return value


def main():
    p=argparse.ArgumentParser()
    for name in ('common-sha256','inventory','inventory-sha256','accepted5-receipt-sha256','native-report','native-report-sha256','native-inspector','native-inspector-sha256','hf-helper','execution-job-id','execution-step-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common(a.common_sha256);value=load(a,c);record=c.write_exclusive(a.output,value)
    print(c.canonical(dict(scope=SCOPE,passed=True,output=record,stored_training_job_id='3304',execution_identity=value['execution_identity'])).decode())


if __name__=='__main__':main()
