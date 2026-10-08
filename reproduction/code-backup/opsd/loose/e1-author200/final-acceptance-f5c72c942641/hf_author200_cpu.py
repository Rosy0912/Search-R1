"""Whole CPU load of the real HF200 export; not an RL/FSDP training restore."""
import argparse
import json
import os
from pathlib import Path
import time

SCOPE='HF200_WHOLE_CPU_MODEL_LOAD_NOT_TRAINING_RESTORE'
HELPER='dfc8ac3e4c9c539756c823ad4ba93c4de3ba7ca6ff8ef18dcd2772abdbcd1525'


def load_inspector(path,sha):
    # Bootstrap only enough stdlib to verify the reviewed inspection source.
    import hashlib
    import re
    import sys
    import types
    p=Path(path)
    if not (p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and p.stat().st_uid==os.getuid() and type(sha) is str and re.fullmatch('[0-9a-f]{64}',sha)):
        raise ValueError('Exact owned native inspector and external SHA required')
    before=p.stat();raw=p.read_bytes();after=p.stat()
    names=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    if tuple(getattr(before,k) for k in names)!=tuple(getattr(after,k) for k in names) or hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('Reviewed native inspector bytes differ')
    module=types.ModuleType('author200_exact_native_inspection');module.__file__=str(p)
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=sys.flags.optimize),module.__dict__)
    if module.stable_sha(p)!=sha:raise ValueError('Native inspector changed while compiling')
    return module


def validate_native(report,inventory,inventory_sha,require):
    require(type(report) is dict and report.get('version')==1 and type(report.get('version')) is int and report.get('passed') is True and
        report.get('scope')=='NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION', 'Whole HF load requires actual passed stored-state inspection')
    for key,target in dict(source_commit=inventory['source_commit'],config_sha256=inventory['config_sha256'],job_id='3261',
        checkpoint_step=200,inventory_sha256=inventory_sha,run_root=inventory['run_root']).items():
        require(type(report.get(key)) is type(target) and report[key]==target, 'Native/HF evidence identity differs: '+key)
    require(report['cuda_is_initialized'] is False and report['hf_all_rank_slices_exact'] is True and report['original_raw_statuses']==inventory['raw_statuses'], 'Actual native CPU scope/status differs')
    ranks=report['ranks'];require(type(ranks) is list and len(ranks)==8 and all(type(r) is dict and type(r.get('rank')) is int for r in ranks) and
        [r['rank'] for r in ranks]==list(range(8)), 'Native all8 typed rank evidence required')
    for rank,value in enumerate(ranks):
        require(value['model_dtensor_entries']==435 and type(value['model_dtensor_entries']) is int and value['model_hf_slice_exact'] is True and
            value['raw_saved_status']=='FULL_STATE_SERIALIZED_OK' and value['pid']==inventory['accepted5']['workers'][rank]['pid'] and
            value['object_ids']==inventory['accepted5']['workers'][rank]['object_ids'] and type(value['optimizer_step_values']) is list and
            value['optimizer_step_values']==[3200] and all(type(v) is int for v in value['optimizer_step_values']), 'Native rank content scope differs')
        require(len(value['model_layouts'])==435 and set(value['model_layouts'])==set(value['model_hf_slice_keys']), 'Native435 per-rank layouts incomplete')
        for key,layout in value['model_layouts'].items():
            require(type(layout['mesh_coordinate']) is list and layout['mesh_coordinate']==[rank] and all(type(v) is int for v in layout['mesh_coordinate']) and
                    layout['local_dtype']=='torch.float32' and layout['global_dtype']=='torch.float32', 'Native rank saved layout differs')


def load(a):
    native=load_inspector(a.native_inspector,a.native_inspector_sha256)
    execution=native.execution_identity(a.execution_job_id,a.execution_step_id);pins={}
    self_path=native.physical(Path(__file__).resolve());self_sha=native.stable_sha(self_path)
    native.pinned(a.native_inspector,a.native_inspector_sha256,pins)
    ip=native.pinned(a.inventory,a.inventory_sha256,pins);inventory=native.read_pinned_json(ip,pins);native.validate_inventory(inventory,200)
    rp=native.pinned(a.native_report,a.native_report_sha256,pins);report=native.read_pinned_json(rp,pins);validate_native(report,inventory,a.inventory_sha256,native.require)
    native.require(report['inspector_source_sha256']==a.native_inspector_sha256, 'Actual native report was produced by different inspector bytes')
    paths=native.observed_files(inventory,pins)
    for role,rec in inventory['files'].items():
        native.require(report['all_files'].get(rec['physical_path'])==rec['sha256'], 'Native report file binding differs: '+role)
    state,profile,gate,source_files,code_count=native.source_modules(Path(a.repo),inventory,pins)
    manifest,payload,config,prior=native.accepted5(inventory,gate,pins)
    from omegaconf import OmegaConf
    profile.validate_profile(OmegaConf.create(config))
    import torch
    native.require(torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized(), 'Native Torch2.4 CPU required')
    helper=native.exact_module(a.helper,HELPER,'author200_exact_immutable_hf_load_helper',pins)
    base=Path(config['actor_rollout_ref']['model']['path']);hf=paths['actor/model.safetensors.index.json'].parent
    native.require(base.resolve(strict=True)==base and hf.resolve(strict=True)==hf, 'Physical HF/base directories required')
    original=native.pinned(base/'VERIFIED-MODEL-MANIFEST.json',native.MODEL,pins)
    profile.verify_official_assets(base,original)
    base_files={str(native.physical(p)):native.stable_sha(p) for p in sorted(base.iterdir()) if p.is_file()}
    # Stored-state inspection has already read all8 model shards and compared
    # every one of 435 model tensors to this exact HF export. The immutable
    # loader now proves actual loaded CPU model values equal those same files.
    start=time.monotonic();result=helper.load_compare(base,hf)
    native.require(result['cuda_is_initialized'] is False and not torch.cuda.is_initialized(), 'Whole CPU model load initialized CUDA')
    native.require(result['actual_unique_parameters']==native.NUMEL and result['all_loaded_parameters_finite'] is True and
        result['actual_cpu_forward_finite'] is True and result['tied_embeddings'] is True and len(result['hf_map'])==435 and
        len(result['canonical_state_keys'])==434 and result['base_named_parameter_digest']==report['base_named_parameter_digest'], 'Loaded whole model/official identity differs')
    actual_hf_files={str(path):pins[str(path)] for role,path in paths.items() if role.startswith('actor/') and '/training_state/' not in role}
    native.require(actual_hf_files==report['hf_files'], 'Whole loaded HF files differ from all-rank native comparison')
    for path,pin in pins.items():native.require(native.stable_sha(path)==pin, 'Pinned input changed during whole model load')
    native.require(base_files=={str(native.physical(p)):native.stable_sha(p) for p in sorted(base.iterdir()) if p.is_file()}, 'Official base files changed during whole model load')
    profile.verify_official_assets(base,original)
    native.require(source_files=={str(p.relative_to(Path(a.repo))):native.stable_sha(p) for namespace in ('search_r1','verl') for p in sorted((Path(a.repo)/namespace).rglob('*.py'))}, 'Frozen source changed during CPU load')
    native.require(native.stable_sha(self_path)==self_sha, 'HF inspector source changed')
    result.update(version=1,scope=SCOPE,passed=True,checkpoint_step=200,source_commit=native.SOURCE,config_sha256=native.CONFIG,
        job_id='3261',run_root=inventory['run_root'],inventory_path=str(ip),inventory_sha256=a.inventory_sha256,
        native_report_path=str(rp),native_report_sha256=a.native_report_sha256,native_inspector_source_sha256=a.native_inspector_sha256,
        inspector_source_sha256=self_sha,immutable_hf_helper_sha256=HELPER,checkpoint=str(hf),official_base=str(base),
        execution_identity=execution,cpu_runtime=dict(torch_version=torch.__version__,torch_file=str(Path(torch.__file__).resolve())),
        actual_import_function_count=code_count,native_import_closure_sha256=native.CLOSURE,native_closure_module_count=210,
        native_all8_435_slices_exact_same_files=True,base_all_files=base_files,hf_files=actual_hf_files,all_files=pins,
        original_raw_statuses=inventory['raw_statuses'],seconds=time.monotonic()-start,cuda_is_initialized=False,
        limitations=['Actual complete CPU model load and finite forward only; no RL/FSDP/Adam continuation is attempted or claimed',
                     'Stored training identities differ from this CPU execution identity',
                     'Native all8 saved slices and loaded model share the exact same serialized HF files',
                     'This mechanical check does not show improved search behavior, reward or final evaluation completion'])
    return result,native


def main():
    p=argparse.ArgumentParser()
    for name in ('inventory','inventory-sha256','native-report','native-report-sha256','native-inspector','native-inspector-sha256','helper','repo','execution-job-id','execution-step-id','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();result,native=load(a);sha=native.write_exclusive(a.output,result)
    print(json.dumps(dict(scope=SCOPE,checkpoint_step=200,stored_training_job_id='3261',execution_job_id=a.execution_job_id,output=a.output,output_sha256=sha,passed=True,cuda_is_initialized=False)))


if __name__=='__main__':main()
