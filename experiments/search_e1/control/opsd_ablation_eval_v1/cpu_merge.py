"""Isolated CPU child: source FP32 + export FP32 LoRA, one FP16 cast. No CUDA."""
import argparse
import os
from pathlib import Path
try:
    from .utils import *
    from . import prepare,gates,worker
except ImportError:
    from utils import *
    import prepare,gates,worker


def cpu_state(torch):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and torch.cuda.device_count()==0,
        'CPU merge must have no visible/initialized CUDA')
    return dict(initialized=False,visible_devices=0)


def to_fp16_once(model,torch):
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 and torch.isfinite(p).all().item() for p in model.parameters()),
        'Before deployment cast requires finite FP32 CPU merged parameters')
    model.to(dtype=torch.float16)
    require(all(p.device.type=='cpu' and p.dtype==torch.float16 and torch.isfinite(p).all().item() for p in model.parameters()),
        'Once-cast FP16 deployment nonfinite')
    return model


def merge(config,config_ref,request,output):
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU child boundary required before model imports')
    prepare.validate_config(config);gates.publication(config)
    require(request['run_id']==config['run_id'] and request['config_sha256']==config_ref['sha256']
        and request['version'] in ('fp16_zero','raw10_fp16') and request['output']==str(Path(output).resolve()),'CPU merge request scope differs')
    controller=gates.process(config,request['controller']['pid']);own=gates.process(config,os.getpid())
    require(gates.same(controller,request['controller']) and controller['run_id']==config['run_id']
        and own['ppid']==controller['pid'] and own['run_id']==config['run_id'], 'CPU merge exact live controller/ancestry differs')
    train,_=original_config(config);adapter=prepare.adapter(config,request['version']);weights=source_files(config)
    require(request['adapter_manifest_sha256']==adapter['manifest']['sha256'] and request['adapter_tensor_sha256']==adapter['tensor_sha256'],'Merge selected another adapter')
    destination=Path(output);require(not destination.exists(),'No overwrite/reuse of a partial CPU merge')
    import torch
    from importlib import metadata as package_metadata
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,get_peft_model_state_dict
    require(torch.__version__.split('+')[0]=='2.4.0' and package_metadata.version('transformers')=='4.46.3'
        and package_metadata.version('peft')=='0.13.2','CPU model environment differs')
    before=cpu_state(torch)
    values=load_file(str(Path(adapter['path'])/'adapter_model.safetensors'),device='cpu')
    adapter_proof=worker.tensor_checks(values,request['version'],adapter['tensor_sha256']);del values
    base=AutoModelForCausalLM.from_pretrained(train['paths']['model'],torch_dtype=torch.float32,attn_implementation='sdpa')
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 and torch.isfinite(p).all().item() for p in base.parameters()),'Source is not actual finite FP32 CPU master')
    base_digest=worker.tensor_digest(base.state_dict()) if request['version']=='fp16_zero' else None
    model=PeftModel.from_pretrained(base,adapter['path'],is_trainable=False)
    actual=worker.tensor_checks(get_peft_model_state_dict(model),request['version'],adapter['tensor_sha256'])
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'Actual merge base/LoRA not FP32 CPU')
    with torch.no_grad():merged=model.merge_and_unload(safe_merge=True)
    require(all(p.dtype==torch.float32 and p.device.type=='cpu' and torch.isfinite(p).all().item() for p in merged.parameters()),'Actual merged FP32 parameters invalid')
    zero_identity=None
    if base_digest is not None:
        merged_digest=worker.tensor_digest(merged.state_dict())
        require(merged_digest==base_digest,'Zero LoRA CPU merge changed source tensors')
        zero_identity=dict(accepted=True,source_fp32_tensor_sha256=base_digest,merged_fp32_tensor_sha256=merged_digest)
    to_fp16_once(merged,torch)  # The only weight dtype conversion after the FP32 merge.
    require(merged.generation_config.repetition_penalty==1.05 and merged.generation_config.eos_token_id==[151645,151643],'CPU deployment native generation config changed')
    digest=worker.tensor_digest(merged.state_dict());merged.save_pretrained(destination,safe_serialization=True)
    AutoTokenizer.from_pretrained(train['paths']['model'],use_fast=True).save_pretrained(destination)
    after=cpu_state(torch);check_metadata(weights['files']+adapter['files']);gates.publication(config)
    files=[fileproof(p) for p in sorted(destination.rglob('*')) if p.is_file()]
    manifest=dict(schema='opsd-ablation-fp16-cpu-merged-model-v1',accepted=True,run_id=config['run_id'],study_id=config['study_id'],
        variant=config['variant'],version=request['version'],training_run_id=train['run_id'],source_model_identity=train['source_model_manifest']['sha256'],
        adapter_manifest_sha256=adapter['manifest']['sha256'],adapter_tensor_sha256=adapter['tensor_sha256'],
        actual_adapter=adapter_proof,actual_loaded_adapter=actual,source=weights,path=str(destination.resolve()),files=files,
        model_tensor_sha256=digest,merge_contract=config['scientific']['deployment_contract'],before_cast_dtype='torch.float32',cast_count=1,
        after_cast_dtype='torch.float16',all_finite=True,all_parameters_cpu=True,cuda_before=before,cuda_after=after,
        zero_effect_identity=zero_identity,owner=gates.process(config,os.getpid()),config=config_ref,no_training=True)
    return atomic_json(destination.parent/'fp16-model-manifest.json',manifest,exclusive=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('config','config-sha256','request','request-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();config,ref=checked_json(dict(path=a.config,sha256=a.config_sha256));request,_=checked_json(dict(path=a.request,sha256=a.request_sha256))
    result=merge(config,ref,request,a.output);print(json.dumps(dict(accepted=True,manifest=result)))


if __name__=='__main__':main()
