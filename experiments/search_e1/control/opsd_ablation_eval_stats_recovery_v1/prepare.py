"""Independent per-group raw10 evaluation contract; CPU preparation never spawns."""
import argparse
from pathlib import Path
import uuid
try:
    from .utils import *
except ImportError:
    from utils import *


def expected_versions(variant, role):
    require(variant in VARIANTS and role in ('group','common_zero'),'Unknown variant/evaluation role')
    return ['bf16_zero','fp16_zero'] if role=='common_zero' else (['raw10_bf16','raw10_fp16'] if variant in ('P','J') else ['raw10_bf16'])


def scientific(variant,role):
    versions=expected_versions(variant,role)
    return dict(versions=versions,validation_questions=512,total_primary_slots=512*len(versions),
        physical_to_logical=list(range(8)),rank_seed=42,temperature=0,max_context=8192,max_new_tokens=500,
        turns=5,max_search_calls=4,topk=3,observation_tokens=480,repetition_penalty=1.05,
        eos_token_id=[151645,151643],attention_backend='sdpa',adapter_dtype='torch.float32',
        primary_base_dtype='torch.bfloat16',deployment_dtype='torch.float16',
        deployment_contract='CPU_FP32_base_FP32_LoRA_merge_then_one_FP16_cast',
        hook_control_first_slots_per_rank=1,zero_effect_first_slots_per_rank=1,
        ordinary_prompt=True,privileged_reference=False,no_training=True)


def template(training_ref,base,helper_root,role='group'):
    train,fp=checked_json(training_ref)
    require(train['schema']=='opsd-ablation-config-v1' and train['scientific']['outer_iterations']==10,'Not fresh reverse ten-outer training')
    base=Path(base).resolve();helper_root=Path(helper_root).resolve()
    inv,_=checked_json(train['helper_source_manifest']);memory_source=inv['files']['memory_observer.py']
    memory_source=dict(memory_source) if isinstance(memory_source,dict) else dict(sha256=memory_source)
    memory_source['path']=str(Path(train['paths']['helper_root'])/'memory_observer.py')
    return dict(schema='opsd-ablation-native512-eval-config-v1',run_id=uuid.uuid4().hex,
        study_id=train['study_id'],variant=train['variant'],evaluation_role=role,source_commit=None,helper_source_manifest=None,
        paths=dict(base=str(base),output_root=str(base/'runs'),helper_root=str(helper_root),python=train['paths']['python'],
            retriever_url=train['paths']['retriever_url']),
        training=dict(config=fp,run_id=train['run_id'],terminal=None,raw10_checkpoint=None,final_state_audit=None),
        models={version:dict(deployment_model=None) for version in expected_versions(train['variant'],role)},
        owner=train['owner'],resources=dict(startup_minimum_free_mib=20480,running_minimum_free_mib=1024,
            minimum_disk_bytes=100<<30,shared_gpu_lease_paths=train['resources']['shared_gpu_lease_paths'],
            group_lock=str(Path(train['paths']['base'])/'online-experiment.lock'),study_lock=train['resources']['study_lease_path']),
        gates=dict(publication=None,storage=None,retriever_admission=train['gates']['retriever_admission'],
            memory_observer=dict(path=train['gates']['memory_binding_path'],sha256=None),
            memory_observer_config=fp,memory_observer_source=memory_source,process_exclusions=[]),
        scientific=scientific(train['variant'],role),state='NOT_READY: exact group10/terminal/resources/source publication required')


def validate_config(config):
    require(config['schema']=='opsd-ablation-native512-eval-config-v1' and len(config['run_id'])==32
        and all(c in '0123456789abcdef' for c in config['run_id']),'Wrong independent evaluation config/run')
    train,_=original_config(config)
    expected=template(config['training']['config'],config['paths']['base'],config['paths']['helper_root'],config['evaluation_role'])
    require(config['run_id']!=train['run_id'] and config['scientific']==expected['scientific'],'Changed native512 science')
    require(config['owner']==train['owner'] and config['resources']==expected['resources'],'Changed eight-card identity/leases')
    require(config['paths']['python']==train['paths']['python'] and config['paths']['retriever_url']==train['paths']['retriever_url'],'Changed environment/oracle')
    require(set(config['models'])==set(config['scientific']['versions']),'Wrong model slots or denominator')
    require(Path(config['paths']['output_root']).resolve()==Path(config['paths']['base']).resolve()/'runs','Output outside independent eval root')
    require(Path(config['paths']['base']).resolve()!=Path(train['paths']['base']).resolve(),'Evaluator cannot write the training root')
    require(config['gates']['retriever_admission']==train['gates']['retriever_admission'],'Changed admitted training oracle')
    return True


def adapter(config,version,full=True):
    require(version in config['scientific']['versions'],'Unexpected evaluation version')
    train,_=original_config(config)
    if version.endswith('zero'):
        value,mp=checked_json(train['common_initial_adapter_manifest'])
        require(value.get('adapter_tensor_sha256',train['common_initial_adapter_tensor_sha256'])==train['common_initial_adapter_tensor_sha256'],'Common initial tensor digest differs')
        files=value['payload_files'];model=value['adapter_model']
        root=Path(model['path']).resolve().parent
        require({Path(x['path']).name for x in files}>={'adapter_config.json','adapter_model.safetensors'},'Common PEFT export incomplete')
        digest=train['common_initial_adapter_tensor_sha256'];checkpoint=None
    else:
        ref=config['training']['raw10_checkpoint'];require(ref is not None,'NOT_READY: raw10 checkpoint missing')
        value,mp=checked_json(ref)
        require(value.get('accepted') is True and value['run_id']==train['run_id'] and value['config_sha256']==config['training']['config']['sha256']
            and value['contract']==dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',
                variant=train['variant'],precision=train['precision']['mode'],mixture_alpha=.1 if train['variant'] in ('M','J') else 0.)
            and value['source_model_identity']==train['source_model_manifest']['sha256'] and value['runtime_identity']==train['runtime_manifest']['sha256']
            and value['common_initial_adapter']==train['common_initial_adapter_manifest']
            and value['cursor']['outer_index']==9 and value['cursor']['next_pair_cursor']==value['cursor']['usable_pair_count']
            and value['complete_rank_count']==8 and value['equal_optimizer_steps'] is True
            and len(value['all_rank_adapter_sha256'])==8 and len(set(value['all_rank_adapter_sha256']))==1,'Not this group raw tenth-outer adapter')
        files=value['adapter']['files'];root=Path(value['adapter']['path']).resolve();digest=value['all_rank_adapter_sha256'][0];checkpoint=mp
        require(root.is_relative_to(Path(train['paths']['output_root']).resolve()/'checkpoints'),'Raw adapter outside this group checkpoints')
    proofs=[]
    for f in files:
        require(Path(f['path']).resolve().parent==root,'Adapter payload escaped exact directory')
        proofs.append(fileproof(f['path'],f['sha256']) if full else f)
    peft,pc=checked_json(next(x for x in files if Path(x['path']).name=='adapter_config.json'))
    require(peft['r']==16 and peft['lora_alpha']==32 and peft['lora_dropout']==0 and peft.get('bias')=='none'
        and set(peft['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},'Actual LoRA contract differs')
    return dict(path=str(root),tensor_sha256=digest,files=proofs,manifest=mp,checkpoint=checkpoint,adapter_config=pc,
        common_initial_adapter=train['common_initial_adapter_manifest'],zero_effect_required=version.endswith('zero'))


def deployment(config,version,ref):
    require(version in ('fp16_zero','raw10_fp16'),'Deployment model allowed only explicit FP16 slots')
    expected=adapter(config,version);train,_=original_config(config);value,mp=checked_json(ref)
    require(value['schema']=='opsd-ablation-fp16-cpu-merged-model-v1' and value['accepted'] is True
        and value['run_id']==config['run_id'] and value['study_id']==config['study_id'] and value['variant']==config['variant'] and value['version']==version
        and value['training_run_id']==train['run_id'] and value['source_model_identity']==train['source_model_manifest']['sha256']
        and value['adapter_tensor_sha256']==expected['tensor_sha256'] and value['adapter_manifest_sha256']==expected['manifest']['sha256']
        and value['merge_contract']==config['scientific']['deployment_contract'] and value['before_cast_dtype']=='torch.float32'
        and value['cast_count']==1 and value['after_cast_dtype']=='torch.float16' and value['all_finite'] is True
        and value['cuda_before']==value['cuda_after']==dict(initialized=False,visible_devices=0)
        and value['all_parameters_cpu'] is True,'FP16 deployment merge provenance/device/dtype differs')
    root=Path(value['path']).resolve()
    require(root.is_relative_to(Path(config['paths']['output_root']).resolve()) and root.is_dir(),'Deployment outside this evaluation output')
    proofs=[fileproof(f['path'],f['sha256']) for f in value['files']]
    require({f['path'] for f in proofs}=={str(p.resolve()) for p in root.rglob('*') if p.is_file()}
        and all(Path(f['path']).resolve().is_relative_to(root) for f in proofs),'Deployment file inventory differs')
    return value,dict(manifest=mp,files=proofs)


def main():
    p=argparse.ArgumentParser()
    for name in ('training-config','training-config-sha256','base','helper-root','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=('group','common_zero'),default='group');a=p.parse_args()
    cfg=template(dict(path=a.training_config,sha256=a.training_config_sha256),a.base,a.helper_root,a.role)
    validate_config(cfg);fp=atomic_json(a.output,cfg,exclusive=True)
    print(json.dumps(dict(config=fp,ready=False,gpu_started=False)))


if __name__=='__main__':main()
