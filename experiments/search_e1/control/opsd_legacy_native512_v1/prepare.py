"""Optional L only: explicit fixed99 qualification, never reverse training metadata."""
from pathlib import Path
try:
    from .utils import *
except ImportError:
    from utils import *


def scientific():
    return dict(versions=list(VERSIONS),validation_questions=512,total_primary_slots=512,
        physical_to_logical=list(range(8)),rank_seed=42,temperature=0,max_context=8192,max_new_tokens=500,
        turns=5,max_search_calls=4,topk=3,observation_tokens=480,repetition_penalty=1.05,
        eos_token_id=[151645,151643],attention_backend='sdpa',adapter_dtype='torch.float32',
        primary_base_dtype='torch.bfloat16',hook_control_first_slots_per_rank=1,
        ordinary_prompt=True,privileged_reference=False,no_training=True)


def qualification(config):
    require(config['qualification']['sha256']==QUALIFICATION_SHA,'Actual accepted adapter qualification SHA differs')
    value,proof=checked_json(config['qualification'])
    require(value['schema']=='optional-L-adapter-only-qualification-v1' and value['accepted'] is True
        and value['historical_run_id']==HISTORICAL_RUN and value['label']=='L_historical_forward_outer10_Adam99'
        and value['target']['path']==TARGET_PATH and value['target']['sha256']==TARGET_SHA
        and value['actual_tensor_sha256']==L_TENSOR_SHA and value['adapter_tensors']==504
        and value['adapter_finite_fp32'] is True and value['common_actual_tensor_sha256']==COMMON_TENSOR_SHA
        and value['cuda_initialized'] is False and value['full_Adam_or_RNG_read'] is False,
        'Wrong historical adapter-only qualification')
    require(config['checkpoint99']['path']==TARGET_PATH and config['checkpoint99']['sha256']==TARGET_SHA,
        'Only the immutable historical CP99 is allowed, no best/CP374')
    train,_=original_config(config)
    require(value['source_model_manifest']['sha256']==train['source_model_manifest']['sha256']
        and value['runtime_manifest']['sha256']==train['runtime_manifest']['sha256']
        and value['common_initial']['sha256']==train['common_initial_adapter_manifest']['sha256'],
        'Historical/common science qualification differs')
    return value,proof


def validate_config(config):
    require(config['schema']=='opsd-legacy-native512-config-v1' and config['state']=='READY'
        and config['variant']=='L' and config['evaluation_role']=='historical_forward_appendix'
        and len(config['run_id'])==32 and all(c in '0123456789abcdef' for c in config['run_id']),
        'Explicit ready independent L config required')
    train,_=original_config(config);qualification(config)
    base=Path(config['paths']['base']).resolve()
    require(base.is_relative_to(Path(train['paths']['base']).resolve().parent)
        and base!=Path(train['paths']['base']).resolve()
        and Path(config['paths']['output_root']).resolve()==base/'runs','Independent L output root required')
    require(config['scientific']==scientific() and config['models']=={VERSIONS[0]:dict(deployment_model=None)},
        'Changed ordinary native BF16512 science/model')
    require(config['owner']==train['owner'] and config['paths']['python']==train['paths']['python']
        and config['paths']['retriever_url']==train['paths']['retriever_url'],'Changed source environment/oracle/UUID owner')
    expected=dict(startup_minimum_free_mib=20480,running_minimum_free_mib=1024,minimum_disk_bytes=100<<30,
        shared_gpu_lease_paths=train['resources']['shared_gpu_lease_paths'],group_lock=str(base/'legacy-experiment.lock'),
        study_lock=train['resources']['study_lease_path'])
    require(config['resources']==expected and len(set(expected['shared_gpu_lease_paths']))==5,'Changed original shared resources/leases')
    require(config['gates']['retriever_admission']==train['gates']['retriever_admission'],
        'Must use the actual admitted common18004 route')
    require(config['run_id']!=train['run_id'] and config['run_id']!=HISTORICAL_RUN,'L evaluation cannot reuse a training run')
    require(set(config['comparison_inputs'])=={'bf16_zero','C0'},'Only prespecified same-BF16 comparisons')
    return True


def adapter(config,version,full=True):
    require(version==VERSIONS[0],'Only the explicit L99 BF16 adapter is allowed')
    q,qp=qualification(config);value,mp=checked_json(config['checkpoint99'])
    train,_=original_config(config)
    require(value['schema']=='opsd-online-checkpoint-v1' and value['accepted'] is True
        and value['run_id']==HISTORICAL_RUN and value['config_sha256']==q['config']['sha256']
        and value['optimizer_updates']==99 and value['cursor']['outer_index']==9
        and value['cursor']['next_pair_cursor']==value['cursor']['usable_pair_count']
        and value['complete_rank_count']==8 and value['equal_optimizer_steps'] is True
        and value['all_rank_adapter_sha256']==[L_TENSOR_SHA]*8
        and value['source_model_identity']==SOURCE_MANIFEST_SHA and value['runtime_identity']==RUNTIME_SHA,
        'Not the qualified actual historical99 endpoint')
    files=value['adapter']['files'];root=Path(value['adapter']['path']).resolve()
    require(root==Path(TARGET_PATH).parent/'adapter' and {Path(f['path']).name for f in files}
        =={'README.md','adapter_config.json','adapter_model.safetensors'} and len(files)==3,
        'Exact fixed three-file adapter inventory required')
    proofs=[]
    for f in files:
        require(Path(f['path']).resolve().parent==root,'Adapter escaped exact99 directory')
        old=next(x for x in q['target_adapter_files'] if x['path']==f['path'])
        require(all(f[k]==old[k] for k in ('sha256','bytes')),'Qualified adapter payload was replaced')
        proofs.append(fileproof(f['path'],f['sha256']) if full else f)
    peft,pc=checked_json(next(f for f in files if Path(f['path']).name=='adapter_config.json'))
    require(peft['r']==16 and peft['lora_alpha']==32 and peft['lora_dropout']==0 and peft.get('bias')=='none'
        and set(peft['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},
        'Actual L LoRA contract differs')
    return dict(path=str(root),tensor_sha256=L_TENSOR_SHA,files=proofs,manifest=mp,checkpoint=mp,
        adapter_config=pc,common_initial_adapter=train['common_initial_adapter_manifest'],zero_effect_required=False,
        qualification=qp,historical_optimizer_updates=99,historical_outer_index=9)
