"""Immutable native512 templates for each completed PR online boundary."""
from pathlib import Path
import uuid
try:
    from .utils import *
except ImportError:
    from utils import *

BOUNDARIES=list(range(10,101,10))

def scientific():
    return dict(versions=['raw_bf16'],validation_questions=512,total_primary_slots=512,
        physical_to_logical=list(range(8)),rank_seed=42,temperature=0,max_context=8192,max_new_tokens=500,
        turns=5,max_search_calls=4,topk=3,observation_tokens=480,repetition_penalty=1.05,
        eos_token_id=[151645,151643],attention_backend='sdpa',adapter_dtype='torch.float32',
        primary_base_dtype='torch.bfloat16',hook_control_first_slots_per_rank=1,
        zero_effect_first_slots_per_rank=0,ordinary_prompt=True,privileged_reference=False,no_training=True)

def template(training_ref,base,helper_root,completed_outers):
    train,fp=checked_json(training_ref)
    require(train['schema']=='opsd-pr100-config-v1' and train['variant']=='PR'
        and train['scientific']['outer_iterations']==100 and completed_outers in BOUNDARIES,'Wrong PR100/boundary')
    base=Path(base).resolve(); helper_root=Path(helper_root).resolve()
    inv,_=checked_json(train['helper_source_manifest']); source=dict(inv['files']['memory_observer.py'])
    source['path']=str(Path(train['paths']['helper_root'])/'memory_observer.py')
    return dict(schema='opsd-pr100-native512-eval-config-v1',run_id=uuid.uuid4().hex,
        study_id=train['study_id'],variant='PR',evaluation_role='periodic_boundary',completed_outers=completed_outers,
        source_commit=None,helper_source_manifest=None,
        paths=dict(base=str(base),output_root=str(base/'runs'),helper_root=str(helper_root),python=train['paths']['python'],retriever_url=train['paths']['retriever_url']),
        training=dict(config=fp,run_id=train['run_id'],terminal=None,boundary_checkpoint=None,boundary=None),
        models={'raw_bf16':{}},owner=train['owner'],
        resources=dict(startup_minimum_free_mib=20480,running_minimum_free_mib=1024,minimum_disk_bytes=100<<30,
            shared_gpu_lease_paths=train['resources']['shared_gpu_lease_paths'],group_lock=str(Path(train['paths']['base'])/'online-experiment.lock'),
            study_lock=train['resources']['study_lease_path']),
        gates=dict(publication=None,storage=None,retriever_admission=train['gates']['retriever_admission'],
            memory_observer=dict(path=train['gates']['memory_binding_path'],sha256=None),memory_observer_config=fp,
            memory_observer_source=source,process_exclusions=[]),scientific=scientific(),state='NOT_READY')

def validate_config(config):
    require(config['schema']=='opsd-pr100-native512-eval-config-v1' and len(config['run_id'])==32
        and all(c in '0123456789abcdef' for c in config['run_id']),'Wrong periodic evaluator identity')
    train,_=original_config(config); expected=template(config['training']['config'],config['paths']['base'],config['paths']['helper_root'],config['completed_outers'])
    require(config['run_id']!=train['run_id'] and config['scientific']==expected['scientific'],'Changed native512 protocol')
    require(config['owner']==train['owner'] and config['resources']==expected['resources'],'Changed eight-card allocation/leases')
    require(config['paths']==expected['paths'] and config['variant']=='PR' and config['evaluation_role']=='periodic_boundary','Changed paths/role')
    require(set(config['models'])=={'raw_bf16'} and config['gates']['retriever_admission']==train['gates']['retriever_admission'],'Changed evaluation slots/oracle')
    require(Path(config['paths']['base']).resolve()!=Path(train['paths']['base']).resolve(),'Evaluation cannot write training base')
    return True

def adapter(config,version,full=True):
    require(version=='raw_bf16','Only requested BF16 ordinary evaluation is enabled')
    train,_=original_config(config); value,mp=checked_json(config['training']['boundary_checkpoint'])
    modules,_=frozen_helpers(config)
    require(value['accepted'] is True and value['run_id']==train['run_id'] and value['config_sha256']==config['training']['config']['sha256']
        and value['contract']==modules['checkpoint'].science_contract(train)
        and value['source_model_identity']==train['source_model_manifest']['sha256'] and value['runtime_identity']==train['runtime_manifest']['sha256']
        and value['common_initial_adapter']==train['common_initial_adapter_manifest']
        and value['cursor']['outer_index']==config['completed_outers']-1
        and value['cursor']['next_pair_cursor']==value['cursor']['usable_pair_count']
        and value['complete_rank_count']==8 and value['equal_optimizer_steps'] is True
        and len(value['all_rank_adapter_sha256'])==8 and len(set(value['all_rank_adapter_sha256']))==1,'Not this completed raw boundary checkpoint')
    root=Path(value['adapter']['path']).resolve();files=value['adapter']['files']
    require(root.is_relative_to(Path(train['paths']['output_root']).resolve()/'checkpoints'),'Boundary adapter escaped training checkpoints')
    proofs=[]
    for f in files:
        require(Path(f['path']).resolve().parent==root,'Adapter payload escaped directory')
        proofs.append(fileproof(f['path'],f['sha256']) if full else f)
    peft,pc=checked_json(next(x for x in files if Path(x['path']).name=='adapter_config.json'))
    require(peft['r']==16 and peft['lora_alpha']==32 and peft['lora_dropout']==0 and peft.get('bias')=='none'
        and set(peft['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'},'Changed actual LoRA contract')
    return dict(path=str(root),tensor_sha256=value['all_rank_adapter_sha256'][0],files=proofs,manifest=mp,checkpoint=mp,
        adapter_config=pc,common_initial_adapter=train['common_initial_adapter_manifest'],zero_effect_required=False)
