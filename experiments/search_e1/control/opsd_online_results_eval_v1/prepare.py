"""Prepare an explicitly NOT_READY independent native512 configuration, CPU only."""
import argparse
from pathlib import Path
import uuid
try:
    from .utils import *
except ImportError:
    from utils import *

def template(training_config, base, helper_root, cp0, cp89):
    train, fp = checked_json(training_config)
    require(fp['sha256']==TRAIN_CONFIG_SHA and train['run_id']==TRAIN_RUN, 'Wrong online config')
    base=Path(base).resolve(); helper_root=Path(helper_root).resolve()
    return dict(schema='opsd-online-results-eval-config-v1',run_id=uuid.uuid4().hex,
        source_commit=None,helper_source_manifest=None,
        paths=dict(base=str(base),output_root=str(base/'runs'),helper_root=str(helper_root),
                   python=train['paths']['python'],retriever_url=train['paths']['retriever_url']),
        training=dict(config=fp,terminal=None,completion200=None,final_state_audit=None,final_state_execution=None,final_reader_source=None,final_reader_publication=None),
        models=dict(source100=dict(checkpoint=cp0),cp89=dict(checkpoint=cp89),final199=dict(checkpoint=None)),
        owner=train['owner'],
        resources=dict(startup_minimum_free_mib=20480,running_minimum_free_mib=1024,minimum_disk_bytes=100<<30,
             shared_gpu_lease_paths=train['resources']['shared_gpu_lease_paths'],
             original_online_lock=str(Path(train['paths']['base'])/'online-experiment.lock')),
        gates=dict(publication=None,storage=None,retriever_admission=train['gates']['retriever_admission'],memory_observer=None,process_exclusions=[]),
        scientific=dict(versions=list(VERSIONS),validation_questions=512,total_primary_slots=1536,
             physical_to_logical=list(range(8)),rank_seed=42,temperature=0,max_context=8192,max_new_tokens=500,
             turns=5,max_search_calls=4,topk=3,observation_tokens=480,repetition_penalty=1.05,
             eos_token_id=[151645,151643],attention_backend='sdpa',base_dtype='torch.bfloat16',adapter_dtype='torch.float32',
             hook_control_first_slots_per_rank=1,source_zero_effect_first_slots_per_rank=1,
             ordinary_prompt=True,privileged_reference=False,no_training=True),
        state='NOT_READY: bind actual final199, terminal, full200 audit, publication and storage before launch')

def checkpoint(config, version, full=True):
    train,_=original_config(config); ref=config['models'][version].get('checkpoint')
    require(ref is not None,'NOT_READY: explicit checkpoint missing for '+version)
    value,fp=checked_json(ref)
    require(value['schema']=='opsd-online-checkpoint-v1' and value['accepted'] is True
        and value['run_id']==TRAIN_RUN and value['config_sha256']==TRAIN_CONFIG_SHA
        and value['runtime_identity']==train['runtime_manifest']['sha256']
        and value['source_model_identity']==train['source_model_manifest']['sha256']
        and value['complete_rank_count']==8 and value['equal_optimizer_steps'] is True
        and len(value['ranks'])==8 and {r['rank'] for r in value['ranks']}==set(range(8))
        and len(value['all_rank_adapter_sha256'])==8 and len(set(value['all_rank_adapter_sha256']))==1,
        'Checkpoint scientific/run/rank identity differs')
    if version=='source100': require(value['optimizer_updates']==0 and value['cursor']['outer_index']==0 and value['cursor']['next_pair_cursor']==0,'Not actual CP0')
    elif version=='cp89': require(fp['sha256']==CP89_SHA and value['optimizer_updates']==89 and value['cursor']['outer_index']==8,'Not protected preselected CP89')
    else: require(value['cursor']['outer_index']==199 and value['cursor']['next_pair_cursor']==value['cursor']['usable_pair_count'],'Not raw outer199 end')
    path=Path(value['adapter']['path']).resolve()
    require(path.is_relative_to(Path(train['paths']['output_root']).resolve()/'checkpoints'),'Adapter outside original checkpoints')
    require({Path(r['path']).name for r in value['adapter']['files']}=={'README.md','adapter_config.json','adapter_model.safetensors'},'Adapter native file inventory differs')
    proofs=[]
    for item in value['adapter']['files']:
        require(Path(item['path']).resolve().parent==path,'Adapter file escaped exact directory')
        if full: proofs.append(fileproof(item['path'],item['sha256']))
        else: proofs.append(item)
    peft,_=checked_json(next(x for x in value['adapter']['files'] if Path(x['path']).name=='adapter_config.json'))
    require(peft['r']==16 and peft['lora_alpha']==32 and peft['lora_dropout']==0
        and set(peft['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        and peft.get('bias')=='none','Actual LoRA configuration differs')
    return value,dict(checkpoint=fp,adapter_files=proofs,adapter_config=peft)

def validate_config(config):
    require(config['schema']=='opsd-online-results-eval-config-v1' and config['run_id']!=TRAIN_RUN,'Wrong independent evaluation run')
    require(len(config['run_id'])==32 and config['owner']['uid']==1005 and config['owner']['physical_gpus']==list(range(8))
        and len(set(config['owner']['gpu_uuids']))==8,'Wrong eight-card execution identity')
    train,_=original_config(config)
    expected=template(config['training']['config'],config['paths']['base'],config['paths']['helper_root'],None,None)
    require(config['scientific']==expected['scientific'],'Evaluation scientific budget/defaults changed')
    require(config['owner']==train['owner'] and config['resources']==expected['resources'],'Execution allocation/leases changed')
    require(config['paths']['python']==train['paths']['python'] and config['paths']['retriever_url']==train['paths']['retriever_url'],'Changed environment/oracle')
    require(config['paths']['retriever_url']=='http://10.120.17.101:18004/retrieve','Only current original admitted18004')
    require(config['gates']['retriever_admission']==train['gates']['retriever_admission'],'Original oracle admission changed')
    return True

def main():
    p=argparse.ArgumentParser()
    for name in ('training-config','training-config-sha256','base','helper-root','cp0','cp0-sha256','cp89','cp89-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(); cfg=template(dict(path=a.training_config,sha256=a.training_config_sha256),a.base,a.helper_root,
       dict(path=a.cp0,sha256=a.cp0_sha256),dict(path=a.cp89,sha256=a.cp89_sha256))
    validate_config(cfg); source_files(cfg); runtime_files(cfg); cohort(cfg)
    checkpoint(cfg,'source100');checkpoint(cfg,'cp89')
    out=atomic_json(a.output,cfg,exclusive=True)
    print(json.dumps(dict(config=out,ready=False,decision='HOLD',gpu_started=False)))

if __name__=='__main__':main()
