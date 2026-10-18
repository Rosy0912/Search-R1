"""Execute only unsubmitted final ablation milestones on PJ from a bound Lumia export.

The export is produced by the existing Lumia inspect_inputs validator. Model,
training/config/data identities and all scientific settings stay immutable.
"""
import copy
import json
from pathlib import Path
import re
import socket
from .contracts import require,sha,digest,DATA_SHA,FINAL_SHA,IDS_SHA,TOTAL,validate_manifest,validate_model_files,validate_expected_rows

BASE='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
ALLOWED={('budget-5',150),('budget-20',150)}
MAPPED={'base','backend','model','inference_model','canonical_tokenizer','data_root','retriever_receipt','retriever_receipt_sha256','training_config','configuration_receipt','checkpoint_manifest','job','output','evaluation_run_id','evaluation_code_commit'}
EXTRA={'pj_export','pj_export_sha256','repeat_index','repeat_count'}


def pj_job_name(group, repeat_index, checkpoint_identity):
    require(group in ('budget-5','budget-20'),'Unknown final ablation')
    require(type(repeat_index) is int and repeat_index in (1,2,3),'Only fixed three repeats')
    require(re.fullmatch('[0-9a-f]{64}',checkpoint_identity) is not None,'Full checkpoint identity required')
    name='zhangjunming-ttt-'+('b5' if group=='budget-5' else 'b20')+'-150-r'+str(repeat_index)+'-'+checkpoint_identity[:4]
    require(len(name)<=32,'PJ job name exceeds API limit')
    return name


def validate_export(export):
    parent=export['plan']
    require(export['status']=='LUMIA_FULL_INPUTS_VERIFIED_UNSUBMITTED' and export['questions']==TOTAL,'Unverified origin export')
    require((parent['group'],parent['outer']) in ALLOWED and parent['backend']=='slurm_lumia' and 'repeat' not in parent,'Unrequested or already repeated origin')
    require(export['plan_sha256']==digest(parent) and export['config_digest']==digest(export['config']),'Origin export identity changed')
    claim=export['exclusive_claim']
    require(claim['group']==parent['group'] and claim['outer']==150 and claim['destination']=='pj_pt' and claim['repeats']==3 and claim['exact_scheduler_matches']==[] and claim['origin_output_absent'] is True,'Origin not exclusive and unsubmitted')
    return parent


def validate_projection(plan,export):
    parent=validate_export(export)
    require(set(plan)==set(parent)|EXTRA,'Unexpected PJ plan fields')
    require({k:v for k,v in plan.items() if k not in MAPPED|EXTRA}=={k:v for k,v in parent.items() if k not in MAPPED},'Changed frozen model/data/science identity')
    require(plan['base']==BASE and plan['backend']=='pj_pt' and plan['data_root']==BASE+'/common-data','Unknown PJ physical roots')
    i=plan['repeat_index'];require(type(i) is int and i in (1,2,3) and plan['repeat_count']==3,'Only fixed three repeats')
    require(re.fullmatch('[0-9a-f]{32}',plan['evaluation_run_id']) is not None and plan['evaluation_run_id']!=parent['evaluation_run_id'],'Independent run identity required')
    require(re.fullmatch('[0-9a-f]{40}',plan['evaluation_code_commit']) is not None,'Frozen source required')
    stage=Path(BASE)/'ablation-imports'/plan['checkpoint_identity']
    expected={'model':stage/'model','canonical_tokenizer':stage/'tokenizer','inference_model':stage/'view','checkpoint_manifest':stage/'manifest.json','training_config':stage/'training.yaml','configuration_receipt':stage/'configuration.json','pj_export':stage/'export.json','retriever_receipt':Path(BASE)/'receipts/pj-retriever-ready.json'}
    require(all(Path(plan[k])==v for k,v in expected.items()),'Physical path projection escapes its stage')
    root=Path(BASE)/'ablation-seven-evaluations'/plan['checkpoint_identity']/('r'+str(i))
    require(Path(plan['output'])==root/'output' and plan['job']==pj_job_name(plan['group'],i,plan['checkpoint_identity']),'Wrong unique repeat namespace')
    return root


def mapped_config(plan,export):
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(copy.deepcopy(export['config']));OmegaConf.set_struct(cfg,False)
    cfg.actor_rollout_ref.model.path=plan['inference_model'];cfg.actor_rollout_ref.ref.model_path=plan['inference_model']
    data=Path(plan['data_root'])
    cfg.data.train_files=str(data/'train.parquet');cfg.data.val_files=str(data/'validation.parquet');cfg.data.final_val_files=str(data/'final.parquet')
    cfg.baselines.bapo.stage_validation_files=str(data/'stage.parquet');cfg.baselines.holdout.data_root=str(data);cfg.baselines.holdout.manifest_path=str(data/'manifest.json')
    cfg.baselines.local_runtime_host=socket.gethostname().split('.')[0];cfg.trainer.default_local_dir=str(Path(plan['output'])/'unused-checkpoints')
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    return cfg


def inspect(plan):
    require(sha(plan['pj_export'])==plan['pj_export_sha256'],'Bound export changed')
    export=json.loads(Path(plan['pj_export']).read_text());root=validate_projection(plan,export)
    for path,key in [('checkpoint_manifest','checkpoint_manifest_sha256'),('training_config','training_config_sha256'),('configuration_receipt','configuration_receipt_sha256'),('retriever_receipt','retriever_receipt_sha256')]:require(sha(plan[path])==plan[key],path+' changed')
    validate_manifest(plan,json.loads(Path(plan['checkpoint_manifest']).read_text()))
    receipt=json.loads(Path(plan['configuration_receipt']).read_text());require(receipt['run_id']==plan['training_run_id'] and receipt['config_sha256']==plan['training_config_sha256'] and receipt['data_identity']==DATA_SHA,'Origin training receipt changed')
    model_digest=validate_model_files(plan)
    require(plan['checkpoint_identity']==digest(dict(group=plan['group'],outer=150,run=plan['training_run_id'],training_config=plan['training_config_sha256'],model_files_sha256=model_digest,data=DATA_SHA)),'Origin checkpoint identity changed')
    data=Path(plan['data_root']);manifest=json.loads((data/'manifest.json').read_text())
    require(sha(data/'manifest.json')==plan['data_manifest_sha256']==DATA_SHA and sha(data/'final.parquet')==FINAL_SHA and plan['ordered_ids_sha256']==IDS_SHA,'PJ data identity changed')
    for split in manifest['splits'].values():
        if 'sha256' in split and 'path' in split:require(sha(data/Path(split['path']).name)==split['sha256'],'PJ split changed')
    import pandas as pd
    from search_r1.baselines.holdout import plain,frame_identity
    frame=pd.read_parquet(data/'final.parquet');rows=[]
    for _,r in frame.iterrows():
        gold=plain(r['reward_model'])['ground_truth']['target'];rows.append(dict(question_id=plain(r['extra_info'])['index'],source=r['data_source'],gold_answers=[gold] if isinstance(gold,str) else gold))
    validate_expected_rows(rows);require(frame_identity(frame,'prompt')==plan['final_content_identity'],'PJ final content changed')
    cfg=mapped_config(plan,export)
    from .model_view import validate_view
    validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
    return cfg,rows,root
