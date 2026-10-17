"""Move only never-started base repeats to PJ using an audited frozen config export."""
import copy
import json
from pathlib import Path
import re
import socket
from .contracts import (require,sha,digest,FINAL_SHA,IDS_SHA,TOTAL,validate_model_files,
                        validate_expected_rows,validate_replacement_state)
from .base_contracts import validate_base_plan,FILES

BASE='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
MODEL='/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261005-PJ/model'
EXPORT_CONFIG_DIGEST='ba2c73f475e9f16c03d53be51f50d3fbb7effc9f7bb35164b0f6c72233562308'
CHANGED={'protocol','base','backend','data_root','retriever_receipt','retriever_receipt_sha256',
         'model','evaluation_code_commit','evaluation_run_id','job','output'}
LINKS={'origin_plan','origin_plan_sha256','origin_submission','origin_submission_sha256',
       'config_export','config_export_sha256','retirement','retirement_sha256'}


def read_bound(path,expected):
    require(sha(path)==expected,'Bound migration evidence changed: '+str(path))
    return json.loads(Path(path).read_text())


def validate_migration(plan,parent,submission,export,retirement=None):
    validate_base_plan(parent,export['template'])
    require(set(plan)==set(parent)|{'migration'} and set(plan['migration'])==LINKS,
            'Unexpected migration fields')
    require({k:v for k,v in plan.items() if k not in CHANGED|{'migration'}}==
            {k:v for k,v in parent.items() if k not in CHANGED},'Migration changes model/data/science/repeat')
    require(export['status']=='EXACT_BASE_PROTOCOL_INPUTS_VERIFIED' and export['questions']==TOTAL
            and digest(export['config'])==export['config_digest']==EXPORT_CONFIG_DIGEST,
            'Unverified reference configuration export')
    require(export['template_sha256']==parent['template_plan_sha256'] and
            export['plan']['checkpoint_identity']==parent['checkpoint_identity'] and
            export['plan']['protocol_identity']==parent['protocol_identity'], 'Reference template binding differs')
    require(submission['job_id'].isdigit() and submission['job']==parent['job'] and
            submission['plan_sha256']==plan['migration']['origin_plan_sha256'] and
            submission['checkpoint_identity']==parent['checkpoint_identity'] and
            submission['output']==parent['output'] and submission['repeat_index']==parent['repeat_index'],
            'Origin allocation differs')
    require(plan['protocol']=='base_seven_pj_migration_v1' and plan['backend']=='pj_pt'
            and plan['base']==BASE and plan['model']==MODEL and plan['data_root']==BASE+'/common-data'
            and plan['retriever_receipt']==BASE+'/receipts/pj-retriever-ready.json', 'Unknown PJ paths/backend')
    require(re.fullmatch('[0-9a-f]{40}',plan['evaluation_code_commit']) is not None
            and re.fullmatch('[0-9a-f]{32}',plan['evaluation_run_id']) is not None
            and plan['evaluation_run_id']!=parent['evaluation_run_id'],'Migration needs new immutable execution identity')
    root=Path(BASE)/'baseline-seven-evaluations'/parent['checkpoint_identity']/parent['protocol_identity']/('r'+str(parent['repeat_index']))/'pj-v1'
    require(plan['job']=='zhangjunming-ttt-base-r'+str(parent['repeat_index'])+'-pj'
            and Path(plan['output'])==root/'output','PJ repeat namespace differs')
    if retirement is not None:
        require(retirement['job_id']==submission['job_id'] and
                retirement['plan_sha256']==submission['plan_sha256'] and
                retirement['prediction_bytes']==0 and retirement['progress_exists'] is False and
                retirement['status']=='CANCELLED_NEVER_STARTED_ZERO_PREDICTIONS' and
                len(retirement['observations'])==2,'Origin was not safely retired')
        for row in retirement['observations']:
            require(row['job_id']==submission['job_id'] and row['active_queue']==[], 'Origin still active')
            validate_replacement_state(row['state'],row['elapsed'],row['start'],'UNSTARTED_DEPENDENCY_BLOCKED')
    return root


def mapped_config(plan,export):
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(copy.deepcopy(export['config']));OmegaConf.set_struct(cfg,False)
    cfg.actor_rollout_ref.model.path=plan['model'];cfg.actor_rollout_ref.ref.model_path=plan['model']
    data=Path(plan['data_root'])
    cfg.data.train_files=str(data/'train.parquet');cfg.data.val_files=str(data/'validation.parquet')
    cfg.data.final_val_files=str(data/'final.parquet');cfg.baselines.bapo.stage_validation_files=str(data/'stage.parquet')
    cfg.baselines.holdout.data_root=str(data);cfg.baselines.holdout.manifest_path=str(data/'manifest.json')
    cfg.baselines.local_runtime_host=socket.gethostname().split('.')[0]
    cfg.trainer.default_local_dir=str(Path(plan['output'])/'unused-checkpoints')
    require(cfg.baselines.tree.token_preservation.tokenizer_sha256==FILES['tokenizer.json'],'Tokenizer mismatch')
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    return cfg


def inspect_pj_base(plan,require_retired=True):
    m=plan['migration'];parent=read_bound(m['origin_plan'],m['origin_plan_sha256'])
    sub=read_bound(m['origin_submission'],m['origin_submission_sha256'])
    export=read_bound(m['config_export'],m['config_export_sha256'])
    retired=read_bound(m['retirement'],m['retirement_sha256']) if require_retired else None
    root=validate_migration(plan,parent,sub,export,retired)
    validate_model_files(plan)
    data=Path(plan['data_root']);require(sha(data/'manifest.json')==plan['data_manifest_sha256']
        and sha(data/'final.parquet')==FINAL_SHA,'PJ data files differ')
    import pandas as pd
    from search_r1.baselines.holdout import plain,frame_identity
    frame=pd.read_parquet(data/'final.parquet');expected=[]
    for _,r in frame.iterrows():
        gold=plain(r['reward_model'])['ground_truth']['target'];gold=[gold] if isinstance(gold,str) else gold
        expected.append(dict(question_id=plain(r['extra_info'])['index'],source=r['data_source'],gold_answers=gold))
    validate_expected_rows(expected)
    require(frame_identity(frame,'prompt')==plan['final_content_identity'],'PJ content identity changed')
    return mapped_config(plan,export),expected,root
