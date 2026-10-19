"""Accepted official Tree-GRPO 3B final actor on the unchanged 3B evaluator.

This input contract is separate from the original method's native manifest and
from the ablation-only entry. Scientific runner/worker/bootstrap stay unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .contracts import (DATA_SHA,FINAL_SHA,IDS_SHA,TOTAL,digest,require,sha,
                        validate_model_files,validate_expected_rows)
from .base_contracts import FILES as BASE_FILES

BASE='/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1'
TRAIN_UID='227708e6-4b7b-4262-9407-7aa8c2644731'
TRAIN_SOURCE='6b7779780e5fa52cf3cf0dcc695bbba44ac6f569'
HF_ID='24af8d792f71de114cc92155aface4fe38c3c1310feda2b995348a0c7e8bbb68'
MODEL=BASE+'/official-treegrpo/exports/fair-replacement-v4-final-v1/huggingface'
INPUTS=BASE+'/official-treegrpo/final-evaluation-inputs-v1'
CONFIG_DIGEST='ba2c73f475e9f16c03d53be51f50d3fbb7effc9f7bb35164b0f6c72233562308'
EVIDENCE={
    'native-acceptance.json':'3ba4225ceebb34a48a3112f7af77bd42d3760dc71458006cde3d5e79befbff38',
    'hf-acceptance.json':'cf3d04d96717cd6c6edc21603ed3806c5aa6f418a24f55cfcb7644961bbfd831',
    'hf-reload.json':'ddf846c54f2510acf332b3341e181492f85376cc23626a5f1e011baa26f3a031',
    'common-config-export.json':'06cb49440638193e2c0e613b4838d74d7b21394a545754d584aba14f943b1d68'}
SCIENCE={
    'reproduction/seven_dataset/worker.py':'7dad706109c9ee08a9d83bd7dd8ffb099f5a768378b9cca4da9888916b39d31c',
    'reproduction/seven_dataset/runner.py':'7004276ae17ce61fbd8b8a95b14f8008a8021d313f22d49d9f08c92d5f0c8fc4',
    'reproduction/seven_dataset/entry.py':'0788c0a261cd9e738e95ee6395c0b84069c052b4a3f272282b23db018570413e',
    'reproduction/seven_dataset/bootstrap.py':'240d666b5974a9470d183fe197944d8bcb82f6d25e764ddf3c031b19b0207cfb',
    'search_r1/baselines/local_runtime.py':'7712a8802f63f148680d59fc8dfa00869f78e3986cdb3fd10004635c8c465a5b'}
ALLOCATION=dict(gpus=8,cpus=64,memory_gib=600,machine_type='n2ls.ie.i60b',
                gpu_model='A100',gpu_memory_gib=80)
PLAN_FIELDS={'protocol','group','outer','repeat_count','repeat_index','base','backend','model',
             'training_uid','checkpoint_identity','job','output','evidence','model_files',
             'data_root','data_manifest_sha256','ordered_ids_sha256','final_content_identity',
             'canonical_tokenizer','canonical_tokenizer_files','inference_model','retriever_receipt',
             'retriever_receipt_sha256','allocation','no_auto_retry','evaluation_code_commit','evaluation_run_id'}


def checkpoint_identity():
    return digest(dict(method='official_treegrpo',training_uid=TRAIN_UID,
                       actual_iterations=179,hf_file_identity=HF_ID,data=DATA_SHA))


def output_root(index):
    require(type(index) is int and index in (1,2,3),'Only fixed three repeats')
    return Path(BASE)/'official-treegrpo/final-seven-evaluations'/HF_ID/('r'+str(index))


def job_name(index):
    output_root(index)
    name='zhangjunming-tg3-final-r'+str(index)+'-'+HF_ID[:4]
    require(len(name)<=32,'PJ name length')
    return name


def validate_lineage(native,hf,reload,export):
    require(native['uid']==hf['uid']==reload['uid']==TRAIN_UID,'Wrong trained allocation')
    require(native['source']==TRAIN_SOURCE and native['actual_iterations']==179
            and native['directory_label']==180 and native['actual_adam_per_rank']==1432,
            'Actual official179 native identity differs')
    require(native['status']=='CPU_ACTOR_NATIVE_FILE_TENSOR_ADAM_RNG_SCHEDULER_AUDIT_PASSED'
            and native['local_small_backup_verified'] and len(native['files'])==24
            and len(native['workers'])==8,'Native audit incomplete')
    require(all(w['adam_min']==w['adam_max']==1432 and w['scheduler']['last_epoch']==179
                for w in native['workers']),'Eight native Adam/scheduler states incomplete')
    require(hf['status']=='FINAL_HF_OFFICIAL_CPU_MERGE_ALL_TENSORS_AND_METADATA_ACCEPTED'
            and hf['local_small_backup_verified'] and hf['target']==MODEL
            and hf['hf_file_identity_sha256']==HF_ID
            and hf['native_acceptance_sha256']==EVIDENCE['native-acceptance.json'],
            'HF export lineage differs')
    comparison=hf['parameter_comparison']
    require(comparison['exact_native_bf16_projection_all_parameters']
            and comparison['logical_parameters']==435
            and comparison['rank_parameter_comparisons']==3480
            and comparison['parameters_different_from_initialization'], 'Incomplete HF tensor comparison')
    require(reload['status']=='ACTUAL_HF_CPU_RELOAD_ALL435_PARAMETERS_AND_TIED_STORAGE_VERIFIED'
            and reload['hf_file_identity_sha256']==HF_ID and reload['logical_parameters_compared']==435
            and reload['physical_tied_tensors_equal'] and reload['loaded_tied_parameter_pointer_shared']
            and reload['export_acceptance_sha256']==EVIDENCE['hf-acceptance.json'], 'Actual HF reload differs')
    require(export['status']=='EXACT_BASE_PROTOCOL_INPUTS_VERIFIED' and export['questions']==TOTAL
            and export['config_digest']==digest(export['config'])==CONFIG_DIGEST,'Common original200/Base config changed')


def validate_plan(plan,hf):
    require(set(plan)==PLAN_FIELDS,'Unknown or missing final evaluation plan fields')
    require(plan['protocol']=='official_treegrpo_final51713_pj_v1' and plan['group']=='Tree-GRPO'
            and type(plan['outer']) is int and plan['outer']==179
            and type(plan['repeat_count']) is int and plan['repeat_count']==3,'Wrong scientific group or repeat count')
    require(plan['base']==BASE and plan['backend']=='pj_pt' and plan['model']==MODEL
            and plan['training_uid']==TRAIN_UID and plan['checkpoint_identity']==checkpoint_identity(),
            'Wrong physical training/HF identity')
    require(plan['job']==job_name(plan['repeat_index']) and Path(plan['output'])==output_root(plan['repeat_index'])/'output',
            'Wrong unique repeat namespace')
    require(plan['evidence']=={n:dict(path=str(Path(INPUTS)/n),sha256=h) for n,h in EVIDENCE.items()},'Unbound evidence paths')
    require(plan['model_files']=={n:m['sha256'] for n,m in hf['files'].items()},'HF file binding differs')
    require(plan['data_root']==BASE+'/common-data' and plan['data_manifest_sha256']==DATA_SHA
            and plan['ordered_ids_sha256']==IDS_SHA,'Different scientific data')
    require(plan['canonical_tokenizer']==BASE+'/assets/model'
            and plan['inference_model']==BASE+'/official-treegrpo/final-seven-model-view-v1','Wrong tokenizer view')
    require(plan['canonical_tokenizer_files']=={n:BASE_FILES[n] for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']},'Different canonical tokenizer files')
    require(plan['retriever_receipt']==BASE+'/receipts/pj-retriever-ready.json','Wrong retriever namespace')
    require(plan['allocation']==ALLOCATION and plan['no_auto_retry'] is True,'Different resource/retry contract')
    require(re.fullmatch('[0-9a-f]{40}',plan['evaluation_code_commit']) is not None
            and re.fullmatch('[0-9a-f]{32}',plan['evaluation_run_id']) is not None,'Unfrozen source/run identity')
    return output_root(plan['repeat_index'])


def inspect(plan):
    evidence={}
    for name,m in plan['evidence'].items():
        require(sha(m['path'])==m['sha256']==EVIDENCE[name],'Evidence SHA differs: '+name)
        evidence[name]=json.loads(Path(m['path']).read_text())
    native,hf,reload,export=[evidence[n] for n in EVIDENCE]
    validate_lineage(native,hf,reload,export)
    root=validate_plan(plan,hf)
    src=Path(BASE)/'sources'/plan['evaluation_code_commit']
    for name,h in SCIENCE.items():require(sha(src/name)==h,'Common scientific source changed: '+name)
    validate_model_files(plan)
    data=Path(plan['data_root']);require(sha(data/'manifest.json')==DATA_SHA and sha(data/'final.parquet')==FINAL_SHA,'Data changed')
    manifest=json.loads((data/'manifest.json').read_text())
    for split in manifest['splits'].values():
        if 'path' in split and 'sha256' in split:require(sha(data/Path(split['path']).name)==split['sha256'],'Data split changed')
    import pandas as pd
    from search_r1.baselines.holdout import plain,frame_identity
    frame=pd.read_parquet(data/'final.parquet');expected=[]
    for _,row in frame.iterrows():
        gold=plain(row['reward_model'])['ground_truth']['target']
        expected.append(dict(question_id=plain(row['extra_info'])['index'],source=row['data_source'],
                             gold_answers=[gold] if isinstance(gold,str) else gold))
    validate_expected_rows(expected)
    require(frame_identity(frame,'prompt')==plan['final_content_identity'],'Final frame identity differs')
    from .base_pj import mapped_config
    from .model_view import validate_view
    from .searchr1_contracts import validate_tokenizer_equivalence
    cfg=mapped_config(plan,export)
    validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
    validate_tokenizer_equivalence(plan)
    cfg.actor_rollout_ref.model.path=plan['inference_model'];cfg.actor_rollout_ref.ref.model_path=plan['inference_model']
    require(cfg.max_turns==4 and cfg.force_final_answer is False and cfg.retriever.topk==3
            and cfg.trainer.n_gpus_per_node==8 and cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==1,
            'Frozen 3B evaluation parameters differ')
    return cfg,expected,root


def validate_gate(plan,submission,gate):
    require(gate['plan_sha256']==submission['plan_sha256'],'Gate plan SHA differs')
    require(gate['source']==plan['evaluation_code_commit'],'Gate source differs')
    actual=gate['actual']
    require(actual['name']==submission['job']==plan['job'] and actual['uid']==submission['uid']
            and actual['state']=='RUNNING' and actual['resources']==ALLOCATION,'Actual scheduler allocation differs')
    return actual


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text())
    if a.preflight_only:
        cfg,expected,root=inspect(plan)
        print(json.dumps(dict(status='OFFICIAL_TREE3B_FINAL_HF_COMMON51713_CPU_PREFLIGHT_PASSED',questions=len(expected),checkpoint_identity=plan['checkpoint_identity'])));return
    require(os.environ.get('SEARCH_R1_EXECUTION_BACKEND')=='pj_pt' and not os.environ.get('SLURM_JOB_ID'),'Wrong backend')
    require(os.environ['SEARCH_R1_JOB_ID']==plan['job'] and os.environ['RECURSIVE_SOURCE_COMMIT']==plan['evaluation_code_commit'],'Wrong execution source/job')
    root=output_root(plan['repeat_index']);deadline=time.monotonic()+120
    while not (root/'scheduler-release-once.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('External exactUID gate missing; no scientific worker started')
        time.sleep(.5)
    sub=json.loads((root/'submission.json').read_text());gate=json.loads((root/'scheduler-release-once.json').read_text())
    require(sub['plan_sha256']==sha(a.plan),'Submission plan changed');actual=validate_gate(plan,sub,gate)
    inv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    require(len(inv)==8 and len({r.split(',')[1].strip() for r in inv})==8
            and all('A100' in r and float(r.split(',')[3])>=80000 for r in inv),'Actual8A10080 required')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'Do not preempt GPU processes')
    with (root/'entry-once.json').open('x') as f:json.dump(dict(at=time.time(),actual=actual,gpus=inv,plan_sha256=sha(a.plan)),f,indent=2)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(r.split(',')[0].strip() for r in inv)
    cfg,expected,root=inspect(plan)
    from .entry import run_frozen
    run_frozen(plan,a.plan,cfg,expected,root)


if __name__=='__main__':main()
