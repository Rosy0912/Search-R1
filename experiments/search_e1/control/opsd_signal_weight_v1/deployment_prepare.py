"""HKUST4 CPU final configuration binding; never imports models or starts jobs.

Big model/checkpoint/rollout/cache products remain at explicit remote paths.
The one immutable deployment identity also makes partial preparation resumable:
no subsequent call invents another group run ID or overwrites a config.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

try:
    from .checkpoint import checked_json,checked_file,fileproof,exclusive_json,require
    from .prepare import template,validate_config,verify_runtime,verify_common,VARIANTS,GPU_UUIDS,OLD
    from . import gates
except ImportError:
    from checkpoint import checked_json,checked_file,fileproof,exclusive_json,require
    from prepare import template,validate_config,verify_runtime,verify_common,VARIANTS,GPU_UUIDS,OLD
    import gates

STUDY='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
RETRIEVER='/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004'
ORACLE_GATE_SHA='f256ee31dea1297a07baa28ca9b722e678fbc3555669a3d2bcebcb00884a9443'
ORACLE_ENTRY_SHA='c1545c96b6b2f7ac50fcda2a7e3ef820170f1bc57d26f4b8a3881f14a0cc7fa6'
INITIAL_TENSOR_SHA='836590b798434f21160418c6471cbce3bdfac5bf4a9947c1e3e7723d057baf2f'
GOAL_CREATED_UNIX=1791564769
BACKUP='HKUST1:/data/shaoshitong/search-e1-opsd-signal-weight-backup-20261010'


def write_once(path,value):
    """Immutable JSON; retries only accept the exact existing value."""
    path=Path(path)
    if path.exists():
        saved,proof=checked_json(fileproof(path));require(saved==value,'Refuse overwrite of immutable deployment artifact: '+str(path))
        return proof
    return exclusive_json(path,value)


def remote_paths(config,study,helper):
    root=Path(study).resolve()
    require(root==Path(STUDY).resolve(),'Deployment is outside the newly authorized HKUST4 study')
    require(Path(helper).resolve().is_relative_to(root/'source') and Path(helper).name=='opsd_signal_weight_v1',
            'Helper is outside this study frozen source tree')
    for key in ('base','output_root','receipts','study_base','runtime','helper_root'):
        path=Path(config['paths'][key])
        require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(root),
                'Output/runtime/helper path escaped the authorized remote root: '+key)
    require(config['paths']['python']==OLD+'/env/bin/python','Use the actual compatible original K4 environment')
    require(config['paths']['model']==OLD+'/runs/archive/round-1-grpo/actor/global_step_100',
            'Only the accepted original K4 source100 weights may seed fresh groups')
    for key in ('train','validation'):
        require(config['paths'][key]==OLD+'/data/'+key+'.jsonl','Original K4 data path changed: '+key)
    require(config['paths']['cache']==str(root/'cache') and config['paths']['tmp']==str(root/'tmp'),
            'Cache and temporary output must be explicit remote paths')
    require(config['owner']['uid']==1001 and config['owner']['physical_gpus']==[4,5,6,7]
            and config['owner']['gpu_uuids']==GPU_UUIDS,'Unauthorized physical resource mapping')


def read_assets(reference):
    assets,proof=checked_json(reference)
    require(assets.get('schema')=='opsd-signal-assets-v1' and assets.get('accepted') is True
            and assets.get('runtime_files')==305 and assets.get('cuda_initialized') is False
            and assets.get('local_large_product_bytes')==0,'Fresh original asset preparation was not accepted')
    for key in ('source_model_manifest','runtime_manifest','common_initial_adapter_manifest','common_cohort_manifest',
                'train_identity','validation_identity'):
        checked_file(assets[key])
    source,_=checked_json(assets['source_model_manifest'])
    require(source.get('accepted') is True and source.get('source_grpo_step')==100 and len(source.get('files',[]))==12,
            'Source manifest does not bind the accepted12-file GRPO100')
    source_files=[checked_file(reference) for reference in source['files']]
    require(len({entry['path'] for entry in source_files})==12,'Duplicate source100 payload inventory')
    initial,_=checked_json(assets['common_initial_adapter_manifest'])
    require(initial.get('accepted') is True and initial.get('tensors')==504 and initial.get('zero_B_tensors')==252
            and initial.get('dtype')=='float32' and initial.get('adapter_tensor_sha256')==INITIAL_TENSOR_SHA
            and initial.get('source_model_identity')==assets['source_model_manifest']['sha256'],
            'Common zero-B FP32 adapter/source identity differs')
    cohort,_=checked_json(assets['common_cohort_manifest'])
    require(cohort.get('outer_count')==25 and cohort.get('questions_per_outer')==512
            and cohort.get('selected_unique_question_count')==12800
            and cohort.get('validation_id_overlap')==0 and cohort.get('validation_exact_question_text_overlap')==0,
            'Complete original25-cohort/no-heldout-overlap contract missing')
    return assets,proof,initial


def fresh_resources(config,retriever_receipt):
    """Read only assigned cards. Other users' GPUs0-3 are outside our scope."""
    oracle=gates.retriever_admission(config,retriever_receipt)
    all_cards,all_apps=gates.inventory()
    cards=[c for c in all_cards if c['index'] in [4,5,6,7]]
    require([c['index'] for c in cards]==[4,5,6,7] and [c['uuid'] for c in cards]==GPU_UUIDS,
            'Actual HKUST4 owned-card UUID mapping changed')
    allowed=set(GPU_UUIDS);apps=[a for a in all_apps if a['gpu_uuid'] in allowed]
    require(all(a['pid']==oracle['owner']['pid'] and a['gpu_uuid']==GPU_UUIDS[-1] and a['used_mib']<=1024 for a in apps),
            'A non-oracle compute process still owns a requested card; do not stop or borrow it')
    gates.resource_limits(config,'cpu-merge',cards,[],set())
    return dict(accepted=True,cluster='HKUST',uid=1001,physical_gpus=[4,5,6,7],gpu_uuids=list(GPU_UUIDS),
        checked_unix=time.time(),devices=cards,permitted_compute_processes=apps,oracle=oracle,no_signals_sent=True,
        scope='Fresh initial deployment snapshot; each actual child launch separately rechecks live ownership/free')


def make_config(assets,initial,identity,variant,helper,manifest,source_commit,goal):
    root=identity['study_base'];base=root+'/group-'+variant
    config=template(base,variant,identity['study_id'],root)
    config['run_id']=identity['run_ids'][variant];config['source_commit']=source_commit
    config['paths'].update(helper_root=helper,runtime=root+'/runtime/frozen-305',cache=root+'/cache',tmp=root+'/tmp')
    for key in ('source_model_manifest','runtime_manifest','common_initial_adapter_manifest','common_cohort_manifest',
                'train_identity','validation_identity'):
        config[key]=assets[key]  # Preserve the original asset-reference bytes/semantics; no cohort reformatting.
    config['common_initial_adapter_tensor_sha256']=initial['adapter_tensor_sha256']
    config['helper_source_manifest']=manifest
    config['authorization']['four_signal_groups_25_outer']=dict(goal_file=goal,goal_created_unix=GOAL_CREATED_UNIX,
        source='Latest explicitly executing human goal',variants=list(VARIANTS),outer_iterations_each=25,
        test_completed_outers=[15,25],ordinary_training_em_full2560=True)
    config['authorization']['four_gpu_execution']=dict(cluster='HKUST',node='HKUST4',uid=1001,
        physical_gpus=[4,5,6,7],gpu_uuids=list(GPU_UUIDS),goal_file=goal,goal_created_unix=GOAL_CREATED_UNIX,
        latest_human_instruction='4卡训练核心只把梯度累积double，其余不变',
        original_eight_rank_accumulation=4,four_rank_accumulation=8,global_batch=32,micro_batch=1)
    return config


def prepare_deployment(assets_ref,publication_ref,source_commit,helper,goal_ref,study=STUDY):
    require(os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'Run only under the authorized HKUST4 UID with empty CUDA visibility')
    require('torch' not in sys.modules,'Final deployment preparation must not import Torch or initialize CUDA')
    require(re.fullmatch('[0-9a-f]{40}',source_commit or '') is not None,'Explicit full published source commit required')
    root=Path(study).resolve();helper=str(Path(helper).resolve())
    require(root==Path(STUDY).resolve() and not Path(study).is_symlink(),'Wrong authorized remote study root')
    root.mkdir(parents=True,exist_ok=True)
    receipts=root/'receipts';receipts.mkdir(exist_ok=True)
    assets,asset_proof,initial=read_assets(assets_ref)
    goal=checked_file(goal_ref);require(goal['bytes']>0,'Actual executing goal text is missing')
    publication,pub_proof=checked_json(publication_ref)
    require(publication.get('accepted') is True and publication.get('branch')=='search-e1'
            and publication.get('source_commit')==publication.get('remote_readback_sha')==source_commit,
            'The latest implementation was not actually pushed and read back')
    manifest=fileproof(Path(helper)/'source-manifest.json')
    require(publication.get('source_manifest_sha256')==manifest['sha256'],'Node payload does not name the published manifest')
    binding=dict(schema='opsd-signal-deployment-identity-v1',assets=asset_proof,publication=pub_proof,
        source_commit=source_commit,helper_root=helper,goal_file=goal,study_base=str(root))
    identity_path=receipts/'deployment-identity.json'
    if identity_path.exists():
        identity,_=checked_json(fileproof(identity_path))
        require(all(identity.get(k)==v for k,v in binding.items()),'Existing deployment inputs changed; preserve original configs')
    else:
        identity=dict(binding,study_id='signal-four-'+uuid.uuid4().hex,run_ids={v:uuid.uuid4().hex for v in VARIANTS},created_unix=time.time())
        write_once(identity_path,identity)
    require(set(identity['run_ids'])==set(VARIANTS) and len(set(identity['run_ids'].values()))==4,
            'Four independent immutable run identities required')
    prototypes={v:make_config(assets,initial,identity,v,helper,manifest,source_commit,goal) for v in VARIANTS}
    sample=prototypes['G_POS'];remote_paths(sample,str(root),helper)
    gates.publication_admission(sample,publication)
    oracle_receipt=dict(accepted=True,retriever_url=sample['paths']['retriever_url'],
        full_gpu_gate_path=RETRIEVER+'/receipts/gpu-oracle-acceptance.json',full_gpu_gate_sha256=ORACLE_GATE_SHA,
        pid=1749337,uid=1001,starttick=641133475,source_identity_sha256=ORACLE_ENTRY_SHA)
    study_manifest_path=receipts/'study-manifest.json'
    already=study_manifest_path.exists()
    if already:
        current_resources=None  # No restart/false empty-card claim while the same immutable study may be running.
    else:
        current_resources=fresh_resources(sample,oracle_receipt)
    transition_common=receipts/'deployment-resource-initial.json'
    if transition_common.exists():
        resource_value,resource_ref=checked_json(fileproof(transition_common))
        require(resource_value.get('accepted') is True and resource_value.get('gpu_uuids')==GPU_UUIDS,'Original deployment card scope changed')
    else:
        require(current_resources is not None,'Existing study has no original resource transition proof')
        resource_ref=write_once(transition_common,current_resources)
        resource_value=current_resources
    study_scope=dict(study_id=identity['study_id'],variants=list(VARIANTS),run_ids=identity['run_ids'],
        output_roots={v:c['paths']['output_root'] for v,c in prototypes.items()})
    study_publication=dict(publication,scope=study_scope,original_actual_publication=pub_proof)
    study_pub_ref=write_once(receipts/'study-publication.json',study_publication)
    groups=[]
    for variant in VARIANTS:
        config=prototypes[variant];remote_paths(config,str(root),helper)
        base=Path(config['paths']['base']);base.mkdir(exist_ok=True);group_receipts=Path(config['paths']['receipts']);group_receipts.mkdir(exist_ok=True)
        scope=dict(study_id=identity['study_id'],variant=variant,run_id=config['run_id'],output_root=config['paths']['output_root'])
        config['gates']['publication']=write_once(group_receipts/'publication.json',dict(publication,scope=scope,original_actual_publication=pub_proof))
        storage=dict(accepted=True,cluster='HKUST',scope=scope,output_root=config['paths']['output_root'],mode='human_hkust_local_exception',
            human_authorization=dict(explicit_exception=True,text='最新人工goal允许HKUST远端节点本地盘；Mac仅保留代码、配置和小型审计摘要',
                goal_file=goal,observed_unix=GOAL_CREATED_UNIX,thread_id='01a11e8d-a0ff-7e13-ae56-dd516a1bb280'),
            backup_policy=dict(durable_target=BACKUP,synchronize_completed_checkpoints=True,large_outputs_remote_only=True,
                synchronization_status='Planned remote-HKUST backup; not claiming any checkpoint already copied'),
            no_mac_large_products=True)
        config['gates']['storage']=write_once(group_receipts/'storage.json',storage)
        config['gates']['resource_transition']=write_once(group_receipts/'resource-transition.json',
            dict(accepted=True,cluster='HKUST',scope=scope,uid=1001,physical_gpus=[4,5,6,7],gpu_uuids=list(GPU_UUIDS),original_fresh_snapshot=resource_ref))
        config['gates']['retriever_admission']=write_once(group_receipts/'retriever-admission.json',oracle_receipt)
        validate_config(config,launch=True)
        gates.publication_admission(config,gates.scoped(config,gates.bound(config['gates']['publication'])))
        gates.storage_admission(config,gates.scoped(config,gates.bound(config['gates']['storage'])))
        gates.retriever_admission(config,gates.bound(config['gates']['retriever_admission']))
        verify_runtime(config);verify_common(config)
        config_ref=write_once(group_receipts/'config.json',config)
        groups.append(dict(variant=variant,config=config_ref))
    for path in (root/'cache',root/'tmp',root/'runs',root/'logs'):
        path.mkdir(exist_ok=True)
    study_manifest=dict(schema='opsd-signal-study-manifest-v1',study_id=identity['study_id'],study_base=str(root),
        helper_root=helper,helper_source_manifest=manifest,publication=study_pub_ref,source_commit=source_commit,
        groups=groups,goal_file=goal,goal_created_unix=GOAL_CREATED_UNIX,assets=asset_proof,
        cache_path=str(root/'cache'),tmp_path=str(root/'tmp'),big_products_remote_only=True)
    final_ref=write_once(study_manifest_path,study_manifest)
    require('torch' not in sys.modules,'CPU final-config verification imported Torch unexpectedly')
    return dict(accepted=True,prepared=True,already_prepared=already,study_manifest=final_ref,groups=groups,
        no_training_started=True,no_cuda_or_model_initialized=True,large_products_local_bytes=0,
        fresh_initial_resource_checked=current_resources is not None,
        scope='Immutable preparation only; live observer/scoped CPU roles and child launch gates remain mandatory')


def main():
    p=argparse.ArgumentParser()
    for name in ('assets-ref','assets-ref-sha256','publication','publication-sha256','source-commit','helper-root','goal-file','goal-file-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--study-base',default=STUDY);a=p.parse_args()
    result=prepare_deployment(dict(path=a.assets_ref,sha256=a.assets_ref_sha256),
        dict(path=a.publication,sha256=a.publication_sha256),a.source_commit,a.helper_root,
        dict(path=a.goal_file,sha256=a.goal_file_sha256),a.study_base)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
