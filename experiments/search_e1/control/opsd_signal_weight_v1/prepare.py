"""CPU preparation and explicit science schema. Never launches CUDA or stops jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil

try:
    from .checkpoint import require, checked_json, checked_file, fileproof, exclusive_json
except ImportError:
    from checkpoint import require, checked_json, checked_file, fileproof, exclusive_json

GPU_UUIDS = ['GPU-ed24344f-9cd8-6d29-e8b9-d7582a8cfd65',
             'GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2',
             'GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f',
             'GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3']
OLD = '/weka_3/shaoshitong/search-e1-config-study-20261004'
VARIANTS = ('G_POS','G_NEG','H_POS','H_NEG')
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'


def template(base, variant='G_POS', study_id=None, study_base=None):
    require(variant in VARIANTS,'Unknown four signal-weight variant')
    base = str(Path(base).absolute())
    precision='bf16'
    study_base=str(Path(study_base or Path(base).parent).absolute())
    return dict(schema='opsd-signal-weight-config-v1', variant=variant, baseline_variant='R', study_id=study_id, run_id=None, source_commit=None,
        precision=dict(mode=precision),
        paths=dict(base=base, output_root=base+'/runs', runtime=base+'/runtime/frozen-305',
                   python=OLD+'/env/bin/python', model=OLD+'/runs/archive/round-1-grpo/actor/global_step_100',
                   train=OLD+'/data/train.jsonl', validation=OLD+'/data/validation.jsonl',
                   receipts=base+'/receipts', helper_root=base+'/source/opsd_signal_weight_v1', study_base=study_base,
                   retriever_url='http://10.123.4.11:18003/retrieve'),
        scientific=dict(loss_reduction='trajectory_mean', learning_rate=1e-6, component_clip=10,
            micro_batch=1, global_batch=32, lora_rank=16, lora_alpha=32, lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            grad_norm_clip=1., chunk=32, max_context=8192, seed=42, warmup_updates=0,
            scheduler='none', outer_iterations=25, world_size=4,
            gradient_accumulation_full_batch_per_rank=8,original_eight_rank_accumulation=4,
            teacher_frozen=True, epochs_per_outer=1,
            optimizer=dict(kind='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=0.01)),
        sampling=dict(questions=512,siblings=5,workers=8,batch_questions=64,temperature=1.,top_p=1.,
            max_tokens=500,turns=5,max_search_calls=4,topk=3,observation_tokens=480,seed=42,
            backend='vllm',attention_backend='XFORMERS',dtype='float16' if precision=='fp32_master_fp16' else 'bfloat16',max_model_len=8192,
            enforce_eager=True,gpu_memory_utilization=0.85,num_gpu_blocks_override=22671,cpu_blocks_expected=7281,
            question_order='shared_frozen_original_seed42_first12800_train_permutation_no_replacement',
            logical_rank_to_slot=[0,1,2,3,0,1,2,3], waves=[[0,1,2,3],[4,5,6,7]],
            model_merge='original_PEFT_CPU_BF16_latest_adapter_no_best_selection'),
        owner=dict(uid=1001,physical_gpus=[4,5,6,7],gpu_uuids=list(GPU_UUIDS)),
        resources=dict(startup_minimum_free_mib=20480,training_minimum_free_mib=6144,minimum_disk_bytes=100<<30,
            shared_gpu_lease_paths=[OLD+'/experiment.lock'],
            study_lease_path=study_base+'/signal-weight-gpu-experiment.lock'),
        signal=dict(version='opsd-five-sibling-centered-signal-weights-v1',
            metric='answer_signed_gap' if variant.startswith('G_') else 'all_policy_student_entropy',
            direction=1 if variant.endswith('_POS') else -1, amplitude=.9,numerical_floor=1e-6,
            timing='current_preupdate',missing_answer='neutral',encoding_failure='whole_group_neutral'),
        evaluation=dict(completed_outers=[15,25],common_zero=True,logical_workers=8,
            primary_rows=512,rank_seed=42,baseline_reuse=None,
            common_zero_receipt_path=study_base+'/receipts/common-zero-evaluation.json'),
        logging=dict(topk=32,topk_max_tokens_per_outer=8192,all_valid_token_losses=True,
            full_vocab_study_root=study_base+'/runs/full-vocabulary'),
        diagnostics=dict(teacher_autoregressive=True,teacher_max_cases_per_outer=8,teacher_seed=42,
            teacher_precision='bf16',diagnostic_precision_common_bf16=True,teacher_temperature=0.,
            teacher_selection_version='opsd-ablation-privileged-teacher-autoregressive-v1',
            teacher_strata=['self_pair','CW','CC'],training_or_validation_denominator_contribution=0),
        retention=dict(mode='keep_all',authorized=False,merged_cache_cleanup_authorized=False),
        gates=dict(publication=None,storage=None,resource_transition=None,retriever_admission=None,
            memory_binding_path=base+'/receipts/memory-binding.json',
            scoped_cpu_processes_path=base+'/receipts/scoped-cpu-processes.json'),
        source_model_manifest=None,runtime_manifest=None,helper_source_manifest=None,
        train_identity=None,validation_identity=None,common_cohort_manifest=None,common_initial_adapter_manifest=None,
        common_initial_adapter_tensor_sha256=None,
        mining=dict(version='search-e1-opsd-signal-weight-miner-v1',random_version='search-e1-opsd-uniform-five-sha256-u64-rejection-v1',
            weights_version='search-e1-opsd-correctness-weights-v1',cohort_version='search-e1-opsd-25-train-cohorts-seed42-v1'),
        authorization=dict(four_signal_groups_25_outer=None,four_gpu_execution=None))


def validate_config(config, launch=False):
    require(config['schema']=='opsd-signal-weight-config-v1','Unknown signal-weight schema')
    fixed = template(config['paths']['base'],config['variant'],config.get('study_id'),config['paths']['study_base'])
    # Physical paths and reviewed utilization may differ, scientific defaults may not.
    sci = config['scientific']
    for key,value in fixed['scientific'].items():
        require(sci.get(key)==value,'Scientific setting changed: '+key)
    clip = sci['component_clip']
    require(type(clip) in (int,float) and clip==10,'Four groups require component upper clip10')
    outer = sci['outer_iterations']
    require(type(outer) is int and outer==25,'Each signal group requires twenty-five online outers')
    require(config.get('baseline_variant')=='R' and config.get('signal')==fixed['signal'],'R baseline / fixed signal weighting contract changed')
    require(config.get('evaluation')==fixed['evaluation'],'Authorized zero/15/25 evaluation contract changed')
    require(config['precision']==fixed['precision'] and config['mining']==fixed['mining'],'Precision/miner version changed')
    require(config['diagnostics']==fixed['diagnostics'],'Autonomous teacher diagnostic protocol changed')
    for key,value in fixed['sampling'].items():
        require(config['sampling'].get(key)==value,'Unreviewed sampling change: '+key)
    require(config['owner']==fixed['owner'],'Unexpected UID / physical GPU UUID allocation')
    require(config['paths']['retriever_url']==fixed['paths']['retriever_url'],'Unreviewed retriever endpoint')
    require(config['logging']['all_valid_token_losses'] is True,'Every valid token loss must be retained')
    require(config['logging']['topk']==32 and type(config['logging']['topk_max_tokens_per_outer']) is int and config['logging']['topk_max_tokens_per_outer']>=0,'Invalid logging budget')
    require(config['logging']['full_vocab_study_root']==fixed['logging']['full_vocab_study_root'],'Full-vocabulary cap scope must be shared by all seven groups')
    require(config['resources']['startup_minimum_free_mib']>=20480 and config['resources']['training_minimum_free_mib']>=6144,'Memory gate weakened')
    require(config['resources']['minimum_disk_bytes']>=100<<30,'Disk gate weakened')
    require(config['resources'].get('shared_gpu_lease_paths')==fixed['resources']['shared_gpu_lease_paths'],'Shared GPU lease set/order changed')
    reasons=[]
    if not isinstance(config.get('study_id'),str) or not config['study_id']:reasons.append('New ablation study identity missing')
    if not isinstance(config.get('run_id'),str) or len(config['run_id'])!=32 or any(c not in '0123456789abcdef' for c in config['run_id']): reasons.append('Fixed 32hex run_id missing')
    if not config['authorization']['four_signal_groups_25_outer']: reasons.append('New four-group twenty-five-outer authorization missing')
    if not config['authorization']['four_gpu_execution']: reasons.append('HKUST4 four-GPU execution authorization missing')
    for key in ('publication','storage','resource_transition','retriever_admission'):
        if not config['gates'].get(key): reasons.append(key+' receipt missing')
    for key in ('source_model_manifest','runtime_manifest','helper_source_manifest','train_identity','validation_identity',
                'common_cohort_manifest','common_initial_adapter_manifest','common_initial_adapter_tensor_sha256'):
        if not config.get(key): reasons.append(key+' identity missing')
    if not isinstance(config.get('source_commit'),str) or len(config['source_commit'])!=40: reasons.append('Published source commit missing')
    require(config['retention']['mode']=='keep_all' and config['retention']['authorized'] is False,'No historical checkpoint deletion authorized')
    require(config['resources']['study_lease_path']==fixed['resources']['study_lease_path'],'Ablation common GPU lease changed')
    require(config['gates']['scoped_cpu_processes_path']==fixed['gates']['scoped_cpu_processes_path'],
            'CPU role registry must be the planned group receipt path')
    if launch: require(not reasons,'HOLD: '+'; '.join(reasons))
    return reasons


def verify_runtime(config):
    manifest,proof=checked_json(config['runtime_manifest'])
    files=manifest.get('files',manifest)
    require(proof['sha256']==RUNTIME_SHA and len(files)==305,'Expected exact old305 manifest')
    root=Path(config['paths']['runtime']).resolve()
    actual=[]
    for name,expected in files.items():
        relative=Path(name); require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe runtime relative path')
        ref=dict(path=str(root/relative),**(expected if isinstance(expected,dict) else dict(sha256=expected)))
        actual.append(checked_file(ref))
    try:from .scoring import verify_scoring
    except ImportError:from scoring import verify_scoring
    scoring_proof=verify_scoring(config)
    return dict(manifest=proof,files=305,verified_file_bytes=sum(f['bytes'] for f in actual),all305_equal=True,
                scoring_identity=scoring_proof)


def verify_common(config):
    try:from .cohort_v1 import verify_from_files
    except ImportError:from cohort_v1 import verify_from_files
    frozen,cohort_proof=checked_json(config['common_cohort_manifest'])
    cohort=verify_from_files(frozen,config['train_identity'],config['validation_identity'])
    initial,initial_proof=checked_json(config['common_initial_adapter_manifest'])
    require(initial['adapter_tensor_sha256']==config['common_initial_adapter_tensor_sha256'],
            'Common initial tensor identity differs')
    payload=[checked_file(f) for f in initial['payload_files']]
    require(initial['adapter_model'] in initial['payload_files'] and len(payload)>=2,'Initial adapter inventory incomplete')
    names={Path(f['path']).name for f in payload}
    require({'adapter_config.json','adapter_model.safetensors'}<=names,'Common adapter lacks config/tensors')
    adapter_root=Path(initial['adapter_model']['path']).parent.resolve()
    require(all(Path(f['path']).parent.resolve()==adapter_root for f in payload),'Initial payload outside common adapter directory')
    return dict(cohort_manifest=cohort_proof,cohort=cohort,common_initial_manifest=initial_proof,
        payload=payload,scope='CPU file identity; trainer/merge child separately verifies actual504 finite FP32 tensors and zero B')


def copy_runtime(source,manifest_ref,destination):
    manifest,_=checked_json(manifest_ref); files=manifest.get('files',manifest)
    require(len(files)==305 and manifest_ref['sha256']==RUNTIME_SHA,'Frozen305 manifest mismatch')
    source=Path(source).resolve(); destination=Path(destination)
    require(not destination.exists(),'New runtime directory already exists')
    destination.mkdir(parents=True)
    for name,expected in files.items():
        rel=Path(name); require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe source path')
        origin=source/rel
        checked_file(dict(path=str(origin),**(expected if isinstance(expected,dict) else dict(sha256=expected))))
        target=destination/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(origin,target)
        checked_file(dict(path=str(target),**(expected if isinstance(expected,dict) else dict(sha256=expected))))
    return dict(files=305,changed_files=[],source=str(source),destination=str(destination.resolve()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',required=True,choices=VARIANTS);p.add_argument('--study-id',required=True)
    p.add_argument('--study-base',required=True)
    p.add_argument('--copy-runtime-source');p.add_argument('--runtime-manifest');p.add_argument('--runtime-manifest-sha256')
    a=p.parse_args();config=template(a.base,a.variant,a.study_id,a.study_base)
    if a.copy_runtime_source:
        require(a.runtime_manifest and a.runtime_manifest_sha256,'CPU copy requires exact bound manifest')
        config['runtime_manifest']=dict(path=a.runtime_manifest,sha256=a.runtime_manifest_sha256)
        config['cpu_preparation_runtime_copy']=copy_runtime(a.copy_runtime_source,config['runtime_manifest'],config['paths']['runtime'])
    config['preparation_scope']='CPU only; missing approvals HOLD. No CUDA/HTTP, no process signals, no legacy optimizer restoration.'
    reasons=validate_config(config)
    proof=exclusive_json(a.output,config)
    print(json.dumps(dict(prepared=True,launch_allowed=False,holds=reasons,config=proof)))


if __name__=='__main__': main()
