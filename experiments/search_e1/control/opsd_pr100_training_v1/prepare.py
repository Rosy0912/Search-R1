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

GPU_UUIDS = ['GPU-3dd530f7-5348-350d-02cc-183445e1ecbe', 'GPU-d1f1df21-5faa-e406-00e0-dddab33148c7',
             'GPU-a291bae9-7444-ba85-f4d0-64ece80f706c', 'GPU-99dbcc6b-11ee-aa49-74bc-f3e73e10e59b',
             'GPU-1b18ee4f-7486-4183-bf3e-3290d8133db6', 'GPU-0f337443-2d5c-cac7-51e9-1f628f90a2bf',
             'GPU-2d1c2f87-07d3-0c7a-ef9c-20f7ee656e45', 'GPU-6f04d45c-a34d-6617-030d-0635365f5395']
OLD = '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007'
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'


def template(base, variant='PR', study_id=None, study_base=None):
    require(variant == 'PR','Only the user-authorized PR100 variant is supported')
    base = str(Path(base).absolute())
    precision='fp32_master_fp16' if variant == 'PR' else 'bf16'
    study_base=str(Path(study_base or Path(base).parent).absolute())
    return dict(schema='opsd-pr100-config-v1', variant=variant, study_id=study_id, run_id=None, source_commit=None,
        precision=dict(mode=precision),
        paths=dict(base=base, output_root=base+'/runs', runtime=base+'/runtime/frozen-305',
                   python=OLD+'/env/bin/python', model=OLD+'/assets/archive100',
                   train=OLD+'/data/train.jsonl', validation=OLD+'/data/validation.jsonl',
                   receipts=base+'/receipts', helper_root=base+'/source/opsd_pr100_training_v1', study_base=study_base,
                   retriever_url='http://10.120.17.101:18004/retrieve'),
        scientific=dict(loss_reduction='trajectory_mean', learning_rate=1e-6, component_clip=10,
            micro_batch=1, global_batch=32, lora_rank=16, lora_alpha=32, lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            grad_norm_clip=1., chunk=32, max_context=8192, seed=42, warmup_updates=0,
            scheduler='none', outer_iterations=100, teacher_frozen=True, epochs_per_outer=1,
            optimizer=dict(kind='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=0.01)),
        sampling=dict(questions=512,siblings=5,workers=8,batch_questions=64,temperature=1.,top_p=1.,
            max_tokens=500,turns=5,max_search_calls=4,topk=3,observation_tokens=480,seed=42,
            backend='vllm',attention_backend='XFORMERS',dtype='float16' if precision=='fp32_master_fp16' else 'bfloat16',max_model_len=8192,
            enforce_eager=True,gpu_memory_utilization=0.85,num_gpu_blocks_override=22671,cpu_blocks_expected=7281,
            question_order='shared_frozen_original_seed42_first51200_train_permutation_no_replacement',
            model_merge='CPU_FP32_base_FP32_adapter_merge_then_one_FP16_cast' if precision=='fp32_master_fp16' else 'original_PEFT_CPU_BF16_latest_adapter_no_best_selection'),
        owner=dict(uid=1005,physical_gpus=list(range(8)),gpu_uuids=GPU_UUIDS),
        resources=dict(startup_minimum_free_mib=20480,training_minimum_free_mib=1024,minimum_disk_bytes=100<<30,
            shared_gpu_lease_paths=['/data/shaoshitong/search-e1-opsd-debug-20261008/experiment.lock',
                '/data/shaoshitong/search-e1-opsd-native512-20261008/experiment.lock',
                '/data/shaoshitong/search-e1-opsd-teacher-counterfactual-20261008/experiment.lock',
                OLD+'/experiment.lock',
                '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/online-experiment.lock'],
            study_lease_path=study_base+'/ablation-gpu-experiment.lock'),
        logging=dict(topk=32,topk_max_tokens_per_outer=0,all_valid_token_losses=True,full_vocab_enabled=False,
            full_vocab_study_root=study_base+'/runs/full-vocabulary'),
        diagnostics=dict(teacher_autoregressive=False,teacher_max_cases_per_outer=0,teacher_seed=42,
            teacher_precision='bf16',diagnostic_precision_common_bf16=True,teacher_temperature=0.,
            teacher_selection_version='opsd-ablation-privileged-teacher-autoregressive-v1',
            teacher_strata=['self_pair','CW','CC'],training_or_validation_denominator_contribution=0),
        retention=dict(mode='keep_key_and_resume',authorized=True,merged_cache_cleanup_authorized=True,
            checkpoint_keep_latest=2,protect_updates=[0,1,5],keep_all_outer_end=True),
        gates=dict(publication=None,storage=None,resource_transition=None,retriever_admission=None,
            memory_binding_path=base+'/receipts/memory-binding.json',
            scoped_cpu_processes_path=base+'/receipts/scoped-cpu-processes.json'),
        source_model_manifest=None,runtime_manifest=None,helper_source_manifest=None,
        train_identity=None,validation_identity=None,common_cohort_manifest=None,common_initial_adapter_manifest=None,
        common_initial_adapter_tensor_sha256=None,
        mining=dict(version='search-e1-opsd-ablation-miner-v1',random_version='search-e1-opsd-uniform-five-sha256-u64-rejection-v1',
            weights_version='search-e1-opsd-correctness-weights-v1',cohort_version='search-e1-opsd-pr100-train-cohorts-seed42-v1'),
        authorization=dict(pr100_periodic=None,eight_gpu_execution=None))


def validate_config(config, launch=False):
    require(config['schema']=='opsd-pr100-config-v1','Unknown ablation schema')
    fixed = template(config['paths']['base'],config['variant'],config.get('study_id'),config['paths']['study_base'])
    # Physical paths and reviewed utilization may differ, scientific defaults may not.
    sci = config['scientific']
    for key,value in fixed['scientific'].items():
        require(sci.get(key)==value,'Scientific setting changed: '+key)
    clip = sci['component_clip']
    require(type(clip) in (int,float) and clip==10,'Seven groups require component upper clip10')
    outer = sci['outer_iterations']
    require(type(outer) is int and outer==100,'PR100 requires the same immutable100-outer budget across boundaries')
    require(config['precision']==fixed['precision'] and config['mining']==fixed['mining'],'Precision/miner version changed')
    require(config['diagnostics']==fixed['diagnostics'],'Autonomous teacher diagnostic protocol changed')
    for key,value in fixed['sampling'].items():
        require(config['sampling'].get(key)==value,'Unreviewed sampling change: '+key)
    require(config['owner']==fixed['owner'],'Unexpected UID / physical GPU UUID allocation')
    require(config['paths']['retriever_url']==fixed['paths']['retriever_url'],'Unreviewed retriever endpoint')
    require(config['logging']['all_valid_token_losses'] is True,'Every valid token loss must be retained')
    require(config['logging']['topk']==32 and type(config['logging']['topk_max_tokens_per_outer']) is int and config['logging']['topk_max_tokens_per_outer']>=0,'Invalid logging budget')
    require(config['logging'].get('full_vocab_enabled') is False and config['logging']['topk_max_tokens_per_outer']==0,
            'This run disables top32/full-vocabulary debug volume')
    require(config['logging']['full_vocab_study_root']==fixed['logging']['full_vocab_study_root'],'Full-vocabulary cap scope must be shared by all seven groups')
    require(config['resources']['startup_minimum_free_mib']>=20480 and config['resources']['training_minimum_free_mib']>=1024,'Memory gate weakened')
    require(config['resources']['minimum_disk_bytes']>=100<<30,'Disk gate weakened')
    require(config['resources'].get('shared_gpu_lease_paths')==fixed['resources']['shared_gpu_lease_paths'],'Shared GPU lease set/order changed')
    reasons=[]
    if not isinstance(config.get('study_id'),str) or not config['study_id']:reasons.append('New ablation study identity missing')
    if not isinstance(config.get('run_id'),str) or len(config['run_id'])!=32 or any(c not in '0123456789abcdef' for c in config['run_id']): reasons.append('Fixed 32hex run_id missing')
    if not config['authorization']['pr100_periodic']: reasons.append('New PR100 and periodic512 authorization missing')
    if not config['authorization']['eight_gpu_execution']: reasons.append('eight-GPU execution authorization missing')
    for key in ('publication','storage','resource_transition','retriever_admission'):
        if not config['gates'].get(key): reasons.append(key+' receipt missing')
    for key in ('source_model_manifest','runtime_manifest','helper_source_manifest','train_identity','validation_identity',
                'common_cohort_manifest','common_initial_adapter_manifest','common_initial_adapter_tensor_sha256'):
        if not config.get(key): reasons.append(key+' identity missing')
    if not isinstance(config.get('source_commit'),str) or len(config['source_commit'])!=40: reasons.append('Published source commit missing')
    require(config['retention']==fixed['retention'],
            'Only new PR100 authorized key/resume state policy and completed cache housekeeping are allowed')
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
    p.add_argument('--variant',required=True,choices=('PR',));p.add_argument('--study-id',required=True)
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
