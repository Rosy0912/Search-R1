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


def template(base):
    base = str(Path(base).absolute())
    return dict(schema='opsd-online-config-v1', run_id=None, source_commit=None,
        paths=dict(base=base, output_root=base+'/runs', runtime=base+'/runtime/frozen-305',
                   python=OLD+'/env/bin/python', model=OLD+'/assets/archive100',
                   train=OLD+'/data/train.jsonl', validation=OLD+'/data/validation.jsonl',
                   receipts=base+'/receipts', helper_root=base+'/source/opsd_online',
                   retriever_url='http://10.120.17.101:18004/retrieve'),
        scientific=dict(loss_reduction='trajectory_mean', learning_rate=5e-6, component_clip='UNRESOLVED',
            micro_batch=1, global_batch=32, lora_rank=16, lora_alpha=32, lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            grad_norm_clip=1., chunk=32, max_context=8192, seed=42, warmup_updates=0,
            scheduler='none', outer_iterations=None, teacher_frozen=True, epochs_per_outer=1,
            optimizer=dict(kind='AdamW',betas=[0.9,0.999],eps=1e-8,weight_decay=0.01)),
        sampling=dict(questions=512,siblings=5,workers=8,batch_questions=64,temperature=1.,top_p=1.,
            max_tokens=500,turns=5,max_search_calls=4,topk=3,observation_tokens=480,seed=42,
            backend='vllm',attention_backend='XFORMERS',dtype='bfloat16',max_model_len=8192,
            enforce_eager=True,gpu_memory_utilization=0.85,num_gpu_blocks_override=22671,cpu_blocks_expected=7281,
            question_order='seed42_permuted_train_stream_skip_boundary_duplicate_with_recorded_cursor',
            model_merge='original_PEFT_CPU_BF16_latest_adapter_no_best_selection'),
        owner=dict(uid=1005,physical_gpus=list(range(8)),gpu_uuids=GPU_UUIDS),
        resources=dict(startup_minimum_free_mib=20480,training_minimum_free_mib=1024,minimum_disk_bytes=100<<30,
            shared_gpu_lease_paths=['/data/shaoshitong/search-e1-opsd-debug-20261008/experiment.lock',
                '/data/shaoshitong/search-e1-opsd-native512-20261008/experiment.lock',
                '/data/shaoshitong/search-e1-opsd-teacher-counterfactual-20261008/experiment.lock',
                OLD+'/experiment.lock']),
        logging=dict(topk=32,topk_max_tokens_per_outer=8192,all_valid_token_losses=True),
        retention=dict(mode='rolling',latest_complete=2,protected_updates=[0,1,5],preserve_outer_end=True,
                       merged_cache_keep=2,authorized=False),
        gates=dict(publication=None,storage=None,resource_transition=None,retriever_admission=None),
        source_model_manifest=None,runtime_manifest=None,helper_source_manifest=None,
        train_identity=None,validation_identity=None,
        authorization=dict(clip_decision=None,outer_iterations_decision=None,eight_gpu_execution=None))


def validate_config(config, launch=False):
    require(config['schema']=='opsd-online-config-v1','Unknown online schema')
    fixed = template(config['paths']['base'])
    # Physical paths and reviewed utilization may differ, scientific defaults may not.
    sci = config['scientific']
    for key,value in fixed['scientific'].items():
        if key not in ('component_clip','outer_iterations'):
            require(sci.get(key)==value,'Scientific setting changed: '+key)
    clip = sci['component_clip']
    require(clip=='UNRESOLVED' or clip is None or (type(clip) in (int,float) and __import__('math').isfinite(clip) and clip>0),'Clip must be explicitly null or positive')
    outer = sci['outer_iterations']
    require(outer is None or (type(outer) is int and outer>0),'Invalid outer budget')
    for key,value in fixed['sampling'].items():
        require(config['sampling'].get(key)==value,'Unreviewed sampling change: '+key)
    require(config['owner']==fixed['owner'],'Unexpected UID / physical GPU UUID allocation')
    require(config['paths']['retriever_url']==fixed['paths']['retriever_url'],'Unreviewed retriever endpoint')
    require(config['logging']['all_valid_token_losses'] is True,'Every valid token loss must be retained')
    require(config['logging']['topk']==32 and type(config['logging']['topk_max_tokens_per_outer']) is int and config['logging']['topk_max_tokens_per_outer']>=0,'Invalid logging budget')
    require(config['resources']['startup_minimum_free_mib']>=20480 and config['resources']['training_minimum_free_mib']>=1024,'Memory gate weakened')
    require(config['resources']['minimum_disk_bytes']>=100<<30,'Disk gate weakened')
    require(config['resources'].get('shared_gpu_lease_paths')==fixed['resources']['shared_gpu_lease_paths'],'Shared GPU lease set/order changed')
    reasons=[]
    if not isinstance(config.get('run_id'),str) or len(config['run_id'])!=32 or any(c not in '0123456789abcdef' for c in config['run_id']): reasons.append('Fixed 32hex run_id missing')
    if clip=='UNRESOLVED' or not config['authorization']['clip_decision']: reasons.append('component_clip requires human decision')
    if outer is None or not config['authorization']['outer_iterations_decision']: reasons.append('outer_iterations requires human decision')
    if not config['authorization']['eight_gpu_execution']: reasons.append('eight-GPU execution authorization missing')
    for key in ('publication','storage','resource_transition','retriever_admission'):
        if not config['gates'].get(key): reasons.append(key+' receipt missing')
    for key in ('source_model_manifest','runtime_manifest','helper_source_manifest','train_identity','validation_identity'):
        if not config.get(key): reasons.append(key+' identity missing')
    if not isinstance(config.get('source_commit'),str) or len(config['source_commit'])!=40: reasons.append('Published source commit missing')
    if config['retention'].get('authorized') is not True: reasons.append('Rolling retention authorization missing')
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
    p.add_argument('--copy-runtime-source');p.add_argument('--runtime-manifest');p.add_argument('--runtime-manifest-sha256')
    a=p.parse_args();config=template(a.base)
    if a.copy_runtime_source:
        require(a.runtime_manifest and a.runtime_manifest_sha256,'CPU copy requires exact bound manifest')
        config['runtime_manifest']=dict(path=a.runtime_manifest,sha256=a.runtime_manifest_sha256)
        config['cpu_preparation_runtime_copy']=copy_runtime(a.copy_runtime_source,config['runtime_manifest'],config['paths']['runtime'])
    config['preparation_scope']='CPU only; missing approvals HOLD. No CUDA/HTTP, no process signals, no legacy optimizer restoration.'
    reasons=validate_config(config)
    proof=exclusive_json(a.output,config)
    print(json.dumps(dict(prepared=True,launch_allowed=False,holds=reasons,config=proof)))


if __name__=='__main__': main()
