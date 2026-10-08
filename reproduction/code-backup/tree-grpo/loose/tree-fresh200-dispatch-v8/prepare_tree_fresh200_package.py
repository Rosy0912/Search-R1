"""Same-h100-9 CPU-only fresh source/archive/config preparation, no submit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def load_common(review_path,review_sha):
    raw=Path(review_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=review_sha:raise ValueError('Helper source-review SHA differs')
    review=json.loads(raw)
    if review['status']!='FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED':raise ValueError('Missing independent helper source review')
    if review['actual_gpu_accepted'] is not False:raise ValueError('Helper source review cannot imply GPU acceptance')
    here=Path(__file__).resolve().parent
    for name in ('prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py','fresh200_dispatch_common.py'):
        content=(here/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=review['helper_files'][name]:raise ValueError('Helper bytes changed: '+name)
        if name=='fresh200_dispatch_common.py':common_raw=content
    module=types.ModuleType('fresh200_dispatch_common');module.__file__=str(here/'fresh200_dispatch_common.py')
    exec(compile(common_raw,module.__file__,'exec'),module.__dict__)
    module.local_path(review_path)
    return module


def selected(config,path):
    value=config
    for key in path.split('.'):value=value[key]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','archive-sha','target','manifest-sha','inputs','inputs-sha','helper-review','helper-review-sha'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args();c=load_common(args.helper_review,args.helper_review_sha);c.assert_host()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    inputs=c.json_bound(dict(path=args.inputs,sha256=args.inputs_sha))
    c.require(inputs['code_commit']==c.COMMIT and inputs['manifest_sha256']==args.manifest_sha
              and inputs['archive_sha256']==args.archive_sha and inputs['stage']==c.STAGE,'Reviewed preparation argv differ')
    c.require(inputs['target']==args.target and inputs['archive']==args.archive,'Reviewed archive/target differ')
    manifest=c.safe_extract(args.archive,args.archive_sha,args.target,args.manifest_sha,inputs['max_uncompressed_bytes'])
    target=c.local_path(args.target);reviews=c.validate_reviews(inputs,manifest)
    # Preserve only same-node assets. Never invoke a download/import/copy path.
    os.environ.update(SEARCH_R1_REPO=str(target/'source/repo'),TREE_COMMON_DATA=str(c.ROOT/'instruct300/common-data'),
        TREE_INSTRUCT_MODEL=str(c.MODEL),SEARCH_R1_ASSETS=str(c.ROOT/'assets-root'),
        SEARCH_R1_BASELINE_OUTPUT=str(target/'CPU_PREP_CONFIG/output'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1')
    sys.path[:0]=[str(target),str(c.ROOT/'tools'),str(target/'source/repo')]
    import search_r1
    c.require(Path(search_r1.__file__).resolve()==(target/'source/repo/search_r1/__init__.py').resolve(),'CPU preparation imported another source')
    from search_r1.baselines.select_teacher import model_manifest
    c.require(model_manifest(c.MODEL)==manifest['model_identity'],'Actual local official model/tokenizer inventory changed')
    import supervisor
    c.require(supervisor.PACKAGE==manifest and Path(supervisor.__file__).resolve()==(target/'supervisor.py').resolve(),
              'CPU configuration consumed another manifest/supervisor')
    prep=target/'CPU_PREP_CONFIG';prep.mkdir()
    value,steps=supervisor.configuration(prep,c.STAGE)
    from omegaconf import OmegaConf
    resolved=OmegaConf.to_container(value,resolve=True)
    declared=c.json_bound(dict(path=str(target/'CONFIGURATION_OVERRIDES.json'),sha256=manifest['files']['CONFIGURATION_OVERRIDES.json']))
    c.require(declared['trial_only'] is True and declared['production_thresholds']=={
        'target_old_policy_kl':.2,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None},'Trial/production scope changed')
    c.require(declared['configuration_overrides']==manifest['configuration_overrides'],'Configuration overrides changed')
    for key,expected in declared['expected_resolved_values'].items():
        actual=selected(resolved,key)
        c.require(actual==expected and (type(expected) is not bool or type(actual) is bool),'Resolved config field changed: '+key)
    c.require(steps==200 and os.environ['CUDA_VISIBLE_DEVICES']=='','CPU preparation budget/GPU visibility changed')
    torch=sys.modules.get('torch');c.require(torch is None or not torch.cuda.is_initialized(),'CPU config preparation initialized CUDA')
    c.verify_package(target,manifest)
    result=dict(status='SAME_NODE_FRESH200_CODE_ASSETS_AND_CPU_CONFIG_VERIFIED_NOT_SUBMITTED',stage=c.STAGE,
        code_commit=c.COMMIT,package=str(target),manifest_sha256=args.manifest_sha,archive_sha256=args.archive_sha,
        inputs_sha256=args.inputs_sha,helper_review_sha256=args.helper_review_sha,source_reviews=reviews,
        resolved_config_path=str(prep/'resolved.yaml'),resolved_config_sha256=c.file_sha(prep/'resolved.yaml'),
        actual_hydra_resolved=True,actual_gpu_accepted=False,cuda_initialized=False,optimizer_writes=0,
        official_model_path=str(c.MODEL),model_identity=manifest['model_identity'],common_data_identity=c.DATA_ID,
        checked_resolved_values=declared['expected_resolved_values'],
        budget=dict(complete_outers=steps,questions=512,leaves=5,total_leaves=2560,mini_global=128,actor_micro_global=16,
                    old_ref_micro_global=32,epochs=1,optimizer_opportunities_per_rank_per_complete_outer=20,minimum_successful_optimizer_writes_per_complete_outer=1,
                    successful_optimizer_writes_counted_separately=True,zero_write_or_abandoned_batch_credits_outer=False,warmup_outers=57),
        datasets=dict(train=selected(resolved,'data.train_files'),validation=selected(resolved,'data.val_files'),
                      final=selected(resolved,'data.final_val_files'),train_rows=169028,validation_rows=512,final_rows=51713),
        guard_optins=dict(policy_safety=selected(resolved,'actor_rollout_ref.actor.policy_safety.enabled'),
            behavior_mu=selected(resolved,'actor_rollout_ref.actor.policy_safety.sampling_probability_correction.enabled'),
            raw_tokens=selected(resolved,'baselines.tree.token_preservation.enabled'),
            raw_emission=selected(resolved,'actor_rollout_ref.rollout.tree_raw_emission')),
        trial_thresholds=manifest['trial_thresholds'],production_thresholds_accepted=False,actual_3b_training_accepted=False,
        original_sampler_actor_gate_accepted=False,long_training_ready=False,submitted=False,
        scope='Same h100-9 CPU configuration and full local asset SHA verification, no fresh 3B forward or probability/training acceptance')
    output=prep/'PREPARATION.json';report_sha=c.write_once(output,result)
    print(json.dumps(dict(status=result['status'],path=str(output),sha256=report_sha,submitted=False)),flush=True)

if __name__=='__main__':main()
