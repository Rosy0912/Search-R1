"""Create seven private CPU candidate configs; never imports models or launches."""
import argparse
import json
from pathlib import Path
import uuid

try:
    from .checkpoint import require,checked_json,fileproof,exclusive_json
    from .prepare import template,validate_config
except ImportError:
    from checkpoint import require,checked_json,fileproof,exclusive_json
    from prepare import template,validate_config

VARIANTS=('C0','C1','M','R','W','P','J')
OLD_CONFIG_SHA='d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
SOURCE_SHA='2ef6070a328357b0f9d3a026010aaed0aa3f8840a241bc1021ed9a183e8d75f7'


def build(old_assets,cohorts,initial,study_base,source_commit,manifest_sha,output):
    require(len(source_commit)==40 and len(manifest_sha)==64,'Exact newly published source identities required')
    require(old_assets['source_model_manifest']['sha256']==SOURCE_SHA,'Source100 twelve-file identity differs')
    require(cohorts['outer_count']==10 and cohorts['questions_per_outer']==512
            and cohorts['validation_id_overlap']==0 and cohorts['validation_exact_question_text_overlap']==0,'Common cohort contract differs')
    require(initial['accepted'] is True and initial['tensors']==504 and initial['dtype']=='float32'
            and initial['zero_B_tensors']==252 and initial['source_model_identity']==SOURCE_SHA,'Actual common initial FP32 template missing')
    study_base=str(Path(study_base).absolute());output=Path(output)
    require(not output.exists(),'Never overwrite a prepared group/config/run identity')
    output.mkdir(parents=True)
    study_id='reverse-seven-'+uuid.uuid4().hex
    helper_root=study_base+'/source/training-'+source_commit[:7]+'/opsd_ablation_v1'
    common=output/'common';common.mkdir()
    common_cohort=exclusive_json(common/'common-cohorts.json',cohorts)
    common_initial=exclusive_json(common/'common-initial-adapter.json',initial)
    def remote_common(proof):return dict(path=study_base+'/receipts/'+Path(proof['path']).name,sha256=proof['sha256'],bytes=proof['bytes'])
    configs=[]
    for variant in VARIANTS:
        base=study_base+'/group-'+variant
        config=template(base,variant,study_id,study_base)
        config['run_id']=uuid.uuid4().hex;config['source_commit']=source_commit
        config['paths']['helper_root']=helper_root
        config['paths']['runtime']=study_base+'/runtime/frozen-305'
        # Only immutable model/runtime/data identities are reused. No old
        # authorization, controller, optimizer, observer or output is reused.
        config['source_model_manifest']=old_assets['source_model_manifest']
        config['runtime_manifest']=old_assets['runtime_manifest']
        for name in ('train_identity','validation_identity'):
            config[name]=cohorts[name]
        config['paths']['train']=cohorts['train_identity']['path']
        config['paths']['validation']=cohorts['validation_identity']['path']
        config['helper_source_manifest']=dict(path=helper_root+'/source-manifest.json',sha256=manifest_sha)
        config['common_cohort_manifest']=remote_common(common_cohort)
        config['common_initial_adapter_manifest']=remote_common(common_initial)
        config['common_initial_adapter_tensor_sha256']=initial['adapter_tensor_sha256']
        config['authorization']['reverse_seven_group_ten_outer']=dict(human_goal_created_unix=1791472699,
            scope='C0/C1/M/R/W/P/J fresh reverse ten online outers then full512',source='latest explicit execution goal')
        config['authorization']['eight_gpu_execution']=dict(cluster='HKUST',node='hkust1',uid=1005,goal_created_unix=1791472699)
        holds=validate_config(config)
        config['preparation_status']=dict(ready=False,holds=holds,no_cuda_started=True,
            storage_scope='New seven-group input still required; old online200 exception not inherited')
        proof=exclusive_json(output/('group-'+variant+'-config-candidate.json'),config)
        configs.append(dict(variant=variant,run_id=config['run_id'],candidate=proof,
                            node_candidate_path=base+'/receipts/config-candidate.json',output_root=config['paths']['output_root'],holds=holds))
    return exclusive_json(output/'bundle.json',dict(schema='opsd-ablation-seven-private-cpu-candidates-v1',study_id=study_id,
        study_base=study_base,source_commit=source_commit,source_manifest_sha256=manifest_sha,groups=configs,
        common_files=[common_cohort,common_initial],launch_ready=False,no_cuda_started=True,
        no_old_adam_or_authorization_inherited=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-assets-config',required=True)
    p.add_argument('--common-cohort',required=True);p.add_argument('--common-initial',required=True)
    p.add_argument('--study-base',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();old,_=checked_json(dict(path=a.old_assets_config,sha256=OLD_CONFIG_SHA))
    cohorts,_=checked_json(fileproof(a.common_cohort));initial,_=checked_json(fileproof(a.common_initial))
    proof=build(old,cohorts,initial,a.study_base,a.source_commit,a.source_manifest_sha256,a.output)
    print(json.dumps(dict(prepared=True,launch_ready=False,bundle=proof)))


if __name__=='__main__':main()
