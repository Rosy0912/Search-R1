"""Prepare one fresh PR100 config and the extended immutable TRAIN cohorts.

Stdlib/data file preparation only. No CUDA, HTTP, signals or experiment launch.
"""
import argparse
import json
from pathlib import Path
import uuid

try:
    from .checkpoint import require,checked_json,fileproof,exclusive_json
    from .prepare import template,validate_config
    from .cohort_v1 import freeze_from_files
except ImportError:
    from checkpoint import require,checked_json,fileproof,exclusive_json
    from prepare import template,validate_config
    from cohort_v1 import freeze_from_files

SOURCE_SHA='2ef6070a328357b0f9d3a026010aaed0aa3f8840a241bc1021ed9a183e8d75f7'


def check_original_prefix(cohorts,original):
    require(original.get('outer_count')==10 and original.get('questions_per_outer')==512,
            'Prior ablation cohort must be the accepted original ten')
    require(cohorts['outer_count']==100 and cohorts['questions_per_outer']==512,
            'PR100 cohort must contain100 complete512-question outers')
    require(cohorts['cohorts'][:10]==original['cohorts']
            and cohorts['permutation_sha256']==original['permutation_sha256']
            and cohorts['train_identity']==original['train_identity']
            and cohorts['validation_identity']==original['validation_identity'],
            'PR100 first ten/order/data differs from original ablation cohort')
    return dict(accepted=True,identical_first_ten_outers=True,original_outer_count=10,new_outer_count=100,
                selected_questions=51200,validation_overlap=0)


def build(assets,initial,original,base,source_commit,manifest_sha,output):
    require(len(source_commit)==40 and len(manifest_sha)==64,'Exact published PR100 source identities required')
    require(assets['source_model_manifest']['sha256']==SOURCE_SHA,'Source10012-file identity differs')
    require(initial['accepted'] is True and initial['tensors']==504 and initial['dtype']=='float32'
            and initial['zero_B_tensors']==252 and initial['source_model_identity']==SOURCE_SHA,
            'Accepted original common504 FP32 zero-B template required')
    train=assets['train_identity'];validation=assets['validation_identity']
    cohorts=freeze_from_files(train,validation)
    prefix=check_original_prefix(cohorts,original)
    output=Path(output);require(not output.exists(),'Never overwrite PR100 preparation')
    output.mkdir(parents=True);base=str(Path(base).absolute())
    helper=base+'/source/training-'+source_commit[:7]+'/opsd_pr100_training_v1'
    proof=exclusive_json(output/'common-cohorts100.json',cohorts)
    initial_proof=exclusive_json(output/'common-initial-adapter.json',initial)
    def remote(ref):return dict(path=base+'/receipts/'+Path(ref['path']).name,sha256=ref['sha256'],bytes=ref['bytes'])
    config=template(base,'PR','pr100-'+uuid.uuid4().hex,base)
    config['run_id']=uuid.uuid4().hex;config['source_commit']=source_commit
    config['paths']['helper_root']=helper
    # Same immutable frozen model/runtime/data assets; no old optimizer/run/authorization.
    config['paths']['runtime']=assets['paths']['runtime']
    config['paths']['model']=assets['paths']['model']
    config['paths']['train']=train['path'];config['paths']['validation']=validation['path']
    config['source_model_manifest']=assets['source_model_manifest'];config['runtime_manifest']=assets['runtime_manifest']
    config['train_identity']=train;config['validation_identity']=validation
    config['helper_source_manifest']=dict(path=helper+'/source-manifest.json',sha256=manifest_sha)
    config['common_cohort_manifest']=remote(proof);config['common_initial_adapter_manifest']=remote(initial_proof)
    config['common_initial_adapter_tensor_sha256']=initial['adapter_tensor_sha256']
    config['preparation_status']=dict(ready=False,holds=validate_config(config),no_cuda_started=True,
        same_original_first_ten=prefix,scope='New exact human authorization and published scoped gates must be filled before launch')
    cfg=exclusive_json(output/'config-candidate.json',config)
    return exclusive_json(output/'bundle.json',dict(schema='opsd-pr100-private-cpu-preparation-v1',
        base=base,run_id=config['run_id'],study_id=config['study_id'],candidate=cfg,common_files=[proof,initial_proof],
        first_ten_prefix=prefix,source_commit=source_commit,source_manifest_sha256=manifest_sha,
        new_fresh_adam=True,no_cuda_started=True,launch_ready=False))


def main():
    p=argparse.ArgumentParser()
    for name in ('assets-config','assets-config-sha256','common-initial','common-initial-sha256',
                 'prior-common-cohort','prior-common-cohort-sha256','base','source-commit','source-manifest-sha256','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    assets,_=checked_json(dict(path=a.assets_config,sha256=a.assets_config_sha256))
    initial,_=checked_json(dict(path=a.common_initial,sha256=a.common_initial_sha256))
    original,_=checked_json(dict(path=a.prior_common_cohort,sha256=a.prior_common_cohort_sha256))
    result=build(assets,initial,original,a.base,a.source_commit,a.source_manifest_sha256,a.output)
    print(json.dumps(dict(prepared=True,no_cuda_started=True,launch_ready=False,bundle=result)))


if __name__=='__main__':main()
