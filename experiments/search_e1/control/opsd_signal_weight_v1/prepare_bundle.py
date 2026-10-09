"""Create four small immutable candidate configs; no tensors or CUDA imports."""
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

VARIANTS=('G_POS','G_NEG','H_POS','H_NEG')


def build(old_assets,cohorts,initial,study_base,source_commit,manifest_sha,output):
    require(len(source_commit)==40 and len(manifest_sha)==64,'Exact newly published source identities required')
    source_sha=old_assets['source_model_manifest']['sha256']
    require(isinstance(source_sha,str) and len(source_sha)==64,'Explicit relocated K4 twelve-file source manifest missing')
    require(cohorts['outer_count']==25 and cohorts['questions_per_outer']==512
            and cohorts['validation_id_overlap']==0 and cohorts['validation_exact_question_text_overlap']==0,'Common cohort contract differs')
    require(initial['accepted'] is True and initial['tensors']==504 and initial['dtype']=='float32'
            and initial['zero_B_tensors']==252 and initial['source_model_identity']==source_sha,'Actual common initial FP32 template missing')
    study_base=str(Path(study_base).absolute());output=Path(output)
    require(not output.exists(),'Never overwrite a prepared group/config/run identity')
    output.mkdir(parents=True)
    study_id='signal-four-'+uuid.uuid4().hex
    helper_root=study_base+'/source/training-'+source_commit[:7]+'/opsd_signal_weight_v1'
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
        config['paths']['model']=old_assets['paths']['model']
        config['paths']['python']=old_assets['paths']['python']
        for name in ('train_identity','validation_identity'):
            config[name]=cohorts[name]
        config['paths']['train']=cohorts['train_identity']['path']
        config['paths']['validation']=cohorts['validation_identity']['path']
        config['helper_source_manifest']=dict(path=helper_root+'/source-manifest.json',sha256=manifest_sha)
        config['common_cohort_manifest']=remote_common(common_cohort)
        config['common_initial_adapter_manifest']=remote_common(common_initial)
        config['common_initial_adapter_tensor_sha256']=initial['adapter_tensor_sha256']
        # Authorization/gates must be fresh execution-goal evidence supplied
        # separately. An old assets config never grants this new four-group run.
        holds=validate_config(config)
        config['preparation_status']=dict(ready=False,holds=holds,no_cuda_started=True,
            storage_scope='Large weights/raw/checkpoints remain HKUST remote; new four-group output receipt required')
        proof=exclusive_json(output/('group-'+variant+'-config-candidate.json'),config)
        configs.append(dict(variant=variant,run_id=config['run_id'],candidate=proof,
                            node_candidate_path=base+'/receipts/config-candidate.json',output_root=config['paths']['output_root'],holds=holds))
    return exclusive_json(output/'bundle.json',dict(schema='opsd-signal-weight-four-private-cpu-candidates-v1',study_id=study_id,
        study_base=study_base,source_commit=source_commit,source_manifest_sha256=manifest_sha,groups=configs,
        common_files=[common_cohort,common_initial],launch_ready=False,no_cuda_started=True,
        no_old_adam_or_authorization_inherited=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-assets-config',required=True)
    p.add_argument('--old-assets-config-sha256',required=True)
    p.add_argument('--common-cohort',required=True);p.add_argument('--common-initial',required=True)
    p.add_argument('--study-base',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();old,_=checked_json(dict(path=a.old_assets_config,sha256=a.old_assets_config_sha256))
    cohorts,_=checked_json(fileproof(a.common_cohort));initial,_=checked_json(fileproof(a.common_initial))
    proof=build(old,cohorts,initial,a.study_base,a.source_commit,a.source_manifest_sha256,a.output)
    print(json.dumps(dict(prepared=True,launch_ready=False,bundle=proof)))


if __name__=='__main__':main()
