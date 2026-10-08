"""Same-h100-9 CPU-only formal package/config preparation, no release or submit.

Four real observation descriptors are mandatory; current NULL drafts cannot run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def load_common(review_path,review_sha):
    raw=Path(review_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=review_sha:
        raise ValueError('Formal helper source-review SHA differs')
    review=json.loads(raw)
    if (review['status']!='FORMAL_V6_CPU_PREPARE_HELPERS_SOURCE_ONLY_ACCEPTED'
            or review['fork_turns']!='none' or review['training_release'] is not False
            or review['actual_gpu_accepted'] is not False):
        raise ValueError('Independent formal helper source-only review required')
    here=Path(__file__).resolve().parent
    for name in ('formal_cpu_common.py','prepare_formal_tree_fresh200_package.py'):
        content=(here/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=review['helper_files'][name]:
            raise ValueError('Formal helper bytes changed: '+name)
        if name=='formal_cpu_common.py':
            common_raw=content
    module=types.ModuleType('formal_cpu_common');module.__file__=str(here/'formal_cpu_common.py')
    exec(compile(common_raw,module.__file__,'exec'),module.__dict__)
    module.local_path(review_path)
    module.require(review['code_commit']==module.COMMIT, 'Formal helper review source pin differs')
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','archive-sha','target','manifest-sha','inputs','inputs-sha','helper-review','helper-review-sha'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    c=load_common(args.helper_review,args.helper_review_sha);c.assert_host()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    inputs=c.json_bound(dict(path=args.inputs,sha256=args.inputs_sha))
    c.require_observation_descriptors(inputs['production_inputs'])
    c.require(inputs['code_commit']==c.COMMIT and inputs['stage']==c.STAGE
              and inputs['manifest_sha256']==args.manifest_sha and inputs['archive_sha256']==args.archive_sha
              and inputs['target']==args.target and inputs['archive']==args.archive, 'Reviewed formal preparation argv differ')
    c.require(not any(name in os.environ for name in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),
              'Formal CPU preparation forbids calibration-only environment')
    manifest=c.safe_extract(args.archive,args.archive_sha,args.target,args.manifest_sha,inputs['max_uncompressed_bytes'])
    c.require(inputs['production_inputs']==manifest['production_inputs'], 'Reviewed actual observation descriptors differ from formal manifest')
    target=c.local_path(args.target)
    sys.path[:0]=[str(target),str(c.ROOT/'tools'),str(target/'source/repo')]
    import production_contract
    c.require(Path(production_contract.__file__).resolve()==(target/'production_contract.py').resolve(), 'Formal preparation imported another production contract')
    values={name:production_contract.descriptor(manifest['production_inputs'][name],c.ROOT) for name in c.INPUT_NAMES}
    thresholds=production_contract.observed_production_inputs(manifest,values)
    c.require(thresholds==manifest['bootstrap_thresholds'] and {name:value[1] for name,value in values.items()}==manifest['production_input_sha256'],
              'Actual bootstrap/observation bytes differ from formal package')
    spec=c.json_bound(dict(path=str(target/'CONFIGURATION_SPEC.json'),sha256=manifest['files']['CONFIGURATION_SPEC.json']))
    c.require(spec['trial_only'] is False and not spec['environment'] and spec['production_thresholds']=={
        'target_old_policy_kl':.2,'negative_tail_limit':None,'initial_actor_repeat_max_abs_delta':None}, 'Formal fixed spec/observed threshold scope changed')
    c.require(production_contract.configuration_overrides(spec,thresholds)==manifest['configuration_overrides'], 'Formal actual configuration overrides differ')
    c.verify_local_assets(manifest)
    prep=target/'CPU_PREP_CONFIG';prep.mkdir()
    os.environ.update(SEARCH_R1_REPO=str(target/'source/repo'),TREE_COMMON_DATA=str(c.ROOT/'instruct300/common-data'),
        TREE_INSTRUCT_MODEL=str(c.MODEL),SEARCH_R1_ASSETS=str(c.ROOT/'assets-root'),
        SEARCH_R1_BASELINE_OUTPUT=str(prep/'output'),SEARCH_R1_BASELINE_CHECKPOINTS=str(prep/'checkpoints'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1')
    import search_r1
    c.require(Path(search_r1.__file__).resolve()==(target/'source/repo/search_r1/__init__.py').resolve(), 'Formal CPU preparation imported another source')
    import supervisor
    c.require(supervisor.PACKAGE==manifest and Path(supervisor.__file__).resolve()==(target/'supervisor.py').resolve(),
              'Formal CPU configuration consumed another manifest/supervisor')
    supervisor.PRODUCTION_THRESHOLDS=dict(thresholds)
    value,steps=supervisor.configuration(prep,c.STAGE)
    from omegaconf import OmegaConf
    resolved=OmegaConf.to_container(value,resolve=True)
    checked=c.check_resolved(resolved,spec,thresholds,prep/'checkpoints')
    c.require(steps==200 and os.environ['CUDA_VISIBLE_DEVICES']=='', 'Formal CPU budget/GPU visibility changed')
    torch=sys.modules.get('torch')
    c.require(torch is None or not torch.cuda.is_initialized(), 'Formal CPU configuration initialized CUDA')
    c.verify_package(target,manifest)
    for name in c.INPUT_NAMES:
        production_contract.descriptor(manifest['production_inputs'][name],c.ROOT)
    result=dict(status='SAME_NODE_FORMAL_FRESH200_CPU_CONFIG_AND_ASSETS_VERIFIED_NOT_RELEASED',stage=c.STAGE,
        code_commit=c.COMMIT,package=str(target),manifest_sha256=args.manifest_sha,archive_sha256=args.archive_sha,
        inputs_sha256=args.inputs_sha,helper_review_sha256=args.helper_review_sha,
        production_observation_descriptors=manifest['production_inputs'],production_input_sha256=manifest['production_input_sha256'],
        bootstrap_thresholds=thresholds,bootstrap_thresholds_accepted=True,production_thresholds_accepted=False,
        same_live_first5_accepted=False,continuation_6_200_accepted=False,
        resolved_config_path=str(prep/'resolved.yaml'),resolved_config_sha256=c.file_sha(prep/'resolved.yaml'),
        actual_hydra_resolved=True,cuda_initialized=False,actual_gpu_accepted=False,
        actual_3b_training_accepted=False,optimizer_writes=0,completed_outers=0,
        official_model_path=str(c.MODEL),model_identity=manifest['model_identity'],common_data_identity=c.DATA_ID,
        checked_resolved_values=checked,checked_resolved_field_count=len(checked),protocol=manifest['protocol'],
        original_sampler_actor_gate_accepted=False,long_training_ready=False,training_release=False,submitted=False,
        scope='Same-h100-9 actual formal Hydra/config and local model/data byte verification only; no GPU/main/optimizer/service invocation or fifth dispatch release')
    output=prep/'PREPARATION.json';report_sha=c.write_once(output,result)
    print(json.dumps(dict(status=result['status'],path=str(output),sha256=report_sha,submitted=False)),flush=True)


if __name__=='__main__':
    main()
