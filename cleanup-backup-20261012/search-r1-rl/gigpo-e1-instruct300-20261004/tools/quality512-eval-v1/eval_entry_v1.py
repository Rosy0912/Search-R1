"""Evaluate saved HF on the immutable512 split; no optimizer or restore."""
import json, os, math
from pathlib import Path
import hydra, ray

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    from search_r1.baselines.preflight import validate_gpu_lease, validate_assets_and_retriever
    from search_r1.baselines.local_runtime import check_local_runtime
    from search_r1.baselines.holdout import prepare_common_holdout
    from search_r1.baselines.controller import validate_config
    from search_r1.baselines.runtime_audit import config_boundary, runtime_env_vars
    from eval_observer_v1 import observed_main_task
    assert config.trainer.val_only and config.trainer.val_before_train
    assert not config.trainer.instruct300.enabled and not config.trainer.resume.enabled
    assert not config.baselines.holdout_eval_only
    validate_gpu_lease(config); check_local_runtime(config)
    split=prepare_common_holdout(config)
    assert split['holdout_rows']==512 and split['train_rows']==169028
    assert split['content_identity']['sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    validate_config(config); config_boundary(config,split); validate_assets_and_retriever(config)
    ray.init(address='127.0.0.1:16381',runtime_env={'env_vars':{'TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN'}})
    check_local_runtime(config,ray.nodes())
    result=ray.get(observed_main_task.options(runtime_env={'env_vars':runtime_env_vars()}).remote(config))
    assert int(result['val/qa_count/all'])==512
    assert int(result['val/qa_count/nq'])==256 and int(result['val/qa_count/hotpotqa'])==256
    for key in ('val/f1/all','val/em/all'):
        assert math.isfinite(float(result[key])) and 0<=float(result[key])<=1
    record=dict(status='ACTUAL_GPU512_EVALUATION_COMPLETE',metrics=result,
        hf_directory=config.actor_rollout_ref.model.path,frozen_reference=config.actor_rollout_ref.ref.model_path,
        data_identity=split['content_identity']['sha256'],job=os.environ['SLURM_JOB_ID'],
        source=os.environ['SEARCH_R1_BASELINE_COMMIT'],config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'])
    path=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'evaluation-result.json'
    with path.open('x') as f: json.dump(record,f,indent=2)
    print('QUALITY_CANDIDATE_EVALUATION_JSON='+json.dumps(record,sort_keys=True),flush=True)

if __name__=='__main__':main()
