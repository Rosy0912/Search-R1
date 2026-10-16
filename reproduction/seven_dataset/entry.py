"""Independent milestone entry: frozen HF weights, fixed data, no trainer.fit()."""
import argparse
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.request
from .contracts import (TOTAL,DATA_SHA,FINAL_SHA,IDS_SHA,sha,digest,require,
                        validate_manifest,validate_model_files,validate_expected_rows,validate_scheduler_binding)


def inspect_inputs(plan):
    import pandas as pd
    from omegaconf import OmegaConf
    from search_r1.baselines.holdout import plain,frame_identity
    require(plan['protocol']=='tree_seven_milestone_v1','Unknown evaluator plan')
    require(plan['ordered_ids_sha256']==IDS_SHA,'Plan question identity differs')
    require(sha(plan['checkpoint_manifest'])==plan['checkpoint_manifest_sha256'],'Checkpoint manifest changed')
    manifest=json.loads(Path(plan['checkpoint_manifest']).read_text())
    validate_manifest(plan,manifest)
    require(sha(plan['training_config'])==plan['training_config_sha256'],'Training config changed')
    require(sha(plan['configuration_receipt'])==plan['configuration_receipt_sha256'],'Configuration receipt changed')
    receipt=json.loads(Path(plan['configuration_receipt']).read_text())
    require(receipt['run_id']==plan['training_run_id'] and receipt['config_sha256']==plan['training_config_sha256']
            and receipt.get('group',receipt.get('arm'))==plan['group'],'Configuration/run/group binding changed')
    require(receipt['data_identity']==DATA_SHA,'Training data identity differs')
    model_digest=validate_model_files(plan)
    expected_identity=digest(dict(group=plan['group'],outer=plan['outer'],run=plan['training_run_id'],
        training_config=plan['training_config_sha256'],model_files_sha256=model_digest,data=DATA_SHA))
    require(plan['checkpoint_identity']==expected_identity,'Checkpoint identity changed')
    data=Path(plan['data_root']).resolve(strict=True)
    require(sha(data/'manifest.json')==plan['data_manifest_sha256'],'Data manifest changed')
    manifest_data=json.loads((data/'manifest.json').read_text())
    final=manifest_data['splits']['final']
    require(final['sha256']==FINAL_SHA and sha(data/'final.parquet')==FINAL_SHA and
            final['question_ids_sha256']==IDS_SHA and final['rows']==TOTAL,'Final data identity differs')
    frame=pd.read_parquet(data/'final.parquet');expected=[]
    for _,row in frame.iterrows():
        gold=plain(row['reward_model'])['ground_truth']['target']
        if isinstance(gold,str):gold=[gold]
        expected.append(dict(question_id=plain(row['extra_info'])['index'],source=row['data_source'],gold_answers=gold))
    validate_expected_rows(expected)
    require(frame_identity(frame,'prompt')==final['content_identity']==plan['final_content_identity'],
            'Final frame content differs')
    cfg=OmegaConf.load(plan['training_config'])
    # Ablation fields describe training trees and the training schedule. Native
    # validation always uses one greedy search trajectory. Validate the archived
    # ablation contract, then use the same original validation configuration for
    # every frozen model; no training or checkpoint restore is invoked here.
    if 'recursive_ablation' in cfg.baselines:
        from release.recursive_bran_a.ablation150_v1.contracts import validate_config_dict,projected_original
        archived=OmegaConf.to_container(cfg,resolve=True)
        validate_config_dict(archived)
        cfg=OmegaConf.create(projected_original(archived))
    OmegaConf.set_struct(cfg,False)
    inference_model=plan.get('inference_model',plan['model'])
    if 'inference_model' in plan:
        from .model_view import validate_view
        require(Path(plan['canonical_tokenizer']).resolve(strict=True)==Path(receipt['model']).resolve(strict=True),
                'Canonical tokenizer is not the actual archived training model path')
        validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
    cfg.actor_rollout_ref.model.path=inference_model;cfg.actor_rollout_ref.ref.model_path=inference_model
    # All inference search caps and stop/token semantics are retained.
    cfg.data.train_files=str(data/'train.parquet');cfg.data.val_files=str(data/'validation.parquet')
    cfg.data.final_val_files=str(data/'final.parquet')
    cfg.baselines.bapo.stage_validation_files=str(data/'stage.parquet')
    cfg.baselines.holdout.data_root=str(data);cfg.baselines.holdout.manifest_path=str(data/'manifest.json')
    cfg.baselines.holdout.manifest_sha256=plan['data_manifest_sha256']
    cfg.data.val_batch_size=1024;cfg.actor_rollout_ref.rollout.gpu_memory_utilization=0.65
    cfg.baselines.local_runtime_host=socket.gethostname().split('.')[0]
    cfg.trainer.logger=['console'];cfg.trainer.default_local_dir=str(Path(plan['output'])/'unused-checkpoints')
    cfg.retriever.url='http://127.0.0.1:29147/retrieve'
    require(cfg.baselines.method=='tree_grpo' and cfg.do_search and cfg.max_turns==4 and
            cfg.force_final_answer is False and cfg.retriever.topk==3 and
            [cfg.data.max_start_length,cfg.data.max_prompt_length,cfg.data.max_response_length,cfg.data.max_obs_length]==[2048,4096,500,500]
            and cfg.actor_rollout_ref.rollout.tree_action_tag_stop is True and
            cfg.actor_rollout_ref.rollout.tree_action_stop_mode=='search_only' and
            cfg.trainer.n_gpus_per_node==8 and cfg.trainer.nnodes==1 and
            cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==1,
            'Scientific evaluation protocol changed')
    from search_r1.baselines.controller import validate_config
    validate_config(cfg)
    return cfg,expected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text())
    cfg,expected=inspect_inputs(plan)
    if args.validate_only:
        print(json.dumps(dict(status='INPUTS_VERIFIED_NO_INFERENCE',count=len(expected),checkpoint_identity=plan['checkpoint_identity'])))
        return
    from omegaconf import OmegaConf
    from search_r1.baselines.preflight import validate_gpu_lease
    base=Path(plan['base']).resolve(strict=True);out=Path(plan['output'])
    expected_root=base/'seven-evaluations'/plan['checkpoint_identity']
    if 'replaces_job_id' in plan:
        expected_root=expected_root/('recovery-'+plan['evaluation_code_commit'][:12])
        previous=Path(plan['previous_submission'])
        require(sha(previous)==plan['previous_submission_sha256'],'Previous submission identity differs')
        prior=json.loads(previous.read_text())
        require(prior['job_id']==plan['replaces_job_id'] and prior['checkpoint_identity']==plan['checkpoint_identity'],
                'Recovery targets a different model allocation')
        import subprocess
        states=subprocess.check_output(['sacct','-X','-n','-P','-j',plan['replaces_job_id'],
            '--format=JobIDRaw,State'],text=True)
        matched=[r.split('|')[1] for r in states.splitlines() if r.startswith(plan['replaces_job_id']+'|')]
        require(matched==['FAILED'],'Previous evaluation is not exactly verified FAILED')
        prior_out=Path(prior['output'])
        require(not (prior_out/'progress.json').exists() and
                (not (prior_out/'predictions.jsonl').exists() or (prior_out/'predictions.jsonl').stat().st_size==0),
                'Prediction-producing evaluation requires a separate audited result continuation')
    require(out==expected_root/'output','Evaluation output namespace differs')
    intent=out.parent/'submission-intent.json'
    require(intent.is_file(),'No matching unique submission intent')
    submission_path=out.parent/'submission.json'
    if plan['backend']=='slurm_lumia':
        deadline=time.monotonic()+120
        while not submission_path.exists() and time.monotonic()<deadline:time.sleep(1)
    submission=json.loads(submission_path.read_text()) if submission_path.exists() else None
    actual_job=validate_scheduler_binding(plan,json.loads(intent.read_text()),submission,os.environ,sha(args.plan))
    require(os.statvfs(base).f_bavail*os.statvfs(base).f_frsize>=408_000_000_000,'Training checkpoint reserve unavailable')
    validate_gpu_lease(cfg)
    out.mkdir(exist_ok=False)
    config_text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
    (out/'evaluation-config.yaml').write_text(config_text)
    eval_sha=sha(out/'evaluation-config.yaml')
    os.environ.update(TREE_SEVEN_EVAL_MODE='frozen_milestone_v1',TREE_SEVEN_CHECKPOINT_IDENTITY=plan['checkpoint_identity'],
        SEARCH_R1_BASELINE_RUN_ID=plan['evaluation_run_id'],E1_RUN_ID=plan['evaluation_run_id'],
        SEARCH_R1_BASELINE_COMMIT=plan['evaluation_code_commit'],SEARCH_R1_BASELINE_CONFIG_SHA256=eval_sha,
        SEARCH_R1_BASELINE_DATA_IDENTITY=DATA_SHA,SEARCH_R1_MODEL_IDENTITY=plan['checkpoint_identity'],
        SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
        SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),SEARCH_R1_PJ_NODE=socket.gethostname().split('.')[0],
        TREE_COMMON_DATA=plan['data_root'],TREE_INSTRUCT_MODEL=plan.get('inference_model',plan['model']),TREE_LINEAR_PREFIX_SCAN='1')
    for key in ('RECURSIVE_LUMIA_PAIRED_EVALUATION','RECURSIVE_ABLATION_RESUME_MANIFEST','RECURSIVE_ABLATION_PJ_BINDING'):
        os.environ.pop(key,None)
    (out/'plan.json').write_bytes(Path(args.plan).read_bytes())
    retriever=json.loads(Path(plan['retriever_receipt']).read_text())
    require(sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256'] and
            retriever['encoder_dtype']==retriever['index_dtype']=='float32' and retriever['validation']['top3_identical'],
            'Retriever identity differs')
    with urllib.request.urlopen(retriever['url']+'/health',timeout=15) as response:
        require(json.load(response)['job']==retriever['job'],'Retriever job differs')
    from http.server import ThreadingHTTPServer
    from reproduction.recursive_lumia.proxy import Handler
    server=ThreadingHTTPServer(('127.0.0.1',29147),Handler);server.remote=retriever['url']
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
    prepare_click()
    from .contracts import communication_transport
    transport=communication_transport(plan)
    os.environ.update(transport)
    import ray
    from .runner import run,worker_setup
    started=time.time();outcome='FAILED'
    try:
        context=ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,
            object_store_memory=8*1024**3,_node_ip_address='127.0.0.1',_temp_dir='/tmp/seven-'+plan['evaluation_run_id'][:12],
            runtime_env={'worker_process_setup_hook':worker_setup,'env_vars':{k:v for k,v in os.environ.items()
                if k.startswith(('TREE_','SEARCH_R1_','E1_','NCCL_')) or k in ('PYTHONPATH','TOKENIZERS_PARALLELISM')}})
        os.environ['RAY_ADDRESS']=context.address_info['gcs_address']
        from search_r1.baselines.local_runtime import check_local_runtime
        check_local_runtime(cfg,ray.nodes())
        report=ray.get(ray.remote(run).remote(cfg,plan,expected))
        validate_model_files(plan)
        if 'inference_model' in plan:
            from .model_view import validate_view
            report['inference_model_files']=validate_view(plan,cfg.baselines.tree.token_preservation.tokenizer_sha256)
        report.update(status='COMPLETE_FULL_51713_ACCEPTED',group=plan['group'],outer=plan['outer'],job=plan['job'],actual_job_id=actual_job,
            checkpoint_identity=plan['checkpoint_identity'],evaluation_code_commit=plan['evaluation_code_commit'],
            evaluation_config_sha256=eval_sha,training_config_sha256=plan['training_config_sha256'],
            communication_transport=transport,
            data_manifest_sha256=plan['data_manifest_sha256'],plan_sha256=sha(args.plan),
            inference_wall_gpu_hours_lower_bound=8*(time.time()-started)/3600,
            cost_status='SCHEDULER_ALLOCATION_COST_PENDING_POST_EXIT_AUDIT')
        with (out/'summary.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        outcome='COMPLETE_FULL_51713_ACCEPTED'
    finally:
        ray.shutdown();server.shutdown();server.server_close()
        (out/'entry-exit.json').write_text(json.dumps(dict(status=outcome,job=plan['job'],actual_job_id=actual_job,time=time.time())))

if __name__=='__main__':main()
