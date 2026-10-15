"""Derive only the authorized ablation and explicitly enumerated PJ storage paths."""
import copy
import json
from pathlib import Path
import socket
import uuid
from release.recursive_bran_a.ablation150_v1.contracts import sha,projected_original
from reproduction.recursive_lumia.prepare_ablation import (
    derive as lumia_derive, BASELINE_CONFIG_SHA, BASELINE_RUN_ID, BASELINE_SOURCE_SHA, DATA_SHA,condition_for)
from release.recursive_bran_a.ablation150_v1 import PROTOCOL,EVALUATION_STEPS

MODEL='/plm-shared/zhangjunming/Workspace/Search-Self-Evolution-20261006-PJ-runtime-v1/model'
BASE=Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-20261015-v1')


def derive(template,group,base,host,data,model):
    candidate=lumia_derive(template,group,base,'gpu-l40s-2')
    candidate['baselines']['local_runtime_host']=host
    for key,name in [('train_files','train.parquet'),('val_files','validation.parquet'),('final_val_files','final.parquet')]:
        candidate['data'][key]=str(Path(data)/name)
    candidate['baselines']['holdout'].update(manifest_path=str(Path(data)/'manifest.json'),data_root=str(data))
    candidate['baselines']['bapo']['stage_validation_files']=str(Path(data)/'stage.parquet')
    candidate['actor_rollout_ref']['model']['path']=str(model)
    candidate['actor_rollout_ref']['ref']['model_path']=str(model)
    candidate['critic']['model']['tokenizer_path']=str(model)
    candidate['reward_model']['model']['input_tokenizer']=str(model)
    candidate['trainer']['experiment_name']='recursive-pj-ablation-'+group
    return candidate


def prepare(group,commit,job,node):
    from omegaconf import OmegaConf
    from search_r1.baselines.controller import validate_config
    template_path=BASE/'receipts/original-resolved-template.yaml'
    assert sha(template_path)==BASELINE_CONFIG_SHA
    data=BASE/'common-data';assert sha(data/'manifest.json')==DATA_SHA
    retriever=BASE/'receipts/pj-retriever-ready.json';r=json.loads(retriever.read_text())
    assert r['execution_backend']=='pj_pt' and r['encoder_dtype']==r['index_dtype']=='float32'
    assert r['validation']['top3_identical'] and r['validation']['max_score_error']<=2e-5
    base=BASE/'runs'/group;host=socket.gethostname().split('.')[0]
    template=OmegaConf.to_container(OmegaConf.load(template_path),resolve=True)
    cfg=derive(template,group,base,host,data,MODEL)
    validate_config(OmegaConf.create(projected_original(cfg)))
    base.mkdir(parents=True,exist_ok=False)
    for name in ('receipts','output','logs','tmp','cache','checkpoints'):(base/name).mkdir()
    path=base/'receipts/resolved.yaml';path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    record=dict(protocol=PROTOCOL,group=group,run_id=uuid.uuid4().hex,config=str(path),config_sha256=sha(path),
      template=str(template_path),template_sha256=BASELINE_CONFIG_SHA,baseline_run_id=BASELINE_RUN_ID,
      baseline_source_commit=BASELINE_SOURCE_SHA,source_commit=commit,model=MODEL,data=str(data),data_identity=DATA_SHA,
      retriever_receipt=str(retriever),retriever_receipt_sha256=sha(retriever),retriever_url=r['url'],
      base=str(base),output=str(base/'output'),fresh_authorized=True,parent_checkpoint=None,
      execution_backend='pj_pt',allocation=job,scheduler_node=node,host=host,condition=condition_for(group),
      actual_update_limit=150,evaluation_steps=list(EVALUATION_STEPS),scheduler_scale=200,warmup_rounds=57,
      original_recipe_bytes_available=False,training_submitted=True)
    receipt=base/'receipts/configuration.json';receipt.write_text(json.dumps(record,indent=2)+'\n')
    return receipt
