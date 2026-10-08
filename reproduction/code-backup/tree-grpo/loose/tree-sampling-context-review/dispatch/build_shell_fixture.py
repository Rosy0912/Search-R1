"""Build tiny CPU assets for the real three-payload shell execution review.

This fixture is never deployed. A sitecustomize hook maps only the remote task
root to the fixture and supplies a fake hostname, so genuine source files can
execute on fabricated tensors without any actual remote/model asset reads.
"""
import gzip
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
REMOTE_ROOT='/dataset/shaoshitong/search-r1-rl/tree-grpo'
SOURCE='a632ba91840fb116aacfc0e770681a860647b295'
DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return sha(path)
def tensors(rows=1,raw=False):
    out=dict(input_ids=torch.tensor([[1,2,3]]).repeat(rows,1),position_ids=torch.tensor([[0,1,2]]).repeat(rows,1),
        attention_mask=torch.ones(rows,3,dtype=torch.int64),responses=torch.tensor([[3]]).repeat(rows,1),
        old_log_probs=torch.full((rows,1),-2.,dtype=torch.float64))
    if raw:out.update(sampled_log_probs=torch.full((rows,1),-2.,dtype=torch.float64),distribution={'temperature':1.})
    else:out.update(loss_mask=torch.ones(rows,1,dtype=torch.int64),policy_weights=torch.ones(rows,1,dtype=torch.float64))
    return out

def build(root):
    root=Path(root).resolve();run=root/'runs/tree-safety300-3268';package_dir=root/'instruct300/candidates/safety110-a632ba9-84e3e186'
    tools=root/'instruct300/context-analysis-28e665eb-v2';tools.mkdir(parents=True)
    shim=root/'fixture-python-path';shim.mkdir()
    (shim/'sitecustomize.py').write_text('''import os,pathlib,socket
_original_new=pathlib.Path.__new__
_fake=os.environ['TREE_CONTEXT_REVIEW_FAKE_ROOT']
_remote='/dataset/shaoshitong/search-r1-rl/tree-grpo'
def _new(cls,*args,**kwargs):
 if args:
  text=os.fspath(args[0])
  if text==_remote or text.startswith(_remote+'/'):
   args=(_fake+text[len(_remote):],)+args[1:]
 return _original_new(cls,*args,**kwargs)
pathlib.Path.__new__=staticmethod(_new)
socket.gethostname=lambda:'h100-9'
''')
    metadata=json.loads((WORK/'tree-safety110-review/actual3268/FINAL_METADATA.json').read_bytes())
    config=metadata['files']['resolved.yaml']['text'].encode();cfg_path=run/'resolved.yaml';cfg_path.parent.mkdir(parents=True)
    cfg_path.write_bytes(config);config_sha=sha(cfg_path)
    assert config_sha=='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b'
    entry_cfg=run/'output/entry-audit/entry-config.yaml';entry_cfg.parent.mkdir(parents=True);entry_cfg.write_bytes(config)
    parent_identity={'completed_step':110};parent_path=root/'fixture-parent-manifest.json';parent_sha=json_file(parent_path,{'identity':parent_identity})
    package=dict(stage='calibration110',probability_calibration={'environment':{'TREE_PPO_CALIBRATION_ONLY':'1'}},
        code_commit=SOURCE,common_data_identity=DATA,model_identity={'sha256':MODEL},files={},
        parent_manifest={'path':str(parent_path),'sha256':parent_sha,'identity':parent_identity})
    names=['search_r1/baselines/sampling_consistency.py','search_r1/baselines/policy_safety_runtime.py','verl/workers/fsdp_workers.py',
        'verl/trainer/ppo/ray_trainer.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/trainer/config/ppo_tree_safety300.yaml']
    for name in names:
        p=package_dir/'source/repo'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((WORK/'tree-safety110-a632ba9-v2/source/repo'/name).read_bytes());package['files']['source/repo/'+name]=sha(p)
    package_sha=json_file(package_dir/'manifest.json',package)
    json_file(run/'output/recovery-loaded.json',{'provenance':{'parent_identity':parent_identity,'parent_cursor':57600,'exact_executor_resume':False}})
    identity=dict(job_id='3268',run_id=run.name,code_commit=SOURCE,config_sha256=config_sha,data_identity=DATA,model_identity=MODEL)
    ready=dict(status='CALIBRATION_OBSERVED_ZERO_WRITES',parent_completed_outer=110,observed_outer=111,new_completed_outer=0,
        optimizer_committed=0,scheduler_advanced=False,workers=[],**identity)
    for rank in range(8):
        snapshot=run/'output/sampling-consistency'/f'rank-{rank:02d}-call-00001.pt';snapshot.parent.mkdir(parents=True,exist_ok=True)
        torch.save(tensors(raw=True),snapshot)
        record=dict(rank=rank,world_size=8,generation_call=1,scheduler_step=110,optimizer_committed=0,saved_rows=1,local_rows=1,enforced=False,
            snapshot_path=str(snapshot),snapshot_sha256=sha(snapshot),distribution={'temperature':1.},
            decision={'valid_tokens':8,'invalid_tokens':0,'numeric_bound_exceeded':0,'max_abs_delta':0.,'mean_abs_delta':0.,'rms_delta':0.,'sampled_k3_mean':0.,'allowed':True,'reasons':[]},**identity)
        path=snapshot.with_suffix('.json');record_sha=json_file(path,record)
        monitor=run/'output/probability-calibration'/f'rank-{rank:02d}-monitor.pt';monitor.parent.mkdir(parents=True,exist_ok=True);torch.save(tensors(32),monitor)
        ready['workers'].append(dict(rank=rank,world_size=8,optimizer_committed=0,scheduler_advanced=False,scheduler_step=110,
            restored_optimizer_steps=[2500],diagnostic_records=[str(path)],diagnostic_entries=[dict(path=str(path),sha256=record_sha,snapshot_path=str(snapshot),snapshot_sha256=sha(snapshot))],
            monitor_path=str(monitor),monitor_sha256=sha(monitor),**identity))
    ready_sha=json_file(run/'output/probability-calibration/ready.json',ready)
    json_file(run/'run-result.json',{'status':'CALIBRATION_OBSERVED_ZERO_WRITES_UNREVIEWED','cleanup_residual':{}})
    actor_identity=dict(outer_step=111,job_id='3268',source=SOURCE,config=config_sha,data=DATA)
    actor=tensors(3840);actor['policy_weights']=actor['policy_weights'].float()
    actor.update(info_mask=torch.tensor([[0,0,1]],dtype=torch.int64).repeat(3840,1),
        ref_log_prob=torch.full((3840,1),-2.,dtype=torch.float64),advantages=torch.zeros(3840,1),token_level_rewards=torch.ones(3840,1))
    action={'action_token_ids':[3],'observation_token_ids':[],'action_tokens':1,'observation_tokens':0,'scoring_context_matches_generation':True,
        'raw_generation_token_ids':[3],'observation_text':'','valid_search':False}
    meta=dict(uid=[f'q-{i//6}' for i in range(3840)],tree_uid=[f'root-{(i%6)//3}' for i in range(3840)],
        trajectory_uid=[f'leaf-{i}' for i in range(3840)],action_records=[[action] for i in range(3840)])
    directory=run/'output/trajectories/outer-000111';directory.mkdir(parents=True)
    actor_path=directory/'actor-inputs.pt';torch.save({'identity':actor_identity,'rows':3840,'tensors':actor,'non_tensor_batch':meta},actor_path)
    trajectory_path=directory/'trajectories.jsonl.gz'
    with gzip.open(trajectory_path,'wt') as stream:
        for i in range(3840):
            stream.write(json.dumps({'row':i,'identity':actor_identity,'metadata':{k:v[i] for k,v in meta.items()},'response_token_ids':[3],'token_ids':[1,2,3],'reward':1.})+'\n')
    json_file(directory/'manifest.json',{'status':'ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED','identity':actor_identity,'rows':3840,
        'payloads':{'actor-inputs.pt':sha(actor_path),'trajectories.jsonl.gz':sha(trajectory_path)}})
    audit=run/'output/tree-audit/step_111_rollout_fixture.json'
    fields={f'f{i}':i for i in range(16)}
    rollout={'leaf_count':3840,'question_groups':[{'roots':2,'leaves':6} for i in range(640)],
        'branches':[{'inherited':True,'parent_fields':fields,'child_fields':fields,'source':'fixture-source'} for i in range(2)],
        'leaves':[{'context_matches':True,'generation_prefix_hash':'fixture-context','scoring_prefix_hash':'fixture-context'} for i in range(3840)]}
    rollout_sha=json_file(audit,rollout)
    json_file(audit.with_name(audit.stem+'_policy_batch.json'),{'rollout_file_sha256':rollout_sha,'leaf_count':3840,
        'rows':[{'policy_weight_sum':1.,'empty_leaf':False} for i in range(3840)]})
    return dict(root=root,run=run,package=package_dir,tools=tools,shim=shim,ready_sha=ready_sha,package_sha=package_sha,
        fixture_tensor_max_input_shape=[3840,3],real_model_assets_loaded=False)
