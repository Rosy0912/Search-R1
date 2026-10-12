"""CPU-only immutable B resume config after actual complete3 technical stop."""
import argparse,hashlib,json,os,time,uuid
from pathlib import Path
from omegaconf import OmegaConf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--terminal',type=Path,required=True);p.add_argument('--retry-terminal',type=Path);p.add_argument('--driver-fault-terminal',type=Path);a=p.parse_args()
root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011');repo=Path(__file__).parents[2].resolve();source=(repo/'SOURCE_COMMIT').read_text().strip()
terminal=json.loads(a.terminal.read_text());assert terminal['status'] in ('TECHNICAL_SWITCH_ALL_OWNED_ABSENT','TECHNICAL_SWITCH_ALL_OWNED_KERNEL_EXITED') and terminal['all_original_owned_kernel_exited']
parent=root/'checkpoints/training_state/global_step_3/manifest.json';m=json.loads(parent.read_text());assert m['identity']['completed_step']==3
old=root/'receipts/configuration-d79908be70f74dd2a0ce38b5ef880f1e/resolved.yaml';assert sha(old)=='95d9f8f2195a35f189ff06af8e449d419040ae96c4cde4fc50f9d7062db414f7'
run=uuid.uuid4().hex;target=root/'receipts'/('speed-resume3-'+run);target.mkdir(exist_ok=False)
binding=dict(protocol='recursive_b_sparse_schedule_continuation_v1',parent_completed_step=3,
 parent_commit='3da904591125f117b1eae1ba09546ebe793cb2d4',parent_run_id='ef09290e69454f20b6a9594ad8e485b6',
 parent_config=str(old),parent_config_sha256=sha(old),parent_manifest=str(parent),parent_manifest_sha256=sha(parent),parent_identity=m['identity'],
 sampling_streams=[],sampling_stream_status='PARENT_STREAM_NOT_RECORDED_COLD_EXECUTOR_REINITIALIZED',
 latest_human_authority='Stop current B, fix according to acceleration-handoff, resume same complete checkpoint; retain partial4, no repeated completed updates')
bpath=target/'binding.json';bpath.write_text(json.dumps(binding,sort_keys=True))
cfg=OmegaConf.load(old);cfg.trainer.evaluation_steps=[5,50,100,150,200];cfg.trainer.checkpoint_save_steps=[5,50,100,150,200];cfg.actor_rollout_ref.rollout.tree_action_tag_stop=True;cfg.actor_rollout_ref.rollout.tree_action_stop_mode='search_only';cpath=target/'resolved.yaml';cpath.write_text(OmegaConf.to_yaml(cfg))
os.environ.update(RECURSIVE_B_CONTINUATION_BINDING=str(bpath),RECURSIVE_B_CONTINUATION_SHA256=sha(bpath))
from search_r1.baselines.recursive_b_schedule_resume import validate_entry,parent_manifest
validate_entry(cfg);parent_manifest()
record=dict(run_id=run,config=str(cpath),config_sha256=sha(cpath),binding=str(bpath),binding_sha256=sha(bpath),
 source=str(repo),source_commit=source,model=cfg.actor_rollout_ref.model.path,data_identity=m['identity']['data_identity'],
 output=str(root/('output-speed-resume3-'+run)),parent_terminal=str(a.terminal),parent_terminal_sha256=sha(a.terminal),
 preparation_time=time.time(),parent_completed_step=3,ready=True,scientific_config_change='Only explicit eval/checkpoint lists; runtime acceleration preserves native mu objective')
if a.retry_terminal:
 from search_r1.baselines.recursive_b_retry import validate_retry_terminal
 validate_retry_terminal(a.retry_terminal,sha(a.retry_terminal))
 record.update(retry_terminal=str(a.retry_terminal),retry_terminal_sha256=sha(a.retry_terminal))
if a.driver_fault_terminal:
 from release.recursive_bran_b.capture_driver_stage_terminal import validate
 validate(a.driver_fault_terminal,sha(a.driver_fault_terminal))
 record.update(driver_fault_terminal=str(a.driver_fault_terminal),driver_fault_terminal_sha256=sha(a.driver_fault_terminal))
rpath=target/'record.json';rpath.write_text(json.dumps(record,sort_keys=True));print(json.dumps(dict(record_path=str(rpath),record_sha256=sha(rpath),record=record)))
