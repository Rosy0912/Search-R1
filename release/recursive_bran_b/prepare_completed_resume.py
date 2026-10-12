"""Prepare an isolated continuation from this task's complete8 native state."""
import argparse,hashlib,json,os,time,uuid
from pathlib import Path
from omegaconf import OmegaConf
from search_r1.baselines.recursive_b_completed_resume import (
    ROOT,PARENT_RUN,PARENT_COMMIT,PARENT_CONFIG_SHA,PROTOCOL,validate_terminal)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal',type=Path,required=True)
    a=parser.parse_args();validate_terminal(json.loads(a.terminal.read_text()))
    repo=Path(__file__).parents[2].resolve();source=(repo/'SOURCE_COMMIT').read_text().strip()
    old=ROOT/'receipts'/('speed-resume3-'+PARENT_RUN)/'resolved.yaml'
    if sha(old)!=PARENT_CONFIG_SHA:raise ValueError('Parent config changed')
    parent=ROOT/('output-speed-resume3-'+PARENT_RUN)/'checkpoints/training_state/global_step_8/manifest.json'
    m=json.loads(parent.read_text())
    streams=[r['sampling_stream'] for r in m['worker_states']]
    run=uuid.uuid4().hex;target=ROOT/'receipts'/('speed-resume3-'+run);target.mkdir(exist_ok=False)
    out=ROOT/('output-speed-resume3-'+run)
    binding=dict(protocol=PROTOCOL,parent_completed_step=8,parent_optimizer_calls=160,
        parent_commit=PARENT_COMMIT,parent_run_id=PARENT_RUN,parent_config=str(old),
        parent_config_sha256=sha(old),parent_manifest=str(parent),parent_manifest_sha256=sha(parent),
        parent_identity=m['identity'],sampling_streams=streams,
        sampling_stream_status='COMPLETE_NATIVE_SAMPLING_STREAMS',child_run_id=run,
        child_checkpoint_dir=str(out/'checkpoints'),current_parent_terminal=str(a.terminal),
        current_parent_terminal_sha256=sha(a.terminal),
        latest_human_authority='Continue own complete native state after programming repair; do not replay completed updates.')
    bpath=target/'binding.json';bpath.write_text(json.dumps(binding,sort_keys=True))
    cfg=OmegaConf.load(old);cfg.trainer.default_local_dir=str(out/'checkpoints')
    cpath=target/'resolved.yaml';cpath.write_text(OmegaConf.to_yaml(cfg))
    os.environ.update(RECURSIVE_B_CONTINUATION_BINDING=str(bpath),RECURSIVE_B_CONTINUATION_SHA256=sha(bpath))
    from search_r1.baselines.recursive_b_schedule_resume import validate_entry,parent_manifest
    validate_entry(cfg);parent_manifest()
    record=dict(recovery_kind='complete8_native_rng',run_id=run,config=str(cpath),config_sha256=sha(cpath),
        binding=str(bpath),binding_sha256=sha(bpath),source=str(repo),source_commit=source,
        model=cfg.actor_rollout_ref.model.path,data_identity=m['identity']['data_identity'],output=str(out),
        parent_terminal=str(a.terminal),parent_terminal_sha256=sha(a.terminal),parent_completed_step=8,
        preparation_time=time.time(),ready=True,scientific_config_change='Only isolated artifact root; full parameters/Adam/scheduler/training and sampling RNG/cursor restored.')
    rpath=target/'record.json';rpath.write_text(json.dumps(record,sort_keys=True))
    print(json.dumps(dict(record_path=str(rpath),record_sha256=sha(rpath),record=record)))

if __name__=='__main__':main()
