"""Audit actual production verify-only witnesses. Never signs a training permit."""
import argparse,json
from pathlib import Path
from search_r1.baselines.recovery import read,require,validate_native
from search_r1.baselines.checkpoint_gate import digest

def verify(manifest,directory):
 m=read(manifest);d=read(Path(directory)/'driver.json');ranks=[read(Path(directory)/f'rank_{i}.json')for i in range(8)]
 require(d.get('status')=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and d.get('verify_only')is True and d.get('step')==25 and d.get('next_step')==26 and d.get('actual_outer_updates')==25 and d.get('sampler_cursor')==12800 and d.get('epoch')==0,'Actual verify-only driver incomplete')
 require(d.get('new_pid')!=d.get('old_pid')and d.get('bapo')==read(m['algorithm']['path'])['bapo']and d.get('loaded_state_digest')==read(m['driver_report']['path'])['state_digest'],'Driver content/lineage mismatch')
 for i,w in enumerate(ranks):
  entry=next(x for x in m['ranks']if x['rank']==i);old=read(entry['report']['path'])
  require(w.get('status')=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'and type(w.get('rank'))is int and w['rank']==i and w.get('world_size')==8 and w.get('step')==25 and w.get('next_step')==26,'Actual rank restoration incomplete')
  require(w.get('old_pid')==old['pid']and w.get('new_pid')!=old['pid']and w.get('state_digest')==old['state_digest']and w.get('old_state_sha256')==entry['state']['sha256']and w.get('scheduler_last_epoch')==25 and w.get('optimizer_state_entries')==old['optimizer_state_entries'],'Rank content/optimizer/scheduler mismatch')
  require(w.get('actual_env',{}).get('SLURM_JOB_ID')==str(m['target_job_id'])and w['actual_env'].get('BAPO_RUN_ID')==m['target_run_root'],'Actual worker job/run mismatch')
 for w in ranks+[d]:
  require(w.get('target_source_commit')==m['target_source_commit']and w.get('target_run_root')==m['target_run_root']and w.get('sampling_continuity')=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT','Witness source/run/sampling scope mismatch')
 require(len({w['new_pid']for w in ranks+[d]})==9,'Fresh nine process identity required')
 require(d['rank_witnesses']==ranks,'Driver does not bind eight actual rank witnesses')
 return dict(status='GPU_WITNESSES_REQUIRE_INDEPENDENT_REVIEW',manifest_sha256=digest(manifest),source=m['target_source_commit'],job_id=m['target_job_id'],run=m['target_run_root'],files=[dict(path=str(p),sha256=digest(p))for p in sorted(Path(directory).glob('*.json'))])
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--directory',required=True);a=p.parse_args();print(json.dumps(verify(a.manifest,a.directory),allow_nan=False))
if __name__=='__main__':main()
