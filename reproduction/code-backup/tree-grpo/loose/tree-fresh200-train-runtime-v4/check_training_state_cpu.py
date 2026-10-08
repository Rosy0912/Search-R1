"""Future same-node CPU full saved-state verification; never restores training.

Only its source is prepared here. Current local tests do not load any PT file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from production_contract import read_bound,require


def held_state(path, expected):
    import torch
    with Path(path).open('rb') as stream:
        before=os.fstat(stream.fileno()); digest=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
        require(digest.hexdigest()==expected,'Saved state SHA mismatch')
        stream.seek(0); state=torch.load(stream,map_location='cpu',weights_only=False)
        after=os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Saved state changed while deserializing')
    return state


def verify(manifest_path, expected_sha, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Complete state verification must be CPU-only')
    import torch
    from search_r1.baselines.training_state import digest,finite
    torch.set_num_threads(8); start=time.perf_counter(); cpu=time.process_time()
    manifest,observed,_=read_bound(manifest_path,expected_sha); identity=manifest['identity']; outer=identity['completed_step']
    accounting=identity['guard_continue_accounting']
    from attempt_accounting import sampler_position
    require(accounting['completed_outer_updates']==outer and accounting['protocol']=='tree_guard_continue_fresh_rollout_v1','Saved actual attempt accounting missing')
    epoch,cursor=sampler_position(accounting['training_attempts'])
    require(outer in (5,50,100,150,200) and identity['actual_update_limit']==200,'Wrong fresh saved-state boundary')
    records=sorted(manifest['worker_states'],key=lambda x:x['identity']['rank'])
    require(len(records)==8 and [x['identity']['rank'] for x in records]==list(range(8)), 'Full saved worker coverage incomplete')
    results=[]
    for rank,record in enumerate(records):
        state=held_state(record['path'],record['sha256']); finite(state)
        require(digest(state)==record['logical_state_sha256'] and state['version']==2
                and state['identity']==record['identity']
                and record['identity']==dict(identity,rank=rank,world_size=8,cuda_visible_devices=record['identity']['cuda_visible_devices']),'Complete worker logical/source identity mismatch')
        steps=[int(value['step'].item()) for value in state['optimizer']['state'].values()]
        actual_calls=accounting['total_optimizer_calls_per_rank'][str(rank)]
        require(steps and set(steps)=={actual_calls} and state['scheduler']['last_epoch']==outer,'Saved actual cumulative Adam calls or credit-only scheduler differs')
        require(state['parameters'] and set(state['gradients'])==set(state['parameters']),'Saved parameter/gradient coverage incomplete')
        results.append(dict(rank=rank,world_size=8,adam_step_values=sorted(set(steps)),scheduler_step=outer,all_saved_state_finite=True,
                            saved_state_sha256=record['sha256'],logical_sha256=record['logical_state_sha256']))
        del state
    driver=held_state(manifest['driver']['path'],manifest['driver']['sha256']); finite(driver)
    require(digest(driver)==manifest['driver']['logical_state_sha256'] and driver['identity']==identity
            and driver['completed_step']==driver['actual_outer_updates']==outer and driver['next_step']==outer+1
            and driver['recovery_provenance'] is None,'Actual fresh saved driver boundary differs')
    sampler=driver['sampler']
    require(driver['guard_continue_accounting']==accounting,'Driver saved attempts differ from saved worker/accounting identity')
    require(sampler['rows']==169028 and sampler['batch_size']==512 and sampler['cursor']==cursor and sampler['epoch']==epoch
            and sampler['identity']==identity['data_identity'] and sampler['shuffle'] is True
            and len(sampler['order'])==169028 and set(sampler['order'])==set(range(169028)),'Actual fresh sampler permutation/cursor differs')
    require(driver['algorithm']['method']=='tree_grpo' and driver['algorithm']['tree_rng'] is not None,'Actual Tree driver RNG/state missing')
    report=dict(status='FRESH200_COMPLETE_SAVED_STATE_CPU_VERIFIED',identity=identity,manifest_sha256=observed,
                workers=results,completed_outer=outer,driver_actual_outer=driver['actual_outer_updates'],sampler_cursor=sampler['cursor'],
                sampler_epoch=sampler['epoch'],training_attempts=accounting['training_attempts'],actual_optimizer_calls_per_rank=accounting['total_optimizer_calls_per_rank'],
                wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
                actual_pt_deserialization=True,all_saved_state_finite=True,live_reload_verified_by_this_checker=False,
                resume_permitted=False,automatic_rollback=False,cuda_initialized=torch.cuda.is_initialized())
    require(report['cuda_initialized'] is False,'CPU state checker initialized CUDA')
    with Path(output).open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); print(json.dumps(verify(args.manifest,args.manifest_sha,args.output),sort_keys=True))
