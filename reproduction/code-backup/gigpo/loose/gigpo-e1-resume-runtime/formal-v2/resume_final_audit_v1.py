"""Strict cross-job final evidence for the explicitly authorized full75 continuation."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
OLD = ROOT/'runs/gigpo-e1-instruct300-job3185-1791112239036063948'
OLD_COMMIT = 'bb1adde13c7fd489aef4feb68a36aeb0d29d5da3'
CONTENT = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
OLD_RAW = OLD/'ray/session_2026-10-04_19-12-00_641361_3784093/logs/worker-cfb1ea721289fc22eacf497a3cf8bd280824ea553fed2567ebcf14da-02000000-3787949.out'
OLD_RAW_SHA = '37d648918890c4e7ec437dc632d4d03b30c9dcf5c13b6f8d4c716d36fc30041a'
OLD_MANIFESTS = {5:'3394d7fcde453e03e7ad13e084d32c739a8b4c292c54cab614baf7270c8acda2',
25:'3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',
50:'db39371eda1180bc77b0d2cde24984211b8887d5360dcb24ed73b78804a37159',
75:'65f5eef374d0eb5378aa50d56b9c883755e961b154526a26c05eefcd127968a9'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def validate_update_sequence(records, start, stop):
    require([v['outer_step'] for v in records] == list(range(start, stop+1)),
            'Missing/duplicate/extra actual outer update')


def audit_gate(run, steps, commit, config_hash):
    from search_r1.baselines.instruct300 import check_manifest
    require(steps == 300, 'Wrong final target')
    require(sha(run/'audit/entry-config.yaml') == config_hash, 'Actual config changed')
    entry = rows(run/'audit/entry.jsonl')
    require(len(entry)==1 and entry[0]['kind']=='config_boundary'
            and entry[0]['actual_config_sha256']==config_hash
            and entry[0]['actual_data_identity']==CONTENT
            and entry[0]['entry_total_training_steps']==300, 'Missing actual input boundary')
    driver=rows(run/'audit/driver.jsonl');validate_update_sequence(driver,76,300)
    require(len({v['pid'] for v in driver})==1, 'Driver restarted')
    for record in driver:
        require(record['commit']==commit and record['config_sha256']==config_hash
                and record['data_identity']==CONTENT, 'Driver identity drift')
        require(record['questions']==512 and record['rows']==2560 and record['method']=='gigpo', 'Sampling budget changed')
        for item in record['tensors'].values():require(item['finite']==item['count'], 'Nonfinite driver tensors')
    require(any(v['observation_tokens']>0 for v in driver), 'No real retrieval')
    restored=json.loads((run/'resume75-restored.json').read_text())
    require(restored['status']=='RESTORED_FULL_STATE75' and restored['source_manifest']['sha256']==OLD_MANIFESTS[75]
            and restored['driver_cursor']==38400 and restored['retained_logical_step']==75
            and restored['resumed_actual_updates']==0, 'Wrong actual full restore')
    restored_workers={v['rank']:v for v in restored['workers']}
    require(set(restored_workers)==set(range(8)), 'Incomplete restored workers')
    worker_pids=[]
    for rank in range(8):
        records=rows(run/'audit'/f'{rank}.jsonl');validate_update_sequence(records,76,300)
        require({v['pid'] for v in records}=={restored_workers[rank]['pid']}, 'Restored worker restarted')
        worker_pids.append(records[0]['pid'])
        for r in records:
            require(r['commit']==commit and r['config_sha256']==config_hash and r['data_identity']==CONTENT,'Actor identity drift')
            require(r['rank']==rank and r['epochs']==1 and r['optimizer_steps']==20 and len(r['micro_stats'])==160, 'Actor optimization budget changed')
            require(r['grad_finite']==20 and r['grad_nonzero']==20 and r['skipped_empty_minibatches']==0,'Bad/empty gradient')
            require(r['fixed_unchanged'] and r['shard_layout_unchanged'] and r['mask_binary'], 'Mask/group/layout changed')
            require(len(r['grad_norms'])==20 and all(v is not None and math.isfinite(v) and v>0 for v in r['grad_norms']), 'Bad gradient norm')
            require(r['shard_delta']['nonzero']>0, 'Restored outer parameters did not change')
            for v in list(r['tensors'].values())+[r['shard_delta']]:require(v['finite']==v['count'], 'Nonfinite actor tensors')
            for micro in r['micro_stats']:
                require('joint_action_ratios' in micro, 'Missing strict joint ratio evidence')
                for v in micro.values():require(v['finite']==v['count'], 'Nonfinite logp/loss/ratio')
        require(any(v['shard_delta']['nonzero']>0 for v in records), 'Unchanged trained shard')
    require(len(set(worker_pids))==8, 'Duplicate actor PID')
    expected_new=[80]+list(range(100,301,25))
    actor=run/'checkpoints/actor'
    require({p.name for p in actor.iterdir() if p.is_dir()}=={f'global_step_{s}' for s in expected_new}, 'Missing/extra resumed checkpoint')
    receipts=[]
    for point in list(OLD_MANIFESTS)+expected_new:
        old=point in OLD_MANIFESTS
        path=(OLD if old else run)/f'checkpoints/actor/global_step_{point}/training_state/manifest.json'
        digest=sha(path);m=json.loads(path.read_text())
        if old: require(digest==OLD_MANIFESTS[point], 'Original checkpoint manifest changed')
        require(m['logical_step']==point and m['actual_new_updates']==(point if old else point-75), 'Checkpoint logical/actual identity changed')
        require(m['job_id']==('3185' if old else os.environ['SLURM_JOB_ID']) and m['code_commit']==(OLD_COMMIT if old else commit), 'Checkpoint job/source mismatch')
        require(m['recovery_mode']==('fresh_official_instruct' if old else 'full_state_instruct75')
                and m['data_identity']==CONTENT and m['source']==str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'), 'Checkpoint origin changed')
        if not old: require(m['config_sha256']==config_hash, 'Checkpoint config changed')
        check_manifest(dict(path=str(path),sha256=digest,identity=m))
        algorithm_path=next(v['path'] for v in m['files'] if '/algorithm/' in v['path'])
        algorithm=json.loads(Path(algorithm_path).read_text())
        require(algorithm['method']=='gigpo' and algorithm['global_step']==point
                and algorithm['resume_supported'] is True and algorithm['loss_reduction']=='sequence_mean', 'GiGPO saved state changed')
        receipts.append(dict(logical_step=point,path=str(path),sha256=digest,job_id=m['job_id']))
    final=json.loads((run/'final300-state-roundtrip.json').read_text())
    require(final['logical_step']==300 and final['actual_new_updates']==225
            and final['retained_original_updates']==75 and final['driver_cursor']==153600,
            'Missing actual final full-state roundtrip')
    require(final['manifest']['sha256']==receipts[-1]['sha256'], 'Reloaded wrong final state')
    require(len(final['actual_state_roundtrip'])==8 and {v['rank'] for v in final['actual_state_roundtrip']}==set(range(8))
            and all(v['logical_step']==300 for v in final['actual_state_roundtrip']), 'Incomplete final state reload')
    pause = final['checkpoint_pause_identity']
    require(len(pause)==8 and {v['rank'] for v in pause}==set(range(8)), 'Incomplete final pause identity')
    require(final['manifest']['identity']==m and final['manifest']['path']==str(path), 'Final reload receipt differs from physical300 manifest')
    saved_pause = [v['pause_identity'] for v in m['files'] if 'pause_identity' in v]
    require(pause==saved_pause, 'Final reloaded tensors/Adam/scheduler/RNG do not match saved300')
    require({v['rank']:(v['pid'],v['start_ticks']) for v in pause}=={k:(v['pid'],v['start_ticks']) for k,v in restored_workers.items()}, 'Final worker continuity changed')
    (run/'final-checkpoint-physical-audit.json').write_text(json.dumps(receipts,indent=2))
    return actor/'global_step_300'


def raw_driver_snapshot(run):
    driver=rows(run/'audit/driver.jsonl');pids={v['pid'] for v in driver}
    require(len(pids)==1, 'Driver PID changed')
    pid=next(iter(pids));found=[]
    for session in (run/'ray').iterdir():
        if session.is_dir() and not session.is_symlink():
            found.extend((session/'logs').glob(f'worker-*-{pid}.out'))
    require(len(found)==1, 'Missing/ambiguous actual raw driver log')
    path=found[0];stat=path.stat();raw=path.read_text()
    receipt=dict(path=str(path),sha256=sha(path),bytes=stat.st_size,inode=stat.st_ino,device=stat.st_dev,
                 forwarded_path=str(run/'training.log'),forwarded_sha256=sha(run/'training.log'))
    (run/'raw-log-terminal-identity.json').write_text(json.dumps(receipt,indent=2))
    return raw,receipt


def audit_new_evaluations(records, job, commit, config_hash, driver_pid):
    expected=[('stage64',80)]+[('periodic512',s) for s in range(100,301,50)]+[('final51713',300)]
    require([(r['phase'],r['logical_step']) for r in records]==expected, 'Missing/duplicate/wrong-phase evaluations')
    for r in records:
        require(r['job_id']==job and r['commit']==commit and r['config_sha256']==config_hash
                and r['pid']==driver_pid and r['data_identity']==CONTENT
                and r['actual_new_updates']==r['logical_step']-75, 'Evaluation actual provenance changed')
        phase=r['phase']; name={'stage64':'stage','periodic512':'validation','final51713':'final'}[phase]
        expected_file=ROOT/f'common-data/{name}.parquet'
        require(r['dataset_files']==[dict(path=str(expected_file),sha256=sha(expected_file))], 'Evaluated another dataset')
        count={'stage64':64,'periodic512':512,'final51713':51713}[phase]
        require(r['metrics']['val/qa_count/all']==count, 'Wrong evaluation size')
        if count in (64,512):
            require(r['metrics']['val/qa_count/nq']==count//2 and r['metrics']['val/qa_count/hotpotqa']==count//2,'Wrong NQ/Hotpot evaluation balance')
        else:
            require(sum(v for k,v in r['metrics'].items() if k.startswith('val/qa_count/') and k!='val/qa_count/all')==51713
                    and len([k for k in r['metrics'] if k.startswith('val/qa_count/') and k!='val/qa_count/all'])==7, 'Incomplete seven final sets')


def audit_evaluations(run, commit, config_hash, raw):
    text,identity=raw
    complete=[json.loads(line.split('SEARCH_R1_INSTRUCT300_COMPLETE_JSON=',1)[1]) for line in text.splitlines() if 'SEARCH_R1_INSTRUCT300_COMPLETE_JSON=' in line]
    require(len(complete)==1 and complete[0]['logical_step']==300 and complete[0]['actual_new_updates']==225
            and complete[0]['retained_original_updates']==75 and complete[0]['retained_total_updates']==300, 'Missing real final completion marker')
    records=rows(run/'audit/evaluations.jsonl')
    driver_pid=rows(run/'audit/driver.jsonl')[0]['pid']
    audit_new_evaluations(records,os.environ['SLURM_JOB_ID'],commit,config_hash,driver_pid)
    raw_metrics=[json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for line in text.splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in line]
    require(raw_metrics==[r['metrics'] for r in records], 'Raw evaluations differ from explicit phase receipts')
    require(complete[0]['final_metrics']==records[-1]['metrics'], 'Completion final metrics differ')
    require(sha(OLD_RAW)==OLD_RAW_SHA, 'Original raw log changed')
    old_metrics=[json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for line in OLD_RAW.read_text().splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in line]
    require([r['val/qa_count/all'] for r in old_metrics]==[512,64,512], 'Original accepted evaluation sequence differs')
    old_config_hash=json.loads((OLD/'checkpoints/actor/global_step_75/training_state/manifest.json').read_text())['config_sha256']
    require(sha(OLD/'audit/entry-config.yaml')==old_config_hash, 'Original actual config changed')
    old_roles=[('initial512',0),('stage64',5),('periodic512',50)]
    canonical=[dict(phase=role,logical_step=step,job_id='3185',commit=OLD_COMMIT,
        data_identity=CONTENT,config_sha256=old_config_hash,
        dataset_files=[dict(path=str(ROOT/('common-data/stage.parquet' if role=='stage64' else 'common-data/validation.parquet')),sha256=sha(ROOT/('common-data/stage.parquet' if role=='stage64' else 'common-data/validation.parquet')))],
        raw_path=str(OLD_RAW),raw_sha256=OLD_RAW_SHA,metrics=metrics)
        for (role,step),metrics in zip(old_roles,old_metrics)]
    canonical += [dict(r,raw_path=identity['path'],raw_sha256=identity['sha256']) for r in records]
    (run/'canonical-cross-job-evaluations.json').write_text(json.dumps(canonical,indent=2))
