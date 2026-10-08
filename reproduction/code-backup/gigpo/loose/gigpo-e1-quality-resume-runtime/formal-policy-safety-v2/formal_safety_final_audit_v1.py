"""Versioned selected25 production draft; original trace/state/evaluation checks preserved."""
import gzip,json,math,os
from collections import Counter
from pathlib import Path
from formal_protection_contract_v1 import require,sha,validate_actual_native
from formal_audit_utils_v1 import rows,validate_update_sequence,raw_driver_snapshot
from formal_safety_streams_v1 import bounded_micro,audit_production_streams
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CONTENT='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'

def trace_receipt(folder,expected_count,expected_status,optimizer_steps=None):
    path=folder/'complete.json';require(path.exists(),'Missing actual trace completion')
    receipt=json.loads(path.read_text())
    require(receipt['status']==expected_status and receipt['expected_rows']==expected_count
            and receipt['actual_optimizer_steps']==optimizer_steps,'Trace status/count/Adam mismatch')
    physical={str(p) for p in folder.glob('*.jsonl.gz')}
    files=receipt['files'];require({f['path'] for f in files}==physical and len(physical)==len(files),'Trace file coverage differs')
    require(receipt['turn_count']==len(list(folder.glob('turn_*.jsonl.gz'))) and receipt['turn_count']>0,'Missing actual search turns')
    require(not list(folder.glob('*.partial')),'Incomplete trace evidence')
    require({'generated.jsonl.gz','scored.jsonl.gz'} <= {Path(f['path']).name for f in files},'Missing generation/scoring')
    identities={};generated_lengths={};generated_masks={};scored_lengths={};scored_masks={}
    require(folder.name==f"iter_{receipt['logical_step']:06d}",'Trace receipt differs from physical step folder')
    turn_names={path.name for path in folder.glob('turn_*.jsonl.gz')}
    require(turn_names=={f'turn_{i:02d}.jsonl.gz' for i in range(1,receipt['turn_count']+1)},'Missing/repeated turn index')
    n=5 if expected_status=='ACTUAL_OUTER_UPDATE_COMPLETE' else 1
    for f in files:
        path=Path(f['path'])
        require(f['rows']==expected_count and path.stat().st_size==f['bytes']
                and sha(path)==f['sha256'],'Trajectory bytes/count/identity changed')
        actual={}
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);i=row['trajectory_row']
                require(type(i) is int and 0<=i<expected_count and i not in actual,'Duplicate/invalid actual trajectory ID')
                require(row['response_group']==i//n and row['response_index']==i%n
                    and row['logical_step']==receipt['logical_step'],'Actual trajectory group/step changed')
                if path.name=='generated.jsonl.gz':
                    generated_lengths[i]=len(row['responses'])
                    generated_masks[i]=row['info_mask'][-len(row['responses']):] if row['responses'] else []
                    width=len(row['prompts'])+len(row['responses'])
                    require(len(row['attention_mask'])==width and len(row['info_mask'])==width
                        and all(v in (0,1) for v in row['attention_mask']+row['info_mask']),'Generated mask/token width differs')
                if path.name=='scored.jsonl.gz' and n==5:
                    width=len(row['loss_mask']);scored_lengths[i]=width;scored_masks[i]=row['loss_mask']
                    require(all(len(row[k])==width for k in ('step_ids','policy_weights','advantages','token_level_rewards','returns','old_log_probs','ref_log_prob'))
                        and all(v in (0,1) for v in row['loss_mask']),'Scored effective action/token alignment differs')
                if path.name.startswith('turn_'):
                    require(row['turn']==int(path.name[5:7]),'Actual turn content index differs')
                    require(all(k in row and type(row[k]) is bool for k in ('active','done','valid_action','is_search'))
                        and isinstance(row.get('model_action'),str) and isinstance(row.get('raw_tool_observation_before_token_limit'),str),'Missing actual action/tool observation/turn flags')
                actual[i]=row['question_id']
        require(set(actual)==set(range(expected_count)),'Actual decompressed trace rows incomplete')
        identities[path.name]=actual
    if n==5:
        require(scored_lengths==generated_lengths and scored_masks==generated_masks,'Generated/scored action mask/token widths differ')
    generated=identities['generated.jsonl.gz']
    require(all(values==generated for values in identities.values()),'Generation/scoring/tool identity pairing differs')
    for offset in range(0,expected_count,n):
        require(len({generated[i] for i in range(offset,offset+n)})==1,'Question response group differs')
    require(len({generated[i] for i in range(0,expected_count,n)})==expected_count//n,'Repeated/missing actual question groups')
    return receipt

def actual_restored_workers(restored,original):
    values=restored['workers'];require(len(values)==8 and {v['rank'] for v in values}==set(range(8)),'Duplicate/missing restored rank')
    saved=[f['pause_identity'] for f in original['files'] if 'pause_identity' in f]
    require(len(saved)==8 and {v['rank'] for v in saved}==set(range(8)),'Original pause coverage differs')
    expected={v['rank']:v for v in saved};workers={v['rank']:v for v in values}
    for rank,value in workers.items():
        require({k:v for k,v in value.items() if k not in ('pid','start_ticks')}==
            {k:v for k,v in expected[rank].items() if k not in ('pid','start_ticks')},'Actual restored full tensor/Adam/scheduler/RNG identity differs')
    return workers

def mandatory_evaluation_phases(records,origin_step):
    periodic=[r['logical_step'] for r in records if r['phase']=='periodic512']
    stages=[r['logical_step'] for r in records if r['phase']=='stage64']
    require(periodic==list(range(((origin_step//50)+1)*50,301,50)),'Missing/duplicate periodic512')
    require(stages==[origin_step]+list(range(origin_step+5,301,5)),'Missing/duplicate five-outer64')
    quality=[r['logical_step'] for r in records if r['phase']=='quality512']
    require(len(quality)==len(set(quality)) and all(s in stages[1:] for s in quality),'Invalid/duplicate quality confirmation')
    require(all(origin_step<=r['logical_step']<=300 for r in records),'Evaluation outside actual retained branch')
    serials=[r['validation_trace_serial'] for r in records]
    require(len(serials)==len(set(serials)),'Reused validation trace serial')

def audit_gate(run,steps,commit,config_hash,origin,policy):
    from search_r1.baselines.instruct300 import check_manifest
    require(steps==300,'Wrong target')
    require(sha(run/'audit/entry-config.yaml')==config_hash,'Actual config changed')
    entry=rows(run/'audit/entry.jsonl');require(len(entry)==1 and entry[0]['kind']=='config_boundary'
        and entry[0]['actual_config_sha256']==config_hash and entry[0]['actual_data_identity']==CONTENT
        and entry[0]['entry_total_training_steps']==300,'Missing actual entry identity')
    start=origin['step']+1;driver=rows(run/'audit/driver.jsonl');validate_update_sequence(driver,start,300)
    require(len({r['pid'] for r in driver})==1,'Driver restarted')
    for r in driver:
        require(r['commit']==commit and r['config_sha256']==config_hash and r['data_identity']==CONTENT,'Driver identity drift')
        require(r['questions']==512 and r['rows']==2560 and r['method']=='gigpo','Sampling budget changed')
        require(all(v['finite']==v['count'] for v in r['tensors'].values()),'Nonfinite driver tensors')
        trace_receipt(run/'trajectories'/f"iter_{r['outer_step']:06d}",2560,'ACTUAL_OUTER_UPDATE_COMPLETE',10)
    require(any(r['observation_tokens']>0 for r in driver),'No actual local retrieval')
    restored=json.loads((run/f"resume{origin['step']}-restored.json").read_text())
    require(restored['status']==f"RESTORED_FULL_STATE{origin['step']}" and restored['source_manifest']['sha256']==origin['manifest']
        and restored['driver_cursor']==origin['step']*512 and restored['retained_logical_step']==origin['step']
        and restored['resumed_actual_updates']==0,'Wrong complete-state restore')
    original_path=Path(origin['checkpoint'])/'training_state/manifest.json'
    original=json.loads(original_path.read_text())
    check_manifest(dict(path=str(original_path),sha256=origin['manifest'],identity=original))
    require(restored['source_manifest']==dict(path=str(original_path),sha256=origin['manifest'],identity=original),'Embedded restore manifest differs from physical source')
    workers=actual_restored_workers(restored,original)
    for rank in range(8):
        records=rows(run/'audit'/f'{rank}.jsonl');validate_update_sequence(records,start,300)
        require({r['pid'] for r in records}=={workers[rank]['pid']},'Rank process restarted')
        for r in records:
            require(r['commit']==commit and r['config_sha256']==config_hash and r['data_identity']==CONTENT,'Rank identity drift')
            require(r['rank']==rank and r['rows']==320 and r['epochs']==1 and r['optimizer_steps']==10
                and len(r['micro_stats'])==160 and r['skipped_empty_minibatches']==0,'Wrong actual ten-Adam/160micro budget')
            require(r['grad_finite']==10 and r['grad_nonzero']==10 and len(r['grad_norms'])==10
                and all(math.isfinite(v) and v>0 for v in r['grad_norms']),'Invalid actual gradients')
            require(r['fixed_unchanged'] and r['shard_layout_unchanged'] and r['mask_binary']
                and r['shard_delta']['nonzero']>0,'Mask/group or actual weight change failed')
            require(r['row_denominators']==[dict(mini_index=i,local_rows=32,actual_denominator=32.,world_size=8) for i in range(10)],'Production sequence_mean denominator differs')
            for v in list(r['tensors'].values())+[r['shard_delta']]:require(v['finite']==v['count'],'Nonfinite rank tensors')
            for micro in r['micro_stats']:bounded_micro(micro)
    require(len({v['pid'] for v in workers.values()})==8,'Duplicate workers')
    audit_production_streams(run,start,300,commit,config_hash,workers,policy)
    expected=sorted({origin['step']+5}|set(range(((origin['step']//25)+1)*25,301,25)))
    actor=run/'checkpoints/actor';require({p.name for p in actor.iterdir() if p.is_dir()}=={f'global_step_{s}' for s in expected},'Missing/extra checkpoint')
    receipts=[]
    original_path=Path(origin['checkpoint'])/'training_state/manifest.json'
    original=json.loads(original_path.read_text());check_manifest(dict(path=str(original_path),sha256=origin['manifest'],identity=original))
    for step in expected:
        path=actor/f'global_step_{step}/training_state/manifest.json';m=json.loads(path.read_text());digest=sha(path)
        require(m['logical_step']==step and m['actual_new_updates']==step-origin['step'] and m['job_id']==os.environ['SLURM_JOB_ID']
            and m['code_commit']==commit and m['config_sha256']==config_hash and m['data_identity']==CONTENT
            and m['source']==str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725')
            and m['recovery_mode']==f"full_state_instruct{origin['step']}",'Checkpoint provenance drift')
        check_manifest(dict(path=str(path),sha256=digest,identity=m));receipts.append(dict(step=step,path=str(path),sha256=digest))
        algo=json.loads(Path(next(v['path'] for v in m['files'] if '/algorithm/' in v['path'])).read_text())
        require(algo['method']=='gigpo' and algo['global_step']==step and algo['resume_supported']
            and algo['loss_reduction']=='sequence_mean','Wrong GiGPO saved state')
    final=json.loads((run/'final300-state-roundtrip.json').read_text())
    require(final['logical_step']==300 and final['actual_new_updates']==300-origin['step']
        and final['retained_original_updates']==origin['step'] and final['driver_cursor']==153600,'Wrong actual final roundtrip')
    require(final['manifest']['sha256']==receipts[-1]['sha256'] and final['manifest']['identity']==m and final['manifest']['path']==str(path),'Wrong reloaded final checkpoint')
    loaded=final['actual_state_roundtrip'];require(len(loaded)==8 and {r['rank'] for r in loaded}==set(range(8))
        and all(r['logical_step']==300 for r in loaded),'Incomplete final eight-rank reload')
    pause=final['checkpoint_pause_identity'];require(pause==[v['pause_identity'] for v in m['files'] if 'pause_identity' in v],'Final tensors/Adam/RNG changed')
    require({v['rank']:(v['pid'],v['start_ticks']) for v in pause}=={k:(v['pid'],v['start_ticks']) for k,v in workers.items()},'Final PID continuity changed')
    (run/'final-checkpoint-physical-audit.json').write_text(json.dumps(receipts,indent=2))
    return actor/'global_step_300'

def audit_evaluations(run,commit,config_hash,raw,origin):
    manifest_path=ROOT/'common-data/manifest.json';require(sha(manifest_path)==CONTENT,'Common manifest changed')
    common=json.loads(manifest_path.read_text())
    final_counts=common['splits']['final']['source_counts']
    text,identity=raw;records=rows(run/'audit/evaluations.jsonl');require(bool(records),'No actual evaluations')
    driver_pid=rows(run/'audit/driver.jsonl')[0]['pid'];full512=set();stage64=set();finals=[];restored=[]
    for r in records:
        require(r['job_id']==os.environ['SLURM_JOB_ID'] and r['commit']==commit and r['config_sha256']==config_hash
            and r['pid']==driver_pid and r['data_identity']==CONTENT
            and r['actual_new_updates']==r['logical_step']-origin['step'],'Wrong evaluation identity')
        phase=r['phase'];name={'stage64':'stage','restored512':'validation','quality512':'validation','periodic512':'validation','final51713':'final'}[phase]
        p=ROOT/f'common-data/{name}.parquet';require(r['dataset_files']==[dict(path=str(p),sha256=sha(p))],'Another dataset was evaluated')
        count=64 if phase=='stage64' else 51713 if phase=='final51713' else 512
        require(all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k in ('val/f1/all','val/em/all')),'Invalid actual evaluation quality')
        require(r['metrics']['val/qa_count/all']==count,'Wrong actual evaluation count')
        if count in (64,512):require(r['metrics']['val/qa_count/nq']==count//2 and r['metrics']['val/qa_count/hotpotqa']==count//2,'Wrong NQ/Hotpot balance')
        else:
            sources={k:v for k,v in r['metrics'].items() if k.startswith('val/qa_count/') and k!='val/qa_count/all'}
            require(sources=={'val/qa_count/'+k:v for k,v in final_counts.items()} and len(sources)==7 and sum(sources.values())==51713,'Final exact seven source counts differ');finals.append(r)
        if phase=='periodic512':full512.add(r['logical_step'])
        if phase=='stage64':stage64.add(r['logical_step'])
        if phase=='restored512':restored.append(r)
        folder=run/'validation_trajectories'/f"eval_{r['validation_trace_serial']:04d}";batches=sorted(folder.glob('batch_*'))
        total=0;question_ids=[]
        require([b.name for b in batches]==[f'batch_{i:04d}' for i in range(math.ceil(count/32))],'Invalid validation batch directory coverage')
        for b in batches:
            receipt=json.loads(next(b.glob('iter_*/complete.json')).read_text());n=receipt['expected_rows']
            require(n==min(32,count-total) and receipt['logical_step']==r['logical_step'],'Invalid evaluation batch size/step')
            require(len(list(b.glob('iter_*')))==1,'Extra validation step directory')
            folder_step=next(b.glob('iter_*'));trace_receipt(folder_step,n,'VALIDATION_BATCH_COMPLETE');total+=n
            with gzip.open(folder_step/'generated.jsonl.gz','rt') as stream:
                question_ids.extend(json.loads(line)['question_id'] for line in stream)
        require(total==count and len(set(question_ids))==count,'Actual validation trajectory question coverage differs')
    require(len(restored)==1 and restored[0]['logical_step']==origin['step'],'Missing actual restored512 baseline')
    mandatory_evaluation_phases(records,origin['step'])
    require(len(finals)==1 and records[-1]==finals[0] and finals[0]['logical_step']==300,'Final51713 missing/duplicated')
    printed=[json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]) for line in text.splitlines() if 'BASELINE_EVALUATION_METRICS_JSON=' in line]
    require(printed==[r['metrics'] for r in records],'Actual raw evaluations and receipts differ')
    complete=[json.loads(line.split('SEARCH_R1_INSTRUCT300_COMPLETE_JSON=',1)[1]) for line in text.splitlines() if 'SEARCH_R1_INSTRUCT300_COMPLETE_JSON=' in line]
    require(len(complete)==1 and complete[0]['logical_step']==300 and complete[0]['actual_new_updates']==300-origin['step']
        and complete[0]['retained_original_updates']==origin['step'] and complete[0]['retained_total_updates']==300
        and complete[0]['final_metrics']==finals[0]['metrics'],'Wrong actual final completion marker')
    (run/'canonical-quality-branch-evaluations.json').write_text(json.dumps(records,indent=2))
