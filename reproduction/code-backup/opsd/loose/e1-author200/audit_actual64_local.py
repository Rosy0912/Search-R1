"""Local immutable actual64 progress replay; does not rerun fixed-set50 evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import types

HERE = Path(__file__).resolve().parent
path = HERE / 'audit_actual49_local.py'
source = path.read_bytes()
if hashlib.sha256(source).hexdigest() != 'bb99edbd4f7a772c803f9d2848b9c39498a8c8daf1d33a94c94259d8a608207d':
    raise ValueError('Exact local stable-read helper changed')
base = types.ModuleType('actual64_local_helpers');base.__file__ = str(path)
exec(compile(source,str(path),'exec',dont_inherit=True),base.__dict__)


def inspect():
    common = base.module(base.COMMON,'29592450920eb6bc4f556138f4d0a73757bf3fc05703f7860c3b2f69c02c0295')
    gate = base.module(base.GATE,'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb')
    snapshot = HERE / 'actual-snapshots/3261-20261005T232501896059Z';orig = snapshot / 'original-files'
    raw, summary_sha = base.raw(snapshot / 'observation-summary.json');summary = common.loads(raw)
    _, transport_sha = base.raw(snapshot / 'observed-remote-original.json',summary['transport_original_sha256'])
    base.need(summary['job_id'] == common.JOB and summary['node'] == 'h100-10'
        and summary['expected_source'] == common.SOURCE and summary['run_root'] == str(common.RUN),'Observed run')
    originals = {}
    for name, record in summary['files'].items():
        local = orig / name;base.need(local.is_relative_to(orig),'Local role escape')
        value,digest = base.raw(local,record['sha256']);base.need(len(value)==record['bytes'],'Original length')
        originals[name] = (value,digest)
    base.need(len(originals)==315,'Delivered315 originals required')
    get = lambda name: common.loads(originals[name][0])
    config = get('full_state/global_step_5/resolved-config.json')
    base.need(originals['full_state/global_step_5/resolved-config.json'][1]==common.CONFIG_SHA
        and config['baselines']['method']=='opsd' and config['baselines']['opsd_e1']['profile']=='author200'
        and config['trainer']['total_training_steps']==200,'Pure author200 frozen config')
    receipt = get('acceptance-5-receipt.json')
    base.need(originals['acceptance-5-receipt.json'][1]==common.RECEIPT5_SHA
        and originals['full_state/global_step_5/manifest.json'][1]==common.MANIFEST5_SHA,'Accepted5 immutable pins')
    gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],get('bindings/public.json'))
    base.need(receipt['payload']['status']=='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
        and receipt['payload']['source_commit']==common.SOURCE and receipt['payload']['config_sha256']==common.CONFIG_SHA
        and receipt['payload']['driver_pid']==common.DRIVER_PID and receipt['payload']['global_step']==5
        and receipt['payload']['next_global_step']==6,'Actual first5 release scope')
    old_raw, identity_sha = base.raw(HERE / 'SOURCE_ACTUAL3261_FIRST5_LOG_SAVED_REPORT_JOIN.json',
        '2d5a4dad8067f463c0d1884de59eac2113e021f3fbd3870f9fa46f914b7213e1');old = common.loads(old_raw)
    workers = []
    for record in old['all_rank_saved_report_proof']:
        rank=record['rank'];pid=record['pid'];process=summary['processes'][str(pid)]
        saved=get(f'checkpoints/actor/global_step_50/training_state/rank_{rank}.json')
        base.need(saved['pid']==pid and saved['object_ids']==record['saved_object_ids'],'Live object continuity metadata')
        base.need(process['pid']==pid and process['start_ticks']==record['start_ticks'] and process['uids']==[0]*4
            and process['in_observed_job'] is True and '/job_3261/step_0' in process['cgroup'],'Actual worker identity')
        env=process['environment_allowlist']
        base.need(env['SEARCH_R1_REPO']==str(common.REPO) and env['SEARCH_R1_BASELINE_RUN_ROOT']==str(common.RUN)
            and env['OPSD_E1_RETRIEVER_URL']=='http://127.0.0.1:18084/retrieve','Same-node source/tool worker')
        workers.append(dict(rank=rank,pid=pid,start_ticks=process['start_ticks'],object_ids=saved['object_ids']))
    driver_identity=summary['processes'][str(common.DRIVER_PID)]
    base.need(driver_identity['start_ticks']==str(common.DRIVER_START) and driver_identity['uids']==[0]*4
        and driver_identity['in_observed_job'] is True and '/job_3261/step_0' in driver_identity['cgroup'],'Driver actual identity')
    commands=summary['commands'];source_record=get('bindings/source.json')
    base.need(source_record['source_commit']==common.SOURCE and source_record['clean'] is True
        and commands['source_head']['returncode']==commands['source_porcelain']['returncode']==0
        and commands['source_head']['stdout'].strip()==common.SOURCE and commands['source_porcelain']['stdout']=='','Clean observed source')
    closure=get('bindings/import_closure.json')
    context=dict(gate=gate,workers=workers,checkpoint_warning_path=Path(closure['modules']['torch.utils._contextlib']['path']).parent/'checkpoint.py')
    streams={};boundaries={}
    for name in ['ray-stdout-pid-1533312.out','train.log']:
        record=summary['files'][name];value=originals[name][0];end=value.rfind(b'\n')+1
        base.need(record['complete_original_at_start'] is True and record['source_offset']==0
            and record['unchanged_during_read'] is True and end>0,'Stable complete captured stream')
        streams[name]=common.parse_records(value[:end],context)
        boundaries[name]=dict(captured_sha256=originals[name][1],captured_bytes=len(value),
            completed_prefix_bytes=end,prefix_sha256=hashlib.sha256(value[:end]).hexdigest(),
            unterminated_tail_bytes=len(value)-end,unterminated_tail_not_used=True)
    driver=streams['ray-stdout-pid-1533312.out'];aggregate=streams['train.log']
    join=common.join_log_prefixes(driver,aggregate,context,5)
    base.need(join['authoritative_last_outer']==64 and join['aggregate_last_outer']==63,'Actual delivered progress')
    rows=driver['outers'];per_rank=[]
    for rank in range(8):
        suffix='/rank_'+str(rank);calls=[row['metrics']['actor/optimizer_steps'+suffix] for row in rows]
        samples=[row['metrics']['actor/samples'+suffix] for row in rows]
        base.need(calls==[16]*64 and samples==[64]*64,'Actual perrank optimizer/sample budget')
        base.need(all(row['metrics']['teacher/update_device_is_cuda_per_rank'+suffix]==1 for row in rows),'Actual teacherCUDA path')
        per_rank.append(dict(rank=rank,actual_completed_outers=64,optimizer_calls=1024,samples_per_outer=64,
            loss_gradient_finite=True,teacher_cuda_clock=rows[-1]['metrics']['teacher/cuda_update_entries_cumulative_per_rank'+suffix]))
    base.need(all(row['metrics']['rollout/trajectories']==512 for row in rows),'Actual trajectory budget')
    retrieval=summary['processes']['1531763']
    base.need(retrieval['start_ticks']=='1829074494' and retrieval['uids']==[0]*4 and retrieval['in_observed_job'] is True
        and '/job_3261/step_0' in retrieval['cgroup'],'Actual local GPU retrieval process')
    base.need(retrieval['environment_allowlist']['OPSD_E1_RETRIEVER_URL']=='http://127.0.0.1:18084/retrieve'
        and retrieval['environment_allowlist']['SEARCH_R1_REPO']==str(common.REPO),'Actual retrieval endpoint/source')
    apps=summary['unfiltered_application_csv_rows'];observed_pids={int(row[1]) for row in apps}
    base.need(observed_pids=={1531763}|{w['pid'] for w in workers},'Actual unfiltered NVML membership')
    retriever_gpus={row[0].strip() for row in apps if int(row[1])==1531763}
    gpu=[dict(index=int(row[0]),uuid=row[1].strip(),used_gib=int(row[3])/1024,
        utilization_percent=int(row[5])) for row in summary['gpu_inventory_csv_rows']]
    base.need([row['index'] for row in gpu]==list(range(8)) and retriever_gpus=={row['uuid'] for row in gpu},'Actual local retrieval GPU allocation')
    scontrol=commands['scontrol']['stdout'];base.need('JobState=RUNNING ' in scontrol and 'Restarts=0 ' in scontrol
        and 'NodeList=h100-10 ' in scontrol,'Actual running uninterrupted job')
    _, fixed50sha=base.raw(HERE/'SOURCE_ACTUAL3261_PROGRESS50_FIXED_SET_READ_ONLY_REVIEW.json',
        '99400807820c3a201feda1eeb0039aa8e0b087dc32c76cc6ea6d86e3836dc2a7')
    prior56_raw,prior56sha=base.raw(HERE/'actual3261-progress-20261005T2305.json',
        '8c24cc75b5acd8b40bb458181725660dba3bc245ed4a53efc75a025de96b6faa')
    prior56=common.loads(prior56_raw);base.need(prior56['actual_complete_outer']==56,'Historical56 scope')
    rewards=[row['metrics']['critic/rewards/mean'] for row in rows];latest=rows[-1]['metrics']
    return dict(version=1,scope='SOURCE_AUTHOR_READ_ONLY_ACTUAL64_LOG_PROCESS_RESOURCE_REVIEW_NOT_FIXED_SET_RERUN_OR_FINAL200',
        inspected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot=str(snapshot),
        snapshot_finished_utc=summary['finished_utc'],snapshot_sha256=summary_sha,transport_sha256=transport_sha,
        parser_source_sha256=base.raw(Path(__file__))[1],source_commit=common.SOURCE,config_sha256=common.CONFIG_SHA,job_id=common.JOB,
        all315_original_hashes_matched=True,accepted5_release_signature_and_current_original_pins_valid=True,
        driver_pid=common.DRIVER_PID,driver_start_ticks=common.DRIVER_START,workers=workers,
        first5_identity_reference_sha256=identity_sha,log_join=join,stream_boundaries=boundaries,per_rank_actual_progress=per_rank,
        actual_completed_outer=64,observed_global_trajectories=64*512,derived_questions_not_raw_current_sampler_cursor=True,
        latest_metrics={k:latest[k] for k in ['critic/rewards/mean','actor/opsd_loss','actor/grad_norm','actor/lr','env/finish_ratio','env/ratio_of_valid_action','timing_s/step']},
        last10_training_reward=dict(outers=list(range(55,65)),values=rewards[-10:],mean=statistics.mean(rewards[-10:])),
        instantaneous_gpu_sample=gpu,unfiltered_gpu_process_ids=sorted(observed_pids),retriever_pid=1531763,
        local_retrieval_stdout_sha256=originals['retriever.log'][1],
        observed_local_retrieval_POST200_log_lines=len(re.findall(rb'127\.0\.0\.1:[0-9]+ - "POST /retrieve HTTP/1\.1" 200 OK',originals['retriever.log'][0])),
        preserved_previous56_progress_sha256=prior56sha,preserved_fixed50_negative_review_sha256=fixed50sha,
        fixed50_effect_conclusion_unchanged='Same512 EM18.36%->6.05%, F124.16%->8.32%, complete format52.15%->0.195%; numeric execution success is not effect improvement',
        no_new_native25_or50_content_decision=True,no_training_control_or_source_changes=True,
        limitations=['New train batches differ; recent reward is descriptive, not same-question paired validation',
            'No fixed50 evaluation tests rerun; previous immutable actual50 report is retained by SHA',
            'NVML proves observed local GPU service allocation, not every individual E5 query tensor device',
            'Captured newline-complete prefixes do not prove future budget200 or terminal/resource release',
            'No new native PT/HF/state restoration claim and no SSH/GPU/signing/Slurm control'])


if __name__=='__main__':
    result=inspect();out=HERE/'SOURCE_ACTUAL3261_PROGRESS64_READ_ONLY_REVIEW.json'
    with out.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(output=str(out),sha256=base.raw(out)[1],actual_outer=64,
        last10_reward_mean=result['last10_training_reward']['mean'],latest=result['latest_metrics'])))
