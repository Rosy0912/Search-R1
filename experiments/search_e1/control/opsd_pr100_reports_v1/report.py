"""Read accepted PR100 scalar records and native512 summaries; write public aggregates.

Stdlib CPU only. No models, tensors, tokenizer, GPU, retrieval, subprocess,
training-state restoration, raw export or new bootstrap/scoring computation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

METRICS=('raw_loss','post_loss','weighted_post_loss','student_to_teacher_raw_kl',
         'student_to_outer_old_raw_kl','student_to_mixture_raw_kl')


def require(condition,message):
    if not condition:raise ValueError(message)


def metadata(path):
    path=Path(path).resolve();s=path.stat()
    return dict(path=str(path),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def read_json(reference):
    require(isinstance(reference,dict) and isinstance(reference.get('path'),str)
            and re.fullmatch('[a-f0-9]{64}',reference.get('sha256','')),'Explicit bound JSON path/SHA required')
    before=metadata(reference['path']);raw=Path(before['path']).read_bytes()
    require(metadata(reference['path'])==before,'JSON changed during read')
    sha=hashlib.sha256(raw).hexdigest();require(sha==reference['sha256'],'JSON SHA differs: '+reference['path'])
    return json.loads(raw),dict(before,sha256=sha)


def read_path(path):
    before=metadata(path);raw=Path(before['path']).read_bytes()
    require(metadata(path)==before,'Closed record changed during read')
    ref=dict(before,sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw),ref


def artifact(path):
    before=metadata(path);sha=hashlib.sha256(Path(before['path']).read_bytes()).hexdigest()
    require(metadata(path)==before,'Written report changed while hashing')
    return dict(before,sha256=sha)


def integer(value,low=0):
    require(type(value) is int and value>=low,'Invalid integer recorded scalar')
    return value


def finite(value):
    require(type(value) in (int,float) and math.isfinite(value),'Nonfinite or missing recorded scalar')
    return float(value)


def sampling_counts(stats):
    """Reuse the original accepted strict reward aggregate; no raw rollout read."""
    require(stats['questions']==512 and stats['generated_trajectories']==2560,
            'Closed sampling denominator is not512x5')
    diagnostics=stats['diagnostics'];rewards=diagnostics['strict_reward']
    require(diagnostics['denominator']==2560 and set(rewards)<=set(('0','1')),
            'Original strict reward count schema differs')
    wrong=integer(rewards.get('0',0));hits=integer(rewards.get('1',0))
    require(wrong+hits==2560,'Original recorded strict rewards do not cover2560')
    return dict(questions=512,trajectories=2560,strict_hits=hits,em=hits/2560)


def means(items):
    return {name:(math.fsum(finite(x[name]) for x in items)/len(items) if items else None) for name in METRICS}


def collect_training(config,config_ref):
    root=Path(config['paths']['output_root']);rows=[];updates=[];sources=[]
    previous_update=previous_exposure=cumulative_tokens=0;seen_updates=set()
    present={int(p.parents[1].name.split('-')[-1]) for p in (root/'outers').glob('outer-*/training/outer-complete.json')}
    require(present==set(range(len(present))) and len(present)<=100,'Only a continuous completed outer prefix can be reported')
    for index in range(len(present)):
        outer=root/'outers'/f'outer-{index:05d}'
        completed,cref=read_path(outer/'training'/'outer-complete.json')
        require(completed['schema']=='opsd-online-outer-complete-v1' and completed['accepted'] is True
                and completed['run_id']==config['run_id'] and completed['outer_index']==index,'Wrong accepted outer identity')
        manifest,mref=read_json(completed['training_input']['outer_manifest'])
        require(manifest['run_id']==config['run_id'] and manifest['outer_index']==index
                and manifest['config_sha256']==config_ref['sha256'] and manifest['variant']=='PR'
                and manifest['cumulative_optimizer_updates_before']==previous_update,'Completed outer config/update start differs')
        checkpoint,cpref=read_json(completed['checkpoint'])
        require(checkpoint['schema']=='opsd-online-checkpoint-v1' and checkpoint['accepted'] is True
                and checkpoint['run_id']==config['run_id'] and checkpoint['config_sha256']==config_ref['sha256']
                and checkpoint['cursor']['outer_index']==index,'Wrong retained outer-end checkpoint manifest')
        p=integer(completed['P']);k=integer(completed['K']);require(0<=k<=p<=512 and p==manifest['P'],'Wrong eligible/usable pair counts')
        count=math.ceil(k/32)
        require(completed['optimizer_updates_delta']==count and completed['optimizer_updates']==previous_update+count
                and checkpoint['optimizer_updates']==completed['optimizer_updates']
                and checkpoint['cursor']['next_pair_cursor']==checkpoint['cursor']['usable_pair_count']==k,
                'Completed outer true Adam/tail/cursor accounting differs')
        require(completed['cumulative_pair_exposure']==checkpoint['cumulative_pair_exposure']==previous_exposure+k,
                'Completed actual pair exposure is not continuous')
        references=checkpoint['outer_update_receipts'];require(len(references)==count,'Committed outer receipt count differs')
        pairs=[];pair_indices=set();outer_updates=[];committed_retries=0
        for ordinal,reference in enumerate(references):
            value,proof=read_json(reference);sources.append(proof)
            update=previous_update+ordinal+1;actual=min(32,k-ordinal*32)
            require(value['schema']=='opsd-online-real-update-v1' and value['accepted'] is True
                    and value['run_id']==config['run_id'] and value['outer_index']==index
                    and value['optimizer_updates']==update and value['global_actual_pairs']==actual,
                    'Receipt is not the unique committed actual update/tail')
            require(update not in seen_updates,'Repeated cumulative Adam update');seen_updates.add(update)
            ranks=value['all_rank_actual_token_metrics'];require(len(ranks)==8 and {x['rank'] for x in ranks}==set(range(8)),'Receipt rank coverage differs')
            batch=[metric for rank in ranks for metric in rank['pairs']]
            require(len(batch)==actual,'Actual trajectory receipt denominator differs')
            for metric in batch:
                i=integer(metric['pair_index']);require(i<k and i not in pair_indices,'Duplicate or out-of-range committed pair exposure')
                pair_indices.add(i);integer(metric['valid_tokens'],1)
                require(finite(metric['pair_weight'])==1 and metric['target_kind']=='teacher','PR unitweight/teacher-only target differs')
            valid=sum(x['valid_tokens'] for x in batch)
            require(valid==value['global_valid_tokens'],'Actual recorded valid token sum differs')
            retries=integer(value['retries']);committed_retries+=retries
            norm=finite(value['scaler']['grad_norm'])
            require(value['loss_reduction']=='trajectory_mean' and value['loss_scope']=='actual_training_forward_before_optimizer'
                    and finite(value['learning_rate'])==1e-6,'Loss scope/reduction/LR differs')
            row=dict(outer_number=index+1,outer_index=index,actual_adam=update,global_actual_pairs=actual,
                valid_policy_tokens=valid,preclip_grad_norm=norm,recorded_retries_before_committed_update=retries,
                clipped_vocabulary_count=sum(integer(x['clipped_vocabulary_count']) for x in batch),**means(batch))
            outer_updates.append(row);updates.append(row);pairs.extend(batch)
        require(len(pairs)==k and pair_indices==set(range(k)),'Incomplete/duplicate usable trajectory exposure')
        tokens=sum(x['valid_tokens'] for x in pairs)
        require(tokens==completed['global_valid_tokens'],'Closed outer recorded valid-token sum differs')
        sampling,sref=read_path(outer/'sampling-complete.json')
        require(sampling['accepted'] is True and sampling['run_id']==config['run_id']
                and sampling['rollouts']==manifest['rollouts'],'Sampling completion is not this closed outer')
        stats=sampling['stats'];require(stats==manifest['sampling_stats'],'Immutable original sampling stats differ')
        sampled=sampling_counts(stats)
        require(stats['questions']==512 and stats['generated_trajectories']==2560
                and stats['P']==p and stats['variant']=='PR' and stats['outer_index']==index,'Sampling/mining counts differ')
        require(manifest['sampling_model_identity']['optimizer_updates']==previous_update
                and manifest['sampling_model_identity']['checkpoint']==manifest['previous_complete_checkpoint'],
                'TRAIN counts are not from the actual pre-update outer-start policy')
        cumulative_tokens+=tokens
        row=dict(outer_number=index+1,outer_index=index,train_questions=512,sampled_trajectories=2560,
            train_strict_hits=sampled['strict_hits'],train_em=sampled['em'],eligible_P=p,usable_K=k,
            skipped=sum(integer(x) for x in completed['skipped'].values()),actual_adam_before=previous_update,
            actual_adam_this_outer=count,actual_adam_cumulative=completed['optimizer_updates'],
            cumulative_pair_exposure=completed['cumulative_pair_exposure'],valid_policy_tokens=tokens,
            cumulative_valid_policy_tokens=cumulative_tokens,committed_recorded_retries=committed_retries,
            clipped_vocabulary_count=sum(integer(x['clipped_vocabulary_count']) for x in pairs),
            preclip_grad_norm_min=min((x['preclip_grad_norm'] for x in outer_updates),default=None),
            preclip_grad_norm_max=max((x['preclip_grad_norm'] for x in outer_updates),default=None),
            checkpoint_manifest_sha256=cpref['sha256'],**means(pairs))
        for name in ('all_wrong','all_correct','mixed'):row['sample_questions_'+name]=integer(stats['classes'][name])
        rows.append(row);sources.extend([cref,mref,cpref,sref])
        previous_update=completed['optimizer_updates'];previous_exposure=completed['cumulative_pair_exposure']
    return rows,updates,dict(completed_outers=len(rows),expected_outers=100,all100_complete=len(rows)==100,
        actual_adam=previous_update,raw_pair_exposure=previous_exposure,valid_policy_tokens=cumulative_tokens),sources


def model_counts(model,prefix):
    overall=model['overall'];require(overall['questions']==512,'Native512 denominator differs')
    hits=integer(overall['strict_hits']);require(hits<=512 and math.isclose(finite(overall['em']),hits/512),'Overall EM/hit count differs')
    row={prefix+'strict_hits':hits,prefix+'em':hits/512,prefix+'final_invalid':integer(overall['final_invalid']),
        prefix+'valid_gold_mismatch':integer(overall['valid_gold_mismatch']),prefix+'no_executed_search':integer(overall['no_executed_search'])}
    require(hits+row[prefix+'final_invalid']+row[prefix+'valid_gold_mismatch']==512,'Mutually exclusive final outcome counts differ')
    for dataset,label in (('nq','nq'),('hotpotqa','hotpot')):
        part=model['datasets'][dataset];require(part['questions']==256,'Native dataset denominator differs')
        n=integer(part['strict_hits']);require(n<=256 and math.isclose(finite(part['em']),n/256),'Dataset EM/hits differ')
        row[prefix+label+'_hits']=n;row[prefix+label+'_em']=n/256
    require(row[prefix+'nq_hits']+row[prefix+'hotpot_hits']==hits,'Dataset hits do not sum to512 hits')
    return row


def collect_evaluation(pipeline,config,training_completed):
    root=Path(pipeline['output_root']);sources=[];rows=[]
    baseline,bref=read_json(pipeline['baseline_summary']);sources.append(bref)
    require(baseline['accepted'] is True and pipeline['baseline_version']=='bf16_zero','Accepted source100 BF16 baseline required')
    identity=baseline['identities']
    require(identity['source_model_identity']==config['source_model_manifest']['sha256']
            and identity['runtime_identity']==config['runtime_manifest']['sha256']
            and identity['validation_sha256']==config['validation_identity']['sha256'],'Baseline source/runtime/heldout identity differs')
    base=model_counts(baseline['models']['bf16_zero'],'baseline_')
    present=[n for n in range(10,101,10) if (root/f'outer-{n:03d}'/'boundary-result.json').exists()]
    require(present==list(range(10,10*(len(present)+1),10)),'Only a continuous closed evaluation prefix can be reported')
    if present:
        curve,curve_ref=read_path(root/f'curve-through-{present[-1]:03d}.json');sources.append(curve_ref)
        require(curve['accepted'] is True and curve['run_id']==pipeline['run_id']
                and curve['primary_slots']==len(present)*512
                and [x['completed_outers'] for x in curve['rows']]==present,'Native closed evaluation curve differs')
    else:curve=dict(rows=[])
    for n,entry_curve in zip(present,curve['rows']):
        require(n<=training_completed,'Evaluation is beyond accepted completed training prefix')
        entry,eref=read_json(entry_curve['reference']);sources.append(eref)
        require(entry.get('accepted') is True and entry['completed_outers']==n and entry['training_run_id']==config['run_id']
                and entry=={k:v for k,v in entry_curve.items() if k!='reference'},'Native immutable boundary result differs from accepted curve')
        summary,sref=read_json(entry['summary']);exit_value,xref=read_json(entry['actual_exit']);release,rref=read_json(entry['actual_release'])
        sources.extend([sref,xref,rref])
        require(summary['schema']=='opsd-pr100-native512-summary-v1' and summary['accepted'] is True
                and summary['primary_slots']==512 and summary['run_id']==entry['evaluation_run_id']
                and summary['completed_outers']==n and summary['identities']==identity,'Closed native512 model/protocol differs')
        require(exit_value['actual_wait_returncode']==0 and exit_value['run_id']==entry['evaluation_run_id']
                and release['accepted'] is True,'Evaluation actual child0/fullrelease missing')
        paired=entry['paired_against_source100'];require(paired['questions']==512,'Paired comparison denominator differs')
        current=model_counts(summary['models']['raw_bf16'],'')
        require(paired['preserved_hits']+paired['lost_hits']==base['baseline_strict_hits']
                and paired['preserved_hits']+paired['new_hits']==current['strict_hits']
                and sum(paired[k] for k in ('new_hits','lost_hits','preserved_hits','still_wrong'))==512
                and paired['net_hits']==current['strict_hits']-base['baseline_strict_hits'],
                'Native paired hit counts do not match actual source/current512')
        row=dict(completed_outers=n,actual_adam_cumulative=entry['actual_optimizer_updates'],**current,**base)
        row.update({name:integer(paired[name]) for name in ('new_hits','lost_hits','preserved_hits','still_wrong','lost_final_invalid','lost_valid_gold_mismatch')})
        row['net_hits']=paired['net_hits'];row['difference_percentage_points']=finite(paired['net_percentage_points'])
        require(math.isclose(row['difference_percentage_points'],100*paired['net_hits']/512),'Native paired pp difference differs')
        bootstrap=paired['bootstrap'];require(bootstrap['replicates']==10000 and bootstrap['seed']==42 and bootstrap['question_units']==512,'Prespecified native paired bootstrap differs')
        row['paired_bootstrap95_lower_pp']=100*finite(bootstrap['lower']);row['paired_bootstrap95_upper_pp']=100*finite(bootstrap['upper'])
        row['paired_bootstrap_interval_crosses_zero']=bootstrap['lower']<=0<=bootstrap['upper']
        row['nq_difference_percentage_points']=100*(row['nq_em']-base['baseline_nq_em'])
        row['hotpot_difference_percentage_points']=100*(row['hotpot_em']-base['baseline_hotpot_em'])
        row['checkpoint_manifest_sha256']=entry['checkpoint']['sha256'];rows.append(row)
    return rows,dict(completed_evaluations=len(rows),expected_evaluations=10,formal_primary_slots=len(rows)*512,
        all10_complete=len(rows)==10,baseline_strict_hits=base['baseline_strict_hits'],baseline_em=base['baseline_em']),sources


def write_csv(path,rows,default_fields):
    fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else default_fields
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        handle.flush();os.fsync(handle.fileno())


def number(value,digits=6):return 'NA' if value is None else f'{value:.{digits}f}'


def markdown(training,evaluations,coverage):
    parts=['# PR100 已完成记录汇总','',
        f"当前采纳完整训练前缀：{coverage['training']['completed_outers']}/100 outer；实际 Adam {coverage['training']['actual_adam']} 次；"
        f"完整评测 {coverage['evaluation']['completed_evaluations']}/10，共 {coverage['evaluation']['formal_primary_slots']} 个正式题目槽。",'',
        'TRAIN EM 是每轮训练前最新策略对该轮不同 TRAIN512×5 的严格正确率。它不是固定验证集、不是 pass@5，也不是每次 Adam 后重新生成的 EM。',
        'loss 是实际已提交更新中记录的每轨迹有效 token 均值，再按实际轨迹数合并；包含真实尾批。clip 后目标不是原始 KL。这里没有重读词表或重算模型/梯度。','',
        '| outer | TRAIN EM | K | 累计 Adam | raw reverse KL | clip 后 loss |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in training:parts.append(f"| {row['outer_number']} | {100*row['train_em']:.4f}% | {row['usable_K']} | {row['actual_adam_cumulative']} | {number(row['raw_loss'])} | {number(row['post_loss'])} |")
    parts+=['','固定 heldout512 采用普通提示/BF16 base+FP32 原始 adapter 的既定 greedy 协议。配对 bootstrap 直接引用已接受的10000/seed42原生结果，没有重跑或调参。','',
        '| outer | hits/512 | EM | NQ/256 | Hotpot/256 | 新增 / 丢失 | 对 source100 差值 pp | 95% CI pp |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in evaluations:parts.append(f"| {row['completed_outers']} | {row['strict_hits']}/512 | {100*row['em']:.4f}% | {row['nq_hits']} | {row['hotpot_hits']} | {row['new_hits']} / {row['lost_hits']} | {row['difference_percentage_points']:+.4f} | [{row['paired_bootstrap95_lower_pp']:+.4f}, {row['paired_bootstrap95_upper_pp']:+.4f}] |")
    if not evaluations:parts+=['','尚无完整512题评测，不填入来源历史结果或部分预测。']
    parts+=['','报告只覆盖已经闭合的前缀，不把正在运行的轮次或部分512预测当成完成。不会从十轮间隔推测精确退化时刻；单次实验的差值不是算法因果证明。',
        '所有输入原件、轨迹、token记录、检查点及模型保持在 HKUST；本报告只导出公共计数、均值和来源摘要。','']
    return '\n'.join(parts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-config','training-config-sha256','pipeline-config','pipeline-config-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    config,cf=read_json(dict(path=a.training_config,sha256=a.training_config_sha256))
    pipeline,pf=read_json(dict(path=a.pipeline_config,sha256=a.pipeline_config_sha256))
    require(config['schema']=='opsd-pr100-config-v1' and config['variant']=='PR' and config['scientific']['outer_iterations']==100,'Only exact immutable PR100 input accepted')
    require(pipeline['schema']=='opsd-pr100-periodic-pipeline-v1' and pipeline['state']=='READY'
            and pipeline['training_config']['path']==cf['path'] and pipeline['training_config']['sha256']==cf['sha256']
            and pipeline['study_id']==config['study_id'],'Pipeline does not bind this training config/study')
    output=Path(a.output).absolute();allowed=[Path(config['paths']['base']).resolve(),Path(pipeline['output_root']).resolve().parent]
    require(str(output).startswith('/data/') and any(output.is_relative_to(root) for root in allowed),
            'Report output must remain inside this HKUST remote experiment')
    require(not output.exists(),'Never overwrite an aggregate snapshot')
    train,updates,tc,ts=collect_training(config,cf)
    evaluation,ec,es=collect_evaluation(pipeline,config,len(train))
    require(metadata(cf['path'])=={k:cf[k] for k in metadata(cf['path'])}
            and metadata(pf['path'])=={k:pf[k] for k in metadata(pf['path'])},'Immutable config changed during report')
    coverage=dict(training=tc,evaluation=ec,complete_100_and_all10=tc['all100_complete'] and ec['all10_complete'])
    output.mkdir(parents=True)
    train_csv=output/'train-outer-curves.csv';update_csv=output/'committed-adam-loss.csv';eval_csv=output/'native512-curves.csv'
    write_csv(train_csv,train,['outer_number','train_em','usable_K','actual_adam_cumulative','raw_loss','post_loss'])
    write_csv(update_csv,updates,['outer_number','actual_adam','global_actual_pairs','valid_policy_tokens','raw_loss','post_loss'])
    write_csv(eval_csv,evaluation,['completed_outers','strict_hits','em','nq_hits','hotpot_hits'])
    summary=output/'summary.json'
    summary.write_text(json.dumps(dict(schema='opsd-pr100-public-recorded-report-v1',accepted=True,coverage=coverage,
        training_run_id=config['run_id'],pipeline_run_id=pipeline['run_id'],training_config=cf,pipeline_config=pf,
        train_outer_records=train,heldout512_records=evaluation,
        scope='Accepted completed prefix only; recorded equal-trajectory loss and original accepted strict sampled reward counts; no raw rollout reread. No tensor/token-vocabulary reconstruction, no new science evaluation.'),
        sort_keys=True,indent=2,allow_nan=False)+'\n')
    note=output/'report.md';note.write_text(markdown(train,evaluation,coverage))
    final=output/'complete.json';final.write_text(json.dumps(dict(schema='opsd-pr100-readonly-cpu-report-complete-v1',accepted=True,
        coverage=coverage,observed_unix=time.time(),source_code=artifact(Path(__file__)),input_json_references=[cf,pf]+ts+es,
        artifacts=[artifact(x) for x in (train_csv,update_csv,eval_csv,summary,note)],
        no_gpu=True,no_new_scoring=True,no_bootstrap_recomputation=True,no_tensor_or_fullvocab_read=True,
        output_private_questions=False,original_node_outputs_modified=False),sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(accepted=True,coverage=coverage,public_summary=artifact(summary),complete=artifact(final))))


if __name__=='__main__':main()
