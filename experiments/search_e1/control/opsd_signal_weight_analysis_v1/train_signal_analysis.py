"""CPU-only immutable TRAIN signal/loss analysis; no model or HTTP imports.

TRAIN outcomes belong to outer-start sampling. Scores and token losses belong to
later CURRENT preupdate replay. Associations are retrospective, observational,
and measured on the eligible subset; they are not heldout EM/pass predictions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

SCHEMA='opsd-signal-train-analysis-request-v1'
VARIANTS=('G_POS','G_NEG','H_POS','H_NEG')
REGIONS=('think_text','search_query','structural_tags','final_answer','other','boundary')
SIGNAL_KEYS=('answer_gap','student_entropy')
PAIR_KEYS=('answer_gap','student_entropy','raw_teacher_kl','post_loss','weighted_post_loss',
           'selected_weight','answer_student_logp','answer_teacher_logp')


def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()

def object_sha(value):return hashlib.sha256(canonical(value)).hexdigest()

def finite(value,name):
    require(type(value) in (int,float) and math.isfinite(value),'Nonfinite '+name)
    return float(value)

def close(left,right,name):
    require(math.isclose(finite(left,name),finite(right,name),abs_tol=2e-5,rel_tol=2e-5),'Reconstruction differs '+name)

class Reader:
    """Fresh stream SHA on read AND final audit; metadata is not content proof."""
    def __init__(self):self.proofs={}
    @staticmethod
    def metadata(path):
        s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def lines(self,ref):
        path=Path(ref['path']);require(path.is_absolute() and not path.is_symlink(),'Pinned regular absolute file required')
        require(path.is_file() and len(ref['sha256'])==64,'Missing pinned regular input')
        before=self.metadata(path);h=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for raw in stream:
                require(raw.endswith(b'\n') and len(raw)<=16<<20,'Partial/oversized input row')
                h.update(raw);size+=len(raw);yield json.loads(raw)
        require(self.metadata(path)==before and h.hexdigest()==ref['sha256'] and size==before[2],'Input changed/SHA differs')
        if 'bytes' in ref:require(ref['bytes']==size,'Input bytes differ')
        proof=dict(path=str(path),sha256=h.hexdigest(),bytes=size,metadata=list(before))
        require(str(path) not in self.proofs or self.proofs[str(path)]==proof,'Repeated input changed')
        self.proofs[str(path)]=proof
    def json(self,ref):
        # Allow pretty-printed JSON, still bounded in memory, exact bytes pinned.
        path=Path(ref['path']);require(path.is_absolute() and not path.is_symlink() and path.is_file(),'Pinned JSON file required')
        before=self.metadata(path);require(before[2]<=64<<20,'JSON size cap exceeded')
        raw=path.read_bytes();require(self.metadata(path)==before and hashlib.sha256(raw).hexdigest()==ref['sha256'],'JSON changed/SHA differs')
        if 'bytes' in ref:require(ref['bytes']==len(raw),'JSON bytes differ')
        proof=dict(path=str(path),sha256=ref['sha256'],bytes=len(raw),metadata=list(before))
        require(str(path) not in self.proofs or self.proofs[str(path)]==proof,'Repeated JSON changed')
        self.proofs[str(path)]=proof;return json.loads(raw)
    def stable_end(self):
        # Some remote filesystems expose coarse timestamps. A same-size rewrite
        # can therefore leave every recorded metadata field unchanged. Re-read
        # every pinned input in bounded blocks instead of treating stat as SHA.
        for filename,proof in self.proofs.items():
            path=Path(filename)
            require(path.is_absolute() and not path.is_symlink() and path.is_file(),
                    'Previously read input is no longer a pinned regular file')
            before=self.metadata(path)
            require(list(before)==proof['metadata'],'Previously read input metadata changed')
            digest=hashlib.sha256();size=0
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4<<20),b''):
                    digest.update(block);size+=len(block)
            require(self.metadata(path)==before,'Previously read input changed during final SHA audit')
            require(size==proof['bytes'] and digest.hexdigest()==proof['sha256'],
                    'Previously read input content/SHA changed')


def sampling_questions(summary,variant,outer):
    require(summary['schema']=='opsd-signal-weight-train-em-v1' and summary['accepted'] is True and
        summary['variant']==variant and summary['outer_index']==outer and summary['post_training_em'] is False,
        'Only accepted full sampling-preupdate TRAIN summaries')
    require(summary['questions']==512 and summary['unique_slots']==2560 and summary['overall']['denominator']==2560,'Full512x5 TRAIN denominator required')
    questions={}
    for q in summary['per_question']:
        require(q['question_id'] not in questions and len(q['rewards'])==len(q['row_sha256'])==5,'Duplicate/incomplete question')
        require(all(type(r) is int and r in (0,1) for r in q['rewards']),'Strict binary rewards required')
        require(q['success_count']==sum(q['rewards']) and q['s_over_5']==q['success_count']/5,'Question s/5 differs')
        require(all(isinstance(sha,str) and len(sha)==64 for sha in q['row_sha256']),'Original row SHA missing')
        questions[q['question_id']]=q
    require(len(questions)==512,'Incomplete equal-question TRAIN population')
    correct=sum(q['success_count'] for q in questions.values())
    require(correct==summary['overall']['strict_correct'] and summary['overall']['em']==correct/2560,'Full unfiltered EM mismatch')
    require(summary['all5_wrong']==sum(q['success_count']==0 for q in questions.values()),'Hard/noB count differs')
    return questions


def validate_score(score,q,variant,update,metric,run_id):
    require(score.get('scoring_scope')=='no_grad_current_preupdate_hf_replay_not_generation_scores' and
            score.get('original_training_objective') is False,'Posthoc/generation scores cannot replace actual preupdate sibling replay')
    require(score['schema']=='opsd-current-preupdate-five-sibling-scores-v1' and score['accepted'] is True and
        score['variant']==variant and score['question_id']==q['question_id'],'Score identity differs')
    ident=score['preupdate_identity'];require(ident['run_id']==run_id and ident['optimizer_update']==update['optimizer_updates'] and
        ident['adapter_tensor_sha256']==update['pre_update_adapter_sha256'],'Score is not this actual CURRENT preupdate')
    require(score['state_restore']['state_restored'] is True and score['state_restore']['rng_consumed'] is False,'Scorer altered RNG/state')
    siblings=score['siblings'];require(len(siblings)==5 and [s['sibling'] for s in siblings]==list(range(5)),'All-five score slots required')
    require([s['original_row_sha256'] for s in siblings]==q['row_sha256'],'Score rows differ from full sampling cohort')
    n=score['normalization'];require(n['amplitude']==.9 and n['numerical_floor']==1e-6,'Weight normalization contract changed')
    require(score['signal']==('answer_gap' if variant.startswith('G_') else 'student_entropy') and
            n['direction']==(1 if variant.endswith('_POS') else -1),'Signal or weighting direction changed')
    x=[s[score['signal']] for s in siblings];require(n['scores']==x,'Stored normalization signal differs')
    valid=[j for j,v in enumerate(x) if v is not None]
    expected=[1.]*5
    if n['neutral_reason'] is None and len(valid)>=2:
        mu=math.fsum(finite(x[j],'score') for j in valid)/len(valid)
        denom=max(max(abs(x[j]-mu) for j in valid),1e-6)
        expected=[1+n['direction']*.9*(x[j]-mu)/denom if j in valid else 1. for j in range(5)]
    for a,b in zip(n['weights'],expected):close(a,b,'normalized weight')
    close(math.fsum(n['weights']),5.,'five sibling mean1')
    slot=score['selected_sibling'];require(type(slot) is int and 0<=slot<5,'Invalid original selected D')
    close(score['selected_pair_weight'],n['weights'][slot],'selected weight')
    close(metric['pair_weight'],score['selected_pair_weight'],'actual trained weight')
    return slot


def token_features(rows,metric,score,update):
    iterator=iter(rows)
    try:header=next(iterator)
    except StopIteration:raise ValueError('Empty actual training token log')
    require(header.get('loss_scope')=='actual_training_forward_before_optimizer','Posthoc tokens cannot replace online forward')
    require(header['schema']=='opsd-online-training-token-header-v1' and
        header['optimizer_update']==update['optimizer_updates'] and header['run_id']==update['run_id'] and
        header['student_pre_update_adapter_sha256']==update['pre_update_adapter_sha256'],'Not actual preupdate token log')
    require(header['identity']['question_id']==score['question_id'] and header['encoded_sha256']==score['preupdate_identity']['encoded_sha256'],
        'Training/scorer response encoding changed')
    require(metric.get('target_kind')=='teacher','Pure fixed teacher target required; no mixture reinterpretation')
    require(header['frozen_teacher_identity']==score['preupdate_identity']['source_model_identity'] and
            header['identity']['reference_response_sha256']==score['reference_response_sha256'],'Teacher or same B changed')
    weight=finite(metric['pair_weight'],'weight');close(header['pair_weight'],weight,'header weight')
    actual=update['global_actual_pairs'];regions={r:dict(valid_tokens=0,raw_sum=0.,post_sum=0.,entropy_sum=0.,student_logp_sum=0.,teacher_logp_sum=0.) for r in REGIONS}
    sums=defaultdict(float);seen=set();answer_s=[];answer_t=[]
    expected={i for i,m in enumerate(header['policy_mask']) if m}
    for row in iterator:
        if 'student_observed_token_logp' not in row:continue # Explicit vocabulary metadata, not a policy observation.
        i=row['token_index'];require(i in expected and i not in seen and row['policy_mask'] is True,'Token mask/uniqueness differs')
        require(row['token_id']==header['token_ids'][i] and row['token_offsets']==header['token_offsets'][i],'Token ID/offset changed')
        region=row.get('region',header.get('token_regions',[])[i] if header.get('token_regions') else None)
        require(region in REGIONS,'Missing original exact token region')
        seen.add(i);raw=finite(row['pre_clip_full_vocabulary_kl'],'raw KL');post=finite(row['post_clip_full_vocabulary_loss'],'post loss')
        teacher=finite(row['student_to_teacher_raw_kl'],'teacher KL');close(raw,teacher,'pure teacher target')
        close(row['global_batch_contribution'],weight*post/len(expected)/actual,'actual weighted denominator')
        sums['raw']+=raw;sums['post']+=post;sums['entropy']+=finite(row['student_entropy'],'entropy')
        sums['clip_hits']+=finite(row['clipped_vocabulary_count'],'clip count')
        sums['clip_student_mass']+=finite(row['clipped_student_probability_mass'],'clip student mass')
        regions[region]['valid_tokens']+=1;regions[region]['raw_sum']+=raw;regions[region]['post_sum']+=post
        regions[region]['entropy_sum']+=finite(row['student_entropy'],'region entropy')
        regions[region]['student_logp_sum']+=finite(row['student_observed_token_logp'],'region p')
        regions[region]['teacher_logp_sum']+=finite(row['teacher_observed_token_logp'],'region q')
        if region=='final_answer':
            answer_s.append(finite(row['student_observed_token_logp'],'answer p'))
            answer_t.append(finite(row['teacher_observed_token_logp'],'answer q'))
    require(seen==expected and len(seen)==metric['valid_tokens'] and len(seen)>0,'Incomplete actual effective training token inventory')
    total=len(seen);close(sums['raw']/total,metric['raw_loss'],'raw trajectory mean')
    close(sums['post']/total,metric['post_loss'],'post trajectory mean')
    close(weight*sums['post']/total,metric['weighted_post_loss'],'weighted trajectory mean')
    for value in regions.values():
        value.update(raw_contribution=value['raw_sum']/total,post_contribution=value['post_sum']/total,
                     weighted_post_contribution=weight*value['post_sum']/total,
                     mean_post=value['post_sum']/value['valid_tokens'] if value['valid_tokens'] else None,
                     mean_raw_teacher_kl=value['raw_sum']/value['valid_tokens'] if value['valid_tokens'] else None,
                     mean_student_entropy=value['entropy_sum']/value['valid_tokens'] if value['valid_tokens'] else None,
                     mean_student_logp=value['student_logp_sum']/value['valid_tokens'] if value['valid_tokens'] else None,
                     mean_teacher_logp=value['teacher_logp_sum']/value['valid_tokens'] if value['valid_tokens'] else None,
                     mean_signed_gap=(value['student_logp_sum']-value['teacher_logp_sum'])/value['valid_tokens'] if value['valid_tokens'] else None)
    answer_count=len(answer_s)
    return dict(raw_teacher_kl=sums['raw']/total,post_loss=sums['post']/total,
        weighted_post_loss=weight*sums['post']/total,student_entropy=sums['entropy']/total,
        answer_gap=math.fsum(a-b for a,b in zip(answer_s,answer_t))/answer_count if answer_count else None,
        answer_student_logp=math.fsum(answer_s)/answer_count if answer_count else None,
        answer_teacher_logp=math.fsum(answer_t)/answer_count if answer_count else None,
        clipped_vocabulary_count=sums['clip_hits'],mean_clipped_student_probability_mass=sums['clip_student_mass']/total,
        valid_tokens=total,regions=regions,
        self_pair=header['mining']['reference_sibling']==score['selected_sibling'])


def collect_outer(reader,config,entry):
    variant=config['variant'];outer=entry['outer_index'];summary=reader.json(entry['sampling_summary']);completion=reader.json(entry['training_completion'])
    questions=sampling_questions(summary,variant,outer)
    if config.get('_analysis_config_sha256'):
        require(summary['config']['sha256']==config['_analysis_config_sha256'] and
                completion.get('contract',{}).get('variant')==variant,'Sampling/config/science variant differs')
    require(completion.get('schema')=='opsd-online-outer-complete-v1','Actual training completion required')
    require(summary['run_id']==config['run_id'] and completion['accepted'] is True and
        completion['run_id']==config['run_id'] and completion['outer_index']==outer,'Sampling/training run or outer differs')
    P=completion['P'];K=completion['K'];require(0<=K<=P==sum(q['success_count']>0 for q in questions.values()),'P/K and successful B eligibility differ')
    require(completion['optimizer_updates_delta']==math.ceil(K/32),'Actual Adam budget mismatch')
    checkpoint=reader.json(completion['checkpoint']);refs=checkpoint['outer_update_receipts']
    require(checkpoint.get('schema')=='opsd-online-checkpoint-v1' and checkpoint['optimizer_updates']==completion['optimizer_updates'],'Committed actual checkpoint count differs')
    require(checkpoint['accepted'] is True and checkpoint['run_id']==config['run_id'] and
        checkpoint['cursor']['outer_index']==outer and checkpoint['cursor']['next_pair_cursor']==checkpoint['cursor']['usable_pair_count']==K,
        'Only full committed outer checkpoint accepted')
    require(len(refs)==math.ceil(K/32),'Committed update count mismatch')
    features={qid:dict(variant=variant,outer_index=outer,question_id=qid,dataset=q['dataset'],
        sampling_s_over_5=q['s_over_5'],sampling_rewards=q['rewards'],sampling_model_identity=summary['sampling_preupdate_raw_model'],
        teacher_available=q['success_count']>0,trained=False,score_observed=False,missing_reason='all5_wrong_no_successful_B' if q['success_count']==0 else 'selected_D_encoding_skip',
        original_population_denominator=512,original_trajectory_denominator=2560) for qid,q in questions.items()}
    batches=[];seen=set();updates=[]
    for ref in refs:
        update=reader.json(ref);updates.append(update['optimizer_updates'])
        require(update.get('schema')=='opsd-online-real-update-v1','Actual update receipt required')
        require(update['accepted'] is True and update['outer_index']==outer and update['run_id']==config['run_id'],'Committed update identity changed')
        ranks=update['all_rank_actual_token_metrics'];require(len(ranks)==4 and {r['rank'] for r in ranks}==set(range(4)),'Four actual rank metrics required')
        metrics=[m for rank in ranks for m in rank['pairs']];actual=update['global_actual_pairs']
        require(len(metrics)==actual and 1<=actual<=32,'Actual global tail denominator changed')
        batch_weights=[]
        for metric in metrics:
            require(metric['pair_index'] not in seen,'Duplicate committed pair exposure');seen.add(metric['pair_index'])
            score=reader.json(metric['signal_score']);qid=score['question_id'];require(qid in questions and not features[qid]['trained'],'Duplicate/foreign trained question')
            slot=validate_score(score,questions[qid],variant,update,metric,config['run_id'])
            measured=token_features(reader.lines(metric['token_log']),metric,score,update)
            original_score=score['siblings'][slot];available=original_score.get('skipped_reason') is None
            if available:
                for key in SIGNAL_KEYS:
                    if original_score[key] is None:require(measured[key] is None,'Answer NA changed')
                    else:close(measured[key],original_score[key],'actual/scorer '+key)
            mining=metric['signal_scalar'];close(mining['success_fraction'],questions[qid]['s_over_5'],'saved success fraction')
            require(mining['question_id']==qid and mining['sibling']==slot and
                mining['strict_reward']==questions[qid]['rewards'][slot],'Actual selected label changed')
            weight=finite(metric['pair_weight'],'selected weight');batch_weights.append(weight)
            features[qid].update(measured,trained=True,score_observed=available,
                missing_reason=original_score.get('skipped_reason'),selected_sibling=slot,selected_reward=questions[qid]['rewards'][slot],
                selected_weight=weight,self_pair=measured['self_pair'],
                pair_class='CC' if questions[qid]['rewards'][slot] else 'CW',optimizer_update=update['optimizer_updates'],
                current_preupdate_identity=score['preupdate_identity'],score_siblings=score['siblings'],score_normalization=score['normalization'])
        norm=finite(update['scaler']['grad_norm'],'batch preclip norm');total=math.fsum(batch_weights);squares=math.fsum(w*w for w in batch_weights)
        batches.append(dict(variant=variant,outer_index=outer,optimizer_update=update['optimizer_updates'],actual_pairs=actual,
            weight_sum=total,weight_mean=total/actual,weight_ess=total*total/squares,grad_norm_before_clip=norm,
            clip_triggered=norm>1.,derived_clip_factor=min(1.,1./(norm+1e-6)),clip_factor_is_derived_not_direct_measurement=True))
    require(len(seen)==sum(row['trained'] for row in features.values())==K,'K exposure missing')
    require(not updates or updates==list(range(completion['optimizer_updates']-len(updates)+1,completion['optimizer_updates']+1)),
            'Discontinuous committed Adam chain')
    return dict(variant=variant,outer_index=outer,train_correct=summary['overall']['strict_correct'],train_denominator=2560,
        train_em=summary['overall']['em'],P=P,K=K,all5_wrong=summary['all5_wrong'],optimizer_updates=completion['optimizer_updates'],
        sampling_model_identity=summary['sampling_preupdate_raw_model'],raw_post_training_checkpoint=completion['checkpoint'],
        sampling_model_is_not_current_score_model=True),list(features.values()),batches


def pearson(x,y):
    require(len(x)==len(y),'Paired lengths differ')
    n=len(x)
    if n<3 or len(set(x))<2 or len(set(y))<2:return None
    mx=math.fsum(x)/n;my=math.fsum(y)/n
    xx=math.fsum((a-mx)**2 for a in x);yy=math.fsum((b-my)**2 for b in y)
    if xx<=0 or yy<=0:return None
    return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)

def midranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[None]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        rank=(start+1+end)/2
        for i in order[start:end]:result[i]=rank
        start=end
    return result

def auc(x,labels):
    require(all(v in (0,1) for v in labels),'AUC requires strict binary labels')
    positives=sum(labels);negatives=len(labels)-positives
    if not positives or not negatives or len(set(x))<2:return None
    ranks=midranks(x);return (sum(r for r,label in zip(ranks,labels) if label)-positives*(positives+1)/2)/(positives*negatives)

def quantile(values,q):
    values=sorted(values)
    if not values:return None
    point=(len(values)-1)*q;lo=int(point);hi=min(lo+1,len(values)-1)
    return values[lo]+(point-lo)*(values[hi]-values[lo])

def bootstrap_pearson(rows,key,label,replicates=2000,seed=42):
    """Resample qid blocks from FULL population, retaining NA/noB blocks.

    Sufficient statistics make pooled repeated-qid blocks cheap. No global RNG
    is consumed; score NA contributes n=0 rather than an invented score0.
    """
    import numpy as np
    complete=[(row.get(key),row.get(label)) for row in rows if row.get(key) is not None and row.get(label) is not None]
    undefined=('fewer_than_three_measured_questions' if len(complete)<3 else 'zero_feature_variance'
               if len({finite(a,key) for a,_ in complete})<2 else 'zero_label_variance'
               if len({finite(b,label) for _,b in complete})<2 else None)
    clusters={}
    for row in rows:
        values=clusters.setdefault(row['question_id'],np.zeros(6,dtype=np.float64))
        a=row.get(key);b=row.get(label)
        if a is not None and b is not None:
            a=finite(a,key);b=finite(b,label);values+=np.array([1,a,b,a*a,b*b,a*b])
    if undefined or len(clusters)<3 or replicates<=0:
        reason=undefined or ('fewer_than_three_question_blocks' if len(clusters)<3 else 'bootstrap_not_requested')
        return dict(interval95=None,valid_replicates=0,requested_replicates=replicates,blocks=len(clusters),
                    NA_reason=reason)
    matrix=np.array(list(clusters.values()));rng=np.random.default_rng(seed);estimates=[]
    for _ in range(replicates):
        n,sx,sy,sxx,syy,sxy=matrix[rng.integers(0,len(matrix),len(matrix))].sum(axis=0)
        if n<3:continue
        xx=sxx-sx*sx/n;yy=syy-sy*sy/n
        if xx>0 and yy>0:estimates.append(float((sxy-sx*sy/n)/math.sqrt(xx*yy)))
    return dict(interval95=[quantile(estimates,.025),quantile(estimates,.975)] if estimates else None,
        valid_replicates=len(estimates),requested_replicates=replicates,blocks=len(clusters),
        block_unit='question_id; all siblings/observations move together',full_population_NA_blocks_retained=True)

def relationships(rows,replicates=2000,seed=42,feature_keys=PAIR_KEYS):
    result={}
    for key in feature_keys:
        result[key]={}
        for label in ('selected_reward','sampling_s_over_5'):
            measured=[row for row in rows if row.get(key) is not None and row.get(label) is not None]
            x=[finite(row[key],key) for row in measured];y=[finite(row[label],label) for row in measured]
            result[key][label]=dict(measured_questions=len(measured),population_questions=len(rows),
                missing_questions=len(rows)-len(measured),pearson=pearson(x,y),
                spearman=pearson(midranks(x),midranks(y)),
                raw_direction_auc=auc(x,y) if label=='selected_reward' else None,
                pearson_question_block=bootstrap_pearson(rows,key,label,replicates,seed),
                label_scope='outer_start_sampled_response; not future success probability',
                feature_scope='same stored response replayed later at current_preupdate',
                interpretation='observational eligible-subset association; no causal claim')
    return result


def exposure_table(rows,population_questions):
    """Actual selected-D objective scaling, never invented gradients/exposures."""
    require(all(row['trained'] for row in rows),'Only actual committed selected-D pairs have exposure')
    weights=[finite(row['selected_weight'],'actual selected weight') for row in rows]
    total=math.fsum(weights);squares=math.fsum(w*w for w in weights);n=len(rows)
    return dict(actual_pairs=n,fraction_of_full_question_population=n/population_questions if population_questions else None,
        population_unit='question_outer exposure; repeated question_id counts in each observed outer',
        CC_CW_counts={label:sum(row['pair_class']==label for row in rows) for label in ('CC','CW')},
        CC_CW_weight_sum={label:math.fsum(row['selected_weight'] for row in rows if row['pair_class']==label) for label in ('CC','CW')},
        self_pair_count=sum(row['self_pair'] for row in rows),nonself_pair_count=sum(not row['self_pair'] for row in rows),
        weight_sum=total,weight_mean=total/n if n else None,weight_ess=total*total/squares if squares else None,
        mean_sampling_s_over_5=math.fsum(row['sampling_s_over_5'] for row in rows)/n if n else None,
        mean_raw_teacher_KL=math.fsum(row['raw_teacher_kl'] for row in rows)/n if n else None,
        mean_unweighted_post_loss=math.fsum(row['post_loss'] for row in rows)/n if n else None,
        mean_weighted_post_loss=math.fsum(row['weighted_post_loss'] for row in rows)/n if n else None,
        weighted_exposure_is_scalar_objective_scaling_not_gradient_mass=True,
        optimization_divisor_is_actual_pairs_not_weight_sum=True)


def quantile_exposure(population,key):
    """Prespecified raw-score mid-CDF quartiles; never split tied score values.

    LOW u<=.25, HIGH u>=.75, MIDDLE otherwise, u=(midrank-.5)/n.
    Big ties may make bucket counts differ from nominal quartiles. All-equal
    scores have u=.5 and remain one middle tie; no artificial low/high order.
    """
    measured=[row for row in population if row['trained'] and row.get(key) is not None]
    values=[finite(row[key],key) for row in measured];n=len(values)
    percentiles=[(r-.5)/n for r in midranks(values)] if n else []
    groups={name:[] for name in ('LOW','MIDDLE','HIGH')}
    for row,u in zip(measured,percentiles):
        groups['LOW' if u<=.25 else 'HIGH' if u>=.75 else 'MIDDLE'].append(row)
    missing=Counter()
    for row in population:
        if not row['trained']:missing[row['missing_reason']]+=1
        elif row.get(key) is None:missing['no_effective_final_answer_payload_tokens' if key=='answer_gap' else 'actual_feature_unavailable']+=1
    require(sum(missing.values())+n==len(population),'Quantile coverage silently dropped questions')
    trained_count=sum(row['trained'] for row in population)
    return dict(full_population_questions=len(population),actual_trained_questions=trained_count,
        population_unit='question_outer exposure; repeated question_id counts in each observed outer',
        measured_actual_selected_D_questions=n,missing_full_population_questions=len(population)-n,
        missing_reason_counts=dict(missing),metric=key,
        definition=dict(kind='raw_score_midrank_empirical_percentile',formula='u=(midrank-.5)/n',
            low='u<=.25',high='u>=.75',middle='.25<u<.75',ties_never_split=True,
            nominal_bucket_counts_not_forced=True,all_equal_scores='one MIDDLE tie; no ordering',
            raw_score_is_not_within_question_normalized_r=True,
            population='Within one variant scope; actual selected-D current-preupdate forward feature'),
        ordering_NA_reason='no_measured_scores' if not n else 'zero_feature_variance' if len(set(values))<2 else None,
        buckets={name:dict(exposure_table(rows,len(population)),
            fraction_of_measured_actual_pairs=len(rows)/n if n else None,
            fraction_of_all_actual_trained_pairs=len(rows)/trained_count if trained_count else None,
            score_min=min(row[key] for row in rows) if rows else None,score_max=max(row[key] for row in rows) if rows else None)
            for name,rows in groups.items()})


def neutral_and_pair_exposure(population):
    names=('recorded_group_neutral_reason','selected_signal_NA','normalization_selected_signal_NA',
        'actual_forward_selected_signal_NA','actual_forward_answer_gap_NA','actual_forward_entropy_NA',
        'fewer_than_two_available_sibling_scores',
        'zero_score_dispersion','floor_attenuated_nonzero_dispersion','selected_weight_approximately_one',
        'selected_weight_exactly_one','all_five_weights_approximately_one','all_five_weights_exactly_one',
        'incidental_selected_weight_one_in_nonconstant_group',
        'incidental_selected_weight_approximately_one_in_nonconstant_group','selected_available_score_exactly_at_group_mean')
    trained=[row for row in population if row['trained']];n=len(trained);reason=Counter()
    axes=Counter({key:0 for key in names});available_axes=Counter({key:0 for key in names})
    tolerance=1e-12
    for row in trained:
        norm=row['score_normalization'];scores=norm['scores'];slot=row['selected_sibling']
        signal=norm.get('signal') or ('answer_gap' if row.get('variant','G_POS').startswith('G_') else 'student_entropy')
        require(signal in SIGNAL_KEYS,'Actual weighting signal identity missing or unknown')
        reason['none_recorded' if norm['neutral_reason'] is None else norm['neutral_reason']]+=1
        valid=[j for j,x in enumerate(scores) if x is not None]
        raw_maxabs=norm.get('raw_maxabs')
        if raw_maxabs is None and len(valid)>=2 and norm['neutral_reason'] is None:
            anchor=scores[valid[0]];delta=[scores[j]-anchor for j in valid];mu=math.fsum(delta)/len(valid)
            raw_maxabs=max(abs(x-mu) for x in delta)
        dispersion_available=raw_maxabs is not None and len(valid)>=2 and norm['neutral_reason'] is None
        selected_available=scores[slot] is not None and dispersion_available
        for key in ('recorded_group_neutral_reason','selected_signal_NA','normalization_selected_signal_NA',
            'actual_forward_selected_signal_NA','actual_forward_answer_gap_NA','actual_forward_entropy_NA',
            'fewer_than_two_available_sibling_scores',
            'selected_weight_approximately_one','selected_weight_exactly_one','all_five_weights_approximately_one',
            'all_five_weights_exactly_one'):
            available_axes[key]+=1
        for key in ('zero_score_dispersion','floor_attenuated_nonzero_dispersion'):
            available_axes[key]+=int(dispersion_available)
        for key in ('selected_available_score_exactly_at_group_mean','incidental_selected_weight_one_in_nonconstant_group',
            'incidental_selected_weight_approximately_one_in_nonconstant_group'):
            available_axes[key]+=int(selected_available)
        axes['recorded_group_neutral_reason']+=int(norm['neutral_reason'] is not None)
        axes['selected_signal_NA']+=int(scores[slot] is None)
        axes['normalization_selected_signal_NA']+=int(scores[slot] is None)
        axes['actual_forward_selected_signal_NA']+=int(row.get(signal) is None)
        axes['actual_forward_answer_gap_NA']+=int(row.get('answer_gap') is None)
        axes['actual_forward_entropy_NA']+=int(row.get('student_entropy') is None)
        axes['fewer_than_two_available_sibling_scores']+=int(len(valid)<2)
        axes['zero_score_dispersion']+=int(dispersion_available and raw_maxabs==0)
        axes['floor_attenuated_nonzero_dispersion']+=int(dispersion_available and 0<raw_maxabs<norm.get('numerical_floor',1e-6))
        axes['selected_weight_approximately_one']+=int(abs(row['selected_weight']-1.)<=tolerance)
        axes['selected_weight_exactly_one']+=int(row['selected_weight']==1.)
        axes['all_five_weights_approximately_one']+=int(all(abs(w-1.)<=tolerance for w in norm['weights']))
        axes['all_five_weights_exactly_one']+=int(all(w==1. for w in norm['weights']))
        axes['incidental_selected_weight_one_in_nonconstant_group']+=int(selected_available and raw_maxabs>0
            and row['selected_weight']==1. and not all(w==1. for w in norm['weights']))
        axes['incidental_selected_weight_approximately_one_in_nonconstant_group']+=int(selected_available and raw_maxabs>0
            and abs(row['selected_weight']-1.)<=tolerance and not all(w==1. for w in norm['weights']))
        centered=norm.get('centered_scores')
        if centered is None and scores[slot] is not None and valid:
            anchor=scores[valid[0]];delta=[scores[j]-anchor for j in valid];mu=math.fsum(delta)/len(valid)
            at_mean=(scores[slot]-anchor)-mu==0
        else:at_mean=bool(scores[slot] is not None and centered is not None and centered[slot]==0)
        axes['selected_available_score_exactly_at_group_mean']+=int(at_mean and norm['neutral_reason'] is None)
    not_trained=Counter(row['missing_reason'] for row in population if not row['trained'])
    return dict(full_population_questions=len(population),actual_trained_questions=n,
        population_unit='question_outer exposure; repeated question_id counts in each observed outer',
        not_trained_questions=len(population)-n,not_trained_reason_counts=dict(not_trained),
        allwrong_or_skipped_have_no_weight_and_zero_actual_training_exposure=True,
        recorded_group_neutral_reason_counts=dict(reason),
        recorded_group_neutral_reason_rates_over_actual_K={key:count/n if n else None for key,count in reason.items()},
        derived_neutral_axes={key:dict(count=count,rate_over_actual_K=count/n if n else None,
            rate_over_full_question_population=count/len(population) if population else None,
            data_available_denominator=available_axes[key],rate_over_available_data=count/available_axes[key] if available_axes[key] else None)
            for key,count in axes.items()},
        axes_overlap_and_are_not_recorded_receipt_reasons=True,weight_one_tolerance=tolerance,
        selected_signal_NA_definition='Compatibility alias of normalization_selected_signal_NA: normalization.scores[selected_sibling] is None',
        normalization_and_actual_training_forward_availability_are_separate=True,
        floor_positive_dispersion_is_attenuated_not_neutral=True,
        self_vs_nonself={name:exposure_table([row for row in trained if row['self_pair'] is flag],len(population))
            for name,flag in (('self',True),('nonself',False))})


def within_question_associations(rows,replicates=2000,seed=42):
    """Equal-question centered covariance removes BETWEEN-question means.

    An all-correct scored group contributes zero Y variance, rather than being
    silently removed. Hard/noB or no scores retain explicit population coverage.
    """
    result={}
    for key in SIGNAL_KEYS:
        contributions=[];eligible=0;observed=0;mixed=0;na_slots=0
        for row in rows:
            eligible+=int(row['teacher_available'])
            siblings=row.get('score_siblings')
            if siblings is None:continue
            pairs=[(s[key],row['sampling_rewards'][s['sibling']]) for s in siblings if s.get(key) is not None]
            na_slots+=5-len(pairs)
            if len(pairs)<2:continue
            observed+=1;mixed+=int(len({y for _,y in pairs})>1)
            x=[finite(a,key) for a,_ in pairs];y=[b for _,b in pairs];n=len(x)
            mx=math.fsum(x)/n;my=math.fsum(y)/n
            contributions.append((row['question_id'],math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/n if len(set(x))>1 and len(set(y))>1 else 0.,
                math.fsum((a-mx)**2 for a in x)/n if len(set(x))>1 else 0.,
                math.fsum((b-my)**2 for b in y)/n if len(set(y))>1 else 0.))
        def statistic(items):
            if not items:return None,None
            cov=math.fsum(x[1] for x in items)/len(items);vx=math.fsum(x[2] for x in items)/len(items);vy=math.fsum(x[3] for x in items)/len(items)
            return cov,cov/math.sqrt(vx*vy) if vx>0 and vy>0 else None
        cov,r=statistic(contributions);blocks=defaultdict(list)
        for item in contributions:blocks[item[0]].append(item)
        # Include every full population qid, including noB/NA empty blocks.
        for row in rows:blocks[row['question_id']]
        keys=list(blocks);rng=random.Random(seed);est=[]
        if len(keys)>=3 and replicates>0:
            totals=[(math.fsum(x[1] for x in blocks[k]),math.fsum(x[2] for x in blocks[k]),math.fsum(x[3] for x in blocks[k])) for k in keys]
            import numpy as np
            a=np.array(totals);gen=np.random.default_rng(seed)
            for _ in range(replicates):
                c,vx,vy=a[gen.integers(0,len(a),len(a))].sum(axis=0)
                if vx>0 and vy>0:est.append(float(c/math.sqrt(vx*vy)))
        result[key]=dict(population_questions=len(rows),successful_B_questions=eligible,
            questions_with_at_least_two_valid_scores=observed,mixed_label_scored_questions=mixed,
            missing_scored_sibling_slots=na_slots,equal_question_centered_covariance=cov,
            equal_question_centered_pearson=r,interval95=[quantile(est,.025),quantile(est,.975)] if est else None,
            valid_bootstrap_replicates=len(est),block_unit='question_id',
            noB_and_NA_retained_in_coverage=True,
            interpretation='within-question score-vs-old-sampled-label, not between-question performance or causal gradient effect')
    return result


def aggregate_features(features,batches,replicates=2000,seed=42):
    groups={}
    for variant in sorted({r['variant'] for r in features}):
        population=[r for r in features if r['variant']==variant];trained=[r for r in population if r['trained']]
        class_counts=dict(Counter(row['pair_class'] for row in trained));weights=[row['selected_weight'] for row in trained]
        total=math.fsum(weights);square=math.fsum(w*w for w in weights)
        regional={}
        for region in REGIONS:
            raw=math.fsum(row['regions'][region]['raw_contribution'] for row in trained)
            post=math.fsum(row['regions'][region]['post_contribution'] for row in trained)
            weighted=math.fsum(row['regions'][region]['weighted_post_contribution'] for row in trained)
            regional[region]=dict(valid_tokens=sum(row['regions'][region]['valid_tokens'] for row in trained),
                equal_trajectory_raw_teacher_KL_contribution=raw/len(trained) if trained else None,
                equal_trajectory_post_loss_contribution=post/len(trained) if trained else None,
                equal_trajectory_weighted_post_loss_contribution=weighted/len(trained) if trained else None,
                no_parameter_gradient_attribution=True)
        bs=[b for b in batches if b['variant']==variant]
        region_rows=[];regional_features=[]
        for row in population:
            value=dict(row)
            for name in REGIONS:
                for measure in ('mean_raw_teacher_kl','mean_post','mean_student_entropy','mean_student_logp','mean_signed_gap','post_contribution','weighted_post_contribution'):
                    field=name+'_'+measure
                    if field not in regional_features:regional_features.append(field)
                    value[field]=row.get('regions',{}).get(name,{}).get(measure)
            region_rows.append(value)
        groups[variant]=dict(population_questions=len(population),population_sampled_trajectories=5*len(population),
            sampling_strict_correct=sum(sum(row['sampling_rewards']) for row in population),
            sampling_train_em=math.fsum(row['sampling_s_over_5'] for row in population)/len(population),
            all5_wrong_noB=sum(not row['teacher_available'] for row in population),eligible_with_B=sum(row['teacher_available'] for row in population),
            selected_D_encoding_skips=sum(row['teacher_available'] and not row['trained'] for row in population),
            actual_trained_questions=len(trained),all5_score_observed_questions=sum(row['score_observed'] for row in trained),
            gap_measured_questions=sum(row.get('answer_gap') is not None for row in trained),
            entropy_measured_questions=sum(row.get('student_entropy') is not None for row in trained),
            CC_CW_exposure=class_counts,self_pair_exposure=sum(row['self_pair'] for row in trained),
            CC_CW_weighted_exposure={label:math.fsum(row['selected_weight'] for row in trained if row['pair_class']==label) for label in ('CC','CW')},
            selected_weight=dict(sum=total,sumsq=square,mean=total/len(weights) if weights else None,
                ess=total*total/square if square else None,ess_is_weight_concentration_not_gradient_effective_sample_size=True),
            actual_batches=len(bs),clipped_batches=sum(b['clip_triggered'] for b in bs),
            mean_preclip_norm=math.fsum(b['grad_norm_before_clip'] for b in bs)/len(bs) if bs else None,
            regions=regional,within_question=within_question_associations(population,replicates,seed),
            selected_D_relationships=relationships(population,replicates,seed),
            per_dataset_relationships={dataset:relationships([row for row in population if row['dataset']==dataset],0,seed)
                                       for dataset in sorted({row['dataset'] for row in population})},
            regional_selected_D_relationships=relationships(region_rows,0,seed,regional_features),
            regional_correlations_scope='Exploratory points without bootstrap CI; absent region means NA, contribution zero only when actually trained',
            per_outer_relationships={str(o):relationships([row for row in population if row['outer_index']==o],0,seed)
                                    for o in sorted({row['outer_index'] for row in population})},
            selected_D_raw_signal_quantile_exposure={key:quantile_exposure(population,key) for key in SIGNAL_KEYS},
            neutral_and_self_nonself_exposure=neutral_and_pair_exposure(population),
            predeclared_training_periods={name:dict(outer_indices=list(range(start,end)),
                observed_outer_indices=sorted({row['outer_index'] for row in population if start<=row['outer_index']<end}),
                quantile_exposure={key:quantile_exposure([row for row in population if start<=row['outer_index']<end],key)
                    for key in SIGNAL_KEYS},
                neutral_and_pair_exposure=neutral_and_pair_exposure([row for row in population if start<=row['outer_index']<end]))
                for name,start,end in (('through_15',0,15),('16_through_25',15,25))},
            scientific_limits=['TRAIN questions vary across outer; no fixed-heldout claim',
                'Teacher/loss require successful reference; hard/noB remain denominator but numerical features NA',
                'Sampled reward/s/5 belongs to outer-start model; replay belongs to later current-preupdate model',
                'Loss/entropy/logprob are retrospective observations, not calibrated future pass probabilities',
                'Weight-positive/negative interventions change model and generated histories; observational association is not causal proof',
                'Region loss contribution is not per-token parameter-gradient attribution',
                'Strict CW/all5wrong includes protocol errors; not proven semantic impossibility',
                'Pooled correlations can mix dataset/outer effects; per-dataset/outer reports are stratified descriptive checks',
                'One seed; intervals resample observed question blocks, not algorithm-seed uncertainty'])
    return dict(schema='opsd-signal-train-analysis-summary-v1',accepted=True,groups=groups,
                primary_performance_metric='formal fixed512 heldout evaluation is in separate evaluator; not estimated here',
                bootstrap_replicates=replicates,bootstrap_seed=seed)


def run(request_ref,output_dir):
    reader=Reader();request=reader.json(request_ref)
    require(request['schema']==SCHEMA and isinstance(request['study_id'],str),'Wrong pinned study analysis request')
    require(set(request['configs'])==set(VARIANTS),'Four independent variant configs required')
    configs={variant:reader.json(ref) for variant,ref in request['configs'].items()}
    for variant,config in configs.items():config['_analysis_config_sha256']=request['configs'][variant]['sha256']
    require(all(c['variant']==v and c['study_id']==request['study_id'] for v,c in configs.items()),'Study/variant config identity changed')
    common=('source_model_manifest','runtime_manifest','train_identity','validation_identity','common_cohort_manifest','common_initial_adapter_manifest')
    for field in common:
        require(len({configs[v][field]['sha256'] for v in VARIANTS})==1,'Different common model/data/runtime: '+field)
    roots={str(Path(c['paths']['study_base']).resolve()) for c in configs.values()};require(len(roots)==1,'Cross-study output forbidden')
    root=Path(next(iter(roots)));output=Path(output_dir).resolve()
    require(str(root).startswith('/weka_3/shaoshitong/') and output.is_relative_to(root) and not output.exists(),
            'New analysis output must be under approved HKUST remote study, never Mac')
    require(os.getuid()==configs[VARIANTS[0]]['owner']['uid'],'Analysis UID differs from authorized remote study')
    slots={(e['variant'],e['outer_index']) for e in request['outers']}
    require(len(slots)==len(request['outers']) and all(v in VARIANTS and type(o) is int and 0<=o<25 for v,o in slots),'Duplicate/foreign outer entry')
    complete=request.get('require_complete',True)
    if complete:require(slots=={(v,o) for v in VARIANTS for o in range(25)},'Final analysis requires all four25 outers')
    features=[];batches=[];outers=[]
    for entry in sorted(request['outers'],key=lambda x:(VARIANTS.index(x['variant']),x['outer_index'])):
        info,rows,steps=collect_outer(reader,configs[entry['variant']],entry);outers.append(info);features.extend(rows);batches.extend(steps)
    replicates=request.get('bootstrap_replicates',2000);require(type(replicates) is int and 0<=replicates<=10000,'Explicit bounded bootstrap budget required')
    require(len({(row['variant'],row['question_id']) for row in features})==len(features),'TRAIN cohort repeated a question within one variant')
    summary=aggregate_features(features,batches,replicates,42);summary.update(study_id=request['study_id'],complete_100_outers=complete,
        completed_outers=len(outers),created_unix=time.time(),scorer_rewards='Reused immutable accepted frozen-strict TRAIN summary, not rescored by this helper',
        no_gpu_no_http_no_model_load=True,full_input_refs_verified=len(reader.proofs))
    reader.stable_end();output.mkdir(parents=True)
    def write(name,value,lines=False):
        path=output/name
        with path.open('xb') as stream:
            if lines:
                for row in value:stream.write(canonical(row)+b'\n')
            else:stream.write(canonical(value)+b'\n')
            stream.flush();os.fsync(stream.fileno())
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    refs=dict(summary=write('summary.json',summary),outers=write('outers.jsonl',outers,True),
              private_question_features=write('private-question-features.jsonl',features,True),
              batches=write('batches.jsonl',batches,True),inputs=write('input-sha-audit.json',reader.proofs))
    reader.stable_end();proof=dict(accepted=True,study_id=request['study_id'],request=request_ref,outputs=refs,
        source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),complete_100_outers=complete)
    return write('complete.json',proof)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    print(json.dumps(run(dict(path=args.request,sha256=args.request_sha256),args.output_dir),sort_keys=True))

if __name__=='__main__':main()
