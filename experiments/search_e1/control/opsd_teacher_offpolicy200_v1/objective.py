"""All-vocabulary reverse-KL on fixed same-D targets, clip then weight.

Imports are CPU-only. Actual forward records are distinct from sampling.
"""
import hashlib
import json
import math
from contextlib import nullcontext

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_component_clip(value):
    if value is not None and (isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0):
        raise ValueError('Clip requires finite positive value or explicit None')
    return value

def encoded_pair(tokenizer, pair, core, max_context=8192):
    """Use the frozen prompt and response_encoding; never truncate either side."""
    spans = pair.get('student_policy_spans')
    if spans is None:
        raise ValueError('Original student policy span provenance is required')
    previous = 0
    for start, end in spans:
        if not previous <= start <= end <= len(pair['student']):
            raise ValueError('Invalid policy span provenance')
        previous = end
    response, mask = core.response_encoding(tokenizer, pair['student'], spans)
    prefixes = [tokenizer.apply_chat_template(core.prompt(pair['question'], ref),
                   tokenize=True, add_generation_prompt=True)
                for ref in (None, pair['reference'])]
    offsets = tokenizer(pair['student'], add_special_tokens=False,
                        return_offsets_mapping=True)
    if offsets['input_ids'] != response or len(mask) != len(response):
        raise ValueError('Response token/mask alignment changed')
    if not response:
        return None, 'empty_response'
    if not sum(mask):
        return None, 'empty_policy_mask'
    if any(len(p) + len(response) > max_context for p in prefixes):
        return None, 'context_limit'
    identity = dict(question_id=pair['question_id'], pair_row_sha256=canonical_sha(pair),
                    reference_response_sha256=hashlib.sha256(pair['reference'].encode()).hexdigest(),
                    student_response_sha256=hashlib.sha256(pair['student'].encode()).hexdigest(),
                    reference_reward=pair['reference_reward'], student_reward=pair['student_reward'])
    if identity['reference_reward'] != 1:
        raise ValueError('Teacher reference must be a successful original trajectory')
    encoded = dict(prefixes=prefixes, response=response, mask=mask,
                   offsets=offsets['offset_mapping'], identity=identity)
    encoded['sha256'] = canonical_sha(encoded)
    return encoded, None


def vocabulary_metrics(student_logits,teacher_logits,component_clip=10.,old_logits=None,alpha=.1,use_mixture=None):
    import torch
    validate_component_clip(component_clip)
    if not 0<=alpha<=1:raise ValueError('Invalid mixture alpha')
    s=torch.log_softmax(student_logits.float(),dim=-1)
    t=torch.log_softmax(teacher_logits.detach().float(),dim=-1)
    o=None if old_logits is None else torch.log_softmax(old_logits.detach().float(),dim=-1)
    mixed=(o is not None) if use_mixture is None else bool(use_mixture)
    if mixed and o is None:raise ValueError('Mixed target requires frozen outer-old logits')
    target=t if not mixed or alpha==1 else o if alpha==0 else torch.logaddexp(t+math.log(alpha),o+math.log1p(-alpha))
    target=target.detach();p=s.exp();q=t.exp();a=target.exp()
    if any(not torch.isfinite(x).all().item() for x in (s,t,target)) or (o is not None and not torch.isfinite(o).all().item()):
        raise FloatingPointError('Nonfinite full-vocabulary log distributions')
    for distribution in (p,q,a):
        if not torch.allclose(distribution.sum(-1),torch.ones_like(distribution.sum(-1)),atol=2e-5,rtol=2e-5):
            raise FloatingPointError('Probability sum differs from one')
    components=p*(s-target) # BOTH exp(s) and s retain gradient.
    hit=torch.zeros_like(components,dtype=torch.bool) if component_clip is None else components>component_clip
    post=components if component_clip is None else components.clamp(max=component_clip)
    return dict(raw=components.sum(-1),post=post.sum(-1),clipped_count=hit.sum(-1),
        clipped_student_mass=(p*hit).sum(-1),clipped_teacher_mass=(q*hit).sum(-1),
        clipped_mixture_mass=(a*hit).sum(-1),student_logp=s,teacher_logp=t,
        old_logp=o,target_logp=target,student_to_teacher_raw_kl=(p*(s-t)).sum(-1),student_to_outer_old_raw_kl=None if o is None else (p*(s-o)).sum(-1),student_to_mixture_raw_kl=(p*(s-target)).sum(-1))


def pair_objective(model,encoded,component_clip,chunk,global_actual_pairs,
                   record=None,record_topk_positions=(),tokenizer=None,*,precision=None,targets=None,pair_weight=1.,alpha=.1,use_mixture=False,vocabulary_record=None,vocabulary_positions=()):
    import torch
    from torch.utils.checkpoint import checkpoint
    try:
        from .precision import PrecisionContract
        from .targets import SharedTargets
    except ImportError:
        from precision import PrecisionContract
        from targets import SharedTargets
    if chunk!=32 or global_actual_pairs<=0 or not math.isfinite(pair_weight) or pair_weight<=0:
        raise ValueError('chunk32, positive actual batch/weight required')
    precision=precision or PrecisionContract()
    targets=targets or SharedTargets(model,precision)
    if not targets.busy:raise ValueError('Trainer must hold targets.current_graph through backward')
    student,teacher,old=targets.hidden(encoded)
    head=model.get_base_model().get_output_embeddings()
    if any(p.requires_grad for p in head.parameters()):raise ValueError('Frozen output head required')
    positions=torch.tensor(encoded['mask'],device=student.device,dtype=torch.bool).nonzero().flatten()
    count=len(positions)
    full_wanted=set(vocabulary_positions)
    if not count:raise ValueError('Empty policy mask must be filtered before forward')
    loss=student.float().sum()*0.;rawsum=postsum=0.;teachersum=oldsum=mixturesum=0.;hitcount=0;wanted=set(record_topk_positions)
    for indices in positions.split(chunk):
        top=any(int(i) in wanted for i in indices.detach().cpu())
        full_rows=[r for r,pos in enumerate(indices.detach().cpu().tolist()) if pos in full_wanted]
        def calculate(s,t,o,top=top,indices=indices,full_rows=full_rows):
            with precision.context('current_head'):sl=head(s)
            with precision.context('teacher_head'):tl=head(t)
            with precision.context('outer_old_head'):ol=None if o is None else head(o)
            m=vocabulary_metrics(sl,tl,component_clip,ol,alpha,use_mixture)
            distributions=[m[k] for k in ('student_logp','teacher_logp','old_logp','target_logp')]
            outputs=(m['post'].sum(),m['raw'].detach(),m['post'].detach(),m['clipped_count'].detach(),
                m['clipped_student_mass'].detach(),m['clipped_teacher_mass'].detach(),m['clipped_mixture_mass'].detach(),m['student_to_teacher_raw_kl'].detach(),m['student_to_mixture_raw_kl'].detach())
            if o is not None:outputs+=(m['student_to_outer_old_raw_kl'].detach(),)
            # Observed D-token logp and entropy are retained for EVERY valid token.
            tokenids=torch.tensor(encoded['response'],device=s.device)[indices]
            for d in distributions:
                if d is None:continue
                outputs+=(d.detach().gather(-1,tokenids[:,None]).squeeze(-1),-(d.detach().exp()*d.detach()).sum(-1))
                if top:
                    v,i=torch.topk(d.detach(),min(32,d.shape[-1]),dim=-1);outputs+=(v,i)
            if full_rows:
                outputs+=tuple(d.detach()[full_rows].contiguous() for d in distributions if d is not None)
            return outputs
        values=checkpoint(calculate,student[indices],teacher[indices],None if old is None else old[indices],use_reentrant=False)
        if any(not torch.isfinite(v).all().item() for v in values if v.is_floating_point()):raise FloatingPointError('Nonfinite reverse loss/metrics')
        loss=loss+values[0];raw,post,hits,pm,qm,mm=[v.detach().cpu().tolist() for v in values[1:7]]
        rawsum+=sum(raw);postsum+=sum(post);hitcount+=sum(hits)
        teacher_drift,mixture_drift=[v.detach().cpu().tolist() for v in values[7:9]]
        old_drift=None if old is None else values[9].detach().cpu().tolist()
        teachersum+=sum(teacher_drift);mixturesum+=sum(mixture_drift)
        if old_drift is not None:oldsum+=sum(old_drift)
        cursor=9+int(old is not None);dmetrics={}
        for name in ('student','teacher','outer_old','mixture'):
            if name=='outer_old' and old is None:continue
            dmetrics[name]=[v.detach().cpu().tolist() for v in values[cursor:cursor+(4 if top else 2)]];cursor+=4 if top else 2
        if full_rows and vocabulary_record:
            full={name:value.detach().cpu() for name,value in zip(dmetrics,values[cursor:])}
            for full_row,local_row in enumerate(full_rows):
                position=int(indices[local_row])
                vocabulary_record(dict(token_index=position,token_id=encoded['response'][position],
                    scope='actual_training_forward_full_fp32_logp_not_generation_scores',
                    **{('target' if name=='mixture' else name)+'_logp':value[full_row].clone() for name,value in full.items()},
                    outer_old_present=old is not None))
        if record:
            for row,pos in enumerate(indices.detach().cpu().tolist()):
                item=dict(token_index=pos,token_id=encoded['response'][pos],token_offsets=encoded['offsets'][pos],policy_mask=True,
                    pre_clip_full_vocabulary_kl=raw[row],post_clip_full_vocabulary_loss=post[row],student_to_teacher_raw_kl=teacher_drift[row],student_to_outer_old_raw_kl=None if old_drift is None else old_drift[row],student_to_mixture_raw_kl=mixture_drift[row],
                    clipped_vocabulary_count=hits[row],clipped_vocabulary_fraction=hits[row]/head.weight.shape[0],vocabulary_size=head.weight.shape[0],
                    clipped_student_probability_mass=pm[row],clipped_teacher_probability_mass=qm[row],clipped_mixture_probability_mass=mm[row],
                    trajectory_contribution=post[row]/count,global_batch_contribution=pair_weight*post[row]/count/global_actual_pairs,
                    target_kind='mixture' if use_mixture else 'teacher',pair_weight=pair_weight,valid_trajectory_tokens=count,global_actual_pairs=global_actual_pairs,
                    direction='student_to_fixed_target_reverse',loss_scope='actual_training_forward_before_optimizer',
                    teacher_forced_fixed_student_prefix=True,actual_generation_scores=False,outer_old_identity=targets.old_identity)
                for name,metrics in dmetrics.items():
                    item[name+'_observed_token_logp']=metrics[0][row];item[name+'_entropy']=metrics[1][row]
                    if top and pos in wanted:item[name+'_top32_logp']=metrics[2][row];item[name+'_top32_ids']=metrics[3][row]
                if tokenizer is not None:item['token_text']=tokenizer.decode([encoded['response'][pos]])
                record(item)
    return pair_weight*loss/count,dict(valid_tokens=count,raw_loss=rawsum/count,post_loss=postsum/count,
        weighted_post_loss=pair_weight*postsum/count,student_to_teacher_raw_kl=teachersum/count,student_to_outer_old_raw_kl=None if old is None else oldsum/count,student_to_mixture_raw_kl=mixturesum/count,target_kind='mixture' if use_mixture else 'teacher',pair_weight=pair_weight,clipped_vocabulary_count=hitcount,direction='reverse')
