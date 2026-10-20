"""The retained 7B five-condition 50/100/150 scope, with unique repeat slots.

The scope stays byte-equivalent. Model identity consumes the new MN/MH or
AN/AH accepted proof for this exact group and step; no prediction acceptance.
"""
from pathlib import Path
import hashlib,json,re
SOURCE='b366989a90254813b5e790fd1f8c1fdfce7189d6'
SCIENCE='e265dae594e8f0368f889815d7f6c4729257ce6c'
PROFILE='qwen25_7b_common_eval_v2_20261019'
DATA_SHA='a1b95766beecb66753b5ee4c3dd2552d6a47e375c4c117200248dae6b6fc08d5'
ID_GOLD='4953085d4fdaf823fe355972d8dceae9e167e27d264f0f6b7f8bcf64473c6845'
BASE=Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-7B-Ablations-Resume-20261020-v1')
GROUPS={'C1-root-only':'c1','C2-random-final':'c2','budget-5':'b5','budget-20':'b20','generation-2':'d2'}
STEPS=(50,100,150)
FIXED3={('budget-5',100),('budget-5',150),('budget-20',150),('generation-2',150)}
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def repeat_count(group,step):
    assert group in GROUPS and type(step) is int and step in STEPS
    return 3 if (group,step) in FIXED3 else 1
def slot(group,step,repeat):
    count=repeat_count(group,step)
    assert type(repeat) is int and 1<=repeat<=count
    prefix=BASE/'evaluations/ablation-commonv2-v1'/group/('step-%06d'%step)
    return {'group':group,'actual_training_updates':step,'repeat_index':repeat,'repeat_count':count,
            'job_name':'zhangjunming-abl7lf1-%s-s%03d-r%d'%(GROUPS[group],step,repeat),
            'root':str(prefix/('r'+str(repeat))),'model':str(BASE/'fresh-runs'/group/'checkpoints/actor'/('global_step_'+str(step))),
            'profile':PROFILE,'data_sha256':DATA_SHA,'id_gold_identity':ID_GOLD,'questions':51713,
            'greedy':True,'engine_seed':0,'no_auto_retry':True,'gpus':8,'cpus':64,'memory_gib':600}
def all_slots():
    return [slot(g,s,r) for g in GROUPS for s in STEPS for r in range(1,repeat_count(g,s)+1)]
def validate_slot(value):
    expected=slot(value['group'],value['actual_training_updates'],value['repeat_index'])
    assert value==expected,'A fixed slot, method, seed, resource or scientific protocol changed'
    return expected
def statistic_scope(group,step):
    n=repeat_count(group,step)
    return {'n':n,'ddof':1,'sample_variance':None,'sample_SD':None,
            'variance_status':'UNAVAILABLE_N1' if n==1 else 'WAIT_THREE_INDEPENDENT_ACCEPTED_RUNS',
            'per_question_dispersion_is_not_repeat_variance':True}
def model_binding(value,submission,accepted_HF,HF_proof_sha):
    validate_slot(value);g=value['group'];step=value['actual_training_updates']
    assert submission['group']==g and submission['fresh_full150'] is True
    assert submission['source']==SOURCE and submission['science_source']==SCIENCE
    assert submission['job']=='zhangjunming-7b-abl-'+GROUPS[g]+'-lfresh1' and str(submission['job_id']).isdigit()
    assert submission['root']==str(BASE/'fresh-runs'/g) and re.fullmatch('[a-f0-9]{64}',submission['plan_sha256'])
    expected_status = ('NEW_FRESH_ABLATION_FINAL150_HF_NATIVE_PROJECTION_AND_FULL339_CPU_RELOAD_ACCEPTED' if step==150 else 'NEW_FRESH_ABLATION_MILESTONE_HF_NATIVE_PROJECTION_AND_FULL339_CPU_RELOAD_ACCEPTED')
    assert accepted_HF['status']==expected_status
    if step != 150:
        assert accepted_HF['step']==accepted_HF['completed_actual_rounds']==step and accepted_HF['actual_training_limit_remains']==150 and accepted_HF['native_boundary']=='post-evaluation'
    assert accepted_HF['group']==g and accepted_HF['job_id']==submission['job_id'] and accepted_HF['hf']==value['model']
    assert accepted_HF['plan_sha256']==submission['plan_sha256'] and accepted_HF['entry_source']==SOURCE and accepted_HF['science_source']==SCIENCE
    assert accepted_HF['local_raw_sha_verified'] and (accepted_HF['post_terminal_recheck_unchanged'] if step==150 else accepted_HF['post_parent_and_payload_recheck_passed']) and accepted_HF['not51713_prediction_acceptance']
    assert accepted_HF['reload']['logical_parameters']==accepted_HF['native_projection']['logical_parameters']==339
    assert accepted_HF['reload']['all_tensors_finite_bf16_and_exact_saved_weight_bf16_projection']
    assert accepted_HF['native_projection']['native_ranks']==8 and accepted_HF['native_projection']['all_logical_elements_exact_native_BF16_projection']
    assert accepted_HF['canonical_encoder_and_vocab_verified'] and accepted_HF['hf_identity']['files']
    assert re.fullmatch('[a-f0-9]{64}',HF_proof_sha)
    identity={'method':'CGRTS-7B-ablation-'+g,'training_job':submission['job_id'],'training_plan':submission['plan_sha256'],
              'entry_source':SOURCE,'science_source':SCIENCE,'actual_training_updates':step,'model_files':accepted_HF['hf_identity']['files'],
              'native_manifest_sha256':accepted_HF['native_manifest_sha256'],'HF_local_proof_sha256':HF_proof_sha,
              'profile':PROFILE,'data_sha256':DATA_SHA,'id_gold_identity':ID_GOLD}
    return digest(identity)
