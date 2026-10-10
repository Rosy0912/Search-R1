"""Read-only remote CPU order checks on the eight original 512x5 groups.

Primary: among original same-question pairs with different binary C, does R
preserve the strict order? Ties are explicit; means never substitute for pairs.
All raw groups remain remote; only aggregate integer counts/rates are printed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys

STEPS=[1,20,40,60,80,100,120,140]
COUNTS=['concordant','discordant','C_tie_only','R_tie_only','both_tied']


class Hold(Exception):pass


def require(ok,reason):
    if not ok:raise Hold(reason)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def bound_path(path):
    p=Path(path);require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'invalid_bound_input_path')
    return p


def sign(x):return (x>0)-(x<0)


def group_counts(C,R):
    require(len(C)==len(R)==5 and all(type(x) is int and x in (0,1) for x in C) and
        all(type(x) in (int,float) and math.isfinite(x) for x in R),'invalid_original_five_C_R_values')
    result={key:0 for key in COUNTS}
    for i in range(5):
        for j in range(i+1,5):
            c,r=sign(C[i]-C[j]),sign(R[i]-R[j])
            key=('both_tied' if c==r==0 else ('C_tie_only' if c==0 else ('R_tie_only' if r==0
                else ('concordant' if c==r else 'discordant'))))
            result[key]+=1
    k=sum(C);comparable=k*(5-k)
    require(sum(result.values())==10 and result['concordant']+result['discordant']+result['R_tie_only']==comparable,
        'same_group_pair_decomposition_failure')
    mean=math.fsum(R)/5
    supplementary=dict(mixed_groups=int(0<k<5),C1_count_in_mixed=0,C1_R_below_group_mean=0,
        C1_R_equal_group_mean=0,C1_R_above_group_mean=0,C1_R_below_group_mean_margin_gt_1e_4=0)
    if 0<k<5:
        for c,r in zip(C,R):
            if c:
                supplementary['C1_count_in_mixed']+=1
                key='C1_R_'+('below' if r<mean else ('above' if r>mean else 'equal'))+'_group_mean'
                supplementary[key]+=1
                supplementary['C1_R_below_group_mean_margin_gt_1e_4']+=int(mean-r>1e-4)
    return result,comparable,supplementary


def rates(counts,comparable):
    P,Q,T,U,V=[counts[k] for k in COUNTS]
    require(P+Q+U==comparable,'strict_C_denominator_failure')
    denominator=(P+Q+T)*(P+Q+U)
    return dict(**counts,total_pairs=P+Q+T+U+V,C_strict_comparable=comparable,
        concordance_rate=P/comparable if comparable else None,
        strict_order_failures_including_R_ties=Q+U,
        non_tied_concordance_rate=P/(P+Q) if P+Q else None,
        half_R_tie_AUC_same_group=(P+.5*U)/comparable if comparable else None,
        optional_Kendall_tau_b=(P-Q)/math.sqrt(denominator) if denominator else None,
        Kendall_tau_b_is_different_tie_sensitive_criterion=True)


def global_correct_wrong_order(buckets):
    """O(M log M) score buckets, not an enumeration of N0*N1 cross pairs."""
    lower_wrong=lower_correct=P=Q=U=0
    for R,(wrong,correct) in sorted(buckets.items()):
        require(math.isfinite(R) and type(wrong) is type(correct) is int and wrong>=0 and correct>=0,
            'invalid_global_score_bucket')
        P+=correct*lower_wrong;Q+=wrong*lower_correct;U+=wrong*correct
        lower_wrong+=wrong;lower_correct+=correct
    total=lower_wrong*lower_correct;require(P+Q+U==total,'global_correct_wrong_pair_count_failure')
    return dict(N_correct=lower_correct,N_incorrect=lower_wrong,C_strict_pairs=total,
        concordant=P,discordant=Q,R_ties=U,strict_order_rate=P/total if total else None,
        AUROC=(P+.5*U)/total if total else None,AUROC_defined=bool(total),
        not_conditioned_on_same_original_question_group=True,not_primary_same_question_criterion=True,
        computation='sorted score buckets; no quadratic pair enumeration')


def row_identity(row,run_id):
    step=row['step'];parent=row['original_parent_occurrence'];attempt=row['attempt_id'];upstream=row['upstream_group_sha256']
    require(row['run_id']==run_id and type(step) is int and step in STEPS and type(parent) is int and 0<=parent<512,
        'group_native_run_step_or_parent_mismatch')
    require(re.fullmatch('[0-9a-f]{32}',attempt) is not None and re.fullmatch('[0-9a-f]{64}',upstream) is not None,
        'invalid_original_group_identity')
    require(row['original_sibling_indices']==list(range(5)) and len(row['member_trajectory_ids'])==5,'incomplete_five_siblings')
    require(row['member_trajectory_ids']==[f'{run_id}-{step}-attempt-{attempt}-{parent}-{upstream}-sample-{i}' for i in range(5)],
        'original_sibling_ID_mismatch')
    C,R=row['C_raw'],row['R_reconstructed'];counts,comparable,supplementary=group_counts(C,R)
    outcome='all_C1' if sum(C)==5 else ('all_C0_including_invalid' if sum(C)==0 else 'mixed_C')
    require(row['outcome']==outcome and row['C_hits']==sum(C),'stored_group_outcome_mismatch')
    mean=math.fsum(R)/5;std=math.sqrt(math.fsum((r-mean)**2 for r in R)/4)
    stored=row['R']['std_sample_correction1']
    require(type(stored) in (int,float) and math.isfinite(stored) and stored>=0 and abs(std-stored)<=1e-12*max(1.,std),
        'stored_sample_std_reconstruction_mismatch')
    require(row['R']['zero_variance_exact_values'] is (min(R)==max(R)),'stored_exact_zero_variance_mismatch')
    return step,parent,attempt,outcome,counts,comparable,supplementary


def analyze(summary_path,summary_sha,groups_sha):
    summary_path=bound_path(summary_path);require(summary_path.stat().st_size<=32*1024*1024,'oversized_summary')
    raw=summary_path.read_bytes();require(hashlib.sha256(raw).hexdigest()==summary_sha,'full_summary_SHA_mismatch')
    summary=json.loads(raw)
    require(summary['status']=='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS' and summary['selected_steps']==STEPS and
        summary['trajectories']==20480 and summary['original_five_sibling_groups']==4096 and
        summary['original_online_R_available'] is False,'full_summary_not_selected8_accepted')
    groups=bound_path(summary['remote_group_artifact_path'])
    require(groups.parent==summary_path.parent and summary['remote_group_artifact_sha256']==groups_sha and
        re.fullmatch('[0-9a-f]{64}',groups_sha) is not None,'group_artifact_binding_mismatch')
    state={step:dict(parents=set(),attempts=set(),counts={k:0 for k in COUNTS},comparable=0,C1=0,
        categories={k:0 for k in ['all_C1','mixed_C','all_C0_including_invalid']},
        supplementary={k:0 for k in ['mixed_groups','C1_count_in_mixed','C1_R_below_group_mean',
            'C1_R_equal_group_mean','C1_R_above_group_mean','C1_R_below_group_mean_margin_gt_1e_4']},buckets={}) for step in STEPS}
    before=groups.stat();digest=hashlib.sha256();total_rows=0;run_id=None
    with groups.open('rb') as stream:
        while True:
            raw=stream.readline(64*1024+1)
            if not raw:break
            require(len(raw)<=64*1024 and raw.endswith(b'\n') and raw.strip(),'partial_or_oversized_group_record')
            digest.update(raw);row=json.loads(raw)
            if run_id is None:run_id=row['run_id'];require(re.fullmatch('[0-9a-f]{32}',run_id) is not None,'invalid_original_run_ID')
            step,parent,attempt,outcome,counts,n,supplementary=row_identity(row,run_id);s=state[step]
            require(parent not in s['parents'],'duplicate_original_group');s['parents'].add(parent);s['attempts'].add(attempt)
            s['categories'][outcome]+=1;s['comparable']+=n;s['C1']+=sum(row['C_raw']);total_rows+=1
            for key in COUNTS:s['counts'][key]+=counts[key]
            for key,value in supplementary.items():s['supplementary'][key]+=value
            for C,R in zip(row['C_raw'],row['R_reconstructed']):
                if R not in s['buckets']:s['buckets'][R]=[0,0]
                s['buckets'][R][C]+=1
    after=groups.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and digest.hexdigest()==groups_sha,
        'complete_group_file_SHA_or_identity_mismatch')
    require(total_rows==4096 and all(s['parents']==set(range(512)) and len(s['attempts'])==1 for s in state.values()),
        'incomplete_selected8_512_group_coverage')
    summaries={s['step']:s for s in summary['steps']};require(sorted(summaries)==STEPS,'summary_step_coverage_mismatch')
    result=[];pooled={k:0 for k in COUNTS};pooled_comparable=0;global_buckets={}
    pooled_supplementary={key:0 for key in state[STEPS[0]]['supplementary']}
    for step,s in state.items():
        original=summaries[step]
        require(s['categories']==original['group_outcomes'] and s['C1']==original['conditional']['all']['C_hits'],
            'class_or_C1_counts_full_summary_mismatch')
        step_rates=rates(s['counts'],s['comparable']);require(step_rates['total_pairs']==5120,'step_ten_pairs_per_group_failure')
        require(sum(s['supplementary'][key] for key in ['C1_R_below_group_mean','C1_R_equal_group_mean','C1_R_above_group_mean'])==
            s['supplementary']['C1_count_in_mixed'],'mixed_C1_partition_failure')
        result.append(dict(step=step,original_groups=512,within_original_question=step_rates,
            supplementary_correct_vs_group_R_mean=s['supplementary'],
            supplementary_global_across_questions=global_correct_wrong_order(s['buckets'])))
        for key in COUNTS:pooled[key]+=s['counts'][key]
        for key in pooled_supplementary:pooled_supplementary[key]+=s['supplementary'][key]
        pooled_comparable+=s['comparable']
        for R,counts in s['buckets'].items():
            if R not in global_buckets:global_buckets[R]=[0,0]
            for c in [0,1]:global_buckets[R][c]+=counts[c]
    pooled_rates=rates(pooled,pooled_comparable);require(pooled_rates['total_pairs']==40960,'pooled_group_pair_count_failure')
    require(file_sha(summary_path)==summary_sha,'full_summary_changed_during_read')
    return dict(status='ACCEPTED_SELECTED8_ORIGINAL_QUESTION_PAIR_ORDER',selected_steps=STEPS,original_groups=4096,
        individual_trajectories=20480,steps=result,pooled_within_original_question=pooled_rates,
        pooled_supplementary_correct_vs_group_R_mean=pooled_supplementary,
        supplementary_global_across_questions=global_correct_wrong_order(global_buckets),
        full_summary_sha256=summary_sha,original_group_artifact_sha256=groups_sha,
        primary='sign(Ci-Cj)==sign(Ri-Rj), conditional on Ci!=Cj, original same-question unordered sibling pairs',
        primary_denominator_includes_R_ties=True,exact_recorded_R_float_ties_no_epsilon=True,
        each_group_C_strict_comparable_equals_k_times_5_minus_k=True,
        pooled_is_sum_of_pair_counts_not_mean_of_rates=True,
        pooled_global_combines_selected_steps=True,
        C0_includes_invalid_and_valid_wrong=True,selected_only_not_dense_1_to_150=True,
        statistics='binary64 posthoc reconstructed statistics; not actual native Torch advantages or gradients',
        supplementary_mean_is_fsum_R_divided_by_5=True,supplementary_margin_comparison_is_strict=True,
        perfect_cross_C_order_does_not_validate_C_tie_ranking_or_probe_utility=True,
        raw_IDs_members_not_exported=True,CPU_only_no_model_GPU_HTTP_or_output_files=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['summary','summary-sha256','groups-sha256','expected-node','helper-sha256']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        source=globals().get('__published_helper_bytes__')
        if source is not None:require(type(source) is bytes and hashlib.sha256(source).hexdigest()==args.helper_sha256,'unpublished_order_helper')
        else:require(file_sha(__file__)==args.helper_sha256,'unpublished_order_helper')
        require(sys.platform.startswith('linux') and socket.gethostname()==args.expected_node and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            'torch' not in sys.modules,'not_bound_remote_CPU_process')
        result=analyze(args.summary,args.summary_sha256,args.groups_sha256);result['helper_source_sha256']=args.helper_sha256
        encoded=json.dumps(result,separators=(',',':'),allow_nan=False).encode();require(len(encoded)<=20*1024,'compact_order_exceeds_20KiB')
        print(encoded.decode());return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD',reason=str(error) if isinstance(error,Hold) else type(error).__name__,raw_exception_not_exported=True)));return 2


if __name__=='__main__':raise SystemExit(main())
