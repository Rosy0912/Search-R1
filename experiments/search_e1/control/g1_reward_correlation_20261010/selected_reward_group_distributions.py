"""Optional remote CPU-only distributions of eight original 512x5 TRAIN groups.

This consumes independently accepted reconstructed rewards. It neither trains
nor reruns a probe. Frozen bridge validation is repeated before raw processing.
Only compact distributions may be exported; member/group artifacts stay remote.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import sys

SELECTED_STEPS = [1, 20, 40, 60, 80, 100, 120, 140]
R_EDGES = [-.11, 0., .1, .5, .9, .98, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.61]
PROBE_EDGES = [i/10 for i in range(11)]


class Hold(Exception):
    """Only fixed, non-private errors are reported."""


def require(ok, reason):
    if not ok: raise Hold(reason)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def load_bridge(path, expected):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink() and
            file_sha(path)==expected, 'bridge_source_binding_mismatch')
    spec=importlib.util.spec_from_file_location('frozen_reward_statistics_bridge',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require('torch' not in sys.modules,'Torch_import_forbidden')
    return module


def quantile(sorted_values, p):
    """Linear sample quantile, identical to numpy quantile(method='linear')."""
    require(0<=p<=1 and bool(sorted_values),'invalid_quantile')
    position=(len(sorted_values)-1)*p; lo=math.floor(position); hi=math.ceil(position)
    return sorted_values[lo]+(sorted_values[hi]-sorted_values[lo])*(position-lo)


def moments(values):
    require(all(type(x) in (int,float) and math.isfinite(x) for x in values),'nonfinite_distribution_value')
    if not values: return dict(N=0,mean=None,std_population=None,min=None,p10=None,p50=None,p90=None,max=None)
    ordered=sorted(values); mean=math.fsum(values)/len(values)
    variance=math.fsum((x-mean)**2 for x in values)/len(values)
    return dict(N=len(values),mean=mean,std_population=math.sqrt(variance),min=ordered[0],
        p10=quantile(ordered,.1),p50=quantile(ordered,.5),p90=quantile(ordered,.9),max=ordered[-1])


def histogram(values, edges):
    require(len(edges)>=2 and edges==sorted(set(edges)),'invalid_histogram_edges')
    counts=[0]*(len(edges)-1)
    for value in values:
        require(math.isfinite(value) and edges[0]<=value<=edges[-1],'value_outside_histogram_support')
        for i,(left,right) in enumerate(zip(edges,edges[1:])):
            if left<=value<right or (i==len(counts)-1 and value==right):
                counts[i]+=1; break
        else: raise Hold('histogram_unassigned_value')
    require(sum(counts)==len(values),'histogram_count_mismatch')
    return dict(edges=edges,counts=counts,interval='[left,right), final interval includes right',N=len(values))


def group_key(row, run_id, step, attempt):
    prefix=f'{run_id}-{step}-attempt-{attempt}-'
    match=re.fullmatch(re.escape(prefix)+r'(\d+)-([0-9a-f]{64})-sample-([0-4])',row['trajectory_id'])
    require(match is not None,'group_native_ID_mismatch')
    occurrence, upstream, sibling=int(match[1]),match[2],int(match[3])
    require(0<=occurrence<512 and upstream==row['upstream_group_sha256'],'group_original_parent_mismatch')
    return occurrence,upstream,sibling


def group_statistics(rows):
    require(len(rows)==5,'group_not_five_original_siblings')
    C=[row['C_raw'] for row in rows]; R=[row['R_reconstructed'] for row in rows]
    require(all(type(x) is int and x in (0,1) for x in C),'nonbinary_group_C')
    require(all(type(row['final_action_valid']) is bool and (not row['C_raw'] or row['final_action_valid']) and
            type(row['model_authored_search_count']) is int and row['model_authored_search_count']>=0 for row in rows),
            'group_outcome_mismatch')
    C_stats=moments(C); R_stats=moments(R)
    for values,stats in [(C,C_stats),(R,R_stats)]:
        stats['std_sample_correction1']=stats['std_population']*math.sqrt(5/4)
        stats['range']=max(values)-min(values); stats['zero_variance_exact_values']=min(values)==max(values)
    invalid=sum(not row['final_action_valid'] for row in rows)
    correct=sum(C)
    outcome='all_C1' if correct==5 else ('all_C0_including_invalid' if correct==0 else 'mixed_C')
    if correct==0:
        subclass='all_invalid' if invalid==5 else ('all_valid_wrong' if invalid==0 else 'valid_wrong_and_invalid')
    else: subclass=None
    return dict(C_hits=correct,invalid_count=invalid,valid_wrong_count=5-correct-invalid,outcome=outcome,
        all_C0_subclass=subclass,C=C_stats,R=R_stats,
        C_zero_R_nonzero=C_stats['zero_variance_exact_values'] and not R_stats['zero_variance_exact_values'],
        C_nonzero_R_nonzero=not C_stats['zero_variance_exact_values'] and not R_stats['zero_variance_exact_values'],
        both_zero=C_stats['zero_variance_exact_values'] and R_stats['zero_variance_exact_values'],
        C_nonzero_R_zero=not C_stats['zero_variance_exact_values'] and R_stats['zero_variance_exact_values'])


def analyze_rows(rows, run_id, step, attempt):
    """Identity-valid rows only; original parent index and sibling order retained."""
    groups={}; ids=set()
    for row in rows:
        require(row['run_id']==run_id and type(row['step']) is int and row['step']==step and
                row['attempt_id']==attempt,'row_native_identity_mismatch')
        require(row['trajectory_id'] not in ids,'duplicate_distribution_row')
        ids.add(row['trajectory_id']); occurrence,upstream,sibling=group_key(row,run_id,step,attempt)
        if occurrence not in groups: groups[occurrence]=(upstream,{})
        require(groups[occurrence][0]==upstream and sibling not in groups[occurrence][1],'group_sibling_or_upstream_mismatch')
        require(all(type(row[name]) in (int,float) and math.isfinite(row[name]) and 0<=row[name]<=1
                    for name in ['b_reconstructed','d_reconstructed']),'invalid_distribution_probe')
        groups[occurrence][1][sibling]=row
    require(len(rows)==len(ids)==2560 and set(groups)==set(range(512)) and
            all(set(siblings)==set(range(5)) for _,siblings in groups.values()),'incomplete_512x5_distribution_coverage')
    categories={'all_C1':0,'all_C0_including_invalid':0,'mixed_C':0}
    subclasses={'all_invalid':0,'all_valid_wrong':0,'valid_wrong_and_invalid':0}
    signals={'C_zero_R_nonzero':0,'C_nonzero_R_nonzero':0,'both_zero':0,'C_nonzero_R_zero':0}
    class_signals={name:dict(C_zero_R_nonzero=0,C_nonzero_R_nonzero=0,both_zero=0,C_nonzero_R_zero=0) for name in categories}
    group_records=[]
    for occurrence,(upstream,siblings) in sorted(groups.items()):
        ordered=[siblings[i] for i in range(5)]; stats=group_statistics(ordered)
        categories[stats['outcome']]+=1
        if stats['all_C0_subclass'] is not None: subclasses[stats['all_C0_subclass']]+=1
        for name in signals:
            signals[name]+=int(stats[name]); class_signals[stats['outcome']][name]+=int(stats[name])
        group_records.append(dict(run_id=run_id,step=step,attempt_id=attempt,original_parent_occurrence=occurrence,
            upstream_group_sha256=upstream,member_trajectory_ids=[row['trajectory_id'] for row in ordered],
            original_sibling_indices=list(range(5)),C_raw=[row['C_raw'] for row in ordered],
            R_reconstructed=[row['R_reconstructed'] for row in ordered],
            b_reconstructed=[row['b_reconstructed'] for row in ordered],d_reconstructed=[row['d_reconstructed'] for row in ordered],
            **stats))
    C=[row['C_raw'] for row in rows]; R=[row['R_reconstructed'] for row in rows]
    predicates={
        'all':lambda row:True,
        'searched':lambda row:row['model_authored_search_count']>0,
        'no_search':lambda row:row['model_authored_search_count']==0,
        'strict_correct':lambda row:row['C_raw']==1,
        'strict_incorrect_including_invalid':lambda row:row['C_raw']==0,
        'valid_wrong':lambda row:row['C_raw']==0 and row['final_action_valid'],
        'invalid_final_action':lambda row:not row['final_action_valid'],
        'searched_strict_correct':lambda row:row['C_raw']==1 and row['model_authored_search_count']>0,
        'no_search_strict_correct':lambda row:row['C_raw']==1 and row['model_authored_search_count']==0}
    conditional={}
    for cohort,predicate in predicates.items():
        members=[row for row in rows if predicate(row)]
        conditional[cohort]=dict(N=len(members),C_hits=sum(row['C_raw'] for row in members),
            b=moments([row['b_reconstructed'] for row in members]),
            d=moments([row['d_reconstructed'] for row in members]),
            grounding_product_d_times_1_minus_b=moments([row['d_reconstructed']*(1-row['b_reconstructed']) for row in members]),
            R=moments([row['R_reconstructed'] for row in members]))
    require(sum(categories.values())==sum(signals.values())==512 and
            sum(subclasses.values())==categories['all_C0_including_invalid'],'group_category_count_mismatch')
    require(conditional['searched']['N']+conditional['no_search']['N']==2560 and
            conditional['strict_correct']['N']+conditional['strict_incorrect_including_invalid']['N']==2560 and
            conditional['strict_correct']['N']+conditional['valid_wrong']['N']+conditional['invalid_final_action']['N']==2560,
            'conditional_partition_count_mismatch')
    distribution={}
    for name in ['C','R']:
        distribution[name]={key:moments([record[name][key] for record in group_records])
            for key in ['mean','std_population','std_sample_correction1','range']}
    summary=dict(step=step,trajectory_count=2560,original_groups=512,siblings_per_group=5,
        pre_normalization=True,C_raw=moments(C),R_reconstructed=moments(R),
        R_histogram=histogram(R,R_EDGES),C_histogram=dict(values=[0,1],counts=[2560-sum(C),sum(C)],N=2560),
        b_histogram=histogram([row['b_reconstructed'] for row in rows],PROBE_EDGES),
        d_histogram=histogram([row['d_reconstructed'] for row in rows],PROBE_EDGES),
        group_outcomes=categories,all_C0_subclasses=subclasses,variance_signals=signals,
        variance_signals_by_outcome=class_signals,group_metric_distributions=distribution,conditional=conditional)
    return summary,group_records


def accepted_bridge_report(config,bridge,pairwise,C_helper):
    bridge_cfg,_=bridge.read_metadata(config['bridge_config_path'],config['bridge_config_sha256'])
    require(bridge_cfg['selected_steps']==SELECTED_STEPS,'not_human_selected8_scope')
    require(all(config[name]==bridge_cfg[name] for name in ['diagnostic_base','pairwise_source_path',
            'pairwise_source_sha256','C_helper_path','C_helper_sha256']), 'bridge_CPU_dependency_or_directory_mismatch')
    accepted,accepted_sha=bridge.read_metadata(config['accepted_pair_statistics_path'],config['accepted_pair_statistics_sha256'])
    require(accepted['status']=='ACCEPTED_SELECTED_TRAIN_STEPS_PAIR_CORRELATION' and
            accepted['selected_steps']==SELECTED_STEPS and accepted['total_pairs']==20480 and
            accepted['original_online_R_available'] is False,'missing_selected_pair_acceptance')
    fresh=bridge.analyze_scope(bridge_cfg,pairwise,C_helper)
    # Original native actual-updates.jsonl may have safely appended new TRAIN rows.
    # The bridge authenticates its contiguous original run and selected native gates
    # again. A different whole-ledger SHA therefore need not invalidate old selected
    # rows; ALL other identity, moments and source proofs must remain byte-equivalent.
    accepted_cmp=dict(accepted); fresh_cmp=dict(fresh)
    accepted_cmp.pop('original_ledger_read_sha256',None); fresh_cmp.pop('original_ledger_read_sha256',None)
    require(canonical(accepted_cmp)==canonical(fresh_cmp),'accepted_selected_pair_artifacts_changed')
    return bridge_cfg,accepted,accepted_sha,fresh['original_ledger_read_sha256']


def analyze(config,bridge,pairwise,C_helper):
    bridge_cfg,accepted,accepted_sha,current_ledger_sha=accepted_bridge_report(config,bridge,pairwise,C_helper)
    assignments={}
    for proof in accepted['source_proofs']:
        for step in proof['selected_steps']:
            require(step not in assignments,'duplicate_accepted_R_assignment'); assignments[step]=proof['R_directory']
    require(sorted(assignments)==SELECTED_STEPS,'accepted_source_map_not_selected8')
    proofs={proof['step']:proof for proof in accepted['step_proofs']}
    summaries=[]; all_group_records=[]; step_proofs=[]
    for step in SELECTED_STEPS:
        proof=proofs[step]; directory=Path(assignments[step]); path=directory/f'TRAIN-{step}-pairs-R-reconstructed.jsonl'
        rows=list(bridge.stream_jsonl(path,proof['R_row_artifact_fresh_sha256'],pairwise))
        summary,records=analyze_rows(rows,accepted['run_id'],step,proof['attempt_id'])
        require(pearson_ids(pairwise,rows)==proof['original_trajectory_ID_set_sha256'],'distribution_original_ID_set_mismatch')
        summary['probe_version']=rows[0]['probe_version']
        require(all(row['probe_version']==summary['probe_version'] and row['original_online_R_available'] is False for row in rows),
                'mixed_probe_version_or_online_provenance')
        require(file_sha(path)==proof['R_row_artifact_fresh_sha256'],'R_changed_after_distribution')
        summaries.append(summary); all_group_records.extend(records)
        step_proofs.append(dict(step=step,R_row_artifact_fresh_sha256=proof['R_row_artifact_fresh_sha256'],
            accepted_R_summary_sha256=proof['accepted_R_summary_sha256'],original_ID_set_sha256=proof['original_trajectory_ID_set_sha256']))
        del rows
    report=dict(status='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS',selected_steps=SELECTED_STEPS,
        trajectories=20480,original_five_sibling_groups=4096,steps=summaries,step_proofs=step_proofs,
        accepted_pair_statistics_sha256=accepted_sha,bridge_config_sha256=config['bridge_config_sha256'],
        current_original_native_ledger_read_sha256=current_ledger_sha,
        selected_only_not_TRAIN_population_1_to_150=True,no_interpolation_or_extrapolation=True,
        original_online_R_available=False,reconstructed_R_not_online_bitwise=True,probe_version_changes_not_policy_only_drift=True,
        TRAIN_batch_cohort_varies_by_step_not_fixed_question_panel=True,
        zero_variance_definition='all five actual values exactly equal; no epsilon threshold',
        statistics_arithmetic='binary64 CPU arithmetic over recorded native-FP32 reward values; not native Torch std bitwise replay',
        group_std_definitions=dict(population='sqrt(sum((x-mean)^2)/5)',sample_correction1='sqrt(sum((x-mean)^2)/4)'),
        probe_quantile_definition='linear sample quantiles at (N-1)*p',
        variance_signal_counts_are_not_gradient_magnitude=True,
        all_C0_includes_invalid_and_valid_wrong=True,
        d_is_observed_probe_output_not_verified_causal_evidence_dependence=True,
        grounding_product_is_rowwise_joint_product_not_product_of_marginal_means=True,
        grounding_product_enters_reward_only_for_valid_correct_searched_branch=True,
        conditioning_denominators_per_step_explicit=True,not_hacking_evidence_or_training_causal_effect=True,
        no_Pearson_mean_or_substitute_for_requested_pair_coefficient=True,
        group_artifacts_remote_only=True,CPU_only_no_probe_model_or_training=True)
    require(len(all_group_records)==4096 and 'torch' not in sys.modules,'final_group_count_or_CPU_failure')
    return report,all_group_records


def pearson_ids(pairwise,rows): return pairwise.trajectory_ids_sha256([row['trajectory_id'] for row in rows])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--helper-sha256',required=True);args=parser.parse_args()
    try:
        require(file_sha(__file__)==args.helper_sha256,'unpublished_group_helper')
        # Use stdlib only until authenticated immutable CPU module imports.
        raw=Path(args.config).read_bytes();require(hashlib.sha256(raw).hexdigest()==args.config_sha256,'group_config_SHA_mismatch')
        config=json.loads(raw)
        require(sys.platform.startswith('linux') and socket.gethostname()==config['expected_node'] and
                os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'not_bound_remote_CPU_process')
        D=Path(config['diagnostic_base']);out=Path(config['output_new'])
        require(D.is_absolute() and D.is_relative_to('/data/shaoshitong') and D.resolve()==D and
                out.resolve()==out and out.parent==D and not out.exists(),'output_not_fresh_remote_directory')
        bridge=load_bridge(config['bridge_source_path'],config['bridge_source_sha256'])
        pairwise=bridge.load_bound_module('frozen_pairwise_dependency_for_groups',config['pairwise_source_path'],config['pairwise_source_sha256'])
        C_helper=bridge.load_bound_module('frozen_C_gate_dependency_for_groups',config['C_helper_path'],config['C_helper_sha256'])
        report,records=analyze(config,bridge,pairwise,C_helper)
        out.mkdir(mode=0o700)
        groups=out/'original-group-statistics.jsonl'
        with groups.open('xb') as stream:
            for record in records:stream.write(canonical(record)+b'\n')
            stream.flush();os.fsync(stream.fileno())
        report['remote_group_artifact_sha256']=file_sha(groups);report['remote_group_artifact_path']=str(groups)
        summary=out/'compact-distributions.json'
        with summary.open('xb') as stream:stream.write(canonical(report));stream.flush();os.fsync(stream.fileno())
        print(json.dumps(dict(status=report['status'],selected_steps=SELECTED_STEPS,trajectories=20480,original_groups=4096,
            compact_summary_remote_path=str(summary),compact_summary_sha256=file_sha(summary),raw_groups_not_exported=True)))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD',reason=str(error) if isinstance(error,Hold) else type(error).__name__,
            raw_exception_not_exported=True)));return 2


if __name__=='__main__':raise SystemExit(main())
