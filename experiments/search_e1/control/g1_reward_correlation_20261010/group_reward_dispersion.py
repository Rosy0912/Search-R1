"""Read-only remote CPU audit of reconstructed reward dispersion magnitude.

Consumes one SHA-bound selected8 full summary and its original-group JSONL.
Exports aggregates only. No output files, model, Torch, HTTP or GPU operation.
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
THRESHOLDS=[0.,1e-6,1e-4,1e-3,1e-2,.1]
CLASSES=['all_C1','mixed_C','all_C0']


class Hold(Exception): pass


def require(ok,reason):
    if not ok:raise Hold(reason)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def bound_path(path):
    p=Path(path)
    require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'invalid_bound_input_path')
    return p


def sample_std(values):
    require(len(values)==5 and all(type(x) in (int,float) and math.isfinite(x) for x in values),'invalid_five_reward_values')
    mean=math.fsum(values)/5
    return math.sqrt(math.fsum((x-mean)**2 for x in values)/4)


def quantile(values,p):
    position=(len(values)-1)*p;lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def distribution(values):
    values=sorted(values)
    result=dict(N=len(values),mean=None,p10=None,p25=None,p50=None,p75=None,p90=None,max=None,
        counts_strictly_greater=[dict(threshold=t,count=sum(x>t for x in values)) for t in THRESHOLDS])
    if values:
        result.update(mean=math.fsum(values)/len(values),max=values[-1])
        result.update({name:quantile(values,p) for name,p in [('p10',.1),('p25',.25),('p50',.5),('p75',.75),('p90',.9)]})
    return result


def row_identity(row,run_id):
    step=row['step'];parent=row['original_parent_occurrence'];attempt=row['attempt_id'];upstream=row['upstream_group_sha256']
    require(row['run_id']==run_id and type(step) is int and step in STEPS and type(parent) is int and 0<=parent<512,
        'group_native_run_step_or_parent_mismatch')
    require(re.fullmatch('[0-9a-f]{32}',attempt) is not None and re.fullmatch('[0-9a-f]{64}',upstream) is not None,
        'invalid_original_group_identity')
    require(row['original_sibling_indices']==list(range(5)) and len(row['member_trajectory_ids'])==5,'incomplete_five_siblings')
    expected=[f'{run_id}-{step}-attempt-{attempt}-{parent}-{upstream}-sample-{i}' for i in range(5)]
    require(row['member_trajectory_ids']==expected,'original_sibling_ID_mismatch')
    C=row['C_raw'];require(len(C)==5 and all(type(x) is int and x in (0,1) for x in C),'nonbinary_group_C')
    category='all_C1' if sum(C)==5 else ('all_C0' if sum(C)==0 else 'mixed_C')
    expected_outcome='all_C0_including_invalid' if category=='all_C0' else category
    require(row['outcome']==expected_outcome and row['C_hits']==sum(C),'stored_group_outcome_mismatch')
    std=sample_std(row['R_reconstructed']);stored=row['R']['std_sample_correction1']
    require(type(stored) in (int,float) and math.isfinite(stored) and stored>=0 and
        abs(std-stored)<=1e-12*max(1.,std),'stored_sample_std_reconstruction_mismatch')
    require(row['R']['zero_variance_exact_values'] is (min(row['R_reconstructed'])==max(row['R_reconstructed'])),
        'stored_exact_zero_variance_mismatch')
    return step,parent,attempt,category,std


def analyze(summary_path,summary_sha,groups_sha):
    summary_path=bound_path(summary_path)
    require(summary_path.stat().st_size<=32*1024*1024 and file_sha(summary_path)==summary_sha,'full_summary_SHA_mismatch')
    summary=json.loads(summary_path.read_bytes())
    require(summary['status']=='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS' and
        summary['selected_steps']==STEPS and summary['trajectories']==20480 and
        summary['original_five_sibling_groups']==4096 and summary['original_online_R_available'] is False,
        'full_summary_not_selected8_accepted')
    group_path=bound_path(summary['remote_group_artifact_path'])
    require(group_path.parent==summary_path.parent and summary['remote_group_artifact_sha256']==groups_sha and
        re.fullmatch('[0-9a-f]{64}',groups_sha) is not None,'group_artifact_binding_mismatch')
    state={step:dict(parents=set(),attempts=set(),classes={name:[] for name in CLASSES}) for step in STEPS}
    before=group_path.stat();digest=hashlib.sha256();rows=0;run_id=None
    with group_path.open('rb') as stream:
        while True:
            raw=stream.readline(64*1024+1)
            if not raw:break
            require(len(raw)<=64*1024 and raw.endswith(b'\n') and raw.strip(),'partial_or_oversized_group_record')
            digest.update(raw);row=json.loads(raw)
            if run_id is None:
                run_id=row['run_id'];require(re.fullmatch('[0-9a-f]{32}',run_id) is not None,'invalid_original_run_ID')
            step,parent,attempt,category,std=row_identity(row,run_id);s=state[step]
            require(parent not in s['parents'],'duplicate_original_group')
            s['parents'].add(parent);s['attempts'].add(attempt);s['classes'][category].append(std);rows+=1
    after=group_path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and digest.hexdigest()==groups_sha,
        'complete_group_file_SHA_or_identity_mismatch')
    require(rows==4096 and all(s['parents']==set(range(512)) and len(s['attempts'])==1 for s in state.values()),
        'incomplete_selected8_512_group_coverage')
    summaries={s['step']:s for s in summary['steps']};require(sorted(summaries)==STEPS,'summary_step_coverage_mismatch')
    result=[]
    for step,s in state.items():
        expected=summaries[step]['group_outcomes']
        counts={name:len(v) for name,v in s['classes'].items()}
        require(counts==dict(all_C1=expected['all_C1'],mixed_C=expected['mixed_C'],all_C0=expected['all_C0_including_invalid']),
            'class_counts_full_summary_mismatch')
        result.append(dict(step=step,original_groups=512,classes={name:distribution(v) for name,v in s['classes'].items()}))
    require(file_sha(summary_path)==summary_sha,'full_summary_changed_during_read')
    return dict(status='ACCEPTED_SELECTED8_GROUP_REWARD_DISPERSION_MAGNITUDE',selected_steps=STEPS,
        original_groups=4096,individual_trajectories=20480,steps=result,full_summary_sha256=summary_sha,
        original_group_artifact_sha256=groups_sha,all_C0_includes_invalid=True,
        sample_std_correction1=True,statistics='binary64 reconstructed five-value statistics; not actual native advantages',
        threshold_comparison='strictly greater, no epsilon replacement for exact zero',
        quantiles='linear sample quantiles at (N-1)*p',raw_IDs_members_not_exported=True,
        selected_only_not_dense_1_to_150=True,CPU_only_no_model_GPU_HTTP_or_output_files=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['summary','summary-sha256','groups-sha256','expected-node','helper-sha256']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        # A root-published stdin executor may provide the exact bytes it hashes
        # before compile/exec. This permits strictly read-only Node execution
        # without deploying a file or trusting a fabricated __file__ value.
        source=globals().get('__published_helper_bytes__')
        if source is not None:
            require(type(source) is bytes and hashlib.sha256(source).hexdigest()==args.helper_sha256,
                'unpublished_dispersion_helper')
        else:
            require(file_sha(__file__)==args.helper_sha256,'unpublished_dispersion_helper')
        require(sys.platform.startswith('linux') and socket.gethostname()==args.expected_node and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,'not_bound_remote_CPU_process')
        result=analyze(args.summary,args.summary_sha256,args.groups_sha256)
        result['helper_source_sha256']=args.helper_sha256
        encoded=json.dumps(result,separators=(',',':'),allow_nan=False).encode()
        require(len(encoded)<=20*1024,'compact_dispersion_exceeds_20KiB')
        print(encoded.decode());return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD',reason=str(error) if isinstance(error,Hold) else type(error).__name__,
            raw_exception_not_exported=True)));return 2


if __name__=='__main__':raise SystemExit(main())
