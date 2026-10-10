"""Read-only binary64 reference GRPO coefficients on selected original five-sibling groups.

No native module import, model, Torch, GPU, HTTP or output file. Reconstructed
rewards and frozen group artifacts remain remote; stdout is compact aggregates.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

STEPS=[1,20,40,60,80,100,120,140]
EPSILON=1e-6


class Hold(Exception):pass


def require(ok,reason):
    if not ok:raise Hold(reason)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def identity(stat):return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns


def bound_path(path):
    p=Path(path);require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'invalid_bound_input_path')
    return p


def load_bound_group(source,expected):
    require(type(source) is bytes and hashlib.sha256(source).hexdigest()==expected,'bound_group_helper_source_SHA_mismatch')
    namespace={'__name__':'frozen_pairwise_order_dependency'}
    exec(compile(source,'<published-pairwise-order-dependency>','exec'),namespace)
    require(namespace['STEPS']==STEPS,'bound_group_selected_steps_mismatch')
    return namespace


def native_formula_proof(path,expected):
    """AST only: do not execute or import the actual native training module."""
    path=bound_path(path);before=path.stat();require(before.st_size<=4*1024*1024,'oversized_native_core_source')
    source=path.read_bytes();tree=ast.parse(source)
    functions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='compute_grpo_outcome_advantage']
    require(len(functions)==1,'native_GRPO_function_missing_or_ambiguous');fn=functions[0]
    dump=ast.dump(fn,include_attributes=False);digest=hashlib.sha256(dump.encode()).hexdigest()
    require(digest==expected,'native_GRPO_function_AST_SHA_mismatch')
    args=fn.args;names=[x.arg for x in args.args]
    require('epsilon' in names,'native_epsilon_argument_missing')
    default_index=names.index('epsilon')-(len(names)-len(args.defaults))
    require(default_index>=0 and isinstance(args.defaults[default_index],ast.Constant) and args.defaults[default_index].value==EPSILON,
        'native_epsilon_default_not_1e_6')
    std_calls=[node for node in ast.walk(fn) if isinstance(node,ast.Call) and ast.dump(node.func,include_attributes=False)==
        ast.dump(ast.parse('torch.std',mode='eval').body,include_attributes=False)]
    require(len(std_calls)==1 and len(std_calls[0].args)==1 and not std_calls[0].keywords,'native_std_not_default_correction1')
    expected_expression=ast.dump(ast.parse('(scores[i]-id2mean[index[i]])/(id2std[index[i]]+epsilon)',mode='eval').body,include_attributes=False)
    require(any(isinstance(node,ast.Assign) and ast.dump(node.value,include_attributes=False)==expected_expression for node in ast.walk(fn)),
        'native_normalization_expression_mismatch')
    expected_mask=ast.dump(ast.parse('scores.unsqueeze(-1).tile([1,response_length])*eos_mask',mode='eval').body,include_attributes=False)
    require(any(isinstance(node,ast.Assign) and ast.dump(node.value,include_attributes=False)==expected_mask for node in ast.walk(fn)),
        'native_shared_token_coefficient_mask_mismatch')
    expected_return=ast.dump(ast.parse('scores,scores',mode='eval').body,include_attributes=False)
    require(any(isinstance(node,ast.Return) and ast.dump(node.value,include_attributes=False)==expected_return for node in ast.walk(fn)),
        'native_advantage_return_mismatch')
    require(identity(before)==identity(path.stat()) and file_sha(path)==hashlib.sha256(source).hexdigest(),'native_core_changed_during_read')
    return dict(core_file_sha256=hashlib.sha256(source).hexdigest(),compute_grpo_outcome_advantage_AST_sha256=digest,
        epsilon_default=EPSILON,std_call_uses_default_correction1=True,std_call_has_no_correction_or_unbiased_override=True,
        sample_std_denominator_for_five=4,shared_scalar_tiled_then_eos_mask=True,return_advantages_and_returns_same=True,
        source_checked_by_AST_only_not_imported=True,native_torch_arithmetic_not_executed=True)


def normalize_five(values):
    require(len(values)==5 and all(type(x) in (int,float) and math.isfinite(x) for x in values),'invalid_original_five_values')
    if min(values)==max(values):return [0.]*5,0.,0.,0.
    mean=math.fsum(values)/5
    sigma=math.sqrt(math.fsum((x-mean)**2 for x in values)/4)
    require(math.isfinite(sigma) and sigma>0,'nonconstant_sigma_underflow_or_nonfinite')
    A=[(x-mean)/(sigma+EPSILON) for x in values]
    require(all(math.isfinite(x) for x in A),'nonfinite_normalized_reference')
    Amean=math.fsum(A)/5
    Avariance=math.fsum((x-Amean)**2 for x in A)/4
    return A,sigma,Avariance,sigma/(sigma+EPSILON)


def distribution(values):
    require(bool(values) and all(math.isfinite(x) for x in values),'empty_or_nonfinite_distribution')
    values=sorted(values);n=len(values)
    def quantile(q):
        position=(n-1)*q;lo=math.floor(position);hi=math.ceil(position)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return dict(N=n,min=values[0],mean=math.fsum(values)/n,p10=quantile(.1),p50=quantile(.5),p90=quantile(.9),max=values[-1])


class RewardMoments:
    def __init__(self):self.A=[];self.sigmas=[];self.group_variances=[];self.shrinks=[]
    def add(self,values):
        A,sigma,variance,shrink=normalize_five(values)
        self.A.extend(A);self.sigmas.append(sigma);self.group_variances.append(variance);self.shrinks.append(shrink)
        return A
    def merge(self,other):
        for name in ['A','sigmas','group_variances','shrinks']:getattr(self,name).extend(getattr(other,name))
    def report(self):
        n=len(self.A);require(n==5*len(self.sigmas) and n>0,'five_group_moment_coverage_failure')
        mean=math.fsum(self.A)/n;variance=math.fsum((a-mean)**2 for a in self.A)/n
        square=math.fsum(a*a for a in self.A);active=[s for s,sigma in zip(self.shrinks,self.sigmas) if sigma>0]
        return dict(N_trajectories=n,N_groups=len(self.sigmas),mean_A=mean,variance_population_A=variance,
            sum_A_squared=square,mean_A_squared=square/n,RMS_A=math.sqrt(square/n),mean_abs_A=math.fsum(abs(a) for a in self.A)/n,
            max_abs_A=max(abs(a) for a in self.A),sigma_nonzero_groups=len(active),sigma_exact_zero_groups=len(self.sigmas)-len(active),
            group_A_sample_variance_correction1=distribution(self.group_variances),
            epsilon_shrink_sigma_over_sigma_plus_epsilon_nonzero_groups=distribution(active) if active else None,
            zero_sigma_groups_use_exact_zero_A_mathematical_reference=True)


class PairedMoments:
    def __init__(self):self.C=RewardMoments();self.R=RewardMoments();self.products=[];self.active_added=0;self.active_removed=0
    def add(self,C,R):
        A_C=self.C.add(C);A_R=self.R.add(R);self.products.extend(c*r for c,r in zip(A_C,A_R))
        self.active_added+=int(self.C.sigmas[-1]==0 and self.R.sigmas[-1]>0)
        self.active_removed+=int(self.C.sigmas[-1]>0 and self.R.sigmas[-1]==0)
    def merge(self,other):
        self.C.merge(other.C);self.R.merge(other.R);self.products.extend(other.products)
        self.active_added+=other.active_added;self.active_removed+=other.active_removed
    def report(self):
        C,R=self.C.report(),self.R.report();require(C['N_trajectories']==R['N_trajectories']==len(self.products),'paired_moment_coverage_failure')
        cross=math.fsum(self.products)/len(self.products);cov=cross-C['mean_A']*R['mean_A']
        cv,rv=C['variance_population_A'],R['variance_population_A'];cq,rq=C['mean_A_squared'],R['mean_A_squared']
        return dict(C=C,R=R,C_R_mean_product_A=cross,C_R_covariance_population_A=cov,
            R_to_C_population_variance_ratio=rv/cv if cv>0 else None,R_to_C_RMS_ratio=math.sqrt(rq/cq) if cq>0 else None,
            R_to_C_sum_squared_ratio=R['sum_A_squared']/C['sum_A_squared'] if cq>0 else None,
            ratios_undefined_if_C_denominator_zero=cv==0,C_zero_R_active_groups=self.active_added,C_active_R_zero_groups=self.active_removed)


def analyze(summary_path,summary_sha,groups_sha,bound_group):
    """First frozen proof pass, then independently SHA-checked aggregate pass."""
    proof=bound_group['analyze'](summary_path,summary_sha,groups_sha) if isinstance(bound_group,dict) else bound_group.analyze(summary_path,summary_sha,groups_sha)
    bound_identity=bound_group['row_identity'] if isinstance(bound_group,dict) else bound_group.row_identity
    summary_path=bound_path(summary_path);raw=summary_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==summary_sha,'full_summary_changed_after_proof');summary=json.loads(raw)
    groups=bound_path(summary['remote_group_artifact_path']);require(groups.parent==summary_path.parent,'group_path_changed_after_proof')
    states={step:PairedMoments() for step in STEPS};parents={step:set() for step in STEPS};attempts={step:set() for step in STEPS}
    before=groups.stat();digest=hashlib.sha256();run_id=None;total=0
    with groups.open('rb') as stream:
        while True:
            raw=stream.readline(64*1024+1)
            if not raw:break
            require(len(raw)<=64*1024 and raw.endswith(b'\n') and raw.strip(),'partial_or_oversized_group_record')
            digest.update(raw);row=json.loads(raw)
            if run_id is None:run_id=row['run_id']
            step,parent,attempt,*unused=bound_identity(row,run_id)
            require(parent not in parents[step],'duplicate_original_group_in_second_pass');parents[step].add(parent);attempts[step].add(attempt)
            states[step].add(row['C_raw'],row['R_reconstructed']);total+=1
    require(identity(before)==identity(groups.stat()) and digest.hexdigest()==groups_sha,'group_SHA_or_identity_changed_in_second_pass')
    require(total==4096 and all(parents[s]==set(range(512)) and len(attempts[s])==1 for s in STEPS),'second_pass_incomplete_original_group_coverage')
    require(file_sha(summary_path)==summary_sha,'summary_changed_after_second_pass')
    pooled=PairedMoments();reports=[]
    for step in STEPS:
        report=states[step].report();require(report['C']['N_trajectories']==report['R']['N_trajectories']==2560,'per_step_trajectory_coverage_failure')
        reports.append(dict(step=step,**report));pooled.merge(states[step])
    result=pooled.report();require(result['C']['N_trajectories']==result['R']['N_trajectories']==20480,'pooled_trajectory_coverage_failure')
    return dict(status='ACCEPTED_SELECTED8_POSTHOC_NORMALIZED_ADVANTAGE_MOMENTS',selected_steps=STEPS,original_groups=4096,
        individual_trajectories=20480,steps=reports,pooled_selected_steps=result,
        full_summary_sha256=summary_sha,original_group_artifact_sha256=groups_sha,
        prior_frozen_original_group_proof_status=proof['status'],
        normalization='A=(x-fsum(x)/5)/(sqrt(fsum((x-fsum(x)/5)^2)/4)+1e-6); constant group A=0',
        arithmetic='posthoc binary64 mathematical reference, not historical native Torch advantages',
        epsilon=EPSILON,sample_std_correction=1,group_size_exactly_five_no_short_tail=True,
        pooled_moments_use_all_20480_trajectory_coefficients_not_mean_of_step_ratios=True,
        R_reconstructed_not_online_row_bitwise=True,selected_only_not_dense_1_to_150=True,
        coefficients_not_actual_gradients_or_PPO_clip_or_token_weighted_loss=True,
        active_group_increase_not_proof_of_higher_policy_gradient=True,raw_members_IDs_A_tensors_not_exported=True,
        CPU_only_no_model_GPU_HTTP_or_output_files=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['summary','summary-sha256','groups-sha256','expected-node','helper-sha256','group-helper-sha256',
        'native-core-algos','native-grpo-ast-sha256']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        source=globals().get('__published_helper_bytes__')
        if source is not None:require(type(source) is bytes and hashlib.sha256(source).hexdigest()==args.helper_sha256,'unpublished_normalized_moments_helper')
        else:require(file_sha(__file__)==args.helper_sha256,'unpublished_normalized_moments_helper')
        require(sys.platform.startswith('linux') and socket.gethostname()==args.expected_node and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            'torch' not in sys.modules,'not_bound_remote_CPU_process')
        bound_group=load_bound_group(globals().get('__bound_group_helper_bytes__'),args.group_helper_sha256)
        native=native_formula_proof(args.native_core_algos,args.native_grpo_ast_sha256)
        result=analyze(args.summary,args.summary_sha256,args.groups_sha256,bound_group)
        result.update(helper_source_sha256=args.helper_sha256,bound_group_helper_sha256=args.group_helper_sha256,native_GRPO_source_proof=native)
        encoded=json.dumps(result,separators=(',',':'),allow_nan=False).encode();require(len(encoded)<=20*1024,'compact_normalized_moments_exceeds_20KiB')
        print(encoded.decode());return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD',reason=str(error) if isinstance(error,Hold) else type(error).__name__,raw_exception_not_exported=True)));return 2


if __name__=='__main__':raise SystemExit(main())
