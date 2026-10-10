"""Root-invoked, published CPU-only group-distribution deployment/execution.

No import or local fixture contacts a node. Row/group artifacts remain remote.
The accepted selected8 bridge report is an explicit SHA-bound dependency, never
inferred from a producer's status or a mutable latest pointer.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREFIX = 'experiments/search_e1/control/g1_reward_correlation_20261010/'
SELECTED = [1,20,40,60,80,100,120,140]
PAYLOAD = ['selected_reward_group_distributions.py', 'test_selected_reward_group_distributions.py']
DEPENDENCIES = ['selected_pair_statistics.py','pairwise_pearson.py','test_selected_pair_statistics.py']
C_SHA = 'c9385df75c06f3c7a28e41e0ae0b502866fa3418e54d3162ec0a7a2996732c34'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(raw): return hashlib.sha256(raw).hexdigest()


def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def context(cohort_path):
    p=Path(cohort_path).resolve();co=json.loads(p.read_bytes())
    assert p.is_relative_to(HERE) and co['selected_steps']==SELECTED and co['final_pair_count']==20480
    assert co['original_run_id']=='db474b6abf654ad9892b02ae67ab6f86'
    assert co['cohort_id']=='ff98085b6918423c9a6d0bdbc903bafa'
    D=Path(co['remote_parent']).parent;tag=co['cohort_id'][:12]
    assert str(D).startswith('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-')
    return dict(cohort=co,diagnostic_base=str(D),source=str(D/('group-statistics-source-'+tag+'-v1')),
        output=str(D/('group-statistics-selected-'+tag+'-v1')),
        stats_source=str(D/('statistics-source-'+tag+'-v2')),
        C_helper=str(D/'source/reconstruct_train_raw_c_preflight_v2.py'),
        payload_sha256={n:sha((HERE/n).read_bytes()) for n in PAYLOAD},
        dependency_sha256={n:sha((HERE/n).read_bytes()) for n in DEPENDENCIES})


def publication(path,names):
    pub=json.loads(Path(path).read_bytes());rows={r['path']:r['sha256'] for r in pub['files']}
    for name in names:
        assert rows[PREFIX+name]==sha((HERE/name).read_bytes()),'Unpublished/changed source: '+name
    return pub


def make_config(ctx,bridge_raw,report_path,report_sha):
    b=json.loads(bridge_raw);D=Path(ctx['diagnostic_base']);S=Path(ctx['stats_source'])
    assert b['selected_steps']==SELECTED and b['diagnostic_base']==str(D)
    assert b['expected_node']==ctx['cohort']['expected_node'] and b['native_run_id']==ctx['cohort']['original_run_id']
    assert b['original_base']==ctx['cohort']['base']
    assert b['pairwise_source_path']==str(S/'pairwise_pearson.py') and b['pairwise_source_sha256']==ctx['dependency_sha256']['pairwise_pearson.py']
    assert b['C_helper_path']==ctx['C_helper'] and b['C_helper_sha256']==C_SHA
    assert b['output_new']==str(D/'statistics-selected-ff98085b6918-v2')
    rp=Path(report_path)
    assert rp.is_absolute() and rp.parent==Path(b['output_new']) and rp.name=='selected-pair-statistics.json'
    assert re.fullmatch('[0-9a-f]{64}',report_sha)
    return dict(expected_node=b['expected_node'],diagnostic_base=str(D),output_new=ctx['output'],
        bridge_source_path=str(S/'selected_pair_statistics.py'),bridge_source_sha256=ctx['dependency_sha256']['selected_pair_statistics.py'],
        bridge_config_path=str(S/'config-selected8.json'),bridge_config_sha256=sha(bridge_raw),
        accepted_pair_statistics_path=str(rp),accepted_pair_statistics_sha256=report_sha,
        pairwise_source_path=b['pairwise_source_path'],pairwise_source_sha256=b['pairwise_source_sha256'],
        C_helper_path=b['C_helper_path'],C_helper_sha256=b['C_helper_sha256'],
        group_artifacts_remote_only=True,selected_scope_not_dense_1_to_150=True)


REMOTE_COMMON = r'''
import base64,datetime,hashlib,json,os,pathlib,re,socket,subprocess,sys
j=json.load(sys.stdin);ctx=j['context'];co=ctx['cohort'];D=pathlib.Path(ctx['diagnostic_base']);S=pathlib.Path(ctx['source']);B=pathlib.Path(ctx['stats_source'])
assert sys.platform.startswith('linux') and socket.gethostname()==co['expected_node'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert D.is_dir() and D.resolve()==D and D.stat().st_uid==os.getuid() and S.parent==D and S.resolve()==S
assert B.is_dir() and B.resolve()==B and B.stat().st_uid==os.getuid()
def fresh(path):
    p=pathlib.Path(path);assert p.is_file() and not p.is_symlink() and p.resolve()==p
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def metadata(path,expected=None):
    p=pathlib.Path(path);assert p.stat().st_size<=32*1024*1024
    h=fresh(p);assert expected is None or h==expected
    return json.loads(p.read_bytes()),h
assert fresh(ctx['C_helper'])==j['C_helper_sha256']
for n,h in ctx['dependency_sha256'].items():assert fresh(B/n)==h
def cpu_env():
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(B)+os.pathsep+str(pathlib.Path(ctx['C_helper']).parent),TMPDIR=str(S/'tmp'))
    return e
'''


REMOTE_DEPLOY = REMOTE_COMMON + r'''
assert not S.exists();S.mkdir(mode=0o700);(S/'tmp').mkdir()
records=[]
for row in j['files']:
    p=S/row['name'];assert p.parent==S and p.resolve()==p and not p.exists()
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert fresh(p)==row['sha256'];records.append(dict(name=row['name'],sha256=row['sha256'],bytes=len(raw)))
r=subprocess.run([co['base']+'/env/bin/python','-B','-m','unittest','discover','-s',str(S),'-p','test_selected_reward_group_distributions.py'],env=cpu_env(),capture_output=True,timeout=180)
for name,raw in [('fixtures.stdout.txt',r.stdout),('fixtures.stderr.txt',r.stderr)]:
    with (S/name).open('xb') as f:f.write(raw)
t=r.stderr.decode(errors='replace');accepted=r.returncode==0 and re.search(r'Ran 32 tests\b',t) is not None and '\nOK\n' in t and 'skipped' not in t
print(json.dumps(dict(phase='DEPLOY_GROUP_CPU_FIXTURES',accepted=accepted,actual_rc=r.returncode,CPU_fixtures=32,zero_skip=accepted,
    source=str(S),node=socket.gethostname(),uid=os.getuid(),files=records,fixture_summary=t[-1800:],GPU_started=False,original_sources_unmodified=True)))
raise SystemExit(0 if accepted else 2)
'''


def compact_report(report):
    """Explicitly selected distributions; full detail and all groups stay remote."""
    assert report['status']=='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS'
    assert report['selected_steps']==SELECTED and report['trajectories']==20480 and report['original_five_sibling_groups']==4096
    assert [s['step'] for s in report['steps']]==SELECTED
    rows=[]
    for s in report['steps']:
        assert s['trajectory_count']==2560 and s['original_groups']==512
        keys=['step','probe_version','trajectory_count','original_groups','C_raw','R_reconstructed','C_histogram','R_histogram',
              'group_outcomes','all_C0_subclasses','variance_signals','variance_signals_by_outcome']
        row={k:s[k] for k in keys}
        row['group_metric_distributions']={name:{metric:{k:v[k] for k in ['mean','p50']}
            for metric,v in stats.items()} for name,stats in s['group_metric_distributions'].items()}
        row['conditional']={name:dict(N=c['N'],C_hits=c['C_hits'],
            b_mean=c['b']['mean'],d_mean=c['d']['mean'],
            joint_d_times_1_minus_b_mean=c['grounding_product_d_times_1_minus_b']['mean'],R_mean=c['R']['mean'])
            for name,c in s['conditional'].items()}
        rows.append(row)
    return dict(selected_steps=SELECTED,trajectories=20480,original_five_sibling_groups=4096,steps=rows,
        selected_only_not_dense_1_to_150=True,reconstructed_R_not_online_bitwise=True,pre_normalization=True,
        raw_groups_not_exported=True,full_detailed_summary_remote_only=True,
        std_population_definition='sqrt(sum((x-mean)^2)/5)',std_sample_correction1_definition='sqrt(sum((x-mean)^2)/4)',
        exact_zero_variance_not_epsilon=True,variance_counts_not_gradient_magnitude=True,
        probe_versions_change_and_TRAIN_cohorts_differ=True,probe_d_not_causally_verified=True)


# Inject only the reviewed pure compact transform, not any local node controller.
import inspect
REMOTE_EXECUTE = REMOTE_COMMON + '\nSELECTED='+repr(SELECTED)+'\n'+inspect.getsource(compact_report)+r'''
assert S.is_dir()
for n,h in ctx['payload_sha256'].items():assert fresh(S/n)==h
raw=base64.b64decode(j['config_base64']);assert hashlib.sha256(raw).hexdigest()==j['config_sha256'];cfg=json.loads(raw)
assert cfg['diagnostic_base']==str(D) and cfg['output_new']==ctx['output'] and not pathlib.Path(cfg['output_new']).exists()
bc,_=metadata(cfg['bridge_config_path'],cfg['bridge_config_sha256']);accepted,_=metadata(cfg['accepted_pair_statistics_path'],cfg['accepted_pair_statistics_sha256'])
assert bc['selected_steps']==SELECTED and accepted['selected_steps']==SELECTED and accepted['total_pairs']==20480 and accepted['status']=='ACCEPTED_SELECTED_TRAIN_STEPS_PAIR_CORRELATION'
target=S/'config-group-selected8.json';assert not target.exists()
with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
r=subprocess.run([co['base']+'/env/bin/python','-B',str(S/'selected_reward_group_distributions.py'),
    '--config',str(target),'--config-sha256',j['config_sha256'],'--helper-sha256',ctx['payload_sha256']['selected_reward_group_distributions.py']],
    env=cpu_env(),capture_output=True,timeout=300)
for suffix,v in [('stdout.json',r.stdout),('stderr.txt',r.stderr)]:
    with (S/('execution-'+j['invocation_id']+'.'+suffix)).open('xb') as f:f.write(v)
assert r.returncode==0,'Group consumer HOLD; preserve independent diagnostics'
result=json.loads(r.stdout);assert result['status']=='ACCEPTED_SELECTED8_RECONSTRUCTED_REWARD_GROUP_DISTRIBUTIONS'
report,h=metadata(result['compact_summary_remote_path'],result['compact_summary_sha256'])
for n,v in ctx['payload_sha256'].items():assert fresh(S/n)==v
for n,v in ctx['dependency_sha256'].items():assert fresh(B/n)==v
compact=compact_report(report);compact.update(phase='ACCEPTED_SELECTED8_GROUP_CPU_STATISTICS',actual_rc=r.returncode,
    observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_summary_remote_path=result['compact_summary_remote_path'],
    full_summary_sha256=h,accepted_pair_statistics_sha256=cfg['accepted_pair_statistics_sha256'],config_sha256=j['config_sha256'],
    raw_group_artifact_sha256=report['remote_group_artifact_sha256'],GPU_started=False,original_sources_unmodified=True)
encoded=json.dumps(compact,separators=(',',':'),allow_nan=False).encode();assert len(encoded)<=32*1024
print(encoded.decode())
'''


def remote_call(code,payload,phase,timeout):
    sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
    token=stamp();payload['invocation_id']=token
    command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',payload['context']['cohort']['base']+'/env/bin/python','-B','-c',code]
    root=HERE/('group-statistics-control-'+phase+'-'+token)
    try:
        r=subprocess.run(control.route('baai-a800',shlex.join(command)),input=canonical(payload),capture_output=True,timeout=timeout)
        for suffix,raw in [('stdout.json',r.stdout),('stderr.txt',r.stderr)]:root.with_suffix('.'+suffix).write_bytes(raw)
        root.with_suffix('.execution.json').write_bytes(canonical(dict(phase=phase,rc=r.returncode,invocation_id=token,controller_sha256=sha(Path(__file__).read_bytes()))))
        assert r.returncode==0,'Preserve failed receipt; do not overwrite or infer completion'
        value=json.loads(r.stdout);print(json.dumps(value,ensure_ascii=False,separators=(',',':')));return value
    except subprocess.TimeoutExpired as e:
        root.with_suffix('.timeout.json').write_bytes(canonical(dict(phase=phase,status='TIMEOUT_RC_UNKNOWN',invocation_id=token)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['deploy','prepare','execute'],required=True)
    p.add_argument('--publication',required=True)
    p.add_argument('--cohort-config',default=str(HERE/'historical_replay/parallel_selected/cohort-config.json'))
    p.add_argument('--bridge-config',default=str(HERE/'statistics-config-prepared-v2.json'))
    p.add_argument('--accepted-report-remote-path');p.add_argument('--accepted-report-sha256')
    p.add_argument('--prepared-config',default=str(HERE/'group-statistics-config-v1.json'));args=p.parse_args()
    cohort=Path(args.cohort_config).resolve();assert cohort==HERE/'historical_replay/parallel_selected/cohort-config.json'
    names=['group_statistics_control_v1.py','test_group_statistics_control_v1.py']+PAYLOAD+DEPENDENCIES+[cohort.relative_to(HERE).as_posix()]
    if args.phase=='prepare':
        assert Path(args.bridge_config).resolve()==HERE/'statistics-config-prepared-v2.json';names+=['statistics-config-prepared-v2.json']
    if args.phase=='execute':names+=['group-statistics-config-v1.json']
    pub=publication(args.publication,names);ctx=context(cohort);out=Path(args.prepared_config)
    assert out.resolve()==HERE/'group-statistics-config-v1.json'
    payload=dict(context=ctx,C_helper_sha256=C_SHA,source_commit=pub['source_commit'])
    if args.phase=='deploy':
        payload['files']=[dict(name=n,sha256=ctx['payload_sha256'][n],base64=base64.b64encode((HERE/n).read_bytes()).decode()) for n in PAYLOAD]
        remote_call(REMOTE_DEPLOY,payload,'deploy',230)
    elif args.phase=='prepare':
        assert not out.exists()
        cfg=make_config(ctx,Path(args.bridge_config).read_bytes(),args.accepted_report_remote_path,args.accepted_report_sha256)
        with out.open('xb') as f:f.write(canonical(cfg))
        print(json.dumps(dict(prepared_config=str(out),config_sha256=sha(out.read_bytes()),requires_publication_before_execute=True,Node_contacted=False)))
    else:
        raw=out.read_bytes();cfg=json.loads(raw);assert cfg['output_new']==ctx['output']
        payload.update(config_base64=base64.b64encode(raw).decode(),config_sha256=sha(raw))
        remote_call(REMOTE_EXECUTE,payload,'execute',350)


if __name__=='__main__':main()
