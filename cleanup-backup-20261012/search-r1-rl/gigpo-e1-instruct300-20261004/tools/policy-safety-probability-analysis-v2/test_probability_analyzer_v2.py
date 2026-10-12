"""Small JSON-only analyzer regressions; no actual3272/native/GPU evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

FILE = Path(__file__).with_name('probability_analyzer_v2.py')
spec = importlib.util.spec_from_file_location('probability_readonly_analyzer', FILE)
analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)


def structure_fixture():
    rows, turns, checks = [], [[]], []
    for source in range(2560):
        action = dict(source=source, step_id=0, generation_prefix=[100, 200], scoring_prefix=[100, 200],
                      retained_action_tokens=[300], sampled_tokens=[300, 0], engine_prompt_token_ids=[100, 200],
                      sampling_action_position_ids=[2], scoring_action_position_ids=[2],
                      sampling_action_mask=[1], scoring_action_mask=[1])
        rows.append(dict(source=source, actions=[action]))
        turns[0].append(copy.deepcopy(action))
        checks.append(dict(source=source, step_id=0, effective_tokens=1,
                           same_position_ids=True, same_effective_mask=True, engine_prefix_matches_generation=True,
                           full_trajectory_vs_sample_prefix_old_max_abs=0., full_vs_prefix_old_joint_sum_delta=0.,
                           full_vs_vllm_joint_sum_delta=0., prefix_old_vs_vllm_joint_sum_delta=0.))
    alignment=dict(scope=analyzer.SCOPE,actions=rows,alignment=dict(trajectory_rows=2560,recorded_actions=2560,
            prefix_mismatch_count=0,action_token_mismatch_count=0,position_ids_mismatch_count=0,
            effective_mask_mismatch_count=0,engine_prefix_mismatch_count=0))
    return alignment,turns,dict(scope=analyzer.SCOPE,checks=checks)


@pytest.mark.parametrize('mutation',['missing_row','duplicate_source','missing_step','duplicate_cross','missing_cross',
    'missing_turn','hidden_prefix','hidden_tokens','hidden_position','hidden_mask','hidden_engine','nonfinite_joint'])
def test_structural_or_summary_contradictions_refuse(mutation):
    alignment,turns,cross=structure_fixture();action=alignment['actions'][0]['actions'][0]
    if mutation=='missing_row':alignment['actions'].pop()
    elif mutation=='duplicate_source':alignment['actions'][-1]['source']=0
    elif mutation=='missing_step':action['step_id']=1
    elif mutation=='duplicate_cross':cross['checks'].append(copy.deepcopy(cross['checks'][0]))
    elif mutation=='missing_cross':cross['checks'].pop()
    elif mutation=='missing_turn':turns[0].pop()
    elif mutation=='hidden_prefix':action['scoring_prefix']=[200];turns[0][0]['scoring_prefix']=[200]
    elif mutation=='hidden_tokens':action['sampled_tokens'][0]=301
    elif mutation=='hidden_position':action['sampling_action_position_ids']=[3]
    elif mutation=='hidden_mask':action['sampling_action_mask']=[0]
    elif mutation=='hidden_engine':action['engine_prompt_token_ids']=[200]
    else:cross['checks'][0]['full_vs_vllm_joint_sum_delta']=float('inf')
    with pytest.raises(ValueError):analyzer.structural_analysis(alignment,turns,cross)


def test_matching2560_and_true_mismatch_are_observations_not_approval():
    alignment,turns,cross=structure_fixture()
    mapping,checks,result=analyzer.structural_analysis(alignment,turns,cross)
    assert len(mapping)==len(checks)==2560 and not any(result['mismatch_counts'].values())
    action=alignment['actions'][0]['actions'][0];action['generation_prefix']=[200]
    action['engine_prompt_token_ids']=[200];action['sampling_action_position_ids']=[1]
    turns[0][0].update(generation_prefix=[200])
    cross['checks'][0]['same_position_ids']=False
    alignment['alignment'].update(prefix_mismatch_count=1,position_ids_mismatch_count=1)
    result=analyzer.structural_analysis(alignment,turns,cross)[2]
    assert result['mismatch_counts']['prefix']==result['mismatch_counts']['position']==1
    assert 'approval' not in result and 'tolerance' not in result


def ready_fixture():
    workers=[dict(rank=rank,pid=1000+rank,start_ticks='123',scheduler_step=25,learning_rates=[1e-6],
        model_shards_sha256='a'*64,optimizer_tensors_sha256='b'*64,rng_tensors_sha256='c'*64,
        local_shards_sha256='d'*64,buffers_sha256='e'*64,complete_state_sha256='f'*64) for rank in range(8)]
    pause=dict(workers=workers,worker_pause_sha256='1'*64,driver_complete_state_sha256='2'*64,
               driver_pid=999,driver_start_ticks='456',raw=dict(sha256='3'*64))
    identity=dict(code_commit=analyzer.SOURCE,job_id='9001',run_id='gigpo-e1-policy-safety-calibration25-job9001-1',native_binding_sha256='4'*64,
        material_manifest_sha256='5'*64,post_sampling_worker_identities=workers,
        post_sampling_worker_pause_sha256='1'*64,post_sampling_driver_complete_state_sha256='2'*64,
        post_sampling_pause_raw_sha256='3'*64)
    ready=dict(scope=analyzer.SCOPE,status='WAITING_BEFORE_FIRST_ADAM',actual_adam_writes=0,original_adam=500,
               scheduler=25,data_cursor=13312,sampled_trajectories=2560,retained_training_updates_contributed=0,
               identity=identity,post_sampling_pause=pause)
    return ready,dict(post_sampling_pause=copy.deepcopy(pause))


@pytest.mark.parametrize('mutation',['worker_missing_digest','duplicate_pid','worker_scheduler','wrong_job','wrong_binding',
    'changed_material','adam_written','missing_rank','pause_contradiction'])
def test_incomplete_or_wrong_live_pause_identity_refuses(mutation):
    ready,manifest=ready_fixture();workers=ready['identity']['post_sampling_worker_identities']
    if mutation=='worker_missing_digest':workers[0].pop('model_shards_sha256')
    elif mutation=='duplicate_pid':workers[1]['pid']=workers[0]['pid']
    elif mutation=='worker_scheduler':workers[0]['scheduler_step']=26
    elif mutation=='wrong_job':ready['identity']['job_id']='3267'
    elif mutation=='wrong_binding':ready['identity']['native_binding_sha256']='9'*64
    elif mutation=='changed_material':ready['identity']['material_manifest_sha256']='9'*64
    elif mutation=='adam_written':ready['actual_adam_writes']=1
    elif mutation=='missing_rank':workers.pop()
    else:ready['post_sampling_pause']['raw']['sha256']='9'*64
    with pytest.raises(ValueError):analyzer.validate_ready_json(ready,{},'4'*64,'5'*64,manifest,'9001',str(analyzer.ROOT/'runs/gigpo-e1-policy-safety-calibration25-job9001-1'))


def test_valid_ready_still_requires_live_os_and_selects_no_tolerance():
    ready,manifest=ready_fixture();result=analyzer.validate_ready_json(ready,{},'4'*64,'5'*64,manifest,'9001',str(analyzer.ROOT/'runs/gigpo-e1-policy-safety-calibration25-job9001-1'))
    assert result['live_os_identity_recheck_required'] and result['actual_adam_writes']==0
    assert 'approval' not in result and 'tolerance' not in result


@pytest.mark.parametrize('mutation',['changed_sha','changed_size','foreign_path','traversal','symlink'])
def test_original_raw_bytes_and_path_closure_refuse(tmp_path,mutation):
    root=tmp_path/'run';root.mkdir();path=root/'raw.pt';path.write_bytes(b'cpu fixture, not torch')
    record=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if mutation=='changed_sha':record['sha256']='a'*64
    elif mutation=='changed_size':record['bytes']+=1
    elif mutation=='foreign_path':record['path']=str(tmp_path/'other.pt')
    elif mutation=='traversal':record['path']=str(root/'..'/'raw.pt')
    else:
        path.unlink();path.symlink_to(tmp_path/'other.pt')
    with pytest.raises(ValueError):analyzer.OriginalStore(str(root)).verify(record)


def test_exact_original_bytes_and_quantiles_are_diagnostic(tmp_path):
    root=tmp_path/'run';root.mkdir();path=root/'raw.pt';path.write_bytes(b'cpu-only')
    record=dict(path=str(path),bytes=8,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert analyzer.OriginalStore(str(root)).verify(record)==path
    summary=analyzer.error_summary([-.2,0.,.5]);assert summary['maximum']==.5 and summary['minimum']==-.2
    assert 'tolerance' not in summary


@pytest.mark.parametrize('mutation',['other_job','traversal','outside_root','other_node','mac','nondecimal_job'])
def test_actual_analysis_requires_explicit_same_job_run_and_h1008_cpu_location(monkeypatch,mutation):
    monkeypatch.setattr(analyzer.platform,'system',lambda:'Linux')
    monkeypatch.setattr(analyzer.socket,'gethostname',lambda:'h100-8')
    job='9001';run=str(analyzer.ROOT/'runs/gigpo-e1-policy-safety-calibration25-job9001-1')
    if mutation=='other_job':job='9002'
    elif mutation=='traversal':run=run+'/../run'
    elif mutation=='outside_root':run='/tmp/gigpo-e1-policy-safety-calibration25-job9001-1'
    elif mutation=='other_node':monkeypatch.setattr(analyzer.socket,'gethostname',lambda:'h100-11')
    elif mutation=='mac':monkeypatch.setattr(analyzer.platform,'system',lambda:'Darwin')
    else:job='9001;echo'
    with pytest.raises(ValueError):analyzer.validate_actual_execution(job,run)


def test_actual_job_interface_is_generic_but_explicit(monkeypatch):
    monkeypatch.setattr(analyzer.platform,'system',lambda:'Linux')
    monkeypatch.setattr(analyzer.socket,'gethostname',lambda:'h100-8')
    for job in ('9001','3276'):
        analyzer.validate_actual_execution(job,str(analyzer.ROOT/('runs/gigpo-e1-policy-safety-calibration25-job'+job+'-1')))


@pytest.mark.parametrize('text,owned',[
 ('7:freezer:/slurm_h100-8/uid_0/job_9001/step_0\n',True),
 ('0::/slurm_h100-8/uid_0/job_9001/step_0/worker\n',True),
 ('7:freezer:/slurm_h100-8/uid_0/job_9002/step_0\n',False),
 ('7:freezer:/foreign/slurm_h100-8/uid_0/job_9001/step_0\n',False),
 ('7:freezer:/slurm_h100-8/uid_0/job_9001/step_01\n',False),
 ('7:cpu:/slurm_h100-8/uid_0/job_9001/step_0\n',False)])
def test_pause_ownership_requires_exact_actual_freezer_or_unified_scope(text,owned):
    assert analyzer.cgroup_owned(text,'9001') is owned


def proc_stat(state='R',start='123'):
    return '1000 (worker) '+state+' '+' '.join(['0']*18+[start,'0'])


@pytest.mark.parametrize('mutation',['pid_reuse','moved_cgroup','fd_other_pid','pidfd_ready','zombie','unreadable_stat'])
def test_live_pause_rejects_changed_kernel_identity_scope_or_exit(monkeypatch,mutation):
    stat_texts=[proc_stat(),proc_stat('S')]
    cgroup_texts=['7:freezer:/slurm_h100-8/uid_0/job_9001/step_0\n']*2
    fdinfo='Pid:\t1000\n';select_values=[([],[],[]),([],[],[])]
    if mutation=='pid_reuse':stat_texts[1]=proc_stat('S','124')
    elif mutation=='moved_cgroup':cgroup_texts[1]='7:freezer:/slurm_h100-8/uid_0/job_9002/step_0\n'
    elif mutation=='fd_other_pid':fdinfo='Pid:\t1001\n'
    elif mutation=='pidfd_ready':select_values[0]=([41],[],[])
    elif mutation=='zombie':stat_texts[0]=proc_stat('Z')
    else:stat_texts[0]='truncated'
    class FakePath:
        def __init__(self,path):self.path=path
        def read_text(self):
            if self.path=='/proc/1000/stat':return stat_texts.pop(0)
            if self.path=='/proc/1000/cgroup':return cgroup_texts.pop(0)
            if self.path=='/proc/self/fdinfo/41':return fdinfo
            raise OSError('Unexpected CPU fixture path')
    monkeypatch.setattr(analyzer,'Path',FakePath)
    monkeypatch.setattr(analyzer.os,'pidfd_open',lambda pid,flags:41,raising=False)
    monkeypatch.setattr(analyzer.select,'select',lambda *args:select_values.pop(0))
    closed=[];monkeypatch.setattr(analyzer.os,'close',closed.append)
    with pytest.raises((ValueError,IndexError)):analyzer.observe_one_live(dict(pid=1000,start_ticks='123',role='worker',rank=0),'9001')
    assert closed==[41]


def test_normal_R_to_S_transition_keeps_same_pidfd_start_cgroup(monkeypatch):
    texts=[proc_stat(),proc_stat('S')]
    class FakePath:
        def __init__(self,path):self.path=path
        def read_text(self):
            if self.path.endswith('/stat'):return texts.pop(0)
            if self.path.endswith('/cgroup'):return '7:freezer:/slurm_h100-8/uid_0/job_9001/step_0\n'
            return 'Pid:\t1000\n'
    monkeypatch.setattr(analyzer,'Path',FakePath)
    monkeypatch.setattr(analyzer.os,'pidfd_open',lambda pid,flags:41,raising=False)
    monkeypatch.setattr(analyzer.select,'select',lambda *args:([],[],[]))
    monkeypatch.setattr(analyzer.os,'close',lambda fd:None)
    result=analyzer.observe_one_live(dict(pid=1000,start_ticks='123',role='worker',rank=0),'9001')
    assert result['state_before']=='R' and result['state_after']=='S' and result['pidfd_same_live_kernel_identity']


@pytest.mark.parametrize('mutation',['wrong_run','worker_bool_pid','missing_driver_start'])
def test_pause_identity_additional_cross_job_and_driver_requirements(mutation):
    ready,manifest=ready_fixture()
    if mutation=='wrong_run':ready['identity']['run_id']='gigpo-e1-policy-safety-calibration25-job9002-1'
    elif mutation=='worker_bool_pid':ready['identity']['post_sampling_worker_identities'][0]['pid']=True
    else:ready['post_sampling_pause'].pop('driver_start_ticks');manifest['post_sampling_pause'].pop('driver_start_ticks')
    with pytest.raises((ValueError,KeyError)):
        analyzer.validate_ready_json(ready,{},'4'*64,'5'*64,manifest,'9001',str(analyzer.ROOT/'runs/gigpo-e1-policy-safety-calibration25-job9001-1'))


from failure_contract_v2 import audit_zero_update_rows


@pytest.mark.parametrize('row',[
 {'kind':'actor_outer','optimizer_steps':10}, {'kind':'optimizer_step'},
 {'kind':'actor_update'}, {'kind':'unknown'}, {'kind':'forward_log_prob','optimizer_steps_completed':0},
 {'kind':'probability_probe','micro_index':1}, {'kind':'driver_outer','run_id':'foreign'}])
def test_zero_adam_is_not_inferred_from_missing_ready_or_zero_counter(row):
    value=dict(commit=analyzer.SOURCE,run_id='actualrun');value.update(row)
    with pytest.raises(ValueError):audit_zero_update_rows([value],analyzer.SOURCE,'actualrun',analyzer.require)


def test_known_forward_diagnostics_are_not_mislabeled_as_gradient_or_adam():
    rows=[dict(kind=kind,commit=analyzer.SOURCE,run_id='actualrun') for kind in
          ('compute_log_prob','reference_log_prob','probability_probe','driver_outer')]
    audit_zero_update_rows(rows,analyzer.SOURCE,'actualrun',analyzer.require)


def test_aggregate_writer_refuses_other_job_traversal_and_existing_output(tmp_path,monkeypatch):
    root=tmp_path/'root';(root/'logs').mkdir(parents=True);monkeypatch.setattr(analyzer,'ROOT',root)
    with pytest.raises(ValueError):analyzer.write_aggregate_once(str(root/'logs/probability-analysis-job9002-v2-cpu.json'),'9001','{}')
    path=root/'logs/probability-analysis-job9001-v2-cpu.json'
    analyzer.write_aggregate_once(str(path),'9001','{}')
    with pytest.raises(FileExistsError):analyzer.write_aggregate_once(str(path),'9001','new')
    assert path.read_text()=='{}'

from types import SimpleNamespace
import json
import sys
from failure_contract_v2 import validate_failure_context


def failure_fixture(tmp_path,monkeypatch,mutation=None):
    """Synthetic existing-artifact schema test, never an actual job receipt."""
    root=tmp_path/'root';(root/'logs').mkdir(parents=True)
    run=root/'runs/gigpo-e1-policy-safety-calibration25-job9001-1';run.mkdir(parents=True)
    (run/'audit').mkdir()
    repo=root/'repo';repo.mkdir()
    binding=dict(common_manifest_sha256=analyzer.DATA,repo=str(repo))
    args=SimpleNamespace(job='9001',run=str(run),binding_sha='4'*64,manifest_sha='5'*64,
        failure_context=str(root/'logs/context.json'),failure_review=str(root/'logs/review.json'))
    configpath=run/'resolved-config.yaml';configpath.write_text('{}')
    expected=dict(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
        SEARCH_R1_BASELINE_COMMIT=analyzer.SOURCE,SEARCH_R1_BASELINE_CONFIG_SHA256=analyzer.sha(configpath),
        SEARCH_R1_BASELINE_DATA_IDENTITY=analyzer.DATA)
    environment=dict(driver=expected,workers=[dict(rank=str(rank),environment=copy.deepcopy(expected),
        imports={'verl.workers.actor.dp_actor':str(repo/'verl/workers/actor/dp_actor.py')}) for rank in range(8)])
    result=dict(status='FAILED',job_id='9001',run_id=run.name,source_commit=analyzer.SOURCE,
        binding_sha256=args.binding_sha,restore_manifest_sha256=analyzer.RESTORE,
        retained_training_updates_contributed=0,cleanup_residual_states={},final_gpu={'processes':[]})
    terminal=dict(fields=dict(JobId='9001',NodeList='h100-8',JobState='FAILED',ExitCode='1:0',Requeue='0',Restarts='0'))
    jobroot='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_9001'
    release=dict(job='9001',node='h100-8',paths=[dict(path=path,exists=False,errno=2,ancestor_readable=True,
        readable_ancestor='/sys/fs/cgroup/freezer/slurm_h100-8') for path in (jobroot,jobroot+'/step_0')])
    log='wait_probability_acceptance\nadmit_alignment\nValueError: Generation prefix differs from full-trajectory old-policy prefix; preserve and investigate\n'
    if mutation=='environment_missing_rank':environment['workers'].pop()
    elif mutation=='environment_wrong_source':environment['driver']['SEARCH_R1_BASELINE_COMMIT']='other'
    elif mutation=='terminal_running':terminal['fields']['JobState']='RUNNING'
    elif mutation=='terminal_zero_exit':terminal['fields']['ExitCode']='0:0'
    elif mutation=='cgroup_live':release['paths'][0]['exists']=True
    elif mutation=='residual_process':result['final_gpu']['processes']=[{'pid':100}]
    elif mutation=='cleanup_residual':result['cleanup_residual_states']={'100':'R'}
    elif mutation=='missing_structural_trace':log='missing ready does not prove zero Adam'
    paths=dict(terminal=root/'logs/terminal.json',result=run/'run-result.json',resolved_config=configpath,
        ray_environment=run/'ray-audit-environment-verification.json',calibration_log=run/'calibration.log',
        cgroup_release=root/'logs/cgroup-release.json')
    for name,data in (('terminal',terminal),('result',result),('ray_environment',environment),('cgroup_release',release)):
        paths[name].write_text(json.dumps(data))
    paths['calibration_log'].write_text(log)
    originals={name:dict(path=str(path),bytes=path.stat().st_size,sha256=analyzer.sha(path)) for name,path in paths.items()}
    observed=[]
    for rank in range(8):
        for relative in (f'audit/{rank}.jsonl',f'audit/numeric/micro-{rank}.jsonl',f'audit/policy-safety/rank-{rank}.jsonl'):
            path=run/relative
            observed.append(dict(path=str(path),stat_outcome='ENOENT',errno=2,
                readable_ancestor=str(run/'audit'),ancestor_readable=True))
    if mutation in ('present_forward','present_update','present_unknown'):
        path=run/'audit/0.jsonl'
        kind={'present_forward':'compute_log_prob','present_update':'actor_outer','present_unknown':'unknown'}[mutation]
        path.write_text(json.dumps(dict(kind=kind,commit=analyzer.SOURCE,run_id=run.name))+'\n')
        observed[0]=dict(path=str(path),stat_outcome='PRESENT',bytes=path.stat().st_size,sha256=analyzer.sha(path))
    if mutation=='missing_audit_path':observed.pop()
    elif mutation=='unreadable_ancestor':observed[0]['ancestor_readable']=False
    elif mutation=='appeared_audit':(run/'audit/0.jsonl').write_text('{}')
    elif mutation=='ready_present':(run/'probability-identity-ready.json').write_text('{}')
    context=dict(schema=1,scope='ACTUAL_CALIBRATION_ZERO_ADAM_STRUCTURAL_FAILURE',job_id=args.job,run=args.run,
        run_id=run.name,code_commit=analyzer.SOURCE,binding_sha256=args.binding_sha,material_manifest_sha256=args.manifest_sha,
        originals=originals,audit_path_observations=observed)
    Path(args.failure_context).write_text(json.dumps(context));args.failure_context_sha=analyzer.sha(args.failure_context)
    review=dict(status='FAILED_CALIBRATION_ZERO_ADAM_DIAGNOSTICS_ONLY',scope=context['scope'],job_id=args.job,
        run=args.run,run_id=run.name,code_commit=analyzer.SOURCE,binding_sha256=args.binding_sha,
        material_manifest_sha256=args.manifest_sha,failure_context_sha256=args.failure_context_sha,
        actual_adam_writes=0,actual_gradient_micro_batches=0,retained_training_updates_contributed=0,
        zero_adam_audit_environment_verified=True,post_sampling_live_pause_available=False,probability_approval_generated=False)
    if mutation=='review_other_job':review['job_id']='9002'
    elif mutation=='review_no_zero_audit':review['zero_adam_audit_environment_verified']=False
    elif mutation=='review_bool_adam':review['actual_adam_writes']=False
    elif mutation=='review_changed_context':review['failure_context_sha256']='a'*64
    Path(args.failure_review).write_text(json.dumps(review));args.failure_review_sha=analyzer.sha(args.failure_review)
    manifest={} if mutation!='manifest_live_pause' else {'post_sampling_pause':{}}
    # Config fixture is JSON, valid YAML. Only schema tests mock parsing dependency.
    monkeypatch.setitem(sys.modules,'yaml',SimpleNamespace(safe_load=json.loads))
    return args,binding,manifest,root


@pytest.mark.parametrize('mutation',['environment_missing_rank','environment_wrong_source','terminal_running',
    'terminal_zero_exit','cgroup_live','residual_process','cleanup_residual','missing_structural_trace',
    'missing_audit_path','unreadable_ancestor','appeared_audit','ready_present','review_other_job',
    'review_no_zero_audit','review_bool_adam','review_changed_context','manifest_live_pause','present_update','present_unknown'])
def test_failure_context_requires_actual_terminal_zero_adam_environment_and_original_closure(tmp_path,monkeypatch,mutation):
    args,binding,manifest,root=failure_fixture(tmp_path,monkeypatch,mutation)
    with pytest.raises((ValueError,KeyError)):
        validate_failure_context(args,binding,manifest,analyzer.read_json,analyzer.sha,analyzer.require,
                                 root,analyzer.SOURCE,analyzer.RESTORE)


@pytest.mark.parametrize('mutation',[None,'present_forward'])
def test_independently_reviewed_failure_diagnostics_never_claim_live_pause_or_approval(tmp_path,monkeypatch,mutation):
    args,binding,manifest,root=failure_fixture(tmp_path,monkeypatch,mutation)
    result,config=validate_failure_context(args,binding,manifest,analyzer.read_json,analyzer.sha,analyzer.require,
                                         root,analyzer.SOURCE,analyzer.RESTORE)
    assert result['actual_adam_writes']==0 and not result['live_pause_available']
    assert not result['probability_approval_generated'] and not result['production_thresholds_selected']
    assert result['audit_observations_count']==24 and config=={}
