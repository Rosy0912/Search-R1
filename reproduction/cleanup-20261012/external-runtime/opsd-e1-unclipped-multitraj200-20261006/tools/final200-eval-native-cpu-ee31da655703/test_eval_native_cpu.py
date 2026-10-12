"""Explicit local CPU software fixtures; no native/GPU/evaluation acceptance."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'opsd-final200-paired-eval-source'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


checker = load('local_eval_checker_software_fixture', 'inspect_eval_native_cpu.py')
once = load('local_eval_once_software_fixture', 'run_eval_native_cpu_once.py')
c = checker.common()


@pytest.fixture(scope='module')
def partial():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(REPO / 'verl/trainer/config')):
        config = compose(config_name=checker.CONFIG_NAME)
    return OmegaConf.to_container(config, resolve=False)


@pytest.fixture(scope='module')
def inventory():
    files = {}
    for name, (mode, blob) in checker.tree(REPO).items():
        raw = (REPO / name).read_bytes()
        files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), git_mode=mode, git_blob=blob)
    return dict(version=1,kind=checker.INVENTORY_KIND,source_commit=checker.SOURCE,parent_commit=checker.PARENT,
        member_count=857,remote_repository=str(REPO),files=files)


def pinned_json(tmp_path, value):
    p = tmp_path / 'source-inventory.LOCAL_SOFTWARE_FIXTURE.json'
    raw = c.canonical(value); p.write_bytes(raw)
    return p, hashlib.sha256(raw).hexdigest()


def test_actual_local_c5bf_tree_inventory_software_only(tmp_path, inventory):
    p, sha = pinned_json(tmp_path, inventory)
    repo, saved, record, members = checker.source_inventory(c,p,sha)
    assert repo == REPO and saved == inventory and len(members) == 857
    assert record['sha256'] == sha
    assert members['search_r1/baselines/final200_evaluation.py']['sha256'] == '2d927c8bb87b55ea140d3d08931528c5b5fcf604902ccf67902c9830f1b4fd52'


@pytest.mark.parametrize('key,value', [('version',True),('kind','TRAINING211'),('source_commit',checker.PARENT),
    ('parent_commit',checker.SOURCE),('member_count',857.0),('remote_repository',None)])
def test_inventory_header_and_future_null_rejected(tmp_path, inventory, key, value):
    bad = copy.deepcopy(inventory); bad[key] = value
    p, sha = pinned_json(tmp_path,bad)
    with pytest.raises((ValueError,TypeError)): checker.source_inventory(c,p,sha)


@pytest.mark.parametrize('key,value', [('sha256','0'*64),('bytes',True),('bytes',0),('git_mode','120000'),('git_blob','0'*40)])
def test_source_member_full_sha_size_blob_and_typed_schema(tmp_path, inventory, key, value):
    bad = copy.deepcopy(inventory); bad['files']['README.md'][key] = value
    p, sha = pinned_json(tmp_path,bad)
    with pytest.raises(ValueError): checker.source_inventory(c,p,sha)


def test_source_inventory_cannot_drop_new_module(tmp_path, inventory):
    bad = copy.deepcopy(inventory); bad['files'].pop('search_r1/baselines/final200_evaluation.py')
    p, sha = pinned_json(tmp_path,bad)
    with pytest.raises(ValueError): checker.source_inventory(c,p,sha)


def test_authored_partial_config_keeps_future_unresolved(partial):
    selected = checker.structural_config(c,partial)
    assert all(checker.selected(partial,k) == v for k,v in checker.FUTURE_INTERPOLATIONS.items())
    assert selected['trainer.total_training_steps'] == 0
    assert 'search_r1.baselines.final200_evaluation' in checker.ADDED_REQUIRED
    assert 'search_r1.baselines.corrected_runtime_capture' in checker.ADDED_REQUIRED
    assert 'verl.utils.reward_score.qa_em' in checker.ADDED_REQUIRED


@pytest.mark.parametrize('key,value', [
    ('trainer.total_training_steps',False), ('trainer.total_training_steps',200),
    ('trainer.val_only',False),('baselines.opsd_e1.enabled',True),
    ('baselines.opsd_e1.profile','multitraj200'),('baselines.corrected_evaluation.enabled',True),
    ('data.val_batch_size',True),('data.val_batch_size',512),('max_turns',5),
    ('retriever.topk',4),('retriever.url','http://other-node/retrieve'),
    ('actor_rollout_ref.rollout.n_agent',5),('actor_rollout_ref.rollout.top_k',20),
    ('actor_rollout_ref.actor.kl_loss_coef',.001),('actor_rollout_ref.actor.entropy_coeff',True),
    ('actor_rollout_ref.model.path','/FAKE/HF200'),
    ('baselines.final200_evaluation.reviewed_stage_manifest','/FAKE/STAGE'),
    ('baselines.final200_evaluation.output_dir','/FAKE/OUTPUT')])
def test_structural_profile_drift_and_fake_future_values_rejected(partial,key,value):
    bad=copy.deepcopy(partial); target=bad
    parts=key.split('.')
    for part in parts[:-1]: target=target[part]
    target[parts[-1]]=value
    with pytest.raises(ValueError): checker.structural_config(c,bad)


def test_actual_authored_compose_and_existing_validator_software_only():
    sys.path.insert(0,str(REPO))
    import search_r1.baselines.final200_evaluation as evaluation
    forbidden={n:None for n in os.environ if n.startswith('OPSD_FINAL200_') or n=='OPSD_E1_RETRIEVER_URL'}
    old=dict(os.environ)
    try:
        for name in forbidden: os.environ.pop(name)
        partial, meaning=checker.compose_partial(c,REPO,evaluation)
    finally:
        os.environ.clear();os.environ.update(old)
    assert meaning['runtime_resolved_config'] is False and meaning['retrieval_service_verified'] is False
    assert set(meaning['future_values'].values())=={None}
    assert meaning['authored_default_budget']['retriever.url']=='http://127.0.0.1:8000/retrieve'
    assert checker.selected(partial,'actor_rollout_ref.model.path')==checker.FUTURE_INTERPOLATIONS['actor_rollout_ref.model.path']


def test_future_environment_is_not_silently_used():
    with patch.dict(os.environ,{'OPSD_FINAL200_ACTUAL_REVIEWED_STAGE':'/FAKE'},clear=False):
        with pytest.raises(ValueError):checker.compose_partial(c,REPO,None)


def test_real_local_process_is_not_native_acceptance():
    with pytest.raises(ValueError):checker.native_runtime(c,'901','3')


def test_pinned_helper_executes_exact_source_not_pyc(tmp_path):
    p=tmp_path/'helper.py';p.write_bytes(b'value=17\n')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert checker.pinned_module(p,digest,'fixture_exact_helper').value==17
    p.write_bytes(b'raise RuntimeError("unapproved bytes executed")\n')
    with pytest.raises(ValueError,match='before execution'):checker.pinned_module(p,digest,'fixture_replaced_helper')


def test_helper_symlink_rejected(tmp_path):
    p=tmp_path/'real.py';p.write_bytes(b'value=17\n');q=tmp_path/'link.py';q.symlink_to(p)
    with pytest.raises(ValueError):checker.pinned_module(q,hashlib.sha256(p.read_bytes()).hexdigest(),'fixture_link')


def test_once_reservation_fsync_precedes_any_child_and_unknown_refuses_retry(tmp_path):
    out=tmp_path/'once'
    events=[]
    original=once.sync_dir
    def observe(path):events.append(str(path));return original(path)
    with patch.object(once,'sync_dir',side_effect=observe):
        once.reserve_once(c,out,{'kind':'LOCAL_SOFTWARE_INTENT_NOT_PRODUCTION'})
    assert events[:2]==[str(tmp_path),str(out)]
    with pytest.raises(FileExistsError):once.reserve_once(c,out,{'kind':'RETRY_FORBIDDEN'})
    assert c.read_json(out/'intent.original.json')[0]['kind']=='LOCAL_SOFTWARE_INTENT_NOT_PRODUCTION'


@pytest.mark.parametrize('exit_code',[0,7])
def test_real_local_cpu_child_stdout_stderr_wait_not_native_qualification(tmp_path,exit_code):
    command=[sys.executable,'-B','-c','import sys;print("actual local software child");print("actual stderr",file=sys.stderr);sys.exit('+str(exit_code)+')']
    with patch.object(once,'child_identity',return_value={'process_sample':'LOCAL_SOFTWARE_CHILD_ONLY'}):
        waited=once.execute_child(c,command,tmp_path,dict(os.environ),{})
    assert waited['returncode']==exit_code and waited['waited'] is True
    assert b'actual local software child' in (tmp_path/'child.stdout.original').read_bytes()
    assert b'actual stderr' in (tmp_path/'child.stderr.original').read_bytes()
    assert c.read_json(tmp_path/'child-wait.original.json')[0]==waited


def test_start_record_failure_still_waits_actual_child(tmp_path):
    original=once.write_once
    def fail_one(c,path,value):
        if Path(path).name=='child-popen.original.json':raise OSError('explicit LOCAL CPU fault injection')
        return original(c,path,value)
    command=[sys.executable,'-B','-c','print("child completes despite recording failure")']
    with patch.object(once,'write_once',side_effect=fail_one):
        with pytest.raises(ValueError):once.execute_child(c,command,tmp_path,dict(os.environ),{})
    waited=c.read_json(tmp_path/'child-wait.original.json')[0]
    assert waited['returncode']==0 and waited['process_identity_error']['type']=='OSError'
    assert b'child completes' in (tmp_path/'child.stdout.original').read_bytes()


@pytest.fixture
def software_report(tmp_path):
    # All values below are explicitly synthetic protocol fixtures. No execution
    # job, native import, future checkpoint or signed receipt is constructed.
    records={}
    for role in ('checker','inventory','closure','partial','source'):
        p=tmp_path/(role+'.LOCAL_SOFTWARE_FIXTURE.json');p.write_bytes(b'{}')
        records[role]=c.read_file(p)[1]
    execution=dict(execution_job_id='901',execution_step_id='3',uid=os.getuid(),uids=[os.getuid()]*4,
        node='h100-10',stored_training_job_id='3304',pid=os.getpid())
    report=dict(version=1,kind='ACTUAL_C5BF_NATIVE_CPU_IMPORT_PARTIAL_CONFIG_PREPARATION',scope=once.SCOPE,
        passed=True,eval_source_commit=checker.SOURCE,stored_training_job_id='3304',
        actual_eval_job_id=None,actual_model_directory=None,actual_stage_manifest=None,
        actual_input_manifest=None,actual_output_directory=None,model_loaded=False,
        gpu_execution_verified=False,evaluation_started=False,
        evaluation_eligibility='BLOCKED_ACTUAL200_AND_REVIEWED_STAGE_REQUIRED',
        tool_source=records['checker'],source_inventory=records['inventory'],
        execution={**execution,'pid':os.getpid()+1},runtime=dict(cuda_is_initialized=False),
        module_count=220,verified_live_function_code_count=750,source_tree_members=857,
        configuration=dict(runtime_resolved_config=False,retrieval_service_verified=False),
        import_closure=records['closure'],partial_config=records['partial'],source_physical_records=records['source'])
    return report,execution,records


def test_synthetic_protocol_qualification_scope_only(software_report):
    report,execution,records=software_report
    once.qualify(c,report,execution,records['checker'],records['inventory'])


@pytest.mark.parametrize('key,value',[('passed',1),('actual_eval_job_id','3304'),
    ('actual_model_directory','/FAKE/HF200'),('evaluation_started',True),('scope','HF200_PASS'),
    ('module_count',True),('source_tree_members',857.0),('eval_source_commit',checker.PARENT)])
def test_synthetic_report_cannot_promote_to_eval_or_old_training_scope(software_report,key,value):
    report,execution,records=software_report;report[key]=value
    with pytest.raises(ValueError):once.qualify(c,report,execution,records['checker'],records['inventory'])


def test_synthetic_report_cannot_claim_live_service(software_report):
    report,execution,records=software_report;report['configuration']['retrieval_service_verified']=True
    with pytest.raises(ValueError):once.qualify(c,report,execution,records['checker'],records['inventory'])


@pytest.mark.parametrize('bad',[False,True])
def test_actual_child_bootstrap_same_pinned_bytes_before_exec(tmp_path,bad):
    source=tmp_path/'LOCAL_SOFTWARE_CHILD.py';source.write_text('import sys\nprint(__file__)\nprint(sys.argv[1])\n')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if bad: source.write_text('raise RuntimeError("unapproved code must never execute")\n')
    command=[sys.executable,'-B','-c',once.CHILD_BOOTSTRAP,str(source),sha,'LOCAL_SOFTWARE_ARG']
    result=subprocess.run(command,capture_output=True)
    if bad:
        assert result.returncode!=0 and b'Child bytes differ before execution' in result.stderr
        assert b'RuntimeError: unapproved' not in result.stderr
    else:
        assert result.returncode==0 and str(source).encode() in result.stdout and b'LOCAL_SOFTWARE_ARG' in result.stdout


def test_no_entrypoint_or_prepare_call_in_checker_source():
    import ast
    node=ast.parse((HERE/'inspect_eval_native_cpu.py').read_bytes())
    forbidden={'configure','load_stage','start_capture','prepare_inputs','run','preflight'}
    calls=[x.func.attr for x in ast.walk(node) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
    assert not (set(calls)&forbidden)
    assert 'collect' in calls and 'verify_imports' in calls and 'validate_eval_config' in calls


def test_no_scheduler_network_or_training_child_command():
    raw=(HERE/'run_eval_native_cpu_once.py').read_text()
    assert all(word not in raw for word in ('sbatch','srun','ssh','torchrun'))
    assert "'inspect_eval_native_cpu.py'" in raw
