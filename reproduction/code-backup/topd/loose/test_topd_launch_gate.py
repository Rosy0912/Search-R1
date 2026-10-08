"""CPU-only supervisor boundaries; never submits Slurm or starts Ray/GPU."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

spec=importlib.util.spec_from_file_location('topd_launch_gate',Path(__file__).with_name('topd-short-run.py'))
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def args_for(path,stage='smoke2'):
    return SimpleNamespace(binding=path,binding_sha=gate.sha(path),stage=stage,smoke_review=None,smoke_review_sha=None)


def test_pending_review_before_any_process(tmp_path):
    path=tmp_path/'binding.json';path.write_text('{"status":"PENDING"}')
    with patch.object(gate.subprocess,'run') as run,patch.object(gate.subprocess,'check_output') as read:
        with pytest.raises(RuntimeError,match='No independent reviewed'):
            gate.dependency_gate(args_for(path))
        run.assert_not_called();read.assert_not_called()


def test_unknown_fix_before_process(tmp_path):
    path=tmp_path/'binding.json';path.write_text(json.dumps(dict(status='REVIEWED',code_commit='a'*40,common_teacher_path_fix=dict(source_commit='PENDING',integrated_commit='b'*40))))
    with patch.object(gate,'git',side_effect=lambda *args:'a'*40 if args==('rev-parse','HEAD') else ''),patch.object(gate.subprocess,'run') as run:
        with pytest.raises(RuntimeError,match='Unknown public/integrated'):
            gate.dependency_gate(args_for(path))
        run.assert_not_called()


def test_resigned_review_cannot_restore_changed_artifact(tmp_path):
    evidence=tmp_path/'report';evidence.write_text('reviewed')
    item=dict(path=str(evidence),sha256=gate.sha(evidence))
    evidence.write_text('drifted')
    with pytest.raises(RuntimeError,match='Reviewed evidence changed'):
        gate.artifact(item)


def test_binding_drift(tmp_path):
    path=tmp_path/'binding';path.write_text('{}');args=args_for(path);path.write_text('{"status":"REVIEWED"}')
    with pytest.raises(RuntimeError,match='Record changed'):
        gate.dependency_gate(args)


def test_same_machine_environment_sanitizes_all_proxies_and_ray(monkeypatch,tmp_path):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    source={key:'http://a800:8000' for key in gate.PROXY_KEYS}
    source.update(PATH='/bin',RAY_ADDRESS='ray://other:10001',PYTHONPATH='/foreign',CUDA_VISIBLE_DEVICES='0,1')
    actual=gate.local_runtime_env(source)
    assert all(key not in actual for key in gate.PROXY_KEYS)
    assert actual['NO_PROXY']==actual['no_proxy']=='127.0.0.1,localhost,::1'
    assert actual['RAY_ADDRESS']=='127.0.0.1:16385'
    assert actual['PATH'].split(os.pathsep)[0]==str(tmp_path/'env/bin')
    assert 'PYTHONPATH' not in actual and actual['CUDA_VISIBLE_DEVICES']=='0,1'
    assert source['RAY_ADDRESS']=='ray://other:10001'


def test_nonlocal_symlink_asset_rejected_before_launch(tmp_path,monkeypatch):
    root=tmp_path/'own';root.mkdir();foreign=tmp_path/'foreign';foreign.mkdir()
    monkeypatch.setattr(gate,'ROOT',root)
    for asset in gate.RETRIEVAL_ASSETS:
        path=root/asset;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to(foreign,target_is_directory=True)
    with pytest.raises(RuntimeError,match='Nonlocal runtime retrieval'):
        gate.local_assets()


def test_fresh_teacher_gate_rechecks_changed_receipt(tmp_path):
    teacher=tmp_path/'teacher';teacher.mkdir();receipt=teacher/'teacher-manifest.json';receipt.write_text('old')
    binding=dict(teacher_receipt_sha256=gate.sha(receipt));receipt.write_text('new')
    with patch.object(gate.subprocess,'run') as launch:
        with pytest.raises(RuntimeError,match='Teacher receipt drift'):
            gate.teacher_gate(binding,teacher,tmp_path/'out')
        launch.assert_not_called()


def test_teacher_full_verifier_is_invoked_every_time_cpu_only(tmp_path):
    teacher=tmp_path/'teacher';teacher.mkdir();receipt=teacher/'teacher-manifest.json';receipt.write_text('{}')
    output=tmp_path/'output';binding=dict(teacher_receipt_sha256=gate.sha(receipt),split_manifest_path='/own/split')
    def verifier(command,**kwargs):
        assert '--teacher' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['check']
        output.write_text(json.dumps(dict(status='PASS',teacher_receipt_sha256=gate.sha(receipt))))
    with patch.object(gate.subprocess,'run',side_effect=verifier) as run:
        gate.teacher_gate(binding,teacher,output);gate.teacher_gate(binding,teacher,output)
        assert run.call_count==2


@pytest.mark.parametrize('budget',[3,6])
def test_real_scheduler_first_used_lr_zero_and_later_updates(budget):
    import torch
    parameter=torch.nn.Parameter(torch.tensor(1.0));optimizer=torch.optim.SGD([parameter],lr=1e-6)
    warmup=int(.285*budget)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1,step/max(1,warmup)))
    used=[]
    for _ in range(budget-1):
        used.append(optimizer.param_groups[0]['lr'])
        for _ in range(20):
            parameter.grad=torch.tensor(1.);optimizer.step()
        scheduler.step()
    assert used[0]==0 and all(lr==1e-6 for lr in used[1:]) and parameter.item()<1


def test_cleanup_never_signals_baseline_or_process_that_left_step(tmp_path):
    owner=gate.OwnedProcesses.__new__(gate.OwnedProcesses)
    owner.scope=tmp_path;owner.relative='/slurm/job_17/step_batch';owner.baseline={10};owner.children=[];owner.streams=[]
    # First scan sees PID20 which migrated out; second sees only baseline.
    with patch.object(gate,'cgroup_pids',side_effect=[{10,20},{10},{10}]),patch.object(gate.os,'pidfd_open',return_value=123,create=True) as opened,patch.object(gate.os,'close'),patch.object(gate.Path,'read_text',return_value='0::/slurm/job_99/step_batch'),patch.object(gate.signal,'pidfd_send_signal',create=True) as sent,patch.object(gate.time,'sleep'):
        owner.cleanup();opened.assert_called_once_with(20);sent.assert_not_called()


def test_cleanup_own_member_uses_pidfd_term(tmp_path):
    owner=gate.OwnedProcesses.__new__(gate.OwnedProcesses)
    owner.scope=tmp_path;owner.relative='/slurm/job_17/step_batch';owner.baseline={10};owner.children=[];owner.streams=[]
    with patch.object(gate,'cgroup_pids',side_effect=[{10,20},{10},{10}]),patch.object(gate.os,'pidfd_open',return_value=123,create=True),patch.object(gate.os,'close'),patch.object(gate.Path,'read_text',return_value='0::/slurm/job_17/step_batch/child'),patch.object(gate.signal,'pidfd_send_signal',create=True) as sent,patch.object(gate.time,'sleep'):
        owner.cleanup();sent.assert_called_once_with(123,gate.signal.SIGTERM)


def test_missing_slurm_before_any_service():
    with patch.dict(os.environ,{},clear=True),patch.object(gate.OwnedProcesses,'spawn') as launch:
        with pytest.raises(RuntimeError,match='EXECUTION_PROOF_MISSING'):
            gate.run(SimpleNamespace(stage='smoke2'),{},Path('/teacher'))
        launch.assert_not_called()


def test_sbatch_cpu_quota_no_global_kill_or_whole_node():
    text=Path(__file__).with_name('topd-five-step.sbatch').read_text()
    assert '--gres=gpu:8' in text and '--cpus-per-task=128' in text and '--mem=384G' in text
    assert '--exclusive' not in text and 'ray stop' not in text and 'pkill' not in text


@pytest.mark.parametrize('stage,budget',[('smoke2',3),('accept5',6)])
def test_real_hydra_compose_preserves_original_inputs_and_post_split_hash(tmp_path,monkeypatch,stage,budget):
    from omegaconf import OmegaConf
    repo=(Path(os.environ['AGENT_ROOT'])/'repo' if 'AGENT_ROOT' in os.environ else Path(__file__).parent/'topd')
    monkeypatch.syspath_prepend(str(repo.resolve()))
    import search_r1.baselines.holdout as holdout
    root=tmp_path/'own';root.mkdir();(root/'repo').symlink_to(repo.resolve(),target_is_directory=True)
    run=root/'run';run.mkdir();teacher=root/'teacher';teacher.mkdir()
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setenv('SEARCH_R1_ASSETS',str(root/'assets-root'))
    import pyarrow as pa
    import pyarrow.parquet as pq
    validation=root/'assets-root/data/paper-grpo/validation-64.parquet'
    validation.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table({'uid':list(range(64))}),validation)
    original_train=str(root/'assets-root/data/paper-grpo/train.parquet')
    retained=str(root/'assets-root/common-data/retained.parquet')
    def split(config):
        assert config.data.train_files==original_train  # second split cannot read retained input
        config.data.train_files=retained
        config.baselines.bapo.stage_validation_files=str(root/'assets-root/common-data/holdout.parquet')
        return dict(content_identity=dict(sha256=gate.CONTENT),holdout_rows=3393)
    monkeypatch.setattr(holdout,'prepare_common_holdout',split)
    actual,_,lr=gate.build_config(run,teacher,stage)
    saved_original=OmegaConf.load(run/'input-config.yaml');saved_resolved=OmegaConf.load(run/'resolved-config.yaml')
    assert saved_original.data.train_files==original_train and saved_resolved.data.train_files==retained
    assert saved_original.baselines.local_runtime_only and saved_original.baselines.local_runtime_host=='h100-11'
    assert saved_original.actor_rollout_ref.actor.ppo_micro_batch_size==64 and saved_original.data.val_data_num==64
    validation_evidence=json.loads((run/'training-validation-inputs.json').read_text())
    assert validation_evidence['input_rows']==64 and validation_evidence['actual_filtered_rows_required']==64
    assert validation_evidence['actual_filtered_rows_status']=='PENDING_TRAINING_LOG'
    assert 'actual_filtered_rows_verified_in_training_log' not in validation_evidence
    assert saved_original.trainer.total_training_steps==budget and saved_original.trainer.smoke_only
    assert saved_original.actor_rollout_ref.actor.ppo_epochs==2 and saved_original.actor_rollout_ref.rollout.n_agent==5
    assert saved_original.data.train_batch_size==512 and saved_original.data.max_prompt_length==4096
    assert saved_original.retriever.url=='http://127.0.0.1:18085/retrieve'
    assert (run/'resolved-config.yaml').read_bytes()==OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(actual,resolve=True))).encode()
    assert lr['used_lr_by_outer'][0]==0 and len(lr['used_lr_by_outer'])==budget-1
    split(saved_original)  # production main receives raw original and splits once, matches
    assert saved_original.data.train_files==retained


def audit_fixture(tmp_path,steps=2):
    run=tmp_path/'run';(run/'audit').mkdir(parents=True)
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(dict(actor_rollout_ref=dict(actor=dict(ppo_micro_batch_size=64,ppo_mini_batch_size=256,use_dynamic_bsz=False)),trainer=dict(n_gpus_per_node=8,nnodes=1),data=dict(val_data_num=64),baselines=dict(method='topd',local_runtime_only=True,local_runtime_host='h100-11'),retriever=dict(url='http://127.0.0.1:18085/retrieve'))),run/'resolved-config.yaml')
    (run/'training.log').write_text('(main_task pid=1) filtered validation dataset size: 64\n')
    (run/'audit/entry-config.yaml').write_bytes((run/'resolved-config.yaml').read_bytes())
    manifest=dict(stage='smoke2' if steps==2 else 'accept5',code_commit='a'*40,teacher_receipt_sha256='b'*64,teacher_path='/own/teacher',run_id='r',config_sha256=gate.sha(run/'resolved-config.yaml'))
    (run/'launch-manifest.json').write_text(json.dumps(manifest))
    identity=dict(run_id='r',commit='a'*40,config_sha256=manifest['config_sha256'],data_identity=gate.CONTENT)
    entry=dict(identity,kind='config_boundary',actual_config_sha256=manifest['config_sha256'],actual_data_identity=gate.CONTENT,entry_total_training_steps=steps+1,config_scope='resolved Hydra + common holdout before trainer optim.total_training_steps injection')
    (run/'audit/entry.jsonl').write_text(json.dumps(entry)+'\n')
    stat=dict(count=1,finite=1,nonzero=1)
    boundary=dict(identity,kind='teacher_boundary',outer_step=0,training=False,training_modules_count=0,requires_grad_parameters=0,grad_parameters=0,devices=['cpu'],parameter_sha256='c'*64)
    for who in ['driver']+list(map(str,range(8))):
        records=[]
        for step in range(1,steps+1):
            row=dict(identity,kind='driver_outer' if who=='driver' else 'actor_outer',outer_step=step,tensors={key:stat for key in ('old_log_probs','advantages','policy_weights','teacher_log_probs','token_level_rewards','returns')})
            if who=='driver':
                row.update(torch_threads=16,teacher_targets_none=True,teacher_scored_tokens_delta=3,observation_tokens=2)
            else:
                row.update(rows=320,rank=int(who),epochs=2,optimizer_steps=20,grad_finite=20,grad_nonzero=20,skipped_empty_minibatches=0,fixed_unchanged=True,shard_layout_unchanged=True,micro_stats=[{key:stat for key in ('current_log_probs','ratios','policy_loss')} for _ in range(80)],grad_norms=[1.]*20,shard_delta=dict(nonzero=0 if step==1 else 1))
            records.append(row)
        if who=='driver':
            records.insert(0,boundary)
            if steps==5:records.append(dict(boundary,outer_step=5))
        (run/'audit'/f'{who}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    checkpoint=run/'checkpoints/actor'/f'global_step_{steps}';checkpoint.mkdir(parents=True);(checkpoint/'config.json').write_text('{}')
    state=run/'checkpoints/algorithm'/f'global_step_{steps}';state.mkdir(parents=True)
    (state/'state.json').write_text(json.dumps(dict(method='topd',global_step=steps,resume_supported=False,teacher_path='/own/teacher')))
    return run,manifest


@pytest.mark.parametrize('steps',[2,5])
def test_audit_structural_gate_valid_and_read_only(tmp_path,steps):
    run,binding=audit_fixture(tmp_path,steps)
    assert gate.audit_gate(run,steps,binding).is_dir()
    assert not (run/'checkpoint-files.json').exists()  # independent read review does not write evidence


@pytest.mark.parametrize('mutation',['actual_file','actual_identity','actual_budget','missing_rank','nonfinite','teacher_grad','zero_update','empty_micro','empty_grad','missing_tensor','boundary_identity','old_micro_count','bad_rows','validation_missing','validation_filtered63'])
def test_audit_rejects_actual_entry_and_update_failures(tmp_path,mutation):
    run,binding=audit_fixture(tmp_path,5)
    if mutation=='actual_file':(run/'audit/entry-config.yaml').write_text('other: config')
    elif mutation in ('actual_identity','actual_budget'):
        path=run/'audit/entry.jsonl';row=json.loads(path.read_text());row['actual_data_identity' if mutation=='actual_identity' else 'entry_total_training_steps']='bad';path.write_text(json.dumps(row)+'\n')
    elif mutation.startswith('validation_'):
        (run/'training.log').write_text('' if mutation=='validation_missing' else 'filtered validation dataset size: 63')
    elif mutation=='missing_rank':(run/'audit/7.jsonl').write_text('')
    else:
        path=run/'audit'/('driver.jsonl' if mutation in ('teacher_grad','boundary_identity') else '0.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if mutation=='nonfinite':rows[0]['tensors']['advantages']['finite']=0
        elif mutation=='teacher_grad':rows[0]['grad_parameters']=1
        elif mutation=='boundary_identity':rows[0]['run_id']='foreign'
        elif mutation=='empty_micro':rows[0]['micro_stats']=[]
        elif mutation=='empty_grad':rows[0]['grad_norms']=[]
        elif mutation=='old_micro_count':rows[0]['micro_stats']=rows[0]['micro_stats']*8
        elif mutation=='bad_rows':rows[0]['rows']=319
        elif mutation=='missing_tensor':rows[0]['tensors'].pop('old_log_probs')
        else:rows[0]['optimizer_steps']=0
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(RuntimeError):gate.audit_gate(run,5,binding)


def test_run_flag_refuses_before_dependency_reads(monkeypatch,tmp_path):
    binding=tmp_path/'binding';binding.write_text('{}')
    monkeypatch.setattr('sys.argv',['topd-short-run.py','--stage','smoke2','--binding',str(binding),'--binding-sha',gate.sha(binding),'--run'])
    with patch.object(gate,'dependency_gate') as read:
        with pytest.raises(RuntimeError,match='EXECUTION_PROOF_MISSING'):
            gate.main()
        read.assert_not_called()


def test_sbatch_read_only_gate_failure_prevents_preflight_and_rejects_run_flag(tmp_path):
    import subprocess
    text=Path(__file__).with_name('topd-five-step.sbatch').read_text()
    root=tmp_path/'own';(root/'repo').mkdir(parents=True);(root/'env/bin').mkdir(parents=True)
    log=root/'calls';python=root/'env/bin/python'
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> '+str(log)+'\nexit 78\n');python.chmod(0o700)
    wrapper=tmp_path/'wrapper';wrapper.write_text(text.replace('/dataset/shaoshitong/search-r1-rl/topd',str(root)))
    result=subprocess.run(['bash',str(wrapper),'--stage','smoke2'],capture_output=True,text=True)
    assert result.returncode==78
    calls=log.read_text().splitlines()
    assert len(calls)==1 and '--check-execution-proof' in calls[0] and 'preflight' not in calls[0] and '--run' not in calls[0]
    rejected=subprocess.run(['bash',str(wrapper),'--run'],capture_output=True,text=True)
    assert rejected.returncode==64 and len(log.read_text().splitlines())==1



@pytest.mark.parametrize('case',['63rows','holdout_selector','holdout3393','wrong_cap'])
def test_training64_scope_cannot_change_teacher3393(tmp_path,monkeypatch,case):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    path=tmp_path/'assets-root/data/paper-grpo/validation-64.parquet';path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'uid':list(range(63 if case=='63rows' else 64))}),path)
    config=OmegaConf.create(dict(baselines=dict(method='topd',holdout_eval_only=case=='holdout_selector'),data=dict(val_data_num=None if case=='wrong_cap' else 64,val_files=str(path))))
    if case=='holdout3393':config.data.val_files=str(tmp_path/'common-holdout3393.parquet')
    with pytest.raises(RuntimeError):gate.training_validation_inputs(config)


def test_public_baseline_audit_disabled_without_losing_run_paths():
    original=dict(SEARCH_R1_BASELINE_AUDIT_DIR='/same/audit',SEARCH_R1_BASELINE_RUN_ID='old',SEARCH_R1_BASELINE_OUTPUT='/own/output',SEARCH_R1_BASELINE_RUN_ROOT='/own/short')
    environment=gate.local_runtime_env(original)
    assert 'SEARCH_R1_BASELINE_AUDIT_DIR' not in environment and 'SEARCH_R1_BASELINE_RUN_ID' not in environment
    assert environment['SEARCH_R1_BASELINE_OUTPUT']=='/own/output' and environment['SEARCH_R1_BASELINE_RUN_ROOT']=='/own/short'


def test_gpu_service_only_has_overlay_and_preserves_allocated_devices(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    for asset in gate.RETRIEVAL_ASSETS:
        path=tmp_path/asset;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    devices=','.join('GPU-'+str(index) for index in range(8))
    with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES=devices,SEARCH_R1_GPU_LEASES='secretfds',SEARCH_R1_GPU_PREFLIGHT='1'),clear=True),patch.object(gate,'inherited_lease_fds',return_value=tuple(range(3,11))) as leases:
        command,environment=gate.gpu_retriever_spec()
    leases.assert_called_once()
    assert environment['CUDA_VISIBLE_DEVICES']==devices and '--faiss_gpu' in command
    assert command[command.index('--device')+1]=='cuda'
    assert command[2].startswith('import torch,runpy,sys;') and 'sys.argv[0]=server' in command[2]
    assert environment['PYTHONPATH']==str(tmp_path/'retriever-gpu-lib')+os.pathsep+str(tmp_path/'repo')
    assert 'SEARCH_R1_GPU_LEASES' not in environment and 'SEARCH_R1_GPU_PREFLIGHT' not in environment
    assert 'PYTHONPATH' not in gate.local_runtime_env(environment)  # trainer never sees overlay


def test_gpu_service_cannot_start_without_actual_lease_validation():
    with patch.object(gate,'inherited_lease_fds',side_effect=ValueError('lease foreign')),patch.object(gate,'local_assets') as assets:
        with pytest.raises(ValueError,match='lease foreign'):gate.gpu_retriever_spec()
        assets.assert_not_called()


def test_idle_wait_retries_only_public_threshold_not_unknown(tmp_path):
    service=SimpleNamespace(poll=lambda:None)
    with patch.object(gate,'inherited_lease_fds',side_effect=[RuntimeError('Allocated GPU0 exceeds the 15% idle thresholds'),tuple(range(3,11))]),patch.object(gate,'gpu_sample',return_value={}),patch.object(gate.time,'sleep'):
        assert len(gate.wait_for_gpu_idle(service,tmp_path/'idle'))==8
        assert [json.loads(line)['kind'] for line in (tmp_path/'idle').read_text().splitlines()]==['idle_wait','idle_guard_pass']
    with patch.object(gate,'inherited_lease_fds',side_effect=RuntimeError('unknown/invalid occupancy')),patch.object(gate.time,'sleep') as sleep:
        with pytest.raises(RuntimeError,match='unknown'):gate.wait_for_gpu_idle(service,tmp_path/'invalid')
        sleep.assert_not_called()


def test_idle_memory_above15_cannot_be_exempted(tmp_path):
    with patch.object(gate,'inherited_lease_fds',side_effect=RuntimeError('Allocated GPU0 exceeds the 15% idle thresholds')),patch.object(gate,'gpu_sample',return_value={}),patch.object(gate.time,'monotonic',side_effect=[0,121]):
        with pytest.raises(RuntimeError,match='cannot meet public absolute'):gate.wait_for_gpu_idle(SimpleNamespace(poll=lambda:None),tmp_path/'idle',120)


def overlay_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    root=tmp_path/'retriever-gpu-lib';root.mkdir()
    (root/'faiss').mkdir();(root/'faiss/__init__.py').write_text('real fixture source')
    (root/'faiss/gpu.so').write_bytes(b'fixture extension bytes')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(files={str(p.relative_to(root)):gate.sha(p) for p in root.rglob('*') if p.is_file()})))
    return root,dict(retriever_gpu_overlay_manifest=dict(path=str(manifest),sha256=gate.sha(manifest)))


def test_gpu_overlay_exact_inventory_and_bytecode_exclusion(tmp_path,monkeypatch):
    overlay,binding=overlay_fixture(tmp_path,monkeypatch)
    cache=overlay/'faiss/__pycache__';cache.mkdir();(cache/'cached.pyc').write_bytes(b'cache')
    assert gate.validate_gpu_overlay(binding)==overlay
    (overlay/'unexpected.so').write_bytes(b'extra')
    with pytest.raises(RuntimeError,match='inventory changed'):gate.validate_gpu_overlay(binding)


def test_gpu_overlay_old_pass_cannot_hide_mutated_binary(tmp_path,monkeypatch):
    overlay,binding=overlay_fixture(tmp_path,monkeypatch)
    (overlay/'faiss/gpu.so').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='file changed'):gate.validate_gpu_overlay(binding)


def test_public_gpu_protocol_source_identity_rejected_before_process():
    binding=dict(common_gpu_protocol_fix=dict(source_commit='wrong'),original_search_r1_commit=gate.ORIGINAL_SEARCH_R1)
    with patch.object(gate.subprocess,'run') as process:
        with pytest.raises(RuntimeError,match='Unreviewed public'):gate.protocol_binding(binding)
        process.assert_not_called()


def test_owned_spawn_actual_descriptor_forwarding_only_when_explicit(tmp_path,monkeypatch):
    import fcntl
    import subprocess
    import sys
    monkeypatch.setattr(gate,'ROOT',tmp_path);(tmp_path/'repo').mkdir()
    lock=tmp_path/'lease';fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=gate.OwnedProcesses.__new__(gate.OwnedProcesses);owner.children=[];owner.streams=[]
    code="import os,sys,fcntl; fd=int(sys.argv[1]);os.fstat(fd);p=os.open(sys.argv[2],os.O_RDWR); blocked=False\ntry:fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:blocked=True\nassert blocked;print('INHERITED_OWN_LOCK_PASS')"
    try:
        with patch.object(gate.os,'pidfd_open',return_value=-1,create=True):
            passed=owner.spawn([sys.executable,'-c',code,str(fd),str(lock)],tmp_path/'passed.log',pass_fds=(fd,))
            assert passed.wait(timeout=10)==0
            denied=owner.spawn([sys.executable,'-c','import os,sys;os.fstat(int(sys.argv[1]))',str(fd)],tmp_path/'denied.log')
            assert denied.wait(timeout=10)!=0
        for stream in owner.streams:stream.close()
        assert 'INHERITED_OWN_LOCK_PASS' in (tmp_path/'passed.log').read_text()
    finally:os.close(fd)


def test_sbatch_readonly_proof_precedes_outer_public_preflight_wrapper():
    text=Path(__file__).with_name('topd-five-step.sbatch').read_text()
    assert '-m search_r1.baselines.preflight --' in text
    assert text.index('--check-execution-proof')<text.index('-m search_r1.baselines.preflight --')
    assert 'exit 78' not in text


@pytest.fixture
def synthetic_execution_proof(tmp_path,monkeypatch):
    """Synthetic negative gate fixtures; never evidence of a native PASS."""
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(gate,'recompute_hardware_evidence',lambda evidence:None)  # Full raw oracle tested independently in normalizer fixtures.
    monkeypatch.setattr(gate,'current_scoring_fingerprint',lambda:'f'*64)
    monkeypatch.setattr(gate,'current_native_runtime',lambda:dict(python='3.10',torch='2.4.0+cu121',tensordict='0.5.0'))
    def save(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value)+'\n' if not isinstance(value,str) else value)
        return dict(path=str(path),sha256=gate.sha(path))
    state=dict(parameter_sha256='c'*64,parameter_bytes=12343754752,devices=['cpu'],buffer_devices=[],dtypes=['torch.float32'],training=False,training_modules_count=0,requires_grad_parameters=0,grad_parameters=0)
    services=[]
    for rank in range(8):
        uuid='GPU-'+str(rank).zfill(32);pid=100+rank
        ready=dict(rank=rank,pid=pid,uuid=uuid,cgroup='/slurm/job_5',source_sha256='f'*64,run_id='HARDWARE_FIXTURE',code_commit='a'*40,config_sha256='d'*64,data_identity=gate.CONTENT,teacher_receipt_sha256='e'*64,loaded_cpu=state)
        rows=[]
        for step in (1,2,3):
            row=dict(ready,outer_step=step,slurm_job='5',targets_none=True,dtype='float32',attention='eager',tf32=False,status='RELEASED',allocator_released=dict(allocated_bytes=0,reserved_bytes=0),rows=2,scored_tokens_delta=100,input_sha256='1'*64,output_sha256='2'*64,row_indices_sha256='3'*64,released_cpu=state,gpu_before=dict(state,devices=['cuda:0']) if step==1 else None,gpu_after=dict(state,devices=['cuda:0']) if step==3 else None)
            if rank==7 and step==3:row['scored_tokens_delta']=6432
            row.update(allocator_peak=dict(allocated_bytes=12343754752,reserved_bytes=13000000000),gpu_pid_before=dict(pid=pid,uuid=uuid,used_mib=300.),gpu_pid_phase_sample=dict(pid=pid,uuid=uuid,used_mib=14000.),gpu_pid_released=dict(pid=pid,uuid=uuid,used_mib=300.))
            rows.append(row)
        services.append(dict(pid=pid,uuid=uuid,ready=save(f'ready{rank}.json',ready),receipts=save(f'score{rank}.jsonl',''.join(json.dumps(row)+'\n' for row in rows))))
    runtime=dict(python='3.10',torch='2.4.0+cu121',tensordict='0.5.0')
    phases=[dict(outer_step=step,rows=16,row_order_verified=True,masked_zero_verified=True,finite=True,all_eight_released_before_return=True,actual_peer_pid_cgroup_verified=True,lease_uuid_ownership_verified=True,gpu_parity=dict(reference='ORIGINAL_GPU_FUNCTION_ORACLE',passed=True,atol=1e-4,rtol=1e-4,max_abs_delta=1e-5,mean_abs_delta=1e-6),cpu_cross_backend_diagnostic=dict(scope='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC',numeric_close=False,atol=1e-4,rtol=1e-4,max_absolute_delta=.001,mean_absolute_delta=.0001,violation_count=4,per_row_violation_counts=[4]+[0]*15)) for step in (1,2,3)]
    phases[-1].update(prompt_width=2048,response_width=4096,cpu_parity_scope='NOT_MEASURED_LONG_CASE')
    hardware=dict(numerical_reference='ORIGINAL_GPU_FUNCTION_ORACLE',status='PASS_NO_TRAINING',kind='HARDWARE_SURROGATE_NO_TRAINING',host='h100-11',source_fingerprint='f'*64,code_commit='a'*40,runtime=runtime,offline=True,services=services,cgroup='/slurm/job_5',slurm_job='5',initial_lease_idle_verified=True,phases=phases,coexistence=dict(local_gpu_e5=True,gpu_fp16_faiss_all8=True,full_wiki18=True,vllm_count=8,vllm_memory_utilization=.6,same_host=True,actual_retrieval_topk=3,uuids=[s['uuid'] for s in services]),cleanup=dict(passed=True,baseline_preserved=True,own_children_remaining=0,owned_model_process_count=18,parent_driver_cuda_initialized=False),probe_script=save('probe.py','synthetic hardware probe only'),raw_evidence=[save('raw-hardware.json',dict(scope='SYNTHETIC_FIXTURE_ONLY'))])
    hardware['coexistence'].update(vllm_services=[dict(uuid=s['uuid'],pid=200+i) for i,s in enumerate(services)],retriever_pid=300,gpu_samples=save('samples.jsonl',json.dumps(dict(processes=[f"{s['uuid']}, {s['pid']}, 12000" for s in services]+[f"{s['uuid']}, {200+i}, 14000" for i,s in enumerate(services)]+[f"{s['uuid']}, 300, 1500" for s in services]))+'\n'))
    tool_names=['topd-short-run.py','test_topd_launch_gate.py','test_topd_local_gpu_ipc.py','verify-topd-teacher.py','verify-topd-checkpoint.py','topd-five-step.sbatch','build_topd_execution_evidence.py','test_build_topd_execution_evidence.py']
    tools={name:save('tools/'+name,Path(gate.__file__).read_text() if name=='topd-short-run.py' else 'synthetic tool '+name) for name in tool_names}
    proof=dict(status='REVIEWED',scope='HARDWARE_SCORER_READY_NO_TRAINING',host='h100-11',code_commit='a'*40,source_fingerprint='f'*64,runtime=runtime,cpu_suites={name:save(name+'.log',f'{count} passed in 1.00s\n') for name,count in [('source',205),('tools',gate.NATIVE_TOOL_FIXTURES)]},tool_files=tools,hardware_evidence=save('hardware.json',hardware))
    proof['cpu_suites']['source']=save('source.log','a'*40+'\nACTUAL_GPU_SCORER_SOURCE '+'f'*64+'\n205 passed in 1.00s\n')
    proof['cpu_suites']['tools']=save('tools.log',''.join(item['sha256']+'  '+item['path']+'\n' for item in tools.values())+f'{gate.NATIVE_TOOL_FIXTURES} passed in 1.00s\n')
    binding=dict(code_commit='a'*40,native_fixture_counts=dict(source=205,tools=gate.NATIVE_TOOL_FIXTURES))
    def resign():
        proof['hardware_evidence']=save('hardware.json',hardware)
        payload={key:value for key,value in proof.items() if key!='independent_review'}
        review=dict(status='PASS',approval_scope='HARDWARE_ONLY_NOT_ALGORITHM_ACCEPTANCE',reviewed_payload_sha256=gate.canonical_sha(payload),report=save('review.md','SYNTHETIC gate test only, no actual approval'))
        proof['independent_review']=save('review.json',review)
        binding['execution_proof']=save('proof.json',proof)
    resign()
    return binding,proof,hardware,resign


def test_complete_synthetic_gate_validates_structure_without_launch(synthetic_execution_proof):
    binding,_,_,_=synthetic_execution_proof
    with patch.object(gate.subprocess,'run') as child,patch.object(gate.OwnedProcesses,'spawn') as launched:
        result=gate.execution_proof_gate(binding)
        assert result['status']=='HARDWARE_PROOF_CHECKED_NOT_ALGORITHM_ACCEPTED'
        child.assert_not_called();launched.assert_not_called()


@pytest.mark.parametrize('change',['source','tool','missing_phase','rows','peer','release','frozenhash','dtype','pid','uuid','longscope','tol','offline','cleanup','scope','raw_drift','cpu_skipped','review_scope','runtime','nvpid','nvpeak','splitpeaks','maskedlong','old_tool_log'])
def test_resigned_synthetic_readiness_rejects_incomplete_or_drifted_evidence(synthetic_execution_proof,change):
    binding,proof,hardware,resign=synthetic_execution_proof
    if change=='source':proof['source_fingerprint']='0'*64
    elif change=='tool':Path(proof['tool_files']['test_topd_launch_gate.py']['path']).write_text('changed')
    elif change=='missing_phase':hardware['phases'].pop()
    elif change=='rows':hardware['phases'][0]['rows']=15
    elif change=='peer':hardware['phases'][0]['actual_peer_pid_cgroup_verified']=False
    elif change in ('release','frozenhash','dtype','nvpid','nvpeak','maskedlong'):
        item=hardware['services'][0]['receipts'];path=Path(item['path']);rows=[json.loads(line) for line in path.read_text().splitlines()]
        if change=='release':rows[1]['allocator_released']['allocated_bytes']=33554432
        elif change=='frozenhash':rows[1]['released_cpu']['parameter_sha256']='0'*64
        elif change=='dtype':rows[1]['dtype']='float16'
        elif change=='nvpid':rows[1]['gpu_pid_phase_sample']['pid']=9999
        elif change=='nvpeak':rows[1]['allocator_peak']['allocated_bytes']=1
        else:
            item=hardware['services'][7]['receipts'];path=Path(item['path']);rows=[json.loads(line) for line in path.read_text().splitlines()];rows[2]['scored_tokens_delta']=288
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows));item['sha256']=gate.sha(path)
    elif change=='pid':hardware['services'][1]['pid']=hardware['services'][0]['pid']
    elif change=='uuid':hardware['services'][1]['uuid']=hardware['services'][0]['uuid']
    elif change=='longscope':hardware['phases'][2]['response_width']=500
    elif change=='tol':hardware['phases'][0]['gpu_parity']['rtol']=1e-3
    elif change=='offline':hardware['offline']=False
    elif change=='runtime':proof['runtime']['torch']='2.8.0'
    elif change=='cleanup':hardware['cleanup']['own_children_remaining']=1
    elif change=='scope':proof['scope']='FIVE_ITER_ACCEPTED'
    elif change=='raw_drift':Path(hardware['raw_evidence'][0]['path']).write_text('drift')
    elif change=='splitpeaks':
        item=hardware['coexistence']['gpu_samples'];path=Path(item['path']);sample=json.loads(path.read_text());lines=sample['processes']
        path.write_text(''.join(json.dumps(dict(processes=[line for line in lines if line.split(',')[1].strip()!=str(100+rank)]))+'\n' for rank in range(8)));item['sha256']=gate.sha(path)
    elif change=='old_tool_log':
        item=proof['tool_files']['test_topd_launch_gate.py'];path=Path(item['path']);path.write_text('new bytes after old native log');item['sha256']=gate.sha(path)
    elif change=='cpu_skipped':
        item=proof['cpu_suites']['tools'];path=Path(item['path']);path.write_text('85 passed, 1 skipped in 1.00s\n');item['sha256']=gate.sha(path)
    resign()
    if change=='review_scope':
        item=proof['independent_review'];path=Path(item['path']);review=json.loads(path.read_text());review['approval_scope']='ALGORITHM_ACCEPTED';path.write_text(json.dumps(review));item['sha256']=gate.sha(path)
        path=Path(binding['execution_proof']['path']);path.write_text(json.dumps(proof));binding['execution_proof']['sha256']=gate.sha(path)
    with pytest.raises(RuntimeError):gate.execution_proof_gate(binding)


def test_read_only_mode_wins_even_if_run_flag_supplied(tmp_path,monkeypatch):
    path=tmp_path/'binding';path.write_text('{}')
    monkeypatch.setattr('sys.argv',['topd-short-run.py','--stage','smoke2','--binding',str(path),'--binding-sha',gate.sha(path),'--check-execution-proof','--run'])
    with patch.object(gate,'execution_proof_gate',return_value={}) as proof,patch.object(gate,'dependency_gate',return_value=({},Path('/teacher'))),patch.object(gate,'run') as launch,patch.object(gate.OwnedProcesses,'spawn') as spawned:
        gate.main();proof.assert_called_once();launch.assert_not_called();spawned.assert_not_called()


def test_valid_proof_still_needs_real_slurm_before_any_service():
    with patch.dict(os.environ,{},clear=True),patch.object(gate,'execution_proof_gate',return_value={}),patch.object(gate.OwnedProcesses,'spawn') as spawned:
        with pytest.raises(RuntimeError,match='Need actual Slurm GPU allocation'):
            gate.run(SimpleNamespace(stage='smoke2'),{},Path('/teacher'))
        spawned.assert_not_called()


def test_native_cpu_log_slice_must_match_frozen_raw_parent(synthetic_execution_proof):
    binding,proof,_,resign=synthetic_execution_proof
    item=proof['cpu_suites']['source'];child=Path(item['path']);parent=child.with_name('raw-source-parent.log')
    prefix=b'actual batch header\n';parent.write_bytes(prefix+child.read_bytes()+b'actual cleanup tail\n')
    item['provenance']=dict(parent=dict(path=str(parent),sha256=gate.sha(parent)),byte_offset=len(prefix),byte_length=child.stat().st_size)
    resign();gate.execution_proof_gate(binding)
    item['provenance']['byte_offset']+=1;resign()
    with pytest.raises(RuntimeError,match='not an exact raw-parent slice'):gate.execution_proof_gate(binding)


def test_cleanup_waits_for_slow_direct_exit_and_reaps_before_final_scope(tmp_path):
    from unittest.mock import Mock
    owner=gate.OwnedProcesses.__new__(gate.OwnedProcesses)
    owner.scope=tmp_path;owner.relative='/slurm/job_17/step_batch';owner.baseline={10};child=Mock();owner.children=[(child,456)];owner.streams=[]
    clock=[0.]
    def sleep(amount):clock[0]+=amount
    # The own direct service remains beyond the old4.75s budget, then exits at7s.
    def pids(scope):return {10,20} if clock[0]<7 else {10}
    def membership(path,*args,**kwargs):return '0::/slurm/job_17/step_batch'
    with patch.object(gate,'cgroup_pids',side_effect=pids),patch.object(gate,'process_identity',return_value=dict(pid=20,state='S',ppid=1,starttime_ticks=123,cgroup=['0::/slurm/job_17/step_batch'])),patch.object(gate.os,'pidfd_open',return_value=123,create=True),patch.object(gate.os,'close'),patch.object(gate.Path,'read_text',membership),patch.object(gate.signal,'pidfd_send_signal',create=True) as sent,patch.object(gate.time,'sleep',side_effect=sleep),patch.object(gate.time,'monotonic',side_effect=lambda:clock[0]):
        owner.cleanup()
    assert owner.cleanup_record['actual_cgroup_empty'] and owner.cleanup_record['elapsed_seconds']==7.
    assert [row['phase'] for row in owner.cleanup_record['signals']]==['TERM','KILL']
    assert sent.call_count==2 and child.poll.call_count>2;child.wait.assert_called_once_with(timeout=10)


def test_cleanup_persistent_member_fails_with_actual_identity_and_closes_own_handles(tmp_path):
    from unittest.mock import Mock
    owner=gate.OwnedProcesses.__new__(gate.OwnedProcesses);owner.scope=tmp_path;owner.relative='/slurm/job_17/step_batch';owner.baseline={10};child=Mock();stream=Mock();owner.children=[(child,456)];owner.streams=[stream]
    clock=[0.]
    def sleep(amount):clock[0]+=amount
    identity=dict(pid=20,state='D',ppid=19,starttime_ticks=123,cgroup=['0::/slurm/job_17/step_batch'])
    with patch.object(gate,'cgroup_pids',return_value={10,20}),patch.object(gate,'process_identity',return_value=identity),patch.object(gate.os,'pidfd_open',return_value=123,create=True),patch.object(gate.os,'close') as close,patch.object(gate.Path,'read_text',return_value='0::/slurm/job_17/step_batch'),patch.object(gate.signal,'pidfd_send_signal',create=True),patch.object(gate.time,'sleep',side_effect=sleep),patch.object(gate.time,'monotonic',side_effect=lambda:clock[0]):
        with pytest.raises(RuntimeError,match='descendants remain'):owner.cleanup()
    assert owner.cleanup_record['remaining']==[identity] and not owner.cleanup_record['actual_cgroup_empty'];assert clock[0]==30.
    close.assert_any_call(456);stream.close.assert_called_once();child.wait.assert_not_called()


def test_process_identity_records_state_ppid_starttime_cgroup(tmp_path):
    # /proc/stat comm may contain spaces or closing parentheses; split at final')'.
    tail=['D','19']+['0']*17+['12345']
    def read(path,*args,**kwargs):return '20 (own service) tricky) '+ ' '.join(tail) if path.name=='stat' else '0::/slurm/job_17/step_batch\n'
    with patch.object(gate.Path,'read_text',read):r=gate.process_identity(20)
    assert r==dict(pid=20,comm='own service) tricky',state='D',ppid=19,starttime_ticks=12345,cgroup=['0::/slurm/job_17/step_batch'])


def test_process_identity_missing_is_unknown_not_zombie():
    with patch.object(gate.Path,'read_text',side_effect=FileNotFoundError):r=gate.process_identity(20)
    assert r['identity_unavailable'] and 'state' not in r and 'ppid' not in r


def test_stop_monitor_drains_child_spawning_thread_before_cleanup():
    import threading
    event=threading.Event();exited=threading.Event()
    def worker():event.wait();exited.set()
    thread=threading.Thread(target=worker);thread.start()
    r=gate.stop_monitor(event,thread,timeout=1.)
    assert r['stopped'] and exited.is_set() and not thread.is_alive()


def test_stop_monitor_alive_fails_closed():
    from unittest.mock import Mock
    thread=Mock();thread.is_alive.return_value=True;event=Mock()
    with pytest.raises(RuntimeError,match='sampler thread did not stop'):gate.stop_monitor(event,thread,timeout=30.)
    event.set.assert_called_once();thread.join.assert_called_once_with(timeout=30.)


def test_stop_monitor_not_started_is_valid():
    import threading
    event=threading.Event();r=gate.stop_monitor(event,None)
    assert r['stopped'] and event.is_set()


def test_gate_cannot_bypass_failed_raw_oracle_recomputation(synthetic_execution_proof,monkeypatch):
    binding,_,_,_=synthetic_execution_proof
    def failed(evidence):raise RuntimeError('original oracle raw input changed')
    monkeypatch.setattr(gate,'recompute_hardware_evidence',failed)
    with pytest.raises(RuntimeError,match='raw input changed'):gate.execution_proof_gate(binding)


@pytest.mark.parametrize('change',['reference','cpu_counts','owner_count','parent_cuda'])
def test_gate_rejects_oracle_contract_drift(synthetic_execution_proof,change):
    binding,_,hardware,resign=synthetic_execution_proof
    if change=='reference':hardware['numerical_reference']='CPU_GPU_PARITY'
    elif change=='cpu_counts':hardware['phases'][0]['cpu_cross_backend_diagnostic']['violation_count']=3
    elif change=='owner_count':hardware['cleanup']['owned_model_process_count']=17
    else:hardware['cleanup']['parent_driver_cuda_initialized']=True
    resign()
    with pytest.raises(RuntimeError):gate.execution_proof_gate(binding)
