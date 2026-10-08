"""Synthetic control metadata plus real CPU safetensors; no native/GPU evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import pytest
import torch
from safetensors.torch import save,save_file

spec=importlib.util.spec_from_file_location('evidence_builder',Path(__file__).with_name('build_topd_execution_evidence.py'))
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def raw(tmp_path,monkeypatch):
    root=tmp_path/'own';directory=root/'logs/fixture';scripts=root/'tools/afunix-scorer-probe'
    scripts.mkdir(parents=True);directory.mkdir(parents=True)
    source=Path(__file__).parent/'topd/search_r1/baselines'
    if not source.is_dir():
        import os
        source=Path(os.environ['AGENT_ROOT'])/'repo/search_r1/baselines'
    for name in builder.SOURCE_FILES:
        path=root/'repo/search_r1/baselines'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,path)
    fingerprint=builder.digest(json.dumps({name:builder.sha(root/'repo/search_r1/baselines'/name) for name in builder.SOURCE_FILES},sort_keys=True).encode())
    launcher=root/'tools/topd-short-run.py';launcher.write_text('fixture launcher; not native tested')
    script=scripts/'probe.py';script.write_text('fixture raw producer; no native claims')
    oracle_script=scripts/'original_gpu_oracle.py';oracle_script.write_text('synthetic original oracle fixture')
    commit='a'*40
    reviewed=dict(code_commit=commit,source_fingerprint=fingerprint,files={str(p.relative_to(root)):builder.sha(p) for p in [launcher]+list((root/'repo/search_r1/baselines').glob('*.py'))})
    write(scripts/'reviewed-sources.json',reviewed)
    monkeypatch.setattr(builder,'native_context',lambda root:dict(host='h100-11',code_commit=commit,runtime=builder.PINNED))
    monkeypatch.setattr(builder,'fp32_storage',lambda root,files:12343754752)  # Synthetic full-model metadata ONLY.
    base=root/'assets-root/assets/qwen2.5-3b';base.mkdir(parents=True);save_file({'tiny_fixture':torch.tensor([1.,2.])},str(base/'model.safetensors'))
    hf_files={'model.safetensors':builder.sha(base/'model.safetensors')};hf_sha=builder.digest(json.dumps(hf_files,sort_keys=True).encode())
    config=dict(rounds=3,rows_per_service=2,prompt_lengths=[256,256,2048],response_lengths=[32,500,4096],temperature=1.,chunk=16,atol=1e-4,rtol=1e-4)
    effective=dict(gpu_memory_utilization=.6,tensor_parallel_size=1,enforce_eager=True);config['rollout']=effective
    write(directory/'hardware-config.json',config);write(directory/'effective-rollout-config.json',effective)
    services=[dict(rank=i,pid=100+i,uuid='GPU-'+str(i).zfill(32)) for i in range(8)]
    manifest=dict(kind='HARDWARE_SURROGATE_NO_TRAINING',backend='local_gpu_fp32_parallel',execution_device='cuda:0',outer_updates=3,rows_per_service=2,source_sha256=fingerprint,code_commit=commit,slurm_job='1',temperature=1.,chunk=16,teacher_path=str(base),student_path=str(base),hardware_model_files=hf_files,teacher_receipt_sha256=hf_sha,frozen_hf_files_sha256=hf_sha,config_sha256=builder.sha(directory/'hardware-config.json'),student_tokenizer=dict(pad_token_id=0),services=services,cgroup='/slurm/job_1',run_id='FIXTURE_ONLY',data_identity='f'*64)
    write(directory/'teacher-ipc/manifest.json',manifest)
    state=dict(parameter_sha256='b'*64,parameter_bytes=12343754752,devices=['cpu'],buffer_devices=[],dtypes=['torch.float32'],training=False,training_modules_count=0,requires_grad_parameters=0,grad_parameters=0)
    oracle_dir=directory/'original-gpu-oracle';oracle_dir.mkdir()
    common={key:manifest[key] for key in builder.IDENTITY}
    common.update(scope=builder.ORACLE_SCOPE,pid=400,uuid=services[4]['uuid'],rank=4,cgroup=manifest['cgroup'],slurm_job='1',host='h100-11',source_kind='HARDWARE_SURROGATE_NO_TRAINING',manifest_sha256=builder.sha(directory/'teacher-ipc/manifest.json'),oracle_script_sha256=builder.sha(oracle_script),torch_version=builder.PINNED['torch'],actual_hf_files_sha256=hf_sha,offline=True,loaded_cpu=state,original_function=builder.ORIGINAL_FUNCTION,original_function_source_sha256=builder.method_sha(root/'repo/search_r1/baselines/distillation.py'),original_function_file_sha256=builder.sha(root/'repo/search_r1/baselines/distillation.py'))
    oracle_ready=dict(common,status='CPU_READY_NO_CUDA_CONTEXT',cuda_initialized=False,ready_epoch=.5);write(oracle_dir/'ready.json',oracle_ready)
    allrows=[[] for _ in range(8)];comparisons=[]
    for step,(pw,rw) in enumerate([(256,32),(256,500),(2048,4096)],1):
        prompts=torch.ones((16,pw),dtype=torch.long);responses=torch.ones((16,rw),dtype=torch.long);mask=torch.ones_like(responses)
        if step==3:
            prompts[:15,:1792]=0
            for row in range(15):mask[row,32 if row%2==0 else 500:]=0
        gpu=torch.full((16,rw),-.5)*mask
        tensors=dict(prompts=prompts,responses=responses,attention=mask,gpu_log_probs=gpu,original_gpu_log_probs=gpu.clone())
        if step<3:tensors['cpu_log_probs']=gpu.clone()
        file=directory/f'parity-round{step}.safetensors';save_file(tensors,str(file))
        phase_rows=[]
        for rank,service in enumerate(services):
            region=slice(rank*2,rank*2+2);indices=torch.arange(rank*2,rank*2+2)
            incoming=save(dict(prompts=prompts[region].contiguous(),responses=responses[region].contiguous(),attention=mask[region].contiguous(),row_indices=indices))
            outgoing=save(dict(teacher_log_probs=gpu[region].contiguous(),row_indices=indices))
            row={key:manifest[key] for key in builder.IDENTITY}
            row.update(rank=rank,pid=service['pid'],uuid=service['uuid'],cgroup=manifest['cgroup'],slurm_job='1',rows=2,outer_step=step,loaded_cpu=state,status='RELEASED',targets_none=True,dtype='float32',attention='eager',tf32=False,allocator_released=dict(allocated_bytes=0,reserved_bytes=0),released_cpu=state,gpu_before=dict(state,devices=['cuda:0']) if step==1 else None,gpu_after=dict(state,devices=['cuda:0']) if step==3 else None,allocator_peak=dict(allocated_bytes=12343754752),input_sha256=builder.digest(incoming),output_sha256=builder.digest(outgoing),row_indices_sha256=builder.digest(save(dict(row_indices=indices))),scored_tokens_delta=int((prompts[region]!=0).sum()+mask[region].sum()))
            for name in ('gpu_pid_before','gpu_pid_phase_sample','gpu_pid_released'):row[name]=dict(pid=service['pid'],uuid=service['uuid'],used_mib=12000.)
            phase_rows.append(row);allrows[rank].append(row)
        aggregate=directory/f'receipts-round{step}.json';write(aggregate,phase_rows)
        incoming=oracle_dir/f'input-round{step}.safetensors';output=oracle_dir/f'reference-round{step}.safetensors'
        save_file(dict(prompts=prompts,responses=responses,attention=mask,row_indices=torch.arange(16)),str(incoming))
        save_file(dict(original_gpu_log_probs=gpu.clone()),str(output))
        request={key:manifest[key] for key in builder.IDENTITY};request.update(outer_step=step,input_sha256=builder.sha(incoming),all8_released_zero=True,candidate_receipts_sha256=builder.sha(aggregate),scorers_released_epoch=step*10+2.)
        request_path=oracle_dir/f'request-round{step}.json';write(request_path,request)
        tokens=[int((prompts[i]!=0).sum()+mask[i].sum()) for i in range(16)]
        receipt=dict(common,outer_step=step,input_sha256=builder.sha(incoming),request_sha256=builder.sha(request_path),output_sha256=builder.sha(output),rows=16,row_indices=list(range(16)),temperature=1.,chunk=16,targets_none=True,dtype='float32',attention='eager',status='RELEASED',expected_scored_tokens=sum(tokens),scored_tokens_delta=sum(tokens),scored_tokens_by_row=tokens,row_indices_sha256=builder.digest(save(dict(row_indices=torch.arange(16)))),attention_sha256=builder.digest(save(dict(attention=mask))),candidate_receipts_sha256=builder.sha(aggregate),scorers_released_epoch=step*10+2.,actual_flags=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,matmul_precision='highest',autocast_enabled=False,cpu_threads=1,environment=dict(CUDA_VISIBLE_DEVICES=services[4]['uuid'],TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',NVIDIA_TF32_OVERRIDE=None)),cuda_started_epoch=step*10+3.,scoring_started_epoch=step*10+4.,scoring_completed_epoch=step*10+5.,completed_epoch=step*10+6.,gpu_before=dict(state,devices=['cuda:0']),gpu_after=dict(state,devices=['cuda:0']),released_cpu=state,allocator_peak=dict(allocated_bytes=12343754752),allocator_released=dict(allocated_bytes=0,reserved_bytes=0),finite=True,masked_zero_verified=True)
        for name in ('gpu_pid_before','gpu_pid_phase_sample','gpu_pid_released'):receipt[name]=dict(pid=400,uuid=services[4]['uuid'],used_mib=12000.)
        receipt_path=oracle_dir/f'receipt-round{step}.json';write(receipt_path,receipt)
        comparison=dict(step=step,rows=16,rows_per_service=2,prompt_length=pw,response_length=rw,row_order_verified=True,masked_zero_verified=True,finite=True,all8_released_zero=True,all8_fixed_parameter_sha256='b'*64,parity_artifact_sha256=builder.sha(file),receipts_sha256=builder.sha(aggregate),numeric_close=True,atol=1e-4,rtol=1e-4,max_absolute_delta=0.,mean_absolute_delta=0.,scope='ORIGINAL_GPU_FUNCTION_PARITY_BLOCKING',numerical_reference=builder.ORACLE,scoring_started_epoch=step*10.,scoring_completed_epoch=step*10+1.,scorers_released_epoch=step*10+2.,original_oracle_completed_epoch=step*10+7.,original_gpu_oracle_pid=400,original_gpu_oracle_uuid=services[4]['uuid'])
        for role,path in [('receipt',receipt_path),('input',incoming),('output',output)]:comparison['original_gpu_oracle_'+role+'_path']=str(path);comparison['original_gpu_oracle_'+role+'_sha256']=builder.sha(path)
        comparison['cpu_cross_backend_diagnostic']=dict(scope='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC',numeric_close=True,atol=1e-4,rtol=1e-4,max_absolute_delta=0.,mean_absolute_delta=0.,violation_count=0,per_row_violation_counts=[0]*16) if step<3 else None
        comparisons.append(comparison)
    oracle_done=dict(common,status='THREE_ORIGINAL_GPU_PHASES_RELEASED_NO_TRAINING',rounds=3,last_receipt_sha256=builder.sha(oracle_dir/'receipt-round3.json'),done_epoch=38.);write(oracle_dir/'done.json',oracle_done)

    for rank,service in enumerate(services):
        ready={key:manifest[key] for key in builder.IDENTITY};ready.update(rank=rank,pid=service['pid'],uuid=service['uuid'],cgroup=manifest['cgroup'],actual_hf_files_sha256=hf_sha,loaded_cpu=state)
        write(directory/'teacher-ipc'/f'scorer-ready-{rank}.json',ready)
        (directory/'teacher-ipc'/f'scorer-{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in allrows[rank]))
    mapping=[dict(index=i,uuid=s['uuid']) for i,s in enumerate(services)]
    inference=[]
    for rank,m in enumerate(mapping):
        write(directory/'eight-vllm'/f'ready-{rank}.json',dict(rank=rank,pid=200+rank,NO_TRAINING=True,actual_pid_gpu_uuids=[m['uuid']],expected_gpu_uuid=m['uuid'],cuda_visible_devices=str(m['index']),gpu_memory_utilization=.6,tp=1))
        done=dict(rank=rank,NO_TRAINING=True,requests=64,output_tokens=64);inference.append(done);write(directory/'eight-vllm'/f'done-{rank}.json',done)
    baseline=dict(time=1.,phase='baseline',processes=[],gpus=[dict(uuid=s['uuid'],memory_used_mib=0.,memory_total_mib=80000.,utilization=0.) for s in services])
    simultaneous=dict(time=2.,phase='afunix-gpu-scoring-round1',processes=[f"{s['uuid']}, {s['pid']}, 12000" for s in services]+[f"{s['uuid']}, {200+i}, 14000" for i,s in enumerate(services)]+[f"{s['uuid']}, 300, 1500" for s in services])
    (directory/'gpu-memory-timeseries.jsonl').write_text(json.dumps(baseline)+'\n'+json.dumps(dict(time=1.5,phase='eight-scorers-and-original-oracle-cpu-ready',processes=[],own_process_samples=[dict(pid=pid) for pid in list(range(100,108))+[400]]))+'\n'+json.dumps(simultaneous)+'\n'+json.dumps(dict(time=40.,phase='released',processes=[],own_gpu_model_pids_remaining=[]))+'\n')
    request=dict(label='scorer-round1-http0',queries=512,topk=3,started_epoch=1.,completed_epoch=2.);write(directory/(request['label']+'.json'),request)
    server=root/'repo/search_r1/search/retrieval_server.py';server.parent.mkdir(parents=True);server.write_text('fixture retrieval server')
    split={};
    for role in ('train','manifest'):
        path=root/(role+'.data');path.write_text('fixture input');split[role+'_path']=str(path);split[role+'_sha256']=builder.sha(path)
    result=dict(status=builder.PASS,NO_TRAINING=True,job_id='1',cleanup_passed=True,own_children_remaining=0,own_children_remaining_pids=[],baseline_preserved=True,original_gpu_process_identities_missing=[],monitor_errors=[],own_cleanup_record=dict(actual_cgroup_empty=True,remaining=[]),monitor_shutdown=dict(stopped=True),actual_code_commit=commit,torch_version=builder.PINNED['torch'],actual_source_fingerprint=fingerprint,launcher_sha256=builder.sha(launcher),script_sha256={'probe.py':builder.sha(script)},common_config_sha256={},hardware_hf_identity_sha256=hf_sha,scorer_startup=[dict(s,cgroup=manifest['cgroup']) for s in services],ipc_rounds=comparisons,worker_gpu_mapping=mapping,coexistence_samples=[dict(time=2.,phase=simultaneous['phase'],retriever_pid=300)],original_baseline=baseline,requests=[request],inference=inference,retrieval_source_sha256=builder.sha(server),split_artifact=split,effective_rollout_config=effective)
    result.update(own_gpu_model_pids_remaining=[],original_gpu_oracle_startup=oracle_ready,original_gpu_oracle_done=oracle_done,original_gpu_oracle_exit_code=0,parent_driver_cuda_initialized=False,owned_gpu_model_process_count=18,owned_gpu_model_pids=dict(scorers=[p['pid'] for p in services],vllm=list(range(200,208)),retriever=300,original_oracle=400))
    write(directory/'result.json',result)
    return root,directory,script,result


def test_recomputes_cpu_tensor_wire_math_and_normalizes_without_approval(raw):
    root,directory,script,_=raw
    result=builder.normalize(directory,root,script)
    assert result['status']=='PASS_NO_TRAINING' and len(result['services'])==8 and len(result['phases'])==3
    assert result['phases'][0]['gpu_parity']['passed'] and result['phases'][2]['cpu_parity_scope']=='NOT_MEASURED_LONG_CASE'
    assert 'independent_review' not in result and 'REVIEWED' not in result.values()


@pytest.mark.parametrize('change',['failed','empty','cleanup','wire','parity','maskedlong','splitpeaks','source','hf','release','scope','surviving_gpu','retrievalsource','compensatedlong','cleanuprecord','monitorrunning'])
def test_actual_failures_and_resigned_false_summary_cannot_normalize(raw,change):
    root,directory,script,result=raw
    if change=='failed':result['status']='FAIL_NO_TRAINING'
    elif change=='empty':result['ipc_rounds']=[]
    elif change=='cleanup':result['own_children_remaining']=1
    elif change=='cleanuprecord':result['own_cleanup_record']['actual_cgroup_empty']=False
    elif change=='monitorrunning':result['monitor_shutdown']['stopped']=False
    elif change in ('wire','parity','maskedlong','compensatedlong'):
        from safetensors.torch import load_file
        step=3 if change in ('maskedlong','compensatedlong') else 1;path=directory/f'parity-round{step}.safetensors';tensors=load_file(str(path))
        if change=='wire':tensors['responses'][0,0]=2
        elif change=='parity':tensors['original_gpu_log_probs'][0,0]=1.
        else:
            tensors['attention'][15,4095]=0;tensors['gpu_log_probs'][15,4095]=0
            if change=='compensatedlong':tensors['attention'][14,32]=1;tensors['gpu_log_probs'][14,32]=-.5
        save_file(tensors,str(path));result['ipc_rounds'][step-1]['parity_artifact_sha256']=builder.sha(path)
    elif change=='splitpeaks':
        path=directory/'gpu-memory-timeseries.jsonl';baseline,ready,sample,final=[json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(json.dumps(baseline)+'\n'+json.dumps(ready)+'\n'+''.join(json.dumps(dict(sample,processes=[line for line in sample['processes'] if line.split(',')[1].strip()!=str(100+rank)]))+'\n' for rank in range(8))+json.dumps(final)+'\n')
    elif change=='surviving_gpu':
        path=directory/'gpu-memory-timeseries.jsonl';samples=[json.loads(line) for line in path.read_text().splitlines()];samples[-1]['processes']=['GPU-'+str(0).zfill(32)+', 100, 12000'];path.write_text(''.join(json.dumps(s)+'\n' for s in samples))
    elif change=='retrievalsource':(root/'repo/search_r1/search/retrieval_server.py').write_text('drift')
    elif change=='source':(root/'repo/search_r1/baselines/controller.py').write_text('drift')
    elif change=='hf':(root/'assets-root/assets/qwen2.5-3b/model.safetensors').write_bytes(b'drift')
    elif change=='release':
        path=directory/'teacher-ipc/scorer-0.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]['allocator_released']['allocated_bytes']=33554432;path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    else:
        path=directory/'teacher-ipc/manifest.json';manifest=builder.load(path);manifest['kind']='TRAINED_TEACHER_FROZEN';write(path,manifest)
    write(directory/'result.json',result)
    with pytest.raises(RuntimeError):builder.normalize(directory,root,script)


def test_disclosed_cpu_cross_backend_failure_is_not_gpu_interface_failure(raw):
    from safetensors.torch import load_file
    root,directory,script,result=raw
    path=directory/'parity-round1.safetensors';tensor=load_file(str(path));tensor['cpu_log_probs'][8,9]-=.001
    save_file(tensor,str(path));comparison=result['ipc_rounds'][0];comparison['parity_artifact_sha256']=builder.sha(path)
    delta=(tensor['gpu_log_probs']-tensor['cpu_log_probs']).abs();violations=delta>(1e-4+1e-4*tensor['cpu_log_probs'].abs())
    comparison['cpu_cross_backend_diagnostic'].update(numeric_close=False,max_absolute_delta=float(delta.max()),mean_absolute_delta=float(delta.mean()),violation_count=int(violations.sum()),per_row_violation_counts=violations.sum(dim=1).tolist())
    write(directory/'result.json',result)
    normalized=builder.normalize(directory,root,script)
    assert normalized['phases'][0]['gpu_parity']['passed']
    assert normalized['phases'][0]['cpu_cross_backend_diagnostic']['violation_count']==1
    assert normalized['phases'][0]['cpu_cross_backend_diagnostic']['numeric_close'] is False


@pytest.mark.parametrize('change',['cpu_hidden_failure','cpu_false_counts','oracle_reuse_pid','oracle_flags','oracle_source','oracle_input','oracle_release','oracle_overlap','oracle_long_truncation','oracle_exit','parent_cuda','owner_missing','oracle_gpu_survives','oracle_not_cpu_ready'])
def test_independent_oracle_rejects_resigned_incomplete_native_claims(raw,change):
    root,directory,script,result=raw
    ready_path=directory/'original-gpu-oracle/ready.json';receipt_path=directory/'original-gpu-oracle/receipt-round3.json'
    ready=builder.load(ready_path);receipt=builder.load(receipt_path)
    if change=='cpu_hidden_failure':result['ipc_rounds'][0]['cpu_cross_backend_diagnostic']['numeric_close']=False
    elif change=='cpu_false_counts':result['ipc_rounds'][0]['cpu_cross_backend_diagnostic']['violation_count']=4
    elif change=='oracle_reuse_pid':ready['pid']=100;result['original_gpu_oracle_startup']=ready;write(ready_path,ready)
    elif change=='oracle_flags':receipt['actual_flags']['matmul_allow_tf32']=True
    elif change=='oracle_source':receipt['original_function_source_sha256']='0'*64
    elif change=='oracle_input':
        from safetensors.torch import load_file
        path=directory/'original-gpu-oracle/input-round3.safetensors';tensor=load_file(str(path));tensor['responses'][15,0]=2;save_file(tensor,str(path));receipt['input_sha256']=builder.sha(path)
    elif change=='oracle_release':receipt['allocator_released']['reserved_bytes']=33554432
    elif change=='oracle_overlap':receipt['cuda_started_epoch']=result['ipc_rounds'][2]['scorers_released_epoch']-1.
    elif change=='oracle_long_truncation':receipt['scored_tokens_by_row'][15]-=1
    elif change=='oracle_exit':result['original_gpu_oracle_exit_code']=1
    elif change=='parent_cuda':result['parent_driver_cuda_initialized']=True
    elif change=='owner_missing':result['owned_gpu_model_process_count']=17
    elif change=='oracle_gpu_survives':
        path=directory/'gpu-memory-timeseries.jsonl';samples=[json.loads(line) for line in path.read_text().splitlines()];samples[-1]['processes']=[f"{ready['uuid']}, 400, 300"];path.write_text(''.join(json.dumps(row)+'\n' for row in samples))
    else:ready['cuda_initialized']=True;result['original_gpu_oracle_startup']=ready;write(ready_path,ready)
    write(receipt_path,receipt);result['ipc_rounds'][2]['original_gpu_oracle_receipt_sha256']=builder.sha(receipt_path)
    write(directory/'result.json',result)
    with pytest.raises(RuntimeError):builder.normalize(directory,root,script)
