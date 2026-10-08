#!/usr/bin/env python3
"""CPU-only normalization of completed raw hardware evidence; no approval or training."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

PASS='HARDWARE_AFUNIX_PARITY_RELEASE_CORESIDENCY_PASS_NO_TRAINING'
ORACLE='ORIGINAL_GPU_FUNCTION_ORACLE'
ORACLE_SCOPE='ORIGINAL_FROZENTEACHER_GPU_FUNCTION_ORACLE_NO_TRAINING'
ORIGINAL_FUNCTION='search_r1.baselines.distillation.FrozenTeacher.log_probs'
IDENTITY=('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')
SOURCE_FILES=('controller.py','distillation.py','topd_runtime_audit.py','topd_local_gpu.py')
PINNED=dict(python='3.10',torch='2.4.0+cu121',tensordict='0.5.0')


def need(condition,message):
    if not condition:raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def reference(path,root):
    path=Path(path).resolve();need(path.is_file() and path.is_relative_to(root),'Nonlocal/missing raw evidence: '+str(path))
    return dict(path=str(path),sha256=sha(path))


def native_context(root):
    import torch,tensordict
    return dict(host=socket.gethostname().split('.')[0],code_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),
                runtime=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',torch=torch.__version__,tensordict=tensordict.__version__))


def fp32_storage(root,files):
    from safetensors import safe_open
    weights=[root/name for name in files if name.endswith('.safetensors')]
    need(weights,'No actual HF safetensor weights')
    count=0;names=set()
    for path in weights:
        with safe_open(str(path),framework='pt',device='cpu') as model:
            for name in model.keys():
                need(name not in names,'Duplicate HF parameter across shards');names.add(name)
                count+=math.prod(model.get_slice(name).get_shape())
    return count*4


def method_sha(path):
    text=Path(path).read_text();lines=text.splitlines(keepends=True)
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='FrozenTeacher')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='log_probs')
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    return digest(''.join(lines[start-1:node.end_lineno]).encode())


def normalize(directory,root,probe_script):
    """Read completed artifacts only; even success is hardware-only and unreviewed."""
    import torch
    from safetensors.torch import load_file,save
    root=Path(root).resolve();directory=Path(directory).resolve();probe_script=Path(probe_script).resolve()
    need(directory.is_relative_to(root) and probe_script.is_relative_to(root),'Probe artifacts/code must be own local')
    context=native_context(root)
    need(context['host']=='h100-11' and context['runtime']==PINNED,'Builder must run on same pinned local node/environment')
    result=load(directory/'result.json')
    need(result.get('status')==PASS and result.get('NO_TRAINING') is True,'No successful complete raw hardware probe')
    need(result.get('cleanup_passed') is True and result.get('own_children_remaining')==0 and result.get('own_children_remaining_pids')==[]
         and result.get('baseline_preserved') is True and result.get('original_gpu_process_identities_missing')==[]
         and not result.get('monitor_errors') and not any(key in result for key in ('error','cleanup_error','cleanup_final_inventory_error','final_snapshot_error')),'Raw cleanup/monitor failure')
    cleanup=result.get('own_cleanup_record',{});monitor=result.get('monitor_shutdown',{})
    need(cleanup.get('actual_cgroup_empty') is True and cleanup.get('remaining')==[]
         and monitor.get('stopped') is True and 'error' not in monitor,'No actual empty-cgroup cleanup/stopped-monitor diagnostics')
    reviewed=load(probe_script.parent/'reviewed-sources.json')
    need(reviewed['code_commit']==context['code_commit']==result['actual_code_commit'] and result['torch_version']==PINNED['torch'],'Raw/source/current runtime mismatch')
    raw=[]
    def ref(path):
        item=reference(path,root);raw.append(item);return item
    for relative,expected in reviewed['files'].items():
        item=ref(root/relative);need(item['sha256']==expected,'Reviewed source/tool bytes drifted')
    fingerprints={name:sha(root/'repo/search_r1/baselines'/name) for name in SOURCE_FILES}
    fingerprint=digest(json.dumps(fingerprints,sort_keys=True).encode())
    need(fingerprint==reviewed['source_fingerprint']==result['actual_source_fingerprint'],'Scoring implementation fingerprint differs')
    need(result['launcher_sha256']==sha(root/'tools/topd-short-run.py'),'Raw launcher differs from current bytes')
    for name,expected in result['script_sha256'].items():need(ref(probe_script.parent/name)['sha256']==expected,'Probe source drifted')
    need('probe.py' in result['script_sha256'] and probe_script.name=='probe.py','Missing actual probe source')
    for name,expected in result['common_config_sha256'].items():need(ref(root/'repo'/name)['sha256']==expected,'Common runtime config drifted')
    need(result['retrieval_source_sha256']==ref(root/'repo/search_r1/search/retrieval_server.py')['sha256'],'Actual retrieval server source drifted')
    manifest=load(directory/'teacher-ipc/manifest.json');ref(directory/'teacher-ipc/manifest.json')
    need(Path(manifest['student_path']).resolve()==root/'assets-root/assets/qwen2.5-3b','Student assets must be own local base')
    need(manifest['kind']=='HARDWARE_SURROGATE_NO_TRAINING' and manifest['backend']=='local_gpu_fp32_parallel'
         and manifest['execution_device']=='cuda:0' and manifest['outer_updates']==3 and manifest['rows_per_service']==2
         and manifest['source_sha256']==fingerprint and manifest['code_commit']==context['code_commit']
         and manifest['slurm_job']==str(result['job_id']) and manifest['temperature']==1. and manifest['chunk']==16,'Wrong actual hardware socket manifest')
    hfroot=Path(manifest['teacher_path']).resolve();need(hfroot==root/'assets-root/assets/qwen2.5-3b','Hardware must use own frozen base, not a trained-teacher substitute')
    files=manifest['hardware_model_files'];actual={str(path.relative_to(hfroot)) for path in hfroot.rglob('*') if path.is_file()}
    need(files and actual==set(files),'HF inventory changed')
    for name,expected in files.items():need(reference(hfroot/name,root)['sha256']==expected,'Actual HF bytes drifted')
    hf_identity=digest(json.dumps(files,sort_keys=True).encode())
    need(hf_identity==manifest['teacher_receipt_sha256']==manifest['frozen_hf_files_sha256']==result['hardware_hf_identity_sha256'],'Hardware HF source hash mismatch')
    parameter_bytes=fp32_storage(hfroot,files);need(parameter_bytes>10*1024**3,'Tiny fixture cannot qualify full3B production hardware')
    config=load(directory/'hardware-config.json');need(sha(directory/'hardware-config.json')==manifest['config_sha256'],'Raw hardware configuration identity changed')
    need(config['rounds']==3 and config['rows_per_service']==2 and config['prompt_lengths']==[256,256,2048]
         and config['response_lengths']==[32,500,4096] and config['temperature']==1. and config['chunk']==16
         and config['atol']==config['rtol']==1e-4,'Hardware math/context protocol differs')
    peers=manifest['services'];need(len(peers)==8 and len({p['pid'] for p in peers})==8 and len({p['uuid'] for p in peers})==8,'Need actual eight scorer PIDs/UUIDs')
    startup=result['scorer_startup'];need(startup==[dict(rank=i,uuid=p['uuid'],pid=p['pid'],cgroup=manifest['cgroup']) for i,p in enumerate(peers)],'Raw scorer startup differs')
    services=[];receipts=[];fixed_hashes=set()
    def frozen(state,expected=None,device=None):
        need(state['dtypes']==['torch.float32'] and state['training'] is False
             and state['training_modules_count']==state['requires_grad_parameters']==state['grad_parameters']==0
             and state['parameter_bytes']==parameter_bytes and re.fullmatch('[0-9a-f]{64}',state['parameter_sha256']), 'Frozen full-model state invalid')
        if expected:need(state['parameter_sha256']==expected,'Frozen teacher parameters changed')
        if device:need(state['devices']==[device] and all(v==device for v in state['buffer_devices']),'Teacher model/buffer device differs')
    for rank,peer in enumerate(peers):
        ready_path=directory/'teacher-ipc'/f'scorer-ready-{rank}.json';ready=load(ready_path)
        need(ready['rank']==rank and ready['pid']==peer['pid'] and ready['uuid']==peer['uuid'] and ready['cgroup']==manifest['cgroup']
             and ready['actual_hf_files_sha256']==hf_identity and all(ready[k]==manifest[k] for k in IDENTITY),'Actual scorer readiness identity differs')
        frozen(ready['loaded_cpu'],device='cpu');fixed=ready['loaded_cpu']['parameter_sha256'];fixed_hashes.add(fixed)
        path=directory/'teacher-ipc'/f'scorer-{rank}.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        need([row['outer_step'] for row in rows]==[1,2,3],'Incomplete or duplicated score/release rounds')
        for row in rows:
            need(all(row[k]==manifest[k] for k in IDENTITY) and row['rank']==rank and row['pid']==peer['pid'] and row['uuid']==peer['uuid']
                 and row['cgroup']==manifest['cgroup'] and row['slurm_job']==manifest['slurm_job'] and row['rows']==2
                 and row['loaded_cpu']==ready['loaded_cpu'] and row['status']=='RELEASED' and row['targets_none'] is True
                 and row['dtype']=='float32' and row['attention']=='eager' and row['tf32'] is False
                 and row['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0),'Raw scorer ownership/math/release differs')
            frozen(row['released_cpu'],fixed,'cpu')
            if row['outer_step']==1:need(row['gpu_before'] is not None,'Missing first GPU hash')
            if row['outer_step']==3:need(row['gpu_after'] is not None,'Missing last GPU hash')
            for state in (row['gpu_before'],row['gpu_after']):
                if state is not None:frozen(state,fixed,'cuda:0')
            for field in ('gpu_pid_before','gpu_pid_phase_sample','gpu_pid_released'):
                sample=row[field];need(sample['pid']==peer['pid'] and sample['uuid']==peer['uuid'] and math.isfinite(sample['used_mib']) and sample['used_mib']>=0,'Raw GPU PID/UUID invalid')
            need(row['allocator_peak']['allocated_bytes']>=parameter_bytes,'Missing actual full-model GPU storage peak')
        services.append(dict(pid=peer['pid'],uuid=peer['uuid'],ready=ref(ready_path),receipts=ref(path)));receipts.append(rows)
    need(len(fixed_hashes)==1,'Eight teacher replicas differ')
    oracle_dir=directory/'original-gpu-oracle';oracle_ready=load(oracle_dir/'ready.json')
    oracle_script=ref(probe_script.parent/'original_gpu_oracle.py');ready_ref=ref(oracle_dir/'ready.json')
    need(oracle_ready==result['original_gpu_oracle_startup'] and all(oracle_ready[k]==manifest[k] for k in IDENTITY)
         and oracle_ready['pid'] not in {p['pid'] for p in peers} and oracle_ready['pid']>0
         and oracle_ready['rank']==4 and oracle_ready['uuid']==peers[4]['uuid'] and oracle_ready['cgroup']==manifest['cgroup']
         and oracle_ready['slurm_job']==manifest['slurm_job'] and oracle_ready['host'].split('.')[0]=='h100-11'
         and oracle_ready['scope']==ORACLE_SCOPE and oracle_ready['source_kind']=='HARDWARE_SURROGATE_NO_TRAINING'
         and oracle_ready['status']=='CPU_READY_NO_CUDA_CONTEXT' and oracle_ready['cuda_initialized'] is False
         and oracle_ready['offline'] is True and oracle_ready['torch_version']==PINNED['torch']
         and oracle_ready['actual_hf_files_sha256']==hf_identity and oracle_ready['loaded_cpu']==load(directory/'teacher-ipc/scorer-ready-4.json')['loaded_cpu'],
         'Original independent oracle startup/CPU/frozen/owner differs')
    need(oracle_ready['manifest_sha256']==sha(directory/'teacher-ipc/manifest.json')
         and oracle_ready['oracle_script_sha256']==oracle_script['sha256'] and oracle_ready['original_function']==ORIGINAL_FUNCTION
         and oracle_ready['original_function_file_sha256']==sha(root/'repo/search_r1/baselines/distillation.py')
         and oracle_ready['original_function_source_sha256']==method_sha(root/'repo/search_r1/baselines/distillation.py'),
         'Oracle does not bind the unchanged original FrozenTeacher function')
    oracle_refs=[]
    def oracle_round(step,comparison,prompts,responses,mask,original):
        receipt_path=oracle_dir/f'receipt-round{step}.json';request_path=oracle_dir/f'request-round{step}.json'
        input_path=oracle_dir/f'input-round{step}.safetensors';output_path=oracle_dir/f'reference-round{step}.safetensors'
        receipt=load(receipt_path);request=load(request_path)
        for role,path in [('receipt',receipt_path),('input',input_path),('output',output_path)]:
            need(Path(comparison['original_gpu_oracle_'+role+'_path']).resolve()==path.resolve()
                 and comparison['original_gpu_oracle_'+role+'_sha256']==sha(path),'Oracle raw summary/artifact path differs')
        need(comparison['original_gpu_oracle_pid']==oracle_ready['pid'] and comparison['original_gpu_oracle_uuid']==oracle_ready['uuid'], 'Round oracle owner differs')
        need(all(receipt[k]==value for k,value in oracle_ready.items() if k not in ('status','cuda_initialized','ready_epoch'))
             and receipt['outer_step']==step and receipt['rows']==16 and receipt['row_indices']==list(range(16))
             and receipt['status']=='RELEASED' and receipt['temperature']==1. and receipt['chunk']==16
             and receipt['targets_none'] is True and receipt['dtype']=='float32' and receipt['attention']=='eager'
             and receipt['finite'] is True and receipt['masked_zero_verified'] is True
             and receipt['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0)
             and not any(k in receipt for k in ('error','release_error')),'Original GPU function oracle math/provenance/release differs')
        frozen(receipt['gpu_before'],next(iter(fixed_hashes)),'cuda:0');frozen(receipt['gpu_after'],next(iter(fixed_hashes)),'cuda:0')
        need(receipt['released_cpu']==oracle_ready['loaded_cpu'],'Original oracle did not return to frozen CPU state')
        flags=receipt['actual_flags'];need(flags['matmul_allow_tf32'] is False and flags['cudnn_allow_tf32'] is False
             and flags['matmul_precision']=='highest' and flags['autocast_enabled'] is False and flags['cpu_threads']==1
             and flags['environment']['CUDA_VISIBLE_DEVICES']==oracle_ready['uuid']
             and flags['environment']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']=='0'
             and flags['environment']['NVIDIA_TF32_OVERRIDE'] in (None,'0'),'Original oracle actual FP32 flags differ')
        for field in ('gpu_pid_before','gpu_pid_phase_sample','gpu_pid_released'):
            value=receipt[field];need(value['pid']==oracle_ready['pid'] and value['uuid']==oracle_ready['uuid']
                 and math.isfinite(value['used_mib']) and value['used_mib']>=0,'Oracle actual GPU process mapping invalid')
        need(receipt['allocator_peak']['allocated_bytes']>=parameter_bytes,'No actual full-model oracle GPU allocation')
        need(set(request)==set(IDENTITY)|{'outer_step','input_sha256','all8_released_zero','candidate_receipts_sha256','scorers_released_epoch'}
             and all(request[k]==manifest[k] for k in IDENTITY) and request['outer_step']==step
             and request['all8_released_zero'] is True and request['candidate_receipts_sha256']==sha(directory/f'receipts-round{step}.json')
             and receipt['candidate_receipts_sha256']==request['candidate_receipts_sha256']
             and request['input_sha256']==receipt['input_sha256']==sha(input_path) and receipt['request_sha256']==sha(request_path)
             and receipt['output_sha256']==sha(output_path),'Original oracle request/input/output/candidate hash binding differs')
        incoming=load_file(str(input_path),device='cpu');output=load_file(str(output_path),device='cpu')
        need(set(incoming)=={'prompts','responses','attention','row_indices'} and set(output)=={'original_gpu_log_probs'}
             and all(value.dtype==torch.int64 for value in incoming.values()) and torch.equal(incoming['prompts'],prompts) and torch.equal(incoming['responses'],responses)
             and torch.equal(incoming['attention'],mask) and torch.equal(incoming['row_indices'],torch.arange(16))
             and output['original_gpu_log_probs'].dtype==torch.float32 and torch.equal(output['original_gpu_log_probs'],original),
             'Original oracle does not independently score all16 exact input/output rows')
        need(receipt['row_indices_sha256']==digest(save(dict(row_indices=torch.arange(16))))
             and receipt['attention_sha256']==digest(save(dict(attention=mask))),'Original oracle row/mask hash differs')
        tokens=[int((prompts[i]!=manifest['student_tokenizer']['pad_token_id']).sum()+mask[i].sum()) for i in range(16)]
        need(receipt['scored_tokens_by_row']==tokens and receipt['expected_scored_tokens']==receipt['scored_tokens_delta']==sum(tokens),
             'Oracle token counts hide truncation or altered rows')
        times=[comparison['scoring_started_epoch'],comparison['scoring_completed_epoch'],comparison['scorers_released_epoch'],receipt['cuda_started_epoch'],receipt['scoring_started_epoch'],receipt['scoring_completed_epoch'],receipt['completed_epoch'],comparison['original_oracle_completed_epoch']]
        need(all(math.isfinite(v) and v>0 for v in times) and times==sorted(times)
             and request['scorers_released_epoch']==receipt['scorers_released_epoch']==comparison['scorers_released_epoch'],
             'Independent oracle CUDA overlaps unreleased candidate scorers or timestamps differ')
        links={role:ref(path) for role,path in [('receipt',receipt_path),('request',request_path),('input',input_path),('output',output_path)]}
        oracle_refs.append(links);return links
    comparisons=result['ipc_rounds'];need([r['step'] for r in comparisons]==[1,2,3],'Raw result lacks all rounds')
    phases=[]
    for step,comparison in enumerate(comparisons,1):
        path=directory/f'parity-round{step}.safetensors';need(sha(path)==comparison['parity_artifact_sha256'],'Actual parity tensor file changed');ref(path)
        tensor=load_file(str(path),device='cpu');required={'prompts','responses','attention','gpu_log_probs','original_gpu_log_probs'}|({'cpu_log_probs'} if step<3 else set())
        need(set(tensor)==required,'Missing/forbidden actual parity tensors')
        prompts,responses,mask,gpu=(tensor[k] for k in ('prompts','responses','attention','gpu_log_probs'))
        pwidth,rwidth=(256,32) if step==1 else (256,500) if step==2 else (2048,4096)
        need(prompts.dtype==responses.dtype==mask.dtype==torch.int64 and prompts.shape==(16,pwidth)
             and responses.shape==mask.shape==gpu.shape==(16,rwidth) and gpu.dtype==torch.float32
             and bool(((mask==0)|(mask==1)).all()) and bool(torch.isfinite(gpu).all()) and bool((gpu[mask==0]==0).all()),'Actual tensor shape/dtype/mask/finite invalid')
        expected_prefix=torch.tensor([256]*16 if step<3 else [256]*15+[2048])
        expected_response=torch.tensor([rwidth]*16 if step<3 else [32 if row%2==0 else 500 for row in range(15)]+[4096])
        need(torch.equal((prompts!=manifest['student_tokenizer']['pad_token_id']).sum(-1),expected_prefix)
             and torch.equal(mask.sum(-1),expected_response),'Actual per-row full context differs; aggregate tokens cannot hide truncation')
        aggregate=load(directory/f'receipts-round{step}.json');need(aggregate==[rows[step-1] for rows in receipts]
             and sha(directory/f'receipts-round{step}.json')==comparison['receipts_sha256'],'Result/reconstructed actual receipt mismatch');ref(directory/f'receipts-round{step}.json')
        for rank,rows in enumerate(receipts):
            row=rows[step-1];region=slice(rank*2,rank*2+2);indices=torch.arange(rank*2,rank*2+2)
            incoming=save(dict(prompts=prompts[region].contiguous(),responses=responses[region].contiguous(),attention=mask[region].contiguous(),row_indices=indices))
            outgoing=save(dict(teacher_log_probs=gpu[region].contiguous(),row_indices=indices))
            need(row['input_sha256']==digest(incoming) and row['output_sha256']==digest(outgoing)
                 and row['row_indices_sha256']==digest(save(dict(row_indices=indices))),'Actual input/output/row-order SHA differs')
            tokens=int((prompts[region]!=manifest['student_tokenizer']['pad_token_id']).sum()+mask[region].sum())
            need(row['scored_tokens_delta']==tokens,'Actual scored-token count differs from input mask')
        if step==3:need(receipts[7][2]['scored_tokens_delta']==6432,'Maximum6144 context not actually fully scored')
        need(comparison['rows']==16 and comparison['rows_per_service']==2 and comparison['prompt_length']==pwidth
             and comparison['response_length']==rwidth and comparison['row_order_verified'] is True
             and comparison['masked_zero_verified'] is True and comparison['finite'] is True and comparison['all8_released_zero'] is True
             and comparison['all8_fixed_parameter_sha256']==next(iter(fixed_hashes)),'Raw round assertions inconsistent')
        phase=dict(outer_step=step,rows=16,prompt_width=pwidth,response_width=rwidth,row_order_verified=True,masked_zero_verified=True,finite=True,
                   all_eight_released_before_return=True,actual_peer_pid_cgroup_verified=True,lease_uuid_ownership_verified=True,
                   facts_source='Recomputed payload hashes/masks/tokens/GPU-original-function parity and raw RELEASED receipts; peer/lease enforced by exact reviewed backend on successful calls')
        original=tensor['original_gpu_log_probs']
        need(original.dtype==torch.float32 and original.shape==gpu.shape and bool(torch.isfinite(original).all())
             and bool((original[mask==0]==0).all()),'Invalid original GPU-function reference')
        delta=float((gpu-original).abs().max());mean=float((gpu-original).abs().mean())
        need(comparison['scope']=='ORIGINAL_GPU_FUNCTION_PARITY_BLOCKING' and comparison['numerical_reference']==ORACLE and bool(torch.allclose(gpu,original,atol=1e-4,rtol=1e-4))
             and comparison['numeric_close'] is True and comparison['atol']==comparison['rtol']==1e-4
             and comparison['max_absolute_delta']==delta and comparison['mean_absolute_delta']==mean,
             'Actual candidate/original GPU function parity failed or summary differs')
        phase['gpu_parity']=dict(passed=True,reference=ORACLE,atol=1e-4,rtol=1e-4,max_abs_delta=delta,mean_abs_delta=mean)
        phase['parity_artifact']=reference(path,root)
        phase['original_gpu_oracle']=oracle_round(step,comparison,prompts,responses,mask,original)
        if step<3:
            cpu=tensor['cpu_log_probs'];need(cpu.dtype==torch.float32 and cpu.shape==gpu.shape and bool(torch.isfinite(cpu).all()) and bool((cpu[mask==0]==0).all()),'Invalid actual CPU diagnostic')
            diagnostic=comparison['cpu_cross_backend_diagnostic']
            measured=dict(scope='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC',numeric_close=bool(torch.allclose(gpu,cpu,atol=1e-4,rtol=1e-4)),
                          max_absolute_delta=float((gpu-cpu).abs().max()),mean_absolute_delta=float((gpu-cpu).abs().mean()),atol=1e-4,rtol=1e-4,
                          violation_count=int(((gpu-cpu).abs()>(1e-4+1e-4*cpu.abs())).sum()),
                          per_row_violation_counts=[int(v) for v in ((gpu-cpu).abs()>(1e-4+1e-4*cpu.abs())).sum(-1)])
            need(diagnostic==measured,'CPU cross-backend diagnostic differs or hides failure')
            phase['cpu_cross_backend_diagnostic']=measured
        else:
            need('cpu_cross_backend_diagnostic' not in comparison or comparison['cpu_cross_backend_diagnostic'] is None,'Long-case CPU comparison must not be fabricated')
            phase['cpu_parity_scope']='NOT_MEASURED_LONG_CASE'
        phases.append(phase)
    vllm=[]
    for rank,mapping in enumerate(result['worker_gpu_mapping']):
        ready=load(directory/'eight-vllm'/f'ready-{rank}.json');done=load(directory/'eight-vllm'/f'done-{rank}.json')
        need(ready['rank']==done['rank']==rank and ready['NO_TRAINING'] is done['NO_TRAINING'] is True
             and ready['actual_pid_gpu_uuids']==[mapping['uuid']] and ready['expected_gpu_uuid']==mapping['uuid']
             and str(ready['cuda_visible_devices'])==str(mapping['index']) and ready['gpu_memory_utilization']==.6
             and ready['tp']==1 and done['requests']>0 and done['output_tokens']>0,'Actual vLLM inference/device invalid')
        vllm.append(dict(uuid=mapping['uuid'],pid=ready['pid']));ref(directory/'eight-vllm'/f'ready-{rank}.json');ref(directory/'eight-vllm'/f'done-{rank}.json')
    need(len(vllm)==8 and len({v['pid'] for v in vllm})==8 and {v['uuid'] for v in vllm}=={p['uuid'] for p in peers},'vLLM mapping incomplete')
    series=directory/'gpu-memory-timeseries.jsonl';samples=[json.loads(line) for line in series.read_text().splitlines()]
    baseline=samples[0];need(baseline==result['original_baseline'] and baseline['phase']=='baseline','Actual baseline mismatch')
    need(len(baseline['gpus'])==8 and {g['uuid'] for g in baseline['gpus']}=={p['uuid'] for p in peers}
         and all(g['memory_used_mib']/g['memory_total_mib']<=.15 and g['utilization']<=15 for g in baseline['gpus']),'Initial actual GPU idle lease window invalid')
    co_samples=result['coexistence_samples'];need(co_samples,'No actual co-residency samples')
    retriever_pids={s['retriever_pid'] for s in co_samples};need(len(retriever_pids)==1,'Retriever PID drift');retriever_pid=next(iter(retriever_pids))
    simultaneous=[]
    for sample in samples:
        memory={}
        for line in sample['processes']:
            uuid,pid,value=[v.strip() for v in line.split(',')];value=float(value)
            need(math.isfinite(value) and value>=0,'Invalid raw GPU sample');memory[(uuid,int(pid))]=value
        if all(memory.get((p['uuid'],p['pid']),0)>=11000 for p in peers) and all(memory.get((p['uuid'],p['pid']),0)>=10000 for p in vllm) and all((p['uuid'],retriever_pid) in memory for p in peers):simultaneous.append(sample)
    need(simultaneous,'Dispersed GPU peaks do not prove simultaneous co-residency')
    owners={p['pid'] for p in peers}|{v['pid'] for v in vllm}|{retriever_pid,oracle_ready['pid']}
    need(len(owners)==18 and result['owned_gpu_model_process_count']==18 and result['parent_driver_cuda_initialized'] is False
         and result['owned_gpu_model_pids']==dict(scorers=[p['pid'] for p in peers],vllm=[v['pid'] for v in vllm],retriever=retriever_pid,original_oracle=oracle_ready['pid']), 'Independent18owned model processes/parent noCUDA differ')
    done=load(oracle_dir/'done.json');need(result['original_gpu_oracle_exit_code']==0 and result['original_gpu_oracle_done']==done
         and all(done[k]==v for k,v in oracle_ready.items() if k not in ('status','cuda_initialized','ready_epoch'))
         and done['status']=='THREE_ORIGINAL_GPU_PHASES_RELEASED_NO_TRAINING' and done['rounds']==3
         and done['last_receipt_sha256']==sha(oracle_dir/'receipt-round3.json'),'Independent original oracle did not finish/release/exit0')
    need(0<oracle_ready['ready_epoch']<=comparisons[0]['scoring_started_epoch'] and done['done_epoch']>=load(oracle_dir/'receipt-round3.json')['completed_epoch'], 'Oracle startup/completion chronology differs')
    done_ref=ref(oracle_dir/'done.json')
    cpu_ready=[s for s in samples if s['phase']=='eight-scorers-and-original-oracle-cpu-ready']
    cpu_owners={p['pid'] for p in peers}|{oracle_ready['pid']}
    need(cpu_ready and any(cpu_owners<={v['pid'] for v in s['own_process_samples']}
         and oracle_ready['ready_epoch']<=s['time']<=comparisons[0]['scoring_started_epoch']
         and not any(int(line.split(',')[1]) in cpu_owners for line in s['processes']) for s in cpu_ready), 'Oracle must be CPU-ready before own GPU occupancy')
    need(all(any(s['time']==raw_s['time'] and s['phase']==raw_s['phase'] for raw_s in simultaneous) for s in co_samples),'Raw reported coexistence sample not present')
    final=samples[-1];need(final['phase']=='released' and result['own_gpu_model_pids_remaining']==[],'Missing actual final released GPU inventory')
    gpu_ids=lambda lines:{(line.split(',')[0].strip(),int(line.split(',')[1])) for line in lines}
    need(gpu_ids(baseline['processes'])<=gpu_ids(final['processes']),'Original GPU baseline disappeared in actual final snapshot')
    need(not any(pid in owners for _,pid in gpu_ids(final['processes'])),'Own GPU process survived claimed cleanup')
    requests=result['requests'];need(requests and all(r['queries']>0 and r['topk'] in (3,50) and r['completed_epoch']>=r['started_epoch'] for r in requests)
         and any(r['topk']==3 and r['label'].startswith('scorer-round') for r in requests),'No actual local GPU retrieval while scoring')
    need(result['inference']==[load(directory/'eight-vllm'/f'done-{i}.json') for i in range(8)],'Raw inference result differs from actual done files')
    effective=load(directory/'effective-rollout-config.json')
    need(effective==result['effective_rollout_config']==config['rollout'] and effective['gpu_memory_utilization']==.6
         and effective['tensor_parallel_size']==1 and effective['enforce_eager'] is True,'Actual rollout/resource config differs')
    need(result['split_artifact'],'Missing actual local retrieval input provenance')
    for role in ('train','manifest'):
        need(ref(result['split_artifact'][role+'_path'])['sha256']==result['split_artifact'][role+'_sha256'],'Retrieval input source drifted')
    for request in requests:
        need(Path(request['label']).name==request['label'],'Nonlocal HTTP result label')
        path=directory/(request['label']+'.json');need(load(path)==request,'Raw HTTP result differs');ref(path)
    for path in (directory/'result.json',directory/'hardware-config.json',directory/'effective-rollout-config.json',probe_script.parent/'reviewed-sources.json'):ref(path)
    return dict(status='PASS_NO_TRAINING',kind='HARDWARE_SURROGATE_NO_TRAINING',host=context['host'],code_commit=context['code_commit'],source_fingerprint=fingerprint,
                runtime=context['runtime'],runtime_provenance='raw torch_version plus same pinned node/env imports after probe; Python/TD were not directly captured by probe',
                offline=True,offline_provenance='Successful exact service_main asserts HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE; probe source enforces loopback/proxy-disabled HTTP',
                numerical_reference=ORACLE,original_gpu_oracle=dict(pid=oracle_ready['pid'],uuid=oracle_ready['uuid'],ready=ready_ref,script=oracle_script,done=done_ref,rounds=oracle_refs,exit_code=0),services=services,cgroup=manifest['cgroup'],slurm_job=manifest['slurm_job'],initial_lease_idle_verified=True,phases=phases,
                coexistence=dict(local_gpu_e5=True,gpu_fp16_faiss_all8=True,full_wiki18=True,vllm_count=8,vllm_memory_utilization=.6,same_host=True,
                                 actual_retrieval_topk=3,uuids=[p['uuid'] for p in peers],vllm_services=vllm,retriever_pid=retriever_pid,gpu_samples=ref(series)),
                cleanup=dict(passed=True,baseline_preserved=True,own_children_remaining=0,owned_model_process_count=18,parent_driver_cuda_initialized=False),probe_script=reference(probe_script,root),
                raw_evidence=list({item['path']:item for item in raw}.values()),facts_scope='Recomputed hardware-only evidence; no independent review, trained teacher, RL smoke or five-step approval')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--probe-dir',required=True);parser.add_argument('--probe-script',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    result=normalize(args.probe_dir,args.root,args.probe_script)
    output=Path(args.output);need(not output.exists(),'Refuse overwriting hardware evidence');need(output.resolve().is_relative_to(Path(args.root).resolve()),'Output must be own local')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print('HARDWARE_EVIDENCE_WRITTEN_NO_REVIEW_NO_ALGORITHM_ACCEPTANCE',str(output))


if __name__=='__main__':main()
