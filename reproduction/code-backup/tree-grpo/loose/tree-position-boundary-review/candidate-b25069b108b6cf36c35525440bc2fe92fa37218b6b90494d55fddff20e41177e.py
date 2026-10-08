"""CPU-only evidence for recorded per-row positions and actual HF packed-boundary rules.

Reads complete, SHA-bound same-node calibration assets. No model forward, CUDA,
optimizer, network or asset export. Potential attention crossings are evidence
of the reconstructed source rule, not a measured cause of log-prob differences.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from types import SimpleNamespace

import torch
import yaml

HF_SHA = '2753ed0bdb9b37f72a4b4d0663e2f2833ad55a874be37d4599d69eb9a826ede0'
HF_QWEN_SHA = '4568d588b45dd28152d97c37300a5e227589bc70b9fef5224af29977ecf6abdc'
FORWARD_FILES = ('verl/workers/actor/dp_actor.py', 'verl/utils/torch_functional.py',
                 'verl/models/transformers/qwen2.py')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()


def context_tools(expected_sha):
    path = Path(__file__).with_name('analyze_tree_sampling_calibration.py')
    require(sha(path) == expected_sha, 'Context helper SHA changed')
    spec = importlib.util.spec_from_file_location('bound_tree_context_tools', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(sha(path) == expected_sha, 'Context helper SHA changed during import')
    return module


def hf_rules(path, expected_sha):
    """Execute only the SHA-bound real CPU-safe prepare function and actual if predicate."""
    path = Path(path)
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == expected_sha, 'Installed HF FlashAttention helper SHA changed')
    tree = ast.parse(source.decode('utf-8'))
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'prepare_fa2_from_position_ids')
    forward = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_flash_attention_forward')
    predicate = next(node.test for node in ast.walk(forward) if isinstance(node, ast.If)
                     and 'torch.diff(position_ids' in ast.unparse(node.test))
    namespace = {'torch': torch}
    # The whole external module is not imported; CUDA attention functions never execute.
    namespace['Optional'] = object
    function = ast.FunctionDef(name=prepare.name, args=prepare.args, body=prepare.body,
                              decorator_list=[], returns=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    code = compile(ast.fix_missing_locations(ast.Expression(predicate)), str(path), 'eval')
    return dict(path=str(path), sha256=expected_sha, predicate=ast.unparse(predicate),
                prepare=namespace[prepare.name], predicate_code=code)


def native_qwen_caller(qwen_path, expected_qwen_sha, worker_path, expected_worker_sha):
    """Bind the native SP1 source path, separate from a live worker function trace."""
    qwen_path,worker_path=Path(qwen_path),Path(worker_path)
    qwen_bytes,worker_bytes=qwen_path.read_bytes(),worker_path.read_bytes()
    require(hashlib.sha256(qwen_bytes).hexdigest()==expected_qwen_sha,'Installed native HF Qwen source changed')
    require(hashlib.sha256(worker_bytes).hexdigest()==expected_worker_sha,'Frozen actor constructor source changed')
    qwen_tree,worker_tree=ast.parse(qwen_bytes.decode()),ast.parse(worker_bytes.decode())
    qwen_class=next(node for node in qwen_tree.body if isinstance(node,ast.ClassDef) and node.name=='Qwen2FlashAttention2')
    forward=next(node for node in qwen_class.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    flash_calls=[node for node in ast.walk(forward) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_flash_attention_forward']
    require(len(flash_calls)==1,'Native Qwen FlashAttention caller is ambiguous')
    call=flash_calls[0];keywords={keyword.arg:keyword.value for keyword in call.keywords}
    require(len(call.args)==5 and ast.unparse(call.args[3])=='attention_mask' and ast.unparse(call.args[4])=='q_len'
        and ast.unparse(keywords.get('position_ids',ast.Constant(None)))=='position_ids','Native Qwen no longer passes recorded positions')
    mapping=next(node.value for node in qwen_tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='QWEN2_ATTENTION_CLASSES' for target in node.targets))
    require(isinstance(mapping,ast.Dict) and any(isinstance(key,ast.Constant) and key.value=='flash_attention_2'
        and isinstance(value,ast.Name) and value.id=='Qwen2FlashAttention2' for key,value in zip(mapping.keys,mapping.values)),
        'Native Qwen FlashAttention class selection changed')
    worker_class=next(node for node in worker_tree.body if isinstance(node,ast.ClassDef) and node.name=='ActorRolloutRefWorker')
    build=next(node for node in worker_class.body if isinstance(node,ast.FunctionDef) and node.name=='_build_model_optimizer')
    constructors=[node for node in ast.walk(build) if isinstance(node,ast.Call) and ast.unparse(node.func)=='AutoModelForCausalLM.from_pretrained']
    require(len(constructors)==1 and any(keyword.arg=='attn_implementation' and isinstance(keyword.value,ast.Constant)
        and keyword.value.value=='flash_attention_2' for keyword in constructors[0].keywords),'Actor construction no longer selects FlashAttention2')
    gates=[node.test for node in ast.walk(build) if isinstance(node,ast.If) and any(isinstance(child,ast.Call)
        and isinstance(child.func,ast.Name) and child.func.id=='apply_monkey_patch' for statement in node.body for child in ast.walk(statement))]
    require(len(gates)==1,'Actor monkey-patch guard is ambiguous')
    guard=compile(ast.fix_missing_locations(ast.Expression(gates[0])),str(worker_path),'eval')
    require(eval(guard,{},dict(use_remove_padding=True,self=SimpleNamespace(ulysses_sequence_parallel_size=1))) is False,
        'Actor monkey patch is active at the recorded SP1 configuration')
    return dict(status='NATIVE_HF_SP1_CALLER_SOURCE_BOUND_NOT_RUNTIME_TRACE',qwen_path=str(qwen_path),qwen_sha256=expected_qwen_sha,
        worker_path=str(worker_path),worker_sha256=expected_worker_sha,native_flash_call=ast.unparse(call),
        actor_monkey_patch_guard=ast.unparse(gates[0]),actor_monkey_patch_at_sp1=False,
        actor_attn_implementation='flash_attention_2',position_ids_keyword=True,live_worker_function_identity_observed=False)


def tensor_identity(tensor):
    require(tensor.device.type == 'cpu', 'Position analysis requires CPU tensors')
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=str(tensor.dtype), shape=list(tensor.shape)), sort_keys=True).encode())
    flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    for block in flat.split(8*1024**2): digest.update(block.numpy().tobytes())
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), shape_and_bytes_sha256=digest.hexdigest())


def inspect_positions(tensors, rules, label, row_base=0, max_examples=4):
    ids, positions, mask = (tensors[key] for key in ('input_ids','position_ids','attention_mask'))
    require(all(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' for tensor in (ids,positions,mask)), 'CPU inputs required')
    require(ids.ndim == 2 and positions.shape == mask.shape == ids.shape, 'Full token/position/mask shapes differ')
    require(positions.dtype in (torch.int8,torch.int16,torch.int32,torch.int64), 'Integral signed positions required')
    require(bool(((mask == 0)|(mask == 1)).all()), 'Nonbinary attention mask')
    rows = []
    row_summary = dict(rows=ids.shape[0], attention_valid_tokens=0, empty_rows=0,
                       nonzero_start_position_rows=0, noncontiguous_valid_position_rows=0,
                       negative_valid_position_rows=0, row_examples=[])
    for row in range(ids.shape[0]):
        valid = positions[row][mask[row].bool()]
        rows.append(valid)
        row_summary['attention_valid_tokens'] += valid.numel()
        if not valid.numel(): row_summary['empty_rows'] += 1; continue
        contiguous = bool((torch.diff(valid) == 1).all())
        nonzero = int(valid[0]) != 0; negative = bool((valid < 0).any())
        row_summary['nonzero_start_position_rows'] += nonzero
        row_summary['noncontiguous_valid_position_rows'] += not contiguous
        row_summary['negative_valid_position_rows'] += negative
        if (nonzero or not contiguous or negative) and len(row_summary['row_examples']) < max_examples:
            row_summary['row_examples'].append(dict(row=row_base+row, valid_length=valid.numel(),
                first_position=int(valid[0]), last_position=int(valid[-1]), contiguous=contiguous,
                zero_position_count=int((valid == 0).sum()), negative_position=negative))
    summaries = {}
    for micro in (4,2):
        summary = dict(micro=micro, groups=0, packed_varlen_groups=0, single_sequence_groups=0,
            zero_token_groups=0, groups_with_boundary_mismatch=0,
            missing_boundaries=0, extra_boundaries=0, missing_sample_boundaries=0,
            groups_missing_start_zero=0, groups_with_possible_cross_sample_attention=0,
            possible_cross_sample_boundaries=0, groups_with_uncovered_prefix=0,
            attention_valid_tokens_in_crossing_segments=0, examples=[])
        for start in range(0,len(rows),micro):
            group = rows[start:start+micro]
            flat = torch.cat(group)
            length = flat.numel(); summary['groups'] += 1
            expected = [0]; positive_rows = []
            for offset,row in enumerate(group):
                if row.numel():
                    positive_rows.append(row_base+start+offset)
                    expected.append(expected[-1]+row.numel())
            expected = sorted(set(expected))
            if not length:
                summary['zero_token_groups'] += 1
                continue
            packed = bool(eval(rules['predicate_code'], {'torch':torch},
                               dict(position_ids=flat.unsqueeze(0), query_length=length)))
            if packed:
                summary['packed_varlen_groups'] += 1
                # Real HF prepare runs only on tiny CPU shape-only Q/K/V placeholders.
                dummy = torch.empty((1,length,1,1),device='cpu')
                result = rules['prepare'](dummy,dummy,dummy,flat.unsqueeze(0))
                actual = result[4][0].tolist()
                actual_max_length = int(result[5][0])
            else:
                summary['single_sequence_groups'] += 1
                actual = [0,length]; actual_max_length = length
            wanted, obtained = set(expected), set(actual)
            missing, extra = sorted(wanted-obtained), sorted(obtained-wanted)
            internal = expected[1:-1]
            crossed = [boundary for boundary in internal if any(a < boundary < b for a,b in zip(actual,actual[1:]))]
            crossing_tokens = sum(b-a for a,b in zip(actual,actual[1:]) if any(a < boundary < b for boundary in internal))
            mismatch = bool(missing or extra)
            summary['groups_with_boundary_mismatch'] += mismatch
            summary['missing_boundaries'] += len(missing)
            summary['extra_boundaries'] += len(extra)
            summary['missing_sample_boundaries'] += sum(boundary not in obtained for boundary in internal)
            summary['groups_missing_start_zero'] += int(actual[0] != 0)
            summary['groups_with_uncovered_prefix'] += int(actual[0] > 0)
            summary['groups_with_possible_cross_sample_attention'] += bool(crossed)
            summary['possible_cross_sample_boundaries'] += len(crossed)
            summary['attention_valid_tokens_in_crossing_segments'] += crossing_tokens
            if mismatch and len(summary['examples']) < max_examples:
                summary['examples'].append(dict(first_row=row_base+start, rows=len(group), positive_rows=positive_rows,
                    packed_varlen=packed, valid_tokens=length, expected_row_boundaries=expected[:32],
                    actual_hf_boundaries=actual[:32], actual_boundary_count=len(actual),
                    boundary_lists_truncated=len(actual)>32 or len(expected)>32,
                    missing_boundaries=missing[:32], extra_boundaries=extra[:32],
                    possible_cross_sample_boundaries=crossed[:32], actual_hf_max_length=actual_max_length,
                    expected_max_row_length=max(row.numel() for row in group)))
        summaries[str(micro)] = summary
    return dict(label=label, full_input_tensor_identity={name:tensor_identity(tensors[name]) for name in ('input_ids','position_ids','attention_mask')},
                rows=row_summary, local_grouping=summaries,
                grouping_scope='Raw micro4 models original old recompute; monitor micro2 models safety measure. Monitor micro4 is an alternative grouping, not the original full-batch old grouping. Full actor inputs are inspected within their contiguous eight-way DP partitions.',
                limits='Counts describe exact recorded full attention-valid positions and the pinned source segmentation rule. Nonzero RoPE offset alone is not a cause finding. Missing cu[0]=0 means the reconstructed segmentation does not cover the prefix; absence of counted crossing segments does not establish valid attention. No CUDA attention execution or measured probability-cause attribution.')


def inspect_bundle(A, verified, package_dir, rules, max_examples=4):
    run,package,ready,identity,sources = verified
    sources=list(sources)
    config_bytes=(run/'resolved.yaml').read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest()==identity['config_sha256'],'Resolved config changed during position inspection')
    config=yaml.safe_load(config_bytes)
    actor,rollout,model=(config['actor_rollout_ref'][name] for name in ('actor','rollout','model'))
    require(actor['ppo_micro_batch_size']==16 and rollout['log_prob_micro_batch_size']==32 and rollout['n']==1
        and actor.get('ulysses_sequence_parallel_size',1)==1 and rollout['tensor_model_parallel_size']==1
        and actor['use_dynamic_bsz'] is False and rollout['log_prob_use_dynamic_bsz'] is False
        and model['use_remove_padding'] is True and actor['policy_safety']['monitor_rows_per_rank']==32,
        'Position reconstruction requires the actual fixed DP8 local old4/actor2 remove-padding protocol')
    for name in FORWARD_FILES:
        path = package_dir/'source/repo'/name
        require(A.file_sha(path) == package['files']['source/repo/'+name], 'Frozen forward source changed: '+name)
        sources.append(dict(path=str(path),sha256=package['files']['source/repo/'+name]))
    reports, evidence, raw_paths = [], [], set()
    common_calls = None
    def inside(path,directory):
        path = Path(path).resolve(); require(path.is_relative_to(directory.resolve()),'Evidence escaped same-node run'); return path
    for rank,worker in enumerate(sorted(ready['workers'],key=lambda record:record['rank'])):
        require(worker['rank']==rank and worker['world_size']==8 and worker['optimizer_committed']==0 and
            worker['scheduler_advanced'] is False and worker['scheduler_step']==110 and worker['restored_optimizer_steps']==[2500]
            and all(worker.get(key)==value for key,value in identity.items()),'Worker calibration identity changed')
        entries = worker['diagnostic_entries']; calls=[]
        require([entry['path'] for entry in entries]==worker['diagnostic_records'],'Raw entry order changed')
        for entry in entries:
            path=inside(entry['path'],run/'output/sampling-consistency'); raw=A.json_bound(path,entry['sha256'])
            require(path.name==f'rank-{rank:02d}-call-{raw["generation_call"]:05d}.json' and path not in raw_paths,'Raw file/call identity changed')
            raw_paths.add(path); calls.append(raw['generation_call'])
            require(raw['rank']==rank and raw['world_size']==8 and raw['scheduler_step']==110 and raw['optimizer_committed']==0
                and raw['saved_rows']==raw['local_rows'] and raw['saved_rows']>0 and raw['enforced'] is False
                and all(raw.get(key)==value for key,value in identity.items()),'Incomplete or foreign raw evidence')
            snapshot=inside(entry['snapshot_path'],run/'output/sampling-consistency')
            require(snapshot.name==path.with_suffix('.pt').name and raw['snapshot_path']==str(snapshot)
                and raw['snapshot_sha256']==entry['snapshot_sha256'],'Raw snapshot binding changed')
            tensors=A.load_verified(snapshot,entry['snapshot_sha256']); A.normalize(tensors,raw=True)
            require(tensors['input_ids'].shape[0]==raw['saved_rows'] and tensors['distribution']==raw['distribution'],'Raw coverage/mode changed')
            reports.append(dict(kind='raw_same_turn',rank=rank,generation_call=raw['generation_call'],
                inspection=inspect_positions(tensors,rules,str(snapshot),max_examples=max_examples)))
            evidence.append(dict(path=str(path),sha256=entry['sha256'],snapshot_path=str(snapshot),snapshot_sha256=entry['snapshot_sha256']))
            del tensors
        if common_calls is None: common_calls=calls
        require(calls==common_calls==list(range(1,len(calls)+1)),'Raw call/rank coverage differs')
        path=inside(worker['monitor_path'],run/'output/probability-calibration')
        require(path.name==f'rank-{rank:02d}-monitor.pt','Monitor rank/path changed')
        tensors=A.load_verified(path,worker['monitor_sha256']); A.normalize(tensors,raw=False)
        require(tensors['input_ids'].shape[0]==32,'Monitor coverage must be 32/rank')
        reports.append(dict(kind='final_leaf_monitor_subset',rank=rank,
            inspection=inspect_positions(tensors,rules,str(path),max_examples=max_examples)))
        evidence.append(dict(path=str(path),sha256=worker['monitor_sha256'])); del tensors
    require(raw_paths=={p.resolve() for p in (run/'output/sampling-consistency').glob('rank-*-call-*.json')},'Unbound raw JSON file set')
    actor_rows=0; actor_files=0
    for path in sorted((run/'output/trajectories').glob('outer-*/actor-inputs.pt')):
        path=inside(path,run/'output/trajectories'); manifest_path=path.parent/'manifest.json'
        manifest_sha=A.file_sha(manifest_path); manifest=A.json_bound(manifest_path,manifest_sha)
        expected=dict(outer_step=111,job_id=identity['job_id'],source=identity['code_commit'],config=identity['config_sha256'],data=identity['data_identity'])
        require(manifest['status']=='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED' and manifest['identity']==expected,'Actor-input receipt changed')
        payload=A.load_verified(path,manifest['payloads'][path.name]); tensors=payload['tensors']; A.normalize(tensors,raw=False)
        require(payload['identity']==expected and payload['rows']==manifest['rows']==tensors['input_ids'].shape[0]==3840,'Full Tree640x6 actor coverage differs')
        for rank in range(8):
            partition={key:value[rank*480:(rank+1)*480] for key,value in tensors.items() if isinstance(value,torch.Tensor) and value.ndim>0 and value.shape[0]==3840}
            reports.append(dict(kind='complete_final_actor_inputs_dp_partition',rank=rank,
                inspection=inspect_positions(partition,rules,str(path),row_base=rank*480,max_examples=max_examples)))
        evidence.append(dict(path=str(path),sha256=manifest['payloads'][path.name],manifest_path=str(manifest_path),manifest_sha256=manifest_sha))
        actor_rows+=payload['rows']; actor_files+=1; del payload,tensors
    require(actor_files==1 and actor_rows==3840,'Exactly one complete zero-write outer111 actor-input file required')
    audit=run/'output/tree-rank-audit'
    require(not list(audit.glob('step-*.json')) and not list((audit/'ratio-tail').glob('*')),
            'Zero-write evidence contains actual actor optimizer audit files')
    require(not (run/'output/checkpoints').exists() or not any((run/'output/checkpoints').rglob('*')),
            'Zero-write evidence unexpectedly saved a new training checkpoint')
    return dict(status='POSITION_BOUNDARY_EVIDENCE_ANALYZED_NO_TRAINING_ACCEPTANCE',identity=identity,
        actual_optimizer_writes=0,actual_model_forwards=0,actor_input_rows=actor_rows,
        grouping_config=dict(world_size=8,sequence_parallel=1,tensor_parallel=1,rollout_n=1,
            global_old_micro=32,rank_old_micro=4,global_actor_micro=16,rank_actor_micro=2,
            monitor_rows_per_rank=32,use_remove_padding=True,config_sha256=identity['config_sha256']),
        source_files=sources,hf_rule_source=dict(path=rules['path'],sha256=rules['sha256'],predicate=rules['predicate']),
        evidence=evidence,inspections=reports,
        limits=['Raw physical generation rows, monitor subset and complete actor rows are separate populations; do not add them.',
                'Grouping reconstructed from retained row order and contiguous eight-way DataProto DP dispatch; engine KV layout is not measured.',
                'Actual HF branch predicate and CPU prepare function are executed, but no GPU attention kernel or model forward is executed.',
                'Potential cross-sample attention counts reflect segments spanning recorded row boundaries, not proof of actual probability-error causation.',
                'Complete tokens/positions/masks stay in original same-node .pt assets; report contains hashes and bounded metadata examples only.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--expected-package-manifest-sha256',required=True)
    parser.add_argument('--expected-ready-sha256',required=True)
    parser.add_argument('--expected-context-helper-sha256',required=True)
    parser.add_argument('--expected-hf-flash-helper-sha256',default=HF_SHA)
    parser.add_argument('--expected-hf-qwen-source-sha256',default=HF_QWEN_SHA)
    parser.add_argument('--max-examples',type=int,default=4)
    args=parser.parse_args(); require(0<=args.max_examples<=8,'Example count out of range')
    require(socket.gethostname().split('.')[0]=='h100-9' and not os.environ.get('CUDA_VISIBLE_DEVICES',''),'Same-node h100-9 CPU execution with CUDA hidden required')
    started=time.perf_counter(); cpu=time.process_time(); torch.set_num_threads(4)
    A=context_tools(args.expected_context_helper_sha256)
    verified=A.verify_bundle(args.run,args.package,args.expected_package_manifest_sha256,args.expected_ready_sha256)
    require(importlib.metadata.version('transformers')=='4.46.3','HF rules version changed')
    path=importlib.metadata.distribution('transformers').locate_file('transformers/modeling_flash_attention_utils.py')
    rules=hf_rules(path,args.expected_hf_flash_helper_sha256)
    qwen_path=importlib.metadata.distribution('transformers').locate_file('transformers/models/qwen2/modeling_qwen2.py')
    worker_relative='source/repo/verl/workers/fsdp_workers.py'
    caller=native_qwen_caller(qwen_path,args.expected_hf_qwen_source_sha256,args.package/worker_relative,verified[1]['files'][worker_relative])
    result=inspect_bundle(A,verified,args.package.resolve(),rules,args.max_examples)
    require(not torch.cuda.is_initialized(),'Position CPU analysis initialized CUDA')
    result.update(package_manifest_sha256=args.expected_package_manifest_sha256,ready_sha256=args.expected_ready_sha256,
        helper_sha256=sha(__file__),context_helper_sha256=args.expected_context_helper_sha256,
        native_caller_source=caller,package_versions=dict(torch=str(torch.__version__),transformers=importlib.metadata.version('transformers')),
        analysis_cost=dict(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            threads=4,cuda_initialized=False,analysis_slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
