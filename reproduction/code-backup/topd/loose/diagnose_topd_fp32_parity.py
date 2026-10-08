#!/usr/bin/env python3
"""Original failed-row hardware diagnosis only. No training or parity approval.

Run through own single-UUID inherited lease, with fresh process per --rank (4, then 0).
Inputs are the unmodified 2969 raw safetensors; row 8 must remain present.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def comparison(torch, actual, reference):
    delta = (actual - reference).abs()
    bound = 1e-4 + 1e-4 * reference.abs()
    bad = (~torch.isfinite(actual)) | (delta > bound)
    return dict(allclose=bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                max_bound_ratio=float((delta / bound).max()), violation_count=int(bad.sum()),
                violations=[dict(token=int(i), reference=float(reference.flatten()[i]),
                                 actual=float(actual.flatten()[i]), delta=float(delta.flatten()[i]),
                                 bound=float(bound.flatten()[i]))
                            for i in bad.flatten().nonzero().flatten().tolist()[:100]])



def trace_boundaries(torch, teacher, ids, mask):
    """Read-only hooks: output None, detached CPU copies, no retained CUDA tensor."""
    model=teacher.model.base_model
    captures={}; handles=[]
    modules=[('embedding',model.embed_tokens)]
    modules += [(f'layer{index:02d}',layer) for index,layer in enumerate(model.layers)]
    modules += [('finalnorm',model.norm)]
    def hook(name):
        def capture(module, inputs, output):
            if name in captures:raise RuntimeError('Boundary ran more than once: '+name)
            value=output[0] if isinstance(output,tuple) else output
            captures[name]=value.detach().cpu().clone()
            return None
        return capture
    def rope_hook(module, inputs, output):
        for key,value in zip(('rope_cos','rope_sin'),output):
            if key in captures:raise RuntimeError('RoPE ran more than once')
            captures[key]=value.detach().cpu().clone()
        captures['rope_inv_freq']=module.inv_freq.detach().cpu().clone()
        return None
    try:
        for name,module in modules:handles.append(module.register_forward_hook(hook(name)))
        handles.append(model.rotary_emb.register_forward_hook(rope_hook))
        with torch.no_grad():
            result=model(input_ids=ids.to(teacher.device),attention_mask=mask.to(teacher.device),use_cache=False)
        del result
        expected={name for name,_ in modules}|{'rope_cos','rope_sin','rope_inv_freq'}
        if set(captures)!=expected:raise RuntimeError('Incomplete boundary capture')
        return captures,[name for name,_ in modules]
    finally:
        for handle in handles:handle.remove()


def boundary_stats(torch, actual, reference):
    if actual.shape!=reference.shape:raise RuntimeError('Boundary shapes differ')
    a,b=actual.double(),reference.double(); delta=a-b
    denominator=float(torch.linalg.vector_norm(b))
    norm=float(torch.linalg.vector_norm(delta))
    return dict(shape=list(actual.shape),count=actual.numel(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                exact_equal=bool(torch.equal(actual,reference)),
                max_abs=float(delta.abs().max()),mean_abs=float(delta.abs().mean()),
                relative_l2=norm/denominator if denominator else (0. if norm==0 else None),
                reference_l2=denominator,delta_l2=norm)


def summarize_trace(torch, cpu, gpu, names):
    stats={name:boundary_stats(torch,gpu[name],cpu[name]) for name in names}
    first=next((name for name in names if not stats[name]['exact_equal']),None)
    previous=0.; increments={}
    for name in names:
        increments[name]=stats[name]['max_abs']-previous
        previous=stats[name]['max_abs']
    largest=max(names,key=lambda name:increments[name])
    return dict(boundaries=stats,first_nonidentical_boundary=first,
                largest_max_abs_increment_boundary=largest,max_abs_increments=increments,
                increment_scope='Difference of boundary max-absolute errors; diagnostic, not causal attribution',
                rope={name:boundary_stats(torch,gpu[name],cpu[name])
                      for name in ('rope_cos','rope_sin','rope_inv_freq')}),set(filter(None,(first,largest,names[-1])))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=(0, 4), default=4)
    p.add_argument('--decompose', action='store_true')
    p.add_argument('--trace-layers-only', action='store_true')
    a = p.parse_args()
    if a.trace_layers_only and (a.rank!=4 or a.decompose):
        raise RuntimeError('Layer trace only supports original rank4 and excludes decompose')
    if socket.gethostname().split('.')[0] != 'h100-11' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires own h100-11 Slurm job')
    root=Path('/dataset/shaoshitong/search-r1-rl/topd')
    if a.repo.resolve()!=root/'repo' or a.probe_dir.resolve()!=root/'logs/afunix-scorer-hardware-2969' or not a.output.resolve().is_relative_to(root/'logs'):
        raise RuntimeError('Require exact own repo, original probe and own logs output')
    if a.output.exists() or a.output.with_suffix('.safetensors').exists():
        raise RuntimeError('Refuse to replace prior diagnosis JSON or tensors')
    commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    if commit!='438af3529b40663aae7b8293080e09fda57f6a87':
        raise RuntimeError('Production source changed since original failure')
    sys.path.insert(0, str(a.repo.resolve()))
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(1)  # Each worker inherits only its single UUID + FD.
    manifest_path=a.probe_dir/'teacher-ipc/manifest.json'
    if sha(manifest_path)!='4a6c0f743291fb3df0af1993562746eba83c2e0616cb07eb5c0c3f4d6a5af558':
        raise RuntimeError('Original manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['kind'] != 'HARDWARE_SURROGATE_NO_TRAINING':
        raise RuntimeError('Only base hardware diagnosis supported')
    uuid = next(r['uuid'] for r in manifest['services'] if r['rank'] == a.rank)
    records=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    if len(records)!=1 or records[0]['uuid']!=uuid or os.environ['CUDA_VISIBLE_DEVICES']!=uuid:
        raise RuntimeError('Worker must inherit exactly its original rank UUID and one own FD')
    lease=records[0]
    scope=Path('/proc/self/cgroup').read_text()
    if '/job_'+os.environ['SLURM_JOB_ID']+'/' not in scope:
        raise RuntimeError('Outside own Slurm cgroup')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[name] = '1'
    import torch
    from safetensors.torch import load_file, save_file, save
    from transformers import AutoTokenizer
    from search_r1.baselines.distillation import FrozenTeacher, teacher_inputs
    from search_r1.baselines.topd_local_gpu import (lease_ownership, snapshot,
        initialize_cuda_context, cuda_process_snapshot, release_cuda, implementation_fingerprint)
    expected_fp='101eb46fdd5cc493d8be3be11cb9d63313b3e94819bd9d2034b37a1f94e3a9a6'
    if implementation_fingerprint()!=expected_fp or manifest['source_sha256']!=expected_fp or manifest['code_commit']!=commit:
        raise RuntimeError('Original implementation fingerprint differs')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before UUID selection')
    if torch.__version__.split('+')[0] != '2.4.0':
        raise RuntimeError('Requires pinned torch2.4.0')
    source = a.probe_dir / 'parity-round1.safetensors'
    if sha(source)!='fd6efd8eafbf984257663a415dae42edcf1a06132b271df760fea78bcfda9e08':
        raise RuntimeError('Original failed raw inputs/logp changed')
    raw = load_file(str(source), device='cpu')
    receipt_path=a.probe_dir/'teacher-ipc/scorer-4.jsonl'
    if sha(receipt_path)!='660e2e072230f991939534d78a9ecb231dfc452cfd1eb0c37acd26efe8effde9':
        raise RuntimeError('Original failed-row rank receipt changed')
    receipt=json.loads(receipt_path.read_text().splitlines()[0])
    wire={k:raw[k][8:10].contiguous() for k in ('prompts','responses','attention')}
    wire['row_indices']=torch.tensor([8,9],dtype=torch.long)
    if hashlib.sha256(save(wire)).hexdigest()!=receipt['input_sha256'] or receipt['outer_step']!=1 or receipt['source_sha256']!=expected_fp:
        raise RuntimeError('Failed row inputs do not match original rank4 IPC receipt')
    rows = [8, 9]  # Real failing row + same-rank control; never substitute samples.
    x = {k: raw[k][rows].clone() for k in ('prompts', 'responses', 'attention')}
    if x['prompts'].shape != (2, 256) or x['responses'].shape != (2, 32):
        raise RuntimeError('Not original failed round1 dimensions')
    if not bool((x['attention'] == 1).all()):
        raise RuntimeError('Not original fully active round1 mask')
    teacher_path = Path(manifest['teacher_path']).resolve()
    if not teacher_path.is_relative_to(a.repo.resolve().parent):
        raise RuntimeError('Teacher must be copied under own local root')
    for name, expected in manifest['hardware_model_files'].items():
        if sha(teacher_path / name) != expected:
            raise RuntimeError('Original hardware HF files changed: ' + name)
    tok = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    tok.pad_token_id = manifest['student_tokenizer']['pad_token_id']
    teacher = FrozenTeacher(str(teacher_path), tok, 'cpu')
    out = dict(scope='HARDWARE_DIAGNOSTIC_NO_TRAINING_NOT_PARITY_APPROVAL',
               original_probe_sha256=sha(source), manifest_sha256=sha(a.probe_dir/'teacher-ipc/manifest.json'),
               script_sha256=sha(__file__), rows=rows, rank=a.rank, uuid=uuid,
               host=socket.gethostname(), pid=os.getpid(), slurm_job=os.environ['SLURM_JOB_ID'],
               cgroup=Path('/proc/self/cgroup').read_text(), torch_version=torch.__version__,
               source_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
               cpu={}, gpu={}, decompose={}, errors=[],
               environment={k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
                   'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
    values = dict(x, original_cpu=raw['cpu_log_probs'][rows], original_gpu=raw['gpu_log_probs'][rows])
    out['loaded_cpu'] = snapshot(teacher)
    if out['loaded_cpu']['parameter_sha256']!=receipt['loaded_cpu']['parameter_sha256']:
        raise RuntimeError('Original frozen parameter semantics differ')
    def flags():
        return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_precision=torch.get_float32_matmul_precision(),
                    autocast_enabled=torch.is_autocast_enabled(), cpu_threads=torch.get_num_threads(),
                    mkl_available=torch.backends.mkl.is_available(), mkldnn_enabled=torch.backends.mkldnn.enabled)
    out['initial_flags'] = flags()
    def score():
        return torch.cat([teacher.log_probs(x['prompts'][i:i+1], x['responses'][i:i+1],
                          x['attention'][i:i+1], temperature=1., chunk=16) for i in range(2)])
    if not a.trace_layers_only:
        for threads in (1, 16):
            torch.set_num_threads(threads)
            for repeat in (1, 2):
                start=time.monotonic(); v=score(); key=f'cpu{threads}_repeat{repeat}'
                values[key]=v; out['cpu'][key]=dict(seconds=time.monotonic()-start, flags=flags(),
                    per_row=[comparison(torch,v[i],values['original_cpu'][i]) for i in range(2)])
    cpu_trace={}
    if a.trace_layers_only:
        if len(teacher.model.base_model.layers)!=36:raise RuntimeError('Original model must have 36 decoder layers')
        torch.set_num_threads(16)
        out['trace_cpu_flags']=flags();out['trace_layers']={}
        for i,row in enumerate(rows):
            ids,mask,_=teacher_inputs(tok,x['prompts'][i:i+1],x['responses'][i:i+1],x['attention'][i:i+1])
            start=time.monotonic();cpu_trace[row]=trace_boundaries(torch,teacher,ids,mask)
            out['trace_layers'][str(row)]={'cpu_wall_seconds':time.monotonic()-start}
    # Capture the unchanged CPU head storage for optional cross-backend localization.
    head=teacher.model.get_output_embeddings()
    weight=head.weight.detach() if a.decompose else None
    bias=head.bias.detach() if a.decompose and head.bias is not None else None
    cpu_hidden=[]
    if a.decompose:
        with torch.no_grad():
            for i in range(2):
                ids,mask,pos=teacher_inputs(tok,x['prompts'][i:i+1],x['responses'][i:i+1],x['attention'][i:i+1])
                h=teacher.model.base_model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                cpu_hidden.append(h.gather(1,pos[...,None].expand(-1,-1,h.shape[-1])))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    try:
        lease_ownership(lease); initialize_cuda_context('cuda:0')
        out['actual_cuda_process']=cuda_process_snapshot(uuid)
        teacher.model.to('cuda:0');teacher.device=torch.device('cuda:0')
        out['loaded_gpu']=snapshot(teacher)
        if out['loaded_gpu']['parameter_sha256'] != out['loaded_cpu']['parameter_sha256']:
            raise RuntimeError('Parameter semantic hash changed')
        if not a.trace_layers_only:
            for repeat in (1,2):
                start=time.monotonic();v=score();torch.cuda.synchronize();key=f'gpu_repeat{repeat}'
                values[key]=v;out['gpu'][key]=dict(seconds=time.monotonic()-start,flags=flags(),
                    per_row=[comparison(torch,v[i],values['cpu16_repeat1'][i]) for i in range(2)])
        if a.trace_layers_only:
            torch.set_num_threads(16)
            out['trace_gpu_flags']=flags()
            for i,row in enumerate(rows):
                ids,mask,_=teacher_inputs(tok,x['prompts'][i:i+1],x['responses'][i:i+1],x['attention'][i:i+1])
                start=time.monotonic();gpu_capture,names=trace_boundaries(torch,teacher,ids,mask)
                cpu_capture,cpu_names=cpu_trace.pop(row)
                if names!=cpu_names:raise RuntimeError('CPU/GPU boundary order differs')
                summary,selected=summarize_trace(torch,cpu_capture,gpu_capture,names)
                out['trace_layers'][str(row)].update(summary,gpu_wall_seconds=time.monotonic()-start,
                    saved_boundary_names=sorted(selected),raw_scope='Only first nonidentical, largest max-abs increment, finalnorm and RoPE paired CPU raw')
                for name in selected|{'rope_cos','rope_sin','rope_inv_freq'}:
                    values[f'row{row}_{name}_cpu']=cpu_capture[name]
                    values[f'row{row}_{name}_gpu']=gpu_capture[name]
                del cpu_capture,gpu_capture
        if a.decompose:
            torch.set_num_threads(16)  # Match CPU16 reference for cross-head CPU GEMMs.
            out["decompose_flags"]=flags()
            def decompose():
                with torch.no_grad():
                    for i,row in enumerate(rows):
                        ids,mask,pos=teacher_inputs(tok,x['prompts'][i:i+1],x['responses'][i:i+1],x['attention'][i:i+1])
                        h=teacher.model.base_model(input_ids=ids.cuda(),attention_mask=mask.cuda(),use_cache=False).last_hidden_state
                        gh=h.gather(1,pos.cuda()[...,None].expand(-1,-1,h.shape[-1]))
                        ch=cpu_hidden[i]; variants={k:[] for k in ('cpu_head_cpu_hidden','cpu_head_gpu_hidden','gpu_head_cpu_hidden','gpu_logits_cpu_softmax','gpu_head_gpu_hidden')}
                        for start in (0,16):
                            c=ch[:,start:start+16];g=gh[:,start:start+16];labels=x['responses'][i:i+1,start:start+16]
                            gl=head(g).float()
                            logits={'cpu_head_cpu_hidden':torch.nn.functional.linear(c,weight,bias),
                                    'cpu_head_gpu_hidden':torch.nn.functional.linear(g.cpu(),weight,bias),
                                    'gpu_head_cpu_hidden':head(c.cuda()).float(),
                                    'gpu_logits_cpu_softmax':gl.cpu(), 'gpu_head_gpu_hidden':gl}
                            for key,z in logits.items():
                                variants[key].append(torch.nn.functional.log_softmax(z,-1).gather(-1,labels.to(z.device)[...,None]).squeeze(-1).cpu())
                        out['decompose'][str(row)]={'hidden':comparison(torch,gh.cpu(),ch),'variants':{}}
                        for key,parts in variants.items():
                            value=torch.cat(parts,-1);values[f'row{row}_{key}']=value
                            out['decompose'][str(row)]['variants'][key]=comparison(torch,value,values['cpu16_repeat1'][i:i+1])
            decompose()
    except Exception as error:
        out['errors'].append(repr(error));raise
    finally:
        try:
            out['released']=release_cuda(teacher);out['released_cpu']=snapshot(teacher)
            if out['released'] != dict(allocated_bytes=0,reserved_bytes=0):
                out['errors'].append('Nonzero own allocator after release')
            if out['released_cpu']['parameter_sha256']!=out['loaded_cpu']['parameter_sha256']:
                out['errors'].append('Released parameter hash differs')
        except Exception as error:out['errors'].append('release: '+repr(error))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        save_file({k:v.contiguous() for k,v in values.items()},str(a.output.with_suffix('.safetensors')))
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    if out['errors']:raise RuntimeError(out['errors'])


if __name__=='__main__':main()
