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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=(0, 4), default=4)
    p.add_argument('--decompose', action='store_true')
    a = p.parse_args()
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
    for threads in (1, 16):
        torch.set_num_threads(threads)
        for repeat in (1, 2):
            start=time.monotonic(); v=score(); key=f'cpu{threads}_repeat{repeat}'
            values[key]=v; out['cpu'][key]=dict(seconds=time.monotonic()-start, flags=flags(),
                per_row=[comparison(torch,v[i],values['original_cpu'][i]) for i in range(2)])
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
        for repeat in (1,2):
            start=time.monotonic();v=score();torch.cuda.synchronize();key=f'gpu_repeat{repeat}'
            values[key]=v;out['gpu'][key]=dict(seconds=time.monotonic()-start,flags=flags(),
                per_row=[comparison(torch,v[i],values['cpu16_repeat1'][i]) for i in range(2)])
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
