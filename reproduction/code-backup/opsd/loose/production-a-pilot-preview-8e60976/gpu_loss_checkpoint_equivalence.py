"""Actual-3B exact numerical loss-checkpoint control; no training optimizer step."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

PARAMETERS = 3085938688


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    require(type(value) in (int,float) and math.isfinite(value), 'Missing/nonfinite observation')
    return value


def validate_result(j):
    """This exact-equality gate has no nonzero tolerance or automatic fallback."""
    require(j['scope'] in ('production-eight','isolated-one'), 'Unspecified allocation scope')
    require(j['leased_gpu_count'] == (8 if j['scope']=='production-eight' else 1), 'Wrong inherited lease count')
    require(j['all_student_parameter_elements_compared'] == PARAMETERS, 'Incomplete actual3B gradient comparison')
    for name in ('loss_absolute_difference','student_gradient_max_abs','student_gradient_difference_l2'):
        require(finite(j[name]) == 0., 'Checkpoint loss/student gradient numerical equality failed')
    require(finite(j['reference_gradient_l2']) > 0 and finite(j['reference_gradient_max_abs']) > 0, 'Zero reference gradients')
    require(j['student_parameter_dtypes'] == ['torch.bfloat16'] and j['student_attention'] == 'flash_attention_2' and
            j['student_activation_checkpointing'] is True, 'Actual student execution precision/attention changed')
    require(j['matmul_tf32'] is False and j['cudnn_tf32'] is False and
            j['teacher_returned_device'] == 'cpu', 'Runtime precision/storage invariant changed')
    modes=j['modes']
    require(len(modes)==2 and [r['loss_checkpointing'] for r in modes]==[False,True], 'Missing explicit off/on modes')
    width=j['response_width']; require(type(width) is int and width>16 and width%16!=0, 'Must include full and final short chunk')
    chunks=[min(16,width-start) for start in range(0,width,16)]
    require(type(j['masked_observation_positions']) is int and 0<j['masked_observation_positions']<width and
            j['supervised_positions']+j['masked_observation_positions']==width, 'Missing mixed observation mask')
    for index, r in enumerate(modes):
        finite(r['loss'])
        require(r['teacher_head_chunk_lengths']==chunks and r['teacher_head_forward_calls']==len(chunks) and
                r['teacher_base_forward_calls']==1, 'Teacher recomputed, omitted or wrong chunks')
        require(r['teacher_parameter_dtypes']==['torch.float32'] and r['teacher_attention']=='eager' and
                r['teacher_active_device']=='cuda:0' and r['teacher_head_grad_enabled']==[False]*len(chunks),
                'Teacher precision/device/frozen-forward changed')
        require(r['generalized_jsd_calls']==len(chunks)*(1 if index==0 else 2), 'Loss recomputation did not execute exactly per chunk')
        require(r['checkpoint_invocations']==([ ] if index==0 else [False]*len(chunks)), 'Checkpoint mode must be non-reentrant')
        for key in ('teacher/trainable_parameters_per_rank','teacher/gradient_tensors_per_rank','teacher/student_optimizer_membership_per_rank'):
            require(type(r['teacher_audit'][key]) is int and r['teacher_audit'][key]==0, 'Teacher lost frozen isolation')
    require(abs(modes[0]['loss']-modes[1]['loss'])==j['loss_absolute_difference'], 'Loss metadata inconsistent')
    require(j['scope']!='production-eight' or j['retriever_same_job_loopback'] is True, 'Missing same-job local GPU retriever coexistence')
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default=os.environ.get('OPSD_TEACHER'))
    p.add_argument('--output',default=str(Path(os.environ.get('OPSD_EVIDENCE','.'))/'gpu-loss-checkpoint-equivalence.json'))
    p.add_argument('--scope',choices=['production-eight','isolated-one'],default='production-eight')
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('owned_diagnostic',Path(__file__).with_name('gpu_teacher_probe-factor-dtype.py'))
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    allocation=previous.allocation_scope()
    path=Path(args.output);require(not path.exists(),'Refuse stale checkpoint-equivalence output')
    path.parent.mkdir(parents=True,exist_ok=True)
    record=dict(passed=False,allocation=allocation,model=args.model,scope=args.scope,optimizer_steps=0,
                purpose='Exact numerical control for per-chunk non-reentrant loss checkpoint only; not an actual outer update or final acceptance')
    started=time.perf_counter()
    try:
        # Diagnostic-only deterministic backward request; parent/trainer env unchanged.
        os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from types import SimpleNamespace
        import search_r1.baselines.distillation as loss_module
        from search_r1.baselines.rollout import padded
        from search_r1.baselines.preflight import validate_inherited_gpu_leases
        expected=8 if args.scope=='production-eight' else 1
        validate_inherited_gpu_leases(expected)
        record['leased_gpu_count']=expected
        require(torch.cuda.device_count()==expected,'Visible CUDA devices differ from declared owned scope')
        if args.scope=='production-eight':
            evidence=Path(os.environ['OPSD_EVIDENCE']);pid=int((evidence/'retriever.pid').read_text())
            require(('job_'+allocation['job_id']+'/') in Path(f'/proc/{pid}/cgroup').read_text(),'Retriever outside current job')
            listeners=subprocess.check_output(['ss','-ltnp','sport = :18084'],text=True)
            require(f'pid={pid},' in listeners and '127.0.0.1:18084' in listeners,'Local same-job retriever not listening')
            record.update(retriever_same_job_loopback=True,retriever_pid=pid)
        else:
            record['retriever_same_job_loopback']=False
        torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(args.model)
        prompts=padded([tok.encode('Question: What is the capital of France?\nAssistant:')],tok.pad_token_id,True)
        responses=padded([tok.encode('<search>France capital</search>\n<information>Paris is the capital.</information>\n<answer>Paris</answer>')],tok.pad_token_id)
        ids=torch.cat((prompts,responses),1).cuda();attention=torch.ones_like(ids)
        mask=torch.ones_like(responses,dtype=torch.float32).cuda();mask[:,8:16]=0
        tensors=dict(prompts=prompts.cuda(),responses=responses.cuda(),input_ids=ids,attention_mask=attention,
                     position_ids=attention.cumsum(-1)-1,loss_mask=mask)
        teacher=loss_module.FrozenTeacher(args.model,tok,'cuda:0')
        student=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
                    attn_implementation='flash_attention_2').cuda().train()
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        for module in student.modules():
            if isinstance(module,torch.nn.Dropout):module.eval()
        require(sum(v.numel() for v in student.parameters())==PARAMETERS,'Unexpected student model')
        optimizer=torch.optim.AdamW(student.parameters(),lr=1e-6)  # Audit membership only, never step.
        settings=SimpleNamespace(chunk_size=16,beta=0.,temperature=1.,component_clip=.05,max_student_logits_gib=16.,loss_checkpointing=False)
        current={};reference=None;original_jsd=loss_module.generalized_jsd;original_checkpoint=loss_module.checkpoint
        def count_jsd(*a,**kw):
            current['generalized_jsd_calls']+=1
            return original_jsd(*a,**kw)
        def count_checkpoint(*a,**kw):
            current['checkpoint_invocations'].append(kw.get('use_reentrant'))
            return original_checkpoint(*a,**kw)
        def head_hook(module,inputs,output):
            current['teacher_head_forward_calls']+=1
            current['teacher_head_chunk_lengths'].append(output.shape[1])
            current['teacher_head_grad_enabled'].append(torch.is_grad_enabled())
        def base_hook(module,inputs,output):current['teacher_base_forward_calls']+=1
        handles=[teacher.model.get_output_embeddings().register_forward_hook(head_hook),teacher.model.base_model.register_forward_hook(base_hook)]
        loss_module.generalized_jsd=count_jsd;loss_module.checkpoint=count_checkpoint
        modes=[]
        try:
            for enabled in (False,True):
                current=dict(loss_checkpointing=enabled,generalized_jsd_calls=0,checkpoint_invocations=[],
                             teacher_head_forward_calls=0,teacher_head_chunk_lengths=[],teacher_base_forward_calls=0,teacher_head_grad_enabled=[])
                settings.loss_checkpointing=enabled;student.zero_grad(set_to_none=True)
                with teacher.on_device():
                    runtime=previous.teacher_runtime_audit(teacher,'cuda:0')
                    loss=loss_module.opsd_loss(student,teacher,tensors,tok,[['Paris']],settings,mask.sum())
                    loss.backward();torch.cuda.synchronize()
                    current.update(loss=float(loss.detach()),teacher_parameter_dtypes=runtime['parameter_dtypes'],
                                   teacher_attention=runtime['attention'],teacher_active_device=runtime['active_device'],
                                   teacher_audit=teacher.audit_metrics(optimizer))
                    if not enabled:
                        reference={name:v.grad.detach().cpu().clone() for name,v in student.named_parameters() if v.grad is not None}
                    else:
                        d2=r2=dmax=rmax=0.;elements=0;seen=set()
                        for name,v in student.named_parameters():
                            if v.grad is None:
                                require(name not in reference,'Gradient coverage changed');continue
                            require(name in reference,'Extra gradient parameter');seen.add(name)
                            actual=v.grad.detach().cpu().float();expected_grad=reference[name].float();diff=actual-expected_grad
                            d2+=float(diff.square().sum());r2+=float(expected_grad.square().sum())
                            dmax=max(dmax,float(diff.abs().max()));rmax=max(rmax,float(expected_grad.abs().max()));elements+=v.numel()
                        require(seen==set(reference),'Missing student gradients')
                        record.update(all_student_parameter_elements_compared=elements,student_gradient_difference_l2=d2**.5,
                                      reference_gradient_l2=r2**.5,student_gradient_max_abs=dmax,reference_gradient_max_abs=rmax)
                modes.append(current)
                del loss
        finally:
            loss_module.generalized_jsd=original_jsd;loss_module.checkpoint=original_checkpoint
            for handle in handles:handle.remove()
        record.update(modes=modes,loss_absolute_difference=abs(modes[0]['loss']-modes[1]['loss']),
                      student_parameter_dtypes=sorted({str(v.dtype) for v in student.parameters()}),
                      student_attention=student.config._attn_implementation,student_activation_checkpointing=student.is_gradient_checkpointing,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                      teacher_returned_device=str(teacher.device),response_width=responses.shape[1],
                      masked_observation_positions=int((mask==0).sum()),supervised_positions=int(mask.sum()),
                      loss_source_sha256=hashlib.sha256(Path(loss_module.__file__).read_bytes()).hexdigest(),
                      runtime_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      equivalence='numeric_exact; not memory byte comparison',actual_outer_iterations=0)
        validate_result(record);record['passed']=True
    except Exception as exc:
        import traceback
        record.update(passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        path.write_text(json.dumps(record,indent=2));raise
    path.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
