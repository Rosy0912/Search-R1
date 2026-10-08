"""Actual-model CPU/GPU FP32 teacher equivalence. Run only inside owned Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import socket
import time


def allocation_scope():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise RuntimeError('Requires an owned Slurm allocation')
    node = socket.gethostname().split('.')[0]
    nodelist = os.environ.get('SLURM_JOB_NODELIST')
    cgroup = Path('/proc/self/cgroup').read_text()
    import re
    if not re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', cgroup, re.MULTILINE):
        raise RuntimeError('Process is outside its declared Slurm job cgroup')
    if node != 'h100-10' or nodelist != 'h100-10':
        raise RuntimeError('Probe requires the single local h100-10 allocation')
    return dict(job_id=job, hostname=socket.gethostname(), node_short=node,
                slurm_job_nodelist=nodelist, own_job_cgroup=cgroup)


def teacher_runtime_audit(teacher, expected_device):
    import torch
    expected_device = torch.device(expected_device)
    dtypes = sorted({str(p.dtype) for p in teacher.model.parameters()})
    devices = sorted({str(p.device) for p in teacher.model.parameters()})
    attention = teacher.model.config._attn_implementation
    if dtypes != ['torch.float32'] or attention != 'eager':
        raise AssertionError('Teacher FP32/eager invariant failed')
    if devices != [str(expected_device)] or teacher.device != expected_device:
        raise AssertionError('Teacher active device invariant failed')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise AssertionError('TF32 unexpectedly enabled')
    return dict(parameter_dtypes=dtypes, parameter_devices=devices, attention=attention,
                active_device=str(teacher.device), matmul_tf32=False, cudnn_tf32=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('OPSD_TEACHER'))
    p.add_argument('--output', default=str(Path(os.environ.get('OPSD_EVIDENCE', '.')) / 'gpu-teacher-equivalence.json'))
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--flash-deterministic', action='store_true', help='Diagnostic only: request deterministic FlashAttention backward before Transformers imports')
    p.add_argument('--factor-bf16-backward', action='store_true', help='Diagnostic only: capture projection gradients and force identical upstream gradient to isolate downstream BF16 backward')
    p.add_argument('--fp32-reference', action='store_true', help='Also isolate teacher-target change using actual FP32/eager student and all parameter gradients')
    args = p.parse_args()
    allocation = allocation_scope()
    if args.flash_deterministic:
        os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from types import SimpleNamespace
    from search_r1.baselines.distillation import FrozenTeacher, opsd_loss, teacher_inputs, generalized_jsd
    from search_r1.baselines.rollout import padded
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = padded([tok.encode('Question: What is the capital of France?\nAssistant:')], tok.pad_token_id, True)
    response = padded([tok.encode('<search>France capital</search>\n<information>Paris is the capital.</information>\n<answer>Paris</answer>')], tok.pad_token_id)
    ids = torch.cat([prompts, response], dim=1).cuda()
    attention = torch.ones_like(ids)
    mask = torch.ones_like(response, dtype=torch.float32).cuda()
    # Observations remain attended while their prediction positions are excluded.
    mask[:, 8:16] = 0
    tensors = dict(prompts=prompts.cuda(), responses=response.cuda(), input_ids=ids,
                   attention_mask=attention, position_ids=attention.cumsum(-1)-1, loss_mask=mask)
    targets = [['Paris']]
    settings = SimpleNamespace(chunk_size=16, beta=0., temperature=1., component_clip=.05,
                               max_student_logits_gib=16.)
    teacher = FrozenTeacher(args.model, tok, 'cpu')
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                  attn_implementation='flash_attention_2').cuda().eval()
    student.requires_grad_(True)
    student_dtypes = sorted({str(p.dtype) for p in student.parameters()})
    if student_dtypes != ['torch.bfloat16']:
        raise AssertionError('Numerical student must have BF16 parameters')
    teacher_cpu_runtime = teacher_runtime_audit(teacher, 'cpu')
    tid, tm, pos = teacher_inputs(tok, prompts, response, torch.ones_like(response), targets,
                                 teacher.model.config.max_position_embeddings)

    def targets_now():
        with teacher.measure(), torch.no_grad():
            h = teacher.model.base_model(input_ids=tid.to(teacher.device),
                    attention_mask=tm.to(teacher.device), use_cache=False).last_hidden_state
            selected = h.gather(1, pos.to(teacher.device)[..., None].expand(-1, -1, h.shape[-1]))
            return teacher.model.get_output_embeddings()(selected).float().cpu()

    def snapshot_grads(model):
        return {name: v.grad.detach().cpu().clone() for name, v in model.named_parameters() if v.grad is not None}

    def compare_grads(model, reference):
        squared_diff = squared_reference = 0.
        max_difference = maximum_reference = 0.
        count = 0
        seen = set()
        per_parameter = []
        for name, param in model.named_parameters():
            if param.grad is None:
                if name in reference:
                    raise AssertionError('Gradient presence changed: ' + name)
                continue
            seen.add(name)
            actual = param.grad.detach().cpu().float()
            expected = reference[name].float()
            difference = actual - expected
            d2, e2 = float(difference.square().sum()), float(expected.square().sum())
            dmax, emax = float(difference.abs().max()), float(expected.abs().max())
            squared_diff += d2
            squared_reference += e2
            max_difference = max(max_difference, dmax)
            maximum_reference = max(maximum_reference, emax)
            count += param.numel()
            per_parameter.append(dict(name=name, difference_l2=d2**.5, reference_l2=e2**.5,
                                      max_absolute_difference=dmax, reference_max_abs=emax))
        if seen != set(reference):
            raise AssertionError('Student gradient coverage changed')
        return dict(relative_l2=(squared_diff / max(squared_reference, 1e-30))**.5,
                    difference_l2=squared_diff**.5, reference_l2=squared_reference**.5,
                    max_absolute_difference=max_difference, reference_max_abs=maximum_reference,
                    parameter_elements_compared=count,
                    largest_difference_parameters=sorted(per_parameter, key=lambda r:r['difference_l2'], reverse=True)[:10])

    def cached_target_loss(model, target):
        # Explicit FP32 model execution, not opsd_loss's BF16 CUDA autocast path.
        with torch.autocast('cuda', enabled=False):
            logits = model(input_ids=ids, attention_mask=attention,
                           position_ids=tensors['position_ids'], use_cache=False).logits
            source = logits[:, -response.shape[1]-1:-1]
            total = source.sum() * 0
            for start in range(0, response.shape[1], settings.chunk_size):
                stop = min(start+settings.chunk_size, response.shape[1])
                valid = mask[:, start:stop].bool()[..., None]
                divergence = generalized_jsd(torch.where(valid, source[:, start:stop], 0.),
                    torch.where(valid, target[:, start:stop].to(source.device), 0.),
                    settings.beta, settings.temperature, settings.component_clip)
                total = total + (divergence*mask[:, start:stop]).sum()
            return total/mask.sum()

    started = time.perf_counter()
    cpu_logits = targets_now()
    loss_cpu = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
    loss_cpu.backward()
    reference_grads = snapshot_grads(student)
    cpu_loss_value = float(loss_cpu.detach())
    student.zero_grad(set_to_none=True)
    # Same teacher placement and weights, same student and trajectory. This is
    # the control for nondeterministic BF16/Flash backward or repeat sensitivity.
    cpu_repeat_controls = []
    for _ in range(3):
        loss_cpu_repeat = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
        loss_cpu_repeat.backward()
        control = compare_grads(student, reference_grads)
        control['loss_absolute_difference'] = abs(float(loss_cpu_repeat.detach())-cpu_loss_value)
        cpu_repeat_controls.append(control)
        student.zero_grad(set_to_none=True)
    teacher.execution_device = torch.device('cuda', torch.cuda.current_device())
    with teacher.on_device():
        teacher_gpu_runtime = teacher_runtime_audit(teacher, teacher.execution_device)
        gpu_logits = targets_now()
        loss_gpu = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
        loss_gpu.backward()
        torch.cuda.synchronize()
        gpu_loss_value = float(loss_gpu.detach())
        migration_gradients = compare_grads(student, reference_grads)
        gpu_reference_grads = snapshot_grads(student)
        student.zero_grad(set_to_none=True)
        gpu_repeat_controls = []
        for _ in range(3):
            loss_gpu_repeat = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
            loss_gpu_repeat.backward()
            control = compare_grads(student, gpu_reference_grads)
            control['loss_absolute_difference'] = abs(float(loss_gpu_repeat.detach())-gpu_loss_value)
            gpu_repeat_controls.append(control)
            student.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
    student.zero_grad(set_to_none=True)
    # Freeze the exact student logits to remove transformer backward entirely.
    # Differentiate full-vocabulary clipped FKL in FP32 and check its explicit
    # derivative, including any components crossing the clipping boundary.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        fixed_source = student(input_ids=ids, attention_mask=attention,
            position_ids=tensors['position_ids'], use_cache=False).logits[:, -response.shape[1]-1:-1].detach().float()
    valid = mask.bool()[..., None]
    fixed_source = torch.where(valid, fixed_source, 0.)
    clip_parts, fixed_gradients, fixed_losses = [], [], []
    derivative_errors = []
    for target in (cpu_logits, gpu_logits):
        leaf = fixed_source.clone().requires_grad_(True)
        target = torch.where(valid, target.cuda(), 0.)
        scalar = (generalized_jsd(leaf, target, 0., 1., .05)*mask).sum()/mask.sum()
        gradient, = torch.autograd.grad(scalar, leaf)
        sl, tl = leaf.detach().log_softmax(-1), target.log_softmax(-1)
        parts = tl.exp()*(tl-sl)
        unclipped_mass = tl.exp()*(parts <= .05)
        explicit = (sl.exp()*unclipped_mass.sum(-1, keepdim=True)-unclipped_mass)*valid/mask.sum()
        derivative_errors.append(float((explicit-gradient).abs().max()))
        clip_parts.append(parts.detach())
        fixed_gradients.append(gradient.detach())
        fixed_losses.append(float(scalar.detach()))
    crossings = ((clip_parts[0] > .05) != (clip_parts[1] > .05)) & valid
    clipping_crossings = int(crossings.sum())
    fixed_delta = fixed_gradients[1]-fixed_gradients[0]
    fixed_comparison = dict(loss_absolute_difference=abs(fixed_losses[0]-fixed_losses[1]),
        gradient_relative_l2=float(fixed_delta.norm()/fixed_gradients[0].norm().clamp_min(1e-30)),
        gradient_max_absolute_difference=float(fixed_delta.abs().max()),
        explicit_derivative_max_errors=derivative_errors, clipping_boundary_crossings=clipping_crossings,
        clipped_components_cpu=int(((clip_parts[0] > .05)&valid).sum()),
        clipped_components_gpu=int(((clip_parts[1] > .05)&valid).sum()),
        source_dtype=str(fixed_source.dtype), vocab_size=fixed_source.shape[-1],
        tolerances=dict(gradient_relative_l2=.001, gradient_max_abs=1e-6, explicit_derivative_max_abs=1e-7),
        scope='Identical cached actual student logits; FP32 full-vocab FKL leaf-logit derivatives isolate teacher target change from transformer BF16 backward')
    fixed_comparison['passed'] = (fixed_comparison['gradient_relative_l2'] <= .001
        and fixed_comparison['gradient_max_absolute_difference'] <= 1e-6
        and max(derivative_errors) <= 1e-7)
    cp, gp = cpu_logits.softmax(-1), gpu_logits.softmax(-1)
    probability_difference = float((cp-gp).abs().max())
    kl = (cp*(cpu_logits.log_softmax(-1)-gpu_logits.log_softmax(-1))).sum(-1)
    loss_difference = abs(cpu_loss_value-gpu_loss_value)
    relative_gradient_l2 = migration_gradients['relative_l2']
    max_grad_difference = migration_gradients['max_absolute_difference']
    result = dict(allocation=allocation, job_id=allocation['job_id'], hostname=allocation['hostname'], model=args.model,
        teacher_cpu_runtime=teacher_cpu_runtime, teacher_gpu_runtime=teacher_gpu_runtime,
        teacher_after_return_runtime=teacher_runtime_audit(teacher, 'cpu'), student_parameter_dtypes=student_dtypes,
        teacher_dtype='float32', teacher_attention='eager', tf32=False,
        flash_attention_deterministic_env=os.environ.get('FLASH_ATTENTION_DETERMINISTIC'),
        flash_deterministic_requested_diagnostic_only=args.flash_deterministic,
        teacher_cpu_gpu_logits_max_abs=float((cpu_logits-gpu_logits).abs().max()),
        teacher_probability_max_abs=probability_difference, teacher_kl_max=float(kl.max()),
        cpu_teacher_loss=cpu_loss_value, gpu_teacher_loss=gpu_loss_value,
        loss_absolute_difference=loss_difference, all_student_gradient_parameters_compared=migration_gradients['parameter_elements_compared'],
        student_gradient_relative_l2=relative_gradient_l2, student_gradient_max_abs=max_grad_difference,
        tolerances=dict(probability_max_abs=1e-5, loss_absolute=1e-5, gradient_relative_l2=.01, gradient_max_abs=5e-4),
        bf16_student_migration_gradient_details=migration_gradients,
        identical_cpu_teacher_bf16_backward_repeat_controls=cpu_repeat_controls,
        identical_gpu_teacher_bf16_backward_repeat_controls=gpu_repeat_controls,
        cached_student_logit_fkl_gradient_comparison=fixed_comparison,
        scope='Actual 3B short-context full-vocabulary clipped forward KL and every student parameter gradient; repeat controls and fixed-logit derivatives diagnose numerical sensitivity, no optimizer step')
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6)
    result['teacher_audit'] = teacher.audit_metrics(optimizer)
    if args.factor_bf16_backward:
        try:
            def tensor_difference(a, b):
                difference = b.float()-a.float()
                return dict(relative_l2=float(difference.norm()/a.float().norm().clamp_min(1e-30)),
                    max_absolute_difference=float(difference.abs().max()),
                    changed_elements=int((a!=b).sum()), total_elements=a.numel(),
                    reference_l2=float(a.float().norm()))

            quantized_cpu = fixed_gradients[0].cpu().to(torch.bfloat16)
            quantized_gpu = fixed_gradients[1].cpu().to(torch.bfloat16)
            quantized_difference = tensor_difference(quantized_cpu, quantized_gpu)
            same_nonzero_sign = (quantized_cpu.sign()==quantized_gpu.sign()) & (quantized_cpu!=0) & (quantized_gpu!=0)
            ulp_distance = (quantized_cpu.view(torch.int16).int()-quantized_gpu.view(torch.int16).int()).abs()
            quantized_difference.update(same_nonzero_sign_changed_elements=int(((quantized_cpu!=quantized_gpu)&same_nonzero_sign).sum()),
                same_nonzero_sign_max_ulp_distance=int(ulp_distance[same_nonzero_sign].max()) if same_nonzero_sign.any() else 0,
                cpu_rounding_error=tensor_difference(fixed_gradients[0].cpu(), quantized_cpu.float()),
                gpu_rounding_error=tensor_difference(fixed_gradients[1].cpu(), quantized_gpu.float()),
                scope='Cached FP32 leaf-logit gradients rounded to BF16, before deep transformer backward')
            result['bf16_backward_factorization'] = dict(passed=False, stage='cached_bf16_rounding',
                cached_leaf_gradient_bf16_rounding=quantized_difference)
            captures = {}
            force_gradient = None

            def projection_forward_hook(module, inputs, output):
                captures['projection_logits'] = output.detach().cpu().clone()
                def gradient_hook(gradient):
                    captures['projection_gradient'] = gradient.detach().cpu().clone()
                    if force_gradient is not None:
                        if gradient.shape != force_gradient.shape:
                            raise AssertionError('Controlled projection gradient shape changed')
                        return force_gradient.to(device=gradient.device, dtype=gradient.dtype)
                output.register_hook(gradient_hook)

            def student_forward_hook(module, inputs, output):
                captures['hf_logits'] = output.logits.detach().cpu().clone()
                output.logits.register_hook(lambda gradient: captures.__setitem__('hf_logits_gradient', gradient.detach().cpu().clone()))

            projection_handle = student.get_output_embeddings().register_forward_hook(projection_forward_hook)
            student_handle = student.register_forward_hook(student_forward_hook)
            try:
                student.zero_grad(set_to_none=True)
                captured_cpu_loss = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
                captured_cpu_loss.backward()
                cpu_capture = dict(captures)
                captures.clear()
                student.zero_grad(set_to_none=True)
                with teacher.on_device():
                    captured_gpu_loss = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
                    captured_gpu_loss.backward()
                    gpu_capture = dict(captures)
                captures.clear()
                actual_projection = tensor_difference(cpu_capture['projection_gradient'], gpu_capture['projection_gradient'])
                result['bf16_backward_factorization'].update(stage='captured_projection_gradients',
                    actual_projection_gradient_difference=actual_projection,
                    cpu_hf_logits_dtype=str(cpu_capture['hf_logits'].dtype),
                    gpu_hf_logits_dtype=str(gpu_capture['hf_logits'].dtype))
                casting_checks = []
                analytical_checks = []
                for index, capture in enumerate((cpu_capture, gpu_capture)):
                    projected_expected = capture['hf_logits_gradient'].to(capture['projection_gradient'].dtype)
                    cast_exact = torch.equal(projected_expected, capture['projection_gradient'])
                    if not cast_exact:
                        raise AssertionError('Actual projection gradient differs from HF-logits gradient cast to projection dtype')
                    casting_checks.append(dict(exact=True, projection_gradient_dtype=str(capture['projection_gradient'].dtype),
                        hf_logits_gradient_dtype=str(capture['hf_logits_gradient'].dtype)))
                    actual_hf_dtype = capture['hf_logits_gradient'].dtype
                    actual_leaf_gradient_native = capture['hf_logits_gradient'][:, -response.shape[1]-1:-1]
                    # Pinned Qwen2 4.46 labels=None keeps BF16 logits; other
                    # versions may cast them to FP32. Loss backward reaches
                    # this actual output dtype before the projection hook.
                    expected_leaf_gradient_native = fixed_gradients[index].cpu().to(actual_hf_dtype)
                    analytical_difference = tensor_difference(expected_leaf_gradient_native, actual_leaf_gradient_native)
                    analytical_difference.update(actual_hf_logits_gradient_dtype=str(actual_hf_dtype),
                        expected_analytic_gradient_dtype_after_cast=str(expected_leaf_gradient_native.dtype),
                        fp32_leaf_gradient_before_cast_dtype=str(fixed_gradients[index].dtype),
                        exact_after_matching_actual_output_dtype=torch.equal(expected_leaf_gradient_native, actual_leaf_gradient_native),
                        comparison_absolute_tolerance=1e-7)
                    analytical_difference['passed_absolute_1e_minus_7'] = analytical_difference['max_absolute_difference'] <= 1e-7
                    analytical_checks.append(analytical_difference)
                result['bf16_backward_factorization'].update(stage='captured_dtype_matched_analytic_checks',
                    actual_projection_cast_checks=casting_checks,
                    actual_hf_logits_gradient_vs_cached_analytic=analytical_checks)
                if not torch.equal(cpu_capture['projection_logits'], gpu_capture['projection_logits']):
                    raise AssertionError('Student forward projection activations differ across teacher placements')
                force_gradient = cpu_capture['projection_gradient']
                student.zero_grad(set_to_none=True)
                forced_cpu_loss = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
                forced_cpu_loss.backward()
                forced_reference = snapshot_grads(student)
                student.zero_grad(set_to_none=True)
                with teacher.on_device():
                    forced_gpu_loss = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
                    forced_gpu_loss.backward()
                    forced_details = compare_grads(student, forced_reference)
                exact_control = forced_details['max_absolute_difference']==0. and forced_details['relative_l2']==0.
                if not exact_control:
                    raise AssertionError('Forced identical projection gradient did not produce identical all-parameter gradients')
                result['bf16_backward_factorization'] = dict(passed=all(r['passed_absolute_1e_minus_7'] for r in analytical_checks),
                    stage='completed_forced_gradient_control',
                    cached_leaf_gradient_bf16_rounding=quantized_difference,
                    actual_projection_gradient_difference=actual_projection,
                    actual_projection_cast_checks=casting_checks,
                    actual_hf_logits_gradient_vs_cached_analytic=analytical_checks,
                    student_projection_forward_identical=True,
                    forced_identical_projection_gradient_all_parameter_comparison=forced_details,
                    forced_control_gradient_source='CPU teacher actual BF16 projection-output gradient, reused byte-identically for both teacher placements',
                    production_gradient_or_loss_modified=False,
                    scope='Diagnostic tensor hook replaces projection-output gradient only during two isolated test backwards. This isolates downstream student backward dependence on its input; no optimizer/training mutation or BF16 parity assertion')
                del forced_reference, cpu_capture, gpu_capture
                if not result['bf16_backward_factorization']['passed']:
                    raise AssertionError('Actual HF-logits gradient differs from dtype-matched analytic reference; forced-gradient control evidence retained')
            finally:
                projection_handle.remove()
                student_handle.remove()
                student.zero_grad(set_to_none=True)
                captures.clear()
                force_gradient = None
                captured_cpu_loss = captured_gpu_loss = forced_cpu_loss = forced_gpu_loss = None
        except BaseException as exc:
            import traceback
            result.setdefault('bf16_backward_factorization', {}).update(passed=False, error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc())
    if args.fp32_reference:
        try:
            # Release BF16 gradient snapshots before storing the FP32 reference.
            del student, optimizer, reference_grads, gpu_reference_grads, loss_cpu, loss_cpu_repeat, loss_gpu, loss_gpu_repeat
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            fp32_student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                        attn_implementation='eager').cuda().eval().requires_grad_(True)
            fp32_dtypes = sorted({str(v.dtype) for v in fp32_student.parameters()})
            if fp32_dtypes != ['torch.float32'] or fp32_student.config._attn_implementation != 'eager':
                raise AssertionError('FP32 student diagnostic execution invariant failed')
            first = cached_target_loss(fp32_student, cpu_logits)
            first.backward()
            fp32_reference = snapshot_grads(fp32_student)
            fp32_student.zero_grad(set_to_none=True)
            repeat = cached_target_loss(fp32_student, cpu_logits)
            repeat.backward()
            repeat_details = compare_grads(fp32_student, fp32_reference)
            fp32_student.zero_grad(set_to_none=True)
            second = cached_target_loss(fp32_student, gpu_logits)
            second.backward()
            fp32_details = compare_grads(fp32_student, fp32_reference)
            fp32_details.update(loss_absolute_difference=abs(float(first.detach())-float(second.detach())),
                parameter_dtypes=fp32_dtypes, attention='eager', repeated_identical_target_control=repeat_details,
                tolerances=dict(relative_l2=.001, gradient_max_abs=1e-4, loss_absolute=1e-5),
                scope='Actual full FP32/eager student with cached CPU/GPU FP32 teacher targets, autocast and TF32 disabled; all student parameter gradients')
            fp32_details['passed'] = (fp32_details['relative_l2'] <= .001
                and fp32_details['max_absolute_difference'] <= 1e-4
                and fp32_details['loss_absolute_difference'] <= 1e-5)
            result['fp32_student_reference'] = fp32_details
        except BaseException as exc:
            import traceback
            result['fp32_student_reference'] = dict(passed=False, error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.perf_counter()-started
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['passed'] = bool(probability_difference <= 1e-5 and loss_difference <= 1e-5
        and relative_gradient_l2 <= .01 and max_grad_difference <= 5e-4
        and fixed_comparison['passed']
        and result.get('fp32_student_reference', {'passed':True})['passed']
        and result.get('bf16_backward_factorization', {'passed':True})['passed'])
    result['diagnostic_controls_do_not_relax_original_gate'] = True
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise AssertionError('CPU/GPU FP32 teacher numerical equivalence gate failed')


if __name__ == '__main__':
    main()
