#!/usr/bin/env python3
"""Read-only controlled ordinary-HF probe, never a sampling admission receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = 'aec5e057093b4c491bfb0a47abd7605d98981b56'
RAW_SHA = '2544e34b622b59e5fadd802cb4d950e7d122073dac2fd1e2c51ff2d525e6c77c'
KEYS = ('responses', 'input_ids', 'attention_mask', 'position_ids')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_points(tensors, count=20):
    import torch
    mask = tensors['sampling_mask'].bool()
    old, native = tensors['old_log_probs'], tensors['sampling_log_probs']
    require(old.shape == native.shape == mask.shape, 'LP/mask shape')
    require(bool(torch.isfinite(old[mask]).all() and torch.isfinite(native[mask]).all()), 'nonfinite selected LP')
    coords = mask.nonzero().tolist()
    differences = (old.double() - native.double()).abs()[mask].tolist()
    ranked = sorted(zip(differences, coords), key=lambda item: (-item[0], item[1][0], item[1][1]))[:count]
    return [dict(row=row, position=position, old=float(old[row, position]), native=float(native[row, position]),
                 abs_saved_delta=value, token=int(tensors['responses'][row, position]))
            for value, (row, position) in ranked]


def locate_turn(turns, position):
    offset = 0
    for ordinal, turn in enumerate(turns):
        n = turn['kept_token_count']
        if offset <= position < offset + n:
            return ordinal, position - offset, turn
        offset += n + len(turn['environment_token_ids'])
    raise ValueError('selected response position is not a raw model token')


def groups(points, total):
    rows = sorted(set(p['row'] for p in points))
    result = []
    for row in rows:
        adjacent = list(range((row // 4) * 4, min((row // 4) * 4 + 4, total)))
        # Keep the tested row in exactly its adjacent-batch slot, changing only neighbors.
        slot = adjacent.index(row)
        changed = []
        for offset in (total // 2, total // 3, total // 4):
            candidate = (row + offset) % total
            if candidate != row and candidate not in changed:
                changed.append(candidate)
        for candidate in range(total):
            if len(changed) >= len(adjacent) - 1:
                break
            if candidate != row and candidate not in changed:
                changed.append(candidate)
        changed = changed[:len(adjacent) - 1]
        changed.insert(slot, row)
        result.extend([('RMPAD_MICRO1', [row]), ('NEW_PACKED_PROBE_MICRO4', adjacent),
                       ('NEW_PACKED_PROBE_CHANGED_NEIGHBORS', changed), ('PADDED_MICRO1', [row]),
                       ('PADDED_NEW_MICRO4', adjacent)])
    unique = []
    for name, members in result:
        if (name, members) not in unique:
            unique.append((name, members))
    return unique


def low_error_controls(tensors, journals):
    best = {}
    for row, turns in enumerate(journals):
        start = 0
        for ordinal, turn in enumerate(turns):
            n = turn['kept_token_count']
            if ordinal in (0, 1):
                differences = (tensors['old_log_probs'][row, start:start+n].double() -
                               tensors['sampling_log_probs'][row, start:start+n].double()).abs()
                offset = int(differences.argmin())
                position = start + offset
                value = float(differences[offset])
                candidate = (value, row, position)
                if ordinal not in best or candidate < best[ordinal]:
                    best[ordinal] = candidate
            start += n + len(turn['environment_token_ids'])
    require(set(best) == {0, 1}, 'need turn0 and turn1 low-error controls')
    return [dict(row=row, position=position, old=float(tensors['old_log_probs'][row, position]),
                 native=float(tensors['sampling_log_probs'][row, position]), abs_saved_delta=value,
                 token=int(tensors['responses'][row, position]), point_role='LOW_ERROR_TURN_' + str(ordinal))
            for ordinal, (value, row, position) in sorted(best.items())]


def turn_batch(turn, through_offset=None):
    import torch
    generated = list(turn['raw_generated_token_ids'][:turn['kept_token_count']])
    if through_offset is not None:
        require(type(through_offset) is int and 0 <= through_offset < len(generated), 'truncated offset')
        generated = generated[:through_offset + 1]
    request = list(turn['actual_request_token_ids'])
    require(bool(request) and bool(generated), 'empty turn')
    ids = torch.tensor([request + generated], dtype=torch.long)
    return dict(input_ids=ids, responses=torch.tensor([generated], dtype=torch.long),
                attention_mask=torch.ones_like(ids), position_ids=torch.arange(ids.shape[1]).unsqueeze(0))


def future_token_control(batch, slot, position, vocab_size):
    """Same shape/slot/masks/positions: alter only tokens strictly after target."""
    result = {k:v.clone() for k,v in batch.items()}
    response_width = batch['responses'].shape[1]
    prompt_width = batch['input_ids'].shape[1] - response_width
    response_attention = batch['attention_mask'][slot, prompt_width:]
    changed = []
    for index in range(position + 1, response_width):
        if int(response_attention[index]) == 1:
            token = (int(batch['responses'][slot, index]) + 1) % vocab_size
            result['responses'][slot, index] = token
            result['input_ids'][slot, prompt_width + index] = token
            changed.append(index)
    return result, changed


def parameter_digest(model):
    """Hash every loaded parameter's name, dtype, shape and complete bytes."""
    import torch
    h = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        h.update(json.dumps([name, str(parameter.dtype), list(parameter.shape)]).encode())
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(memoryview(raw).cast('B'))
        count += parameter.numel()
    return dict(sha256=h.hexdigest(), parameter_elements=count)


def check_approval(path, args):
    approval = json.loads(Path(path).read_text())
    require(approval.get('protocol') == 'BAPO_ZERO_ADAM_DIAGNOSTIC_V1', 'diagnostic approval protocol')
    require(approval.get('status') == 'ALLOW_ZERO_ADAM_PROBABILITY_REPLAY', 'diagnostic approval status')
    require(approval.get('source_commit') == SOURCE and approval.get('raw_sha256') == RAW_SHA, 'approval source/raw binding')
    require(approval.get('hf_probe_sha256') == digest(__file__), 'approval probe byte binding')
    require(approval.get('native_acceptance') is False and approval.get('production_release') is False and
            all(type(approval.get(k)) is int and approval[k] == 0 for k in
                ('optimizer_writes', 'formal_outer_updates', 'scheduler_updates')), 'diagnostic-only approval')
    for key, actual in (('repo', args.repo), ('raw_batch', args.raw_batch), ('model', args.model)):
        require(Path(approval[key]).resolve() == Path(actual).resolve(), 'approval path mismatch: ' + key)
    require(approval.get('allocation_gpu_count') == 8 and approval.get('probe_gpu_count') == 1, 'GPU scope')
    require(os.environ.get('FRESH200_ZERO_ADAM_GUARD_READY') == '1', 'guard readiness missing')
    receipt_path = Path(approval['job_receipt_path']).resolve()
    require(os.environ.get('BAPO_DIAGNOSTIC_RECEIPT') == str(receipt_path), 'receipt path environment mismatch')
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('phase') == 'hf-zero-adam-replay' and receipt.get('source') == SOURCE and
            receipt.get('nonce') == approval.get('nonce'), 'receipt identity')
    job_id = receipt.get('job_id')
    require(type(job_id) is int and job_id > 0 and os.environ.get('SLURM_JOB_ID') == str(job_id), 'receipt Slurm job mismatch')
    require(receipt.get('comment') == 'BAPO_ZERO_ADAM:hf-zero-adam-replay:' + SOURCE + ':' + approval['nonce'], 'receipt comment')
    expected_run = str(Path(approval['root']) / 'runs' / ('hf-zero-adam-replay-' + approval['nonce'] + '-' + str(job_id)))
    require(os.environ.get('BAPO_RUN_ID') == expected_run, 'diagnostic run mismatch')
    require(Path(expected_run).resolve() in Path(args.output).resolve().parents, 'output outside own diagnostic run')
    return dict(path=str(Path(path).resolve()), sha256=digest(path), job_id=job_id,
                receipt_path=str(receipt_path), receipt_sha256=digest(receipt_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'raw-batch', 'model', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--approval')
    args = parser.parse_args()
    import torch
    repo = Path(args.repo).resolve()
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE, 'wrong frozen source')
    require(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(), 'dirty tracked source')
    require(digest(args.raw_batch) == RAW_SHA, 'raw batch SHA mismatch')
    # This is the explicitly trusted, immutable local diagnostic payload, not arbitrary downloaded pickle.
    payload = torch.load(args.raw_batch, map_location='cpu', weights_only=False)
    require(payload['source_commit'] == SOURCE and payload['attempt_id'] == 1 and payload['actual_outer'] == 1, 'raw identity')
    sys.path.insert(0, str(repo))
    from search_r1.baselines.fresh200_contract import critical_source_hashes, verify_official_model_assets
    from search_r1.llm_agent.sampling_tokens import audit_batch
    from verl import DataProto
    require(payload['source_sha256'] == critical_source_hashes(), 'raw/current critical source mismatch')
    tensors = payload['tensors']
    data = DataProto.from_dict(tensors=tensors, non_tensors=payload['non_tensor_batch'], meta_info=payload['meta_info'])
    audit = audit_batch(data)
    require(audit['probability_role'] == 'stochastic_sampling_policy', 'not training sampling')
    require(float(payload['meta_info']['temperature']) == 1., 'temperature is not one')
    official = verify_official_model_assets(args.model)
    points = selected_points(tensors)
    for point in points:
        point['point_role'] = 'TOP20_SAVED_ABS_DELTA'
    points += low_error_controls(tensors, payload['non_tensor_batch']['fresh200_turns'])
    cases = groups(points, len(data))
    report = dict(schema_version=1, status='CPU_PLAN_NOT_GPU_EVIDENCE', native_acceptance=False,
        source_commit=SOURCE, source_sha256=payload['source_sha256'], raw_sha256=RAW_SHA,
        probe_sha256=digest(__file__), official_model=official, context_audit=audit,
        original_run_root=payload['run_root'], policy_snapshot_id=audit['policy_snapshot_id'],
        scope='Controlled ordinary-HF BF16; not FSDP bit-exact replay or sampler admission',
        original_microgroup_membership='UNKNOWN: raw saved after balance, old HF computed before balance',
        points=points, cases=[dict(mode=n, rows=r) for n, r in cases], optimizer_steps=0)
    if args.execute_gpu:
        require(args.approval is not None, 'approval required')
        report['approval'] = check_approval(args.approval, args)
        require(sys.platform.startswith('linux') and torch.cuda.is_available() and torch.cuda.device_count() in (1, 8), 'require Linux and guarded GPU visibility')
        torch.cuda.set_device(0)
        report['gpu_scope'] = dict(reserved=8, compute=1, logical_compute_device=0, visible=torch.cuda.device_count())
        from transformers import AutoModelForCausalLM
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        import verl.utils.torch_functional as functional
        import inspect
        import transformers
        require(Path(inspect.getfile(DataParallelPPOActor)).resolve() == repo / 'verl/workers/actor/dp_actor.py', 'foreign actor module')
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        require(all(p.dtype == torch.bfloat16 and p.device.index == 0 for p in model.parameters()), 'model dtype/device mismatch')
        require(model.config._attn_implementation == 'flash_attention_2', 'attention backend mismatch')
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module, actor.actor_optimizer = model, None
        actor.ulysses_sequence_parallel_size, actor.use_ulysses_sp = 1, False
        # Eager entropy is the identical helper; avoids compilation in a forward-only diagnostic.
        actor.compute_entropy_from_logits = functional.entropy_from_logits
        report['runtime'] = dict(torch=torch.__version__, transformers=transformers.__version__,
            executable=sys.executable, prefix=sys.prefix, gpu=torch.cuda.get_device_name(0),
            actor_forward_sha256=hashlib.sha256(inspect.getsource(DataParallelPPOActor._forward_micro_batch).encode()).hexdigest(),
            model_forward_source=inspect.getsource(type(model).forward), entropy_execution='eager original helper',
            logprob_helper_source=inspect.getsource(functional.logprobs_from_logits),
            attention_backend=model.config._attn_implementation, cuda_version=torch.version.cuda,
            allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            weights='direct BF16 official load; production FSDP begins FP32 with BF16 mixed precision')
        before = parameter_digest(model)
        outputs = []
        def forward(batch, remove_padding):
            actor.use_remove_padding = remove_padding
            with torch.inference_mode():
                _, lp = actor._forward_micro_batch({k:v.cuda() for k,v in batch.items()}, temperature=1.)
            return lp.detach().cpu()
        for mode, rows in cases:
            lp = forward({k:tensors[k][rows] for k in KEYS}, not mode.startswith('PADDED'))
            for point in points:
                if point['row'] in rows:
                    slot = rows.index(point['row'])
                    if mode == 'NEW_PACKED_PROBE_CHANGED_NEIGHBORS' and slot != point['row'] % 4:
                        continue
                    value = float(lp[slot, point['position']])
                    outputs.append(dict(mode=mode, rows=rows, row=point['row'], position=point['position'],
                        tested_batch_slot=slot, hf=value, hf_minus_saved_old=value-point['old'], hf_minus_native=value-point['native']))
        for point in points:
            baseline_batch = {k:tensors[k][[point['row']]] for k in KEYS}
            changed_batch, changed = future_token_control(baseline_batch, 0, point['position'], model.config.vocab_size)
            for remove_padding in (False, True):
                baseline = float(forward(baseline_batch, remove_padding)[0, point['position']])
                mutated = float(forward(changed_batch, remove_padding)[0, point['position']]) if changed else baseline
                outputs.append(dict(mode='SAME_SHAPE_FUTURE_TOKEN_CONTROL_' + ('RMPAD' if remove_padding else 'PADDED'),
                    row=point['row'], position=point['position'], tested_batch_slot=0, changed_future_tokens=len(changed),
                    changed_positions_sha256=hashlib.sha256(json.dumps(changed).encode()).hexdigest(),
                    baseline_hf=baseline, future_changed_hf=mutated, difference=mutated-baseline,
                    comparison_available=bool(changed), shapes_masks_positions_preserved=True))
        turn_cache = {}
        for point in points:
            ordinal, offset, turn = locate_turn(payload['non_tensor_batch']['fresh200_turns'][point['row']], point['position'])
            key = (point['row'], ordinal)
            for remove_padding in (False, True):
                cache_key = key + (remove_padding,)
                if cache_key not in turn_cache:
                    turn_cache[cache_key] = forward(turn_batch(turn), remove_padding)[0]
                value = float(turn_cache[cache_key][offset])
                outputs.append(dict(mode='PER_TURN_' + ('RMPAD' if remove_padding else 'PADDED') + '_MICRO1_EXACT_RAW_REQUEST',
                    row=point['row'], position=point['position'], turn=ordinal, offset=offset,
                    request_sha256=hashlib.sha256(json.dumps(turn['actual_request_token_ids']).encode()).hexdigest(),
                    hf=value, hf_minus_saved_old=value-point['old'], hf_minus_native=value-point['native']))
                truncated = float(forward(turn_batch(turn, offset), remove_padding)[0, offset])
                outputs.append(dict(mode='TRUNCATED_CAUSAL_PREFIX_' + ('RMPAD' if remove_padding else 'PADDED'),
                    row=point['row'], position=point['position'], turn=ordinal, offset=offset,
                    generated_prefix_tokens=offset + 1, hf=truncated, hf_minus_full_turn=truncated-value,
                    hf_minus_saved_old=truncated-point['old'], hf_minus_native=truncated-point['native'],
                    interpretation='Suffix sensitivity observation; BF16 GEMM shape changes also differ, not alone proof of causal leakage'))
        after = parameter_digest(model)
        require(before == after and actor.actor_optimizer is None and not model.training and
                all(p.grad is None for p in model.parameters()), 'zero-Adam/model immutability failed')
        report.update(status='CONTROLLED_HF_GPU_DIAGNOSTIC_NOT_ADMISSION', parameter_before=before, parameter_after=after,
            results=outputs, wall_seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            zero_adam_proof=dict(optimizer_constructed=False, scheduler_constructed=False, parameter_bytes_unchanged=True,
                                 model_eval=True, gradients_absent=True))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(target.resolve()), sha256=digest(target), status=report['status'])))


if __name__ == '__main__':
    main()
