"""All-token, fixed-weight diagnostics of the actual frozen OPSD objective."""
import math
from pathlib import Path

if __package__:
    from .common import append_json, file_sha
else:
    from common import append_json, file_sha


class FullVocabularyBudget:
    """Per-worker partition of a declared global cap; defaults to no full dumps."""
    def __init__(self, cap_bytes, workers, allowed_cases, versions):
        self.cap = cap_bytes // workers
        self.used = 0
        self.allowed_cases = set(allowed_cases)
        self.versions = set(versions)

    def create(self, output, case_id, version, shape, teacher_present, response_origin="preserved_student"):
        if not response_origin.startswith("preserved_"):
            return None, dict(status="not_requested_generated_response_scope", response_origin=response_origin,
                              global_cap_partition_bytes=self.cap)
        if case_id not in self.allowed_cases or version not in self.versions:
            return None, dict(status="not_requested", global_cap_partition_bytes=self.cap)
        import numpy as np
        distributions = ["student", "base_ordinary"] + (["teacher_privileged"] if teacher_present else [])
        names = [(name, quantity) for name in distributions for quantity in ("probabilities", "logprobs")]
        estimated = shape[0] * shape[1] * 4 * len(names) + 4096 * len(names)
        if self.used + estimated > self.cap:
            return None, dict(status="cap_skip", bytes_requested_with_header_allowance=estimated,
                              worker_bytes_reserved=self.used, worker_cap_bytes=self.cap)
        self.used += estimated
        output = Path(output)
        arrays, files = {}, {}
        for name, quantity in names:
            key = name + "__" + quantity
            path = output / f"full-vocabulary-{name}-{quantity}.npy"
            arrays[key] = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)
            files[key] = dict(path=str(path), dtype="float32", shape=list(shape), distribution=name, quantity=quantity,
                              precision="BF16 head logits -> FP32 log_softmax and probabilities; logprobs retained through probability underflow")
        return arrays, dict(status="allocated", files=files, bytes_reserved_with_header_allowance=estimated,
                            worker_cap_bytes=self.cap, worker_bytes_reserved=self.used)


def probe(model, tokenizer, core, opsd, response, policy_spans, question, reference,
          version, case_id, output, budget, topk=32, chunk=32, tolerance_atol=2e-5, tolerance_rtol=2e-4,
          response_origin="preserved_student"):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoding = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    response_ids, valid = core.response_encoding(tokenizer, response, policy_spans)
    if response_ids != encoding["input_ids"]:
        raise AssertionError("Frozen response encoding token IDs diverged")
    prefixes = [tokenizer.apply_chat_template(core.prompt(question, ref), tokenize=True, add_generation_prompt=True)
                for ref in (None, reference)]
    teacher_present = reference is not None
    token_count, valid_count = len(response_ids), sum(valid)
    required_prefixes = prefixes if teacher_present else prefixes[:1]
    summary = dict(kind="fixed_response_teacher_forcing", actual_autoregressive_generation=False,
                   response_origin=response_origin,
                   original_training_objective=response_origin == "preserved_student" and teacher_present,
                   version=version, case_id=case_id, response_tokens=token_count, valid_policy_tokens=valid_count,
                   ordinary_prefix_tokens=len(prefixes[0]), privileged_prefix_tokens=len(prefixes[1]) if teacher_present else None,
                   reference_status="successful_original_reference" if teacher_present else "not_applicable_no_successful_reference",
                   precision="BF16 fixed base/adapter forward and output head; FP32 log_softmax/distribution sums",
                   implicit_temperature=1.0, contribution_upper_clip=10.0, topk=topk,
                   topk_limit="Topk, target probabilities and token loss do not reconstruct the full vocabulary distribution",
                   token_reduction="valid policy token mean per trajectory", max_context=8192)
    if not token_count or not valid_count or any(len(prefix) + token_count > 8192 for prefix in required_prefixes):
        summary.update(status="skipped", reason="empty_response_or_mask_or_joint_context_limit_no_independent_truncation")
        return summary
    device = next(model.parameters()).device
    base = model.get_base_model()
    transformer, head = base.base_model, base.get_output_embeddings()
    tensor = lambda prefix: torch.tensor([prefix + response_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        base_hidden = transformer(input_ids=tensor(prefixes[0]), use_cache=False).last_hidden_state[0]
        base_hidden = base_hidden[len(prefixes[0])-1:len(prefixes[0])+token_count-1].detach()
        teacher_hidden = None
        if teacher_present:
            teacher_hidden = transformer(input_ids=tensor(prefixes[1]), use_cache=False).last_hidden_state[0]
            teacher_hidden = teacher_hidden[len(prefixes[1])-1:len(prefixes[1])+token_count-1].detach()
    model.train()  # Match frozen pair_loss mode; LoRA dropout is exactly zero.
    with torch.no_grad():
        student_hidden = transformer(input_ids=tensor(prefixes[0]), use_cache=False).last_hidden_state[0]
        student_hidden = student_hidden[len(prefixes[0])-1:len(prefixes[0])+token_count-1].detach()
    vocab = int(head.weight.shape[0])
    arrays, full = budget.create(output, case_id, version, (token_count, vocab), teacher_present, response_origin)
    info_spans = core.information_spans(response)
    token_path = output / "tokens.jsonl"
    policy_loss = 0.0
    native_policy_sum = student_hidden.sum() * 0.0
    clipped_policy_contributions = 0
    all_finite = True
    records = [None] * token_count
    # Preserve the native head GEMM row shapes/order on policy positions. A
    # contiguous all-token partition could change BF16 arithmetic at its tail.
    valid_indices = [index for index, selected in enumerate(valid) if selected]
    invalid_indices = [index for index, selected in enumerate(valid) if not selected]
    partitions = [(True, valid_indices), (False, invalid_indices)]
    with torch.no_grad():
        for is_policy, partition in partitions:
            for start in range(0, len(partition), chunk):
                indices = partition[start:start + chunk]
                device_indices = torch.tensor(indices, dtype=torch.long, device=device)
                hidden = dict(student=student_hidden[device_indices], base_ordinary=base_hidden[device_indices])
                if teacher_present:
                    hidden["teacher_privileged"] = teacher_hidden[device_indices]
                logs = {name: torch.log_softmax(head(value).float(), dim=-1) for name, value in hidden.items()}
                probabilities = {name: value.exp() for name, value in logs.items()}
                entropies = {name: -(probabilities[name] * value).sum(dim=-1) for name, value in logs.items()}
                tops = {name: torch.topk(value, k=min(topk, vocab), dim=-1) for name, value in logs.items()}
                kl_base = (probabilities["base_ordinary"] * (logs["base_ordinary"] - logs["student"])).sum(dim=-1)
                if teacher_present:
                    contributions = probabilities["teacher_privileged"] * (logs["teacher_privileged"] - logs["student"])
                    clipped = contributions.clamp(max=10.0)
                    kl = contributions.sum(dim=-1)
                    clipped_kl = clipped.sum(dim=-1)
                    saturated = contributions > 10.0
                    clip_counts = saturated.sum(dim=-1)
                    clip_teacher_mass = (probabilities["teacher_privileged"] * saturated).sum(dim=-1)
                    if is_policy:
                        native_policy_sum = native_policy_sum + clipped_kl.sum()
                if arrays is not None:
                    for key, array in arrays.items():
                        name, quantity = key.split("__")
                        values = probabilities[name] if quantity == "probabilities" else logs[name]
                        array[indices] = values.detach().cpu().numpy()
                # Transfer once per chunk, not once per token/statistic.
                targets = torch.tensor([response_ids[index] for index in indices], device=device).unsqueeze(-1)
                cpu = {name: dict(target=values.gather(-1, targets).squeeze(-1).cpu().tolist(),
                                  entropy=entropies[name].cpu().tolist(),
                                  top_ids=tops[name].indices.cpu().tolist(),
                                  top_logs=tops[name].values.cpu().tolist()) for name, values in logs.items()}
                base_values = kl_base.cpu().tolist()
                if teacher_present:
                    kl_values, clipped_values = kl.cpu().tolist(), clipped_kl.cpu().tolist()
                    count_values, mass_values = clip_counts.cpu().tolist(), clip_teacher_mass.cpu().tolist()
                for offset, index in enumerate(indices):
                    token = response_ids[index]
                    a, b = encoding["offset_mapping"][index]
                    distribution = {}
                    for name, values in cpu.items():
                        ids, log_values = values["top_ids"][offset], values["top_logs"][offset]
                        target_logprob = values["target"][offset]
                        distribution[name] = dict(target_logprob=target_logprob, target_probability=math.exp(target_logprob),
                                                  entropy=values["entropy"][offset],
                                                  topk=[dict(token_id=identifier, token=tokenizer.convert_ids_to_tokens(identifier),
                                                             logprob=logprob, probability=math.exp(logprob))
                                                        for identifier, logprob in zip(ids, log_values)])
                    value = clipped_values[offset] if teacher_present else None
                    actual_contribution = value / valid_count if teacher_present and valid[index] else 0.0 if teacher_present else None
                    if actual_contribution is not None:
                        policy_loss += actual_contribution
                        if valid[index]:
                            clipped_policy_contributions += count_values[offset]
                    record = dict(token_index=index, token_id=token, tokenizer_token=tokenizer.convert_ids_to_tokens(token),
                                  decoded_single_token=tokenizer.decode([token]), character_offset=[a, b], text=response[a:b],
                                  policy_mask=bool(valid[index]), information_mask=any(a < y and b > x for x, y in info_spans),
                                  zero_width_offset=not b > a, distributions=distribution,
                                  teacher_student_unclipped_kl=kl_values[offset] if teacher_present else None,
                                  teacher_student_clipped_kl=value,
                                  clipped_vocabulary_contributions=count_values[offset] if teacher_present else None,
                                  clipped_teacher_probability_mass=mass_values[offset] if teacher_present else None,
                                  base_ordinary_to_student_kl=base_values[offset],
                                  actual_masked_trajectory_loss_contribution=actual_contribution,
                                  causal_alignment="Logits at prefix_length-1+token_index predict this fixed response token")
                    numeric = [record["base_ordinary_to_student_kl"]]
                    for dist in distribution.values():
                        numeric += [dist["target_logprob"], dist["target_probability"], dist["entropy"]]
                        numeric += [entry["logprob"] for entry in dist["topk"]]
                    if teacher_present:
                        numeric += [record["teacher_student_unclipped_kl"], value, actual_contribution]
                    all_finite = all_finite and all(math.isfinite(number) for number in numeric)
                    if not all_finite:
                        raise FloatingPointError("Nonfinite all-token diagnostics")
                    records[index] = record
                del hidden, logs, probabilities, entropies, tops, cpu
    with token_path.open("x") as stream:
        for record in records:
            if record is None:
                raise AssertionError("Missing all-token diagnostic row")
            append_json(stream, record)
    del records
    native_policy_mean = float(native_policy_sum / valid_count) if teacher_present else None
    del native_policy_sum
    del base_hidden, teacher_hidden, student_hidden
    if arrays is not None:
        for name, array in arrays.items():
            array.flush()
            path = Path(full["files"][name]["path"])
            full["files"][name].update(bytes=path.stat().st_size, sha256=file_sha(path))
        arrays.clear()
        full["status"] = "complete"
    summary.update(status="complete", all_token_metrics_finite=all_finite,
                   tokens_file=dict(path=str(token_path), bytes=token_path.stat().st_size, sha256=file_sha(token_path), rows=token_count),
                   full_vocabulary_dump=full)
    if teacher_present:
        encoded = opsd.encoded_pair(tokenizer, dict(question=question, reference=reference, student=response,
                                                   student_policy_spans=policy_spans), 8192)
        if encoded is None:
            raise AssertionError("Probe and frozen encoded_pair skip criteria disagree")
        with torch.no_grad():
            frozen_loss = float(opsd.pair_loss(model, encoded, 10.0, 32).detach().cpu())
        matches = math.isfinite(frozen_loss) and math.isclose(native_policy_mean, frozen_loss, rel_tol=tolerance_rtol, abs_tol=tolerance_atol)
        contribution_matches = math.isclose(policy_loss, native_policy_mean, rel_tol=tolerance_rtol, abs_tol=tolerance_atol)
        summary.update(trajectory_mean=native_policy_mean, per_token_contribution_sum_float64=policy_loss, frozen_pair_loss=frozen_loss,
                       frozen_pair_loss_check=dict(passed=matches, mode="no_grad fixed-weight scalar objective; no backward/optimizer",
                                                   head_partition="native valid indices chunk32 first; unmasked tokens separate chunk32",
                                                   per_token_contribution_sum_matches=contribution_matches,
                                                   atol=tolerance_atol, rtol=tolerance_rtol,
                                                   absolute_error=abs(native_policy_mean - frozen_loss)),
                       clipped_policy_vocabulary_contributions=clipped_policy_contributions)
        if not matches or not contribution_matches:
            raise AssertionError("All-token trajectory mean does not match actual frozen pair_loss: " + repr(summary))
    else:
        summary.update(trajectory_mean=None, frozen_pair_loss=None,
                       frozen_pair_loss_check=dict(status="not_applicable_no_successful_reference"))
    model.eval()
    return summary
