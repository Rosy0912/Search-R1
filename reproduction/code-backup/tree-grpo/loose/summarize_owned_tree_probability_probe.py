"""Aggregate saved JSON diagnostics, without loading any tensor/model assets.

This is an owner summary, not an independent acceptance receipt. Quantiles are
deliberately not combined: the saved per-call quantiles cannot yield a global
quantile. Micros remain separate because each traverses the same occurrences.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path


def load_capture(path):
    raw = path.read_bytes()
    capture = json.loads(raw)
    decoded = {}
    for name, item in capture["files"].items():
        value = gzip.decompress(base64.b64decode(item["gzip_base64"]))
        assert len(value) == item["bytes"]
        assert hashlib.sha256(value).hexdigest() == item["sha256"]
        decoded[name] = value
    return raw, capture, decoded


def combine(accumulator, metric):
    count = metric["physical_effective_token_occurrences"]
    mass = metric["tree_weight_denominator"]
    assert type(count) is int and count >= 0 and math.isfinite(mass) and mass >= 0
    accumulator["calls"] += 1
    accumulator["physical_effective_token_occurrences"] += count
    accumulator["tree_weight_denominator"] += mass
    accumulator["above_trial_001_count"] += metric["above_trial_001_count"]
    accumulator["tree_weighted_abs_numerator"] += metric["tree_weighted_abs_numerator"]
    if count:
        for key in ("max_abs", "mean_abs", "rms"):
            assert math.isfinite(metric[key]) and metric[key] >= 0
        accumulator["max_abs"] = max(accumulator["max_abs"], metric["max_abs"])
        accumulator["absolute_sum"] += count * metric["mean_abs"]
        accumulator["squared_sum"] += count * metric["rms"] ** 2


def aggregate(capture_path):
    raw, capture, files = load_capture(capture_path)
    groups, ranks, warmups, monitor = {}, {}, {}, {}
    native_selected_original_max = 0.0
    native_selected_original_all_exact = True
    dtype_pairs = set()
    excluded = 0
    comparisons = 0
    for name, value in files.items():
        if "/gpu-records/rank-" not in name or not name.endswith(".json"):
            continue
        row = json.loads(value)
        rank = int(name.split("/gpu-records/rank-", 1)[1].split("/", 1)[0])
        if name.endswith("/ready.json"):
            assert rank not in ranks
            ranks[rank] = row
            continue
        if name.endswith("/load-and-warmup.json"):
            warmups[rank] = row
            continue
        if not Path(name).name.startswith("micro"):
            continue
        if row["duplicate_excluded_from_occurrence_totals"]:
            excluded += 1
            continue
        comparisons += 1
        micro, kind = row["micro"], row["selection"]["kind"]
        same = row["same_logits"]
        delta = same["selected_native_vs_original_max_abs_delta"]
        if delta is not None:
            native_selected_original_max = max(native_selected_original_max, delta)
        native_selected_original_all_exact &= same["selected_native_exactly_matches_original"]
        dtype_pairs.add((same["scaled_logits_dtype"], same["original_actor_log_probs_dtype"]))
        for metric_name, metric in row["metrics"].items():
            if metric_name == "monitor_vs_sampled_ranges":
                key = (micro, kind)
                target = monitor.setdefault(key, {field: 0 for field in (
                    "effective_occurrences", "matched_occurrences", "unknown_occurrences",
                    "matched_raw_occurrences_denominator", "matched_contexts_with_duplicate_raw_occurrences")})
                for field in target:
                    target[field] += metric[field]
                # Endpoint/closest/worst distances keep their own definitions.
                for variant, statistics in metric["variants"].items():
                    for comparison_name, statistics_row in statistics.items():
                        full_name = "matched_monitor/" + variant + "/" + comparison_name
                        add_metric(groups, micro, kind, full_name, statistics_row)
            else:
                # Physical row identifiers vary; combine the same micro test.
                if metric_name.startswith("row-"):
                    metric_name = "original_actor_micro%d_vs_original_micro4" % micro
                add_metric(groups, micro, kind, metric_name, metric)
    final_groups = []
    for (micro, kind, name), metric in sorted(groups.items()):
        count = metric.pop("physical_effective_token_occurrences")
        mass = metric["tree_weight_denominator"]
        absolute_sum = metric.pop("absolute_sum")
        squared_sum = metric.pop("squared_sum")
        metric.update(micro=micro, kind=kind, metric=name,
                      physical_effective_token_occurrences=count,
                      mean_abs=absolute_sum / count if count else None,
                      rms=math.sqrt(squared_sum / count) if count else None,
                      tree_weighted_mean_abs=metric["tree_weighted_abs_numerator"] / mass if mass else None,
                      above_trial_001_fraction=metric["above_trial_001_count"] / count if count else None)
        final_groups.append(metric)
    run = "/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-probability-probe-" + capture["job_id"]
    terminal = {name: json.loads(files[run + "/" + name]) for name in (
        "result.json", "termination-cost.json") if run + "/" + name in files}
    ready_scope = {str(rank): {key: row.get(key) for key in (
        "status", "world_size", "optimizer_instantiated", "optimizer_committed", "backward_called",
        "scheduler_advanced", "original_forward_calls", "duplicate_alignment_calls",
        "full_state_unchanged_since_postwarm_baseline", "original_module_modes_restored",
        "own_wall_seconds", "own_cpu_seconds", "maximum_rss_kib",
        "torch_peak_allocated_bytes", "torch_peak_reserved_bytes")}
        for rank, row in sorted(ranks.items())}
    return dict(status="OWNER_SAVED_METADATA_AGGREGATION_NO_ACCEPTANCE", job_id=capture["job_id"],
                captured_epoch=capture["captured_epoch"], input_capture_sha256=hashlib.sha256(raw).hexdigest(),
                captured_file_count=len(files), ready_ranks=sorted(ranks), warmup_ranks=sorted(warmups),
                ready_scope=ready_scope, comparisons_excluding_alignment_duplicates=comparisons,
                excluded_alignment_comparison_calls=excluded, scaled_logits_and_actor_dtype_pairs=sorted(dtype_pairs),
                native_selected_vs_original_max_abs=native_selected_original_max if comparisons else None,
                native_selected_matches_original_every_saved_call=native_selected_original_all_exact if comparisons else None,
                groups=final_groups,
                monitor_occurrences_by_micro=[dict(micro=key[0], kind=key[1], **value)
                    for key, value in sorted(monitor.items())],
                commands=capture["commands"], terminal=terminal,
                tensor_files_loaded=False, global_quantiles_reconstructed=False,
                trial_001_is_diagnostic_only=True, probability_accepted=False, training_accepted=False,
                limits="Fixed selection, not all raw/final inputs. Each micro is a repeated comparison; never sum micros as new sampling. Saved JSON bytes are verified, PT contents are not reloaded by this summarizer.")


def add_metric(groups, micro, kind, name, metric):
    key = (micro, kind, name)
    target = groups.setdefault(key, dict(calls=0, physical_effective_token_occurrences=0,
        tree_weight_denominator=0.0, above_trial_001_count=0, tree_weighted_abs_numerator=0.0,
        max_abs=0.0, absolute_sum=0.0, squared_sum=0.0))
    combine(target, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.capture)
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: summary[key] for key in (
        "job_id", "captured_file_count", "ready_ranks", "warmup_ranks",
        "comparisons_excluding_alignment_duplicates", "native_selected_vs_original_max_abs")}, sort_keys=True))
