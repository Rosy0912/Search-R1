"""CPU postprocessing of accepted selection artifacts, never original raw data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

INPUTS = {"rollouts": ("fb4e94fb1efa600757a604db576807d9fd3dbd65a7c75bf36f11b0d2ba89d124", 845515),
          "pairs": ("a81a3ae61bb402f6b99dd35ae552c4bb76df8da391f3620c17fbd742ba2b1f40", 100974)}
SCIENCE = {"search_e1/core.py": "0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc",
           "upstream/Search-R1/search_r1/qa_protocol.py": "9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd"}
PRODUCER_SHA = "b076faa17ff701aff42b60d5111139f149a1044d8d31454e015ca21ca7cde0b4"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(path):
    path = Path(path).resolve(); item = path.stat()
    return dict(path=str(path), device=item.st_dev, inode=item.st_ino, bytes=item.st_size,
                mtime_ns=item.st_mtime_ns, ctime_ns=item.st_ctime_ns)


def require(value, code):
    if not value:
        raise ValueError(code)


def checked_lines(path, proofs):
    before = stat(path); digest = hashlib.sha256(); rows = 0
    with Path(path).open("rb") as stream:
        for rows, line in enumerate(stream, 1):
            require(line.endswith(b"\n") and len(line) <= 8 << 20, "partial_or_oversize_private_jsonl_line")
            digest.update(line)
            yield json.loads(line)
    require(stat(path) == before, "private_jsonl_changed_during_read")
    proofs[Path(path).name] = dict(before, sha256=digest.hexdigest(), rows=rows, fresh_cpu_hash=True)


def think_stack(text):
    depth = closes = nested = 0
    for tag in re.findall(r"</?think>", text):
        if tag == "<think>":
            nested += int(depth > 0); depth += 1
        elif depth:
            depth -= 1
        else:
            closes += 1
    return dict(balanced=not depth and not closes, unclosed=depth, unexpected_closing=closes, nested_opening=nested)


def role_metrics(row, metrics):
    spans = row["policy_spans"]; text = row["response"]; previous = 0; authored = []
    for start, end in spans:
        require(type(start) is int and type(end) is int and previous <= start <= end <= len(text), "selected_role_bad_authored_span")
        authored.append(text[start:end]); previous = end
    require(len(metrics["policy_structure"]) == len(authored), "selected_role_policy_metrics_count_mismatch")
    strict = metrics["strict_reward"]; failure = metrics["failure"]
    require(type(strict) is int and strict in (0, 1) and row["reward"] == strict and metrics["reward_matches"] is True, "selected_role_stored_reward_mismatch")
    require(failure == (None if strict else "final_invalid" if failure == "final_invalid" else "valid_gold_mismatch"), "selected_role_bad_failure_metric")
    nonfinal = [think_stack(item) for item in authored[:-1]]
    return dict(strict_em=int(strict), final_valid=failure != "final_invalid", valid_final_gold_match=int(strict),
                valid_final_gold_mismatch=int(failure == "valid_gold_mismatch"), final_invalid=int(failure == "final_invalid"),
                gold_match_when_final_invalid="not_defined_by_native_final_parser", search_calls=row["search_calls"],
                termination=row["termination"], response_char_count=len(text), token_count=row.get("token_count"),
                token_count_is_full_response_not_policy_tokens=True, nonfinal_policy_spans=len(nonfinal),
                nonfinal_think_literal_unbalanced=sum(not item["balanced"] for item in nonfinal),
                nonfinal_think_unclosed_openings=sum(item["unclosed"] for item in nonfinal),
                nonfinal_think_unexpected_closings=sum(item["unexpected_closing"] for item in nonfinal),
                nonfinal_think_nested_openings=sum(item["nested_opening"] for item in nonfinal),
                nonfinal_reserved_literal_unbalanced=sum(not item["balanced"] for item in metrics["policy_structure"][:-1]),
                whole_transcript_literal_tag_balanced=metrics["transcript_structure"]["balanced"],
                whole_transcript_nested_reserved=bool(metrics["transcript_structure"]["nested_reserved_opening"]))


def add_role(counter, measures):
    counter["rows"] += 1
    for key in ("strict_em", "final_valid", "valid_final_gold_match", "valid_final_gold_mismatch", "final_invalid",
                "nonfinal_policy_spans", "nonfinal_think_literal_unbalanced", "nonfinal_think_unclosed_openings",
                "nonfinal_think_unexpected_closings", "nonfinal_think_nested_openings", "nonfinal_reserved_literal_unbalanced",
                "whole_transcript_literal_tag_balanced", "whole_transcript_nested_reserved"):
        counter[key] += int(measures[key])
    counter["role_with_any_nonfinal_think_defect"] += int(measures["nonfinal_think_literal_unbalanced"] > 0)
    counter["strict_success_with_unbalanced_whole_transcript"] += int(measures["strict_em"] and not measures["whole_transcript_literal_tag_balanced"])
    counter["search_calls_sum"] += measures["search_calls"]
    counter["search_calls_count/" + str(measures["search_calls"])] += 1
    counter["termination/" + measures["termination"]] += 1
    counter["response_char_count_sum"] += measures["response_char_count"]
    if measures["token_count"] is None:
        counter["token_count_missing_rows"] += 1
    else:
        counter["token_count_present_rows"] += 1; counter["full_response_token_count_sum"] += measures["token_count"]


def finish_role(counter):
    value = dict(counter); rows = value["rows"]; tokens = value.get("token_count_present_rows", 0)
    value.update(denominator=rows, strict_success_count=value["strict_em"], strict_em=value["strict_em"] / rows, final_valid_rate=value["final_valid"] / rows,
                 search_calls_mean=value["search_calls_sum"] / rows, response_char_count_mean=value["response_char_count_sum"] / rows,
                 full_response_token_count_mean=value.get("full_response_token_count_sum", 0) / tokens if tokens else None,
                 token_count_mean_denominator=tokens, nonfinal_think_defect_rate_among_spans=value["nonfinal_think_literal_unbalanced"] / value["nonfinal_policy_spans"] if value["nonfinal_policy_spans"] else None)
    return value


def run(directory, output, production=True, expected_summary_sha256=None):
    directory = Path(directory).resolve(); output = Path(output).resolve(); started = time.time()
    summary_path = directory / "summary.json"; before_summary = stat(summary_path)
    original = json.loads(summary_path.read_text()); summary_sha = sha(summary_path)
    require(stat(summary_path) == before_summary, "summary_changed_during_read")
    if production:
        require(summary_sha == expected_summary_sha256, "expected_accepted_summary_sha_required_or_changed")
    require({str(Path(item).resolve()) for item in original["private_artifacts"]} ==
            {str(directory / name) for name in ("source.private.sqlite", "cases.private.jsonl", "pairs.private.jsonl", "issues.private.jsonl")}, "producer_private_artifact_paths_mismatch")
    require(original["schema"] == "opsd-selection-audit-v1" and original["accepted"] is True and original["status"] == "ACCEPTED", "producer_audit_not_accepted")
    require(not any(original["errors"].values()) and original["fatal"] is None, "producer_errors_present")
    questions = original["counters"]["audited_questions"]; eligible = original["counters"]["native_expected_pairs"]
    require(questions == original["counters"]["unique_rollout_questions"] and eligible == original["counters"]["pair_native_exact"], "producer_counts_disagree")
    identities = {}; proofs = {"summary.json": dict(before_summary, sha256=summary_sha, fresh_cpu_hash=True)}
    issues = directory / "issues.private.jsonl"; issues_before = stat(issues)
    require(issues_before["bytes"] == 0, "accepted_producer_has_private_issues")
    proofs[issues.name] = dict(issues_before, sha256=sha(issues), rows=0, fresh_cpu_hash=True)
    if production:
        require((questions, eligible) == (169103, 100974), "production_question_pair_counts_changed")
        for name, (digest, rows) in INPUTS.items():
            source = original["inputs"][name]
            require(source["sha256"] == digest and source["rows"] == rows and source["stat_stable"] and source["known_sha_matches"], "producer_original_input_identity_differs")
        for name, digest in dict(SCIENCE, audit_source=PRODUCER_SHA).items():
            item = original["frozen_science"][name]; before = stat(item["path"])
            require(item["sha256"] == digest and sha(item["path"]) == digest and stat(item["path"]) == before, "producer_scientific_source_changed")
            identities[name] = dict(before, sha256=digest, fresh_cpu_hash=True)
    require(output != directory and not output.exists(), "independent_output_must_be_new")
    output.mkdir(parents=True, mode=0o700); os.chmod(output, 0o700)
    origins = {}
    for item in checked_lines(directory / "pairs.private.jsonl", proofs):
        qid = item["question_id"]; require(qid not in origins, "duplicate_private_pair_question")
        require(item["native_pair_exact"] is True and not item["mismatch_fields"] and item["previous_pair_rows"] == 0 and
                item["actual_semantic_sha256"] == item["expected_semantic_sha256"], "private_pair_native_identity_invalid")
        roles = {name: item[name + "_matching_source_rows"] for name in ("reference", "student")}
        require(all(roles.values()) and all(len({x["sibling"] for x in matches}) == len(matches) for matches in roles.values()), "private_pair_origin_missing_or_duplicate")
        origins[qid] = dict(matches=roles, semantic_sha256=item["actual_semantic_sha256"])
    require(len(origins) == eligible, "private_pair_denominator_mismatch")
    db_path = directory / "source.private.sqlite"; before_db = stat(db_path)
    require(not any(Path(str(db_path) + suffix).exists() for suffix in ("-wal", "-journal")), "producer_spool_still_writable_or_pending")
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True); db.execute("PRAGMA query_only=ON")
    counters = Counter(); cells = defaultdict(lambda: defaultdict(Counter)); seen = set(); origin_cells = defaultdict(Counter)
    try:
        require(db.execute("SELECT count(*) FROM questions").fetchone()[0] == questions and db.execute("SELECT count(*) FROM rows").fetchone()[0] == questions * 5, "spool_question_sibling_counts_disagree")
        with (output / "roles.private.jsonl").open("x", encoding="utf-8") as private:
            for case in checked_lines(directory / "cases.private.jsonl", proofs):
                qid = case["question_id"]; require(qid not in seen and case["native_shadow_exact"] is True, "private_case_duplicate_or_non_native")
                seen.add(qid); counters["questions"] += 1; key = case["dataset"] + "/" + case["group"]
                meta = db.execute("SELECT valid,eligible,dataset,cell,expected_pair,pair_seen FROM questions WHERE qid=?", (qid,)).fetchone()
                require(meta and meta[:4] == (1, int(case["eligible"]), case["dataset"], case["group"]), "case_spool_group_identity_mismatch")
                metrics = {item["sibling"]: {name: value for name, value in item.items() if name != "sibling"} for item in case["sibling_diagnostics"]}
                provenance = case["provenance"]
                require(len(case["sibling_diagnostics"]) == len(provenance) == 5, "case_private_five_diagnostics_required")
                indexed = list(db.execute("SELECT sibling,provenance,metrics FROM rows WHERE qid=? ORDER BY source_line", (qid,)))
                require(len(indexed) == 5 and [item[0] for item in indexed] == case["sibling_order"] and sorted(metrics) == list(range(5)), "case_spool_sibling_order_mismatch")
                for position, (sibling, prov, metric) in enumerate(indexed):
                    require(json.loads(prov) == provenance[position] and json.loads(metric) == metrics[sibling], "case_spool_metrics_provenance_mismatch")
                successes = sum(item["strict_reward"] for item in metrics.values())
                require(successes == case["successes"] and case["group"] == ("all5_wrong" if not successes else "correct_correct" if successes == 5 else "correct_wrong") and case["eligible"] is bool(successes), "case_success_group_disagrees")
                origin_cells[key]["questions"] += 1
                if not case["eligible"]:
                    require(case["group"] == "all5_wrong" and case["reference"] is None and case["selection"] is None and qid not in origins and meta[4] is None and meta[5] == 0, "hard_case_fabricated_reference_or_pair")
                    counters["hard_questions_without_roles"] += 1; continue
                require(meta[5] == 1 and qid in origins and case["group"] in ("correct_correct", "correct_wrong"), "eligible_case_missing_exact_pair")
                origin_cells[key]["eligible_pairs"] += 1
                expected_pair = json.loads(meta[4]); origin = origins.pop(qid)
                require(hashlib.sha256(canonical(expected_pair)).hexdigest() == origin["semantic_sha256"], "spool_expected_pair_semantic_mismatch")
                chosen = dict(reference=case["reference"]["sibling"], student=case["selection"]["full_chosen"])
                require(type(chosen["reference"]) is int and type(chosen["student"]) is int and chosen["reference"] != chosen["student"] and chosen["student"] in case["selection"]["candidate_siblings"], "native_roles_same_or_invalid_sibling")
                report = dict(question_id=qid, dataset=case["dataset"], group=case["group"], roles={})
                for role, sibling in chosen.items():
                    origin_matches = origin["matches"][role]
                    require(sibling in [item["sibling"] for item in origin_matches], "native_predicted_role_not_in_stored_pair_origins")
                    for item in origin_matches:
                        require(item["sibling"] in range(5) and item["provenance"] == json.loads(next(row[1] for row in indexed if row[0] == item["sibling"])), "stored_pair_origin_provenance_mismatch")
                    row = json.loads(db.execute("SELECT payload FROM rows WHERE qid=? AND sibling=?", (qid, sibling)).fetchone()[0])
                    require(row["question_id"] == qid and row["sibling"] == sibling and row["question"] == expected_pair["question"] and row["answers"] == expected_pair["answers"], "selected_role_question_identity_mismatch")
                    dataset = "hotpotqa" if row["data_source"] in ("hotpot", "hotpotqa") else row["data_source"]
                    require(dataset == case["dataset"] and dataset in ("nq", "hotpotqa"), "selected_role_dataset_mismatch")
                    require(row["response"] == expected_pair[role] and row["reward"] == expected_pair[role + "_reward"], "selected_role_pair_response_mismatch")
                    if role == "student": require(row["policy_spans"] == expected_pair["student_policy_spans"], "selected_student_spans_mismatch")
                    require(row["reward"] == (1 if role == "reference" or successes == 5 else 0), "selected_role_native_reward_mode_mismatch")
                    require(type(row["search_calls"]) is int and row["search_calls"] >= 0 and isinstance(row["termination"], str) and ("token_count" not in row or type(row["token_count"]) is int and row["token_count"] >= 0), "selected_role_counts_or_termination_bad")
                    if role == "reference":
                        require(row["search_calls"] == case["reference"]["search_calls"] and row.get("token_count", len(row["response"])) == case["reference"]["token_count_or_length"], "selected_reference_cost_identity_mismatch")
                    measures = role_metrics(row, metrics[sibling]); add_role(cells[key][role], measures)
                    origin_cells[key][role + "_ambiguous_stored_origin_questions"] += int(len(origin_matches) > 1)
                    origin_cells[key][role + "_stored_origin_matches"] += len(origin_matches)
                    report["roles"][role] = dict(native_predicted_sibling=sibling, stored_matching_siblings=[item["sibling"] for item in origin_matches],
                                                 stored_pair_unique_origin_proven=len(origin_matches) == 1, measures=measures)
                    counters[role + "_rows"] += 1
                private.write(canonical(report).decode() + "\n")
            private.flush(); os.fsync(private.fileno())
        require(len(seen) == questions and not origins and counters["reference_rows"] == counters["student_rows"] == eligible, "role_exact_denominator_or_coverage_mismatch")
        expected_cells = {item["dataset"] + "/" + item["group"]: item for item in original["cells"]}
        require(len(expected_cells) == len(original["cells"]) and all(key in expected_cells for key in origin_cells), "accepted_cell_identities_mismatch")
        for key, population in expected_cells.items():
            require(origin_cells[key]["questions"] == population.get("questions", 0) and origin_cells[key]["eligible_pairs"] == population.get("eligible_pairs", 0), "accepted_cell_denominator_mismatch")
        require(stat(db_path) == before_db and stat(summary_path) == before_summary and sha(summary_path) == summary_sha, "accepted_artifacts_changed")
        db_digest = sha(db_path); require(stat(db_path) == before_db, "spool_changed_during_content_hash")
        proofs[db_path.name] = dict(before_db, sha256=db_digest, fresh_cpu_hash=True, opened_mode="ro", pragma_query_only=True)
    finally:
        db.close()
    for proof in list(proofs.values()) + list(identities.values()):
        require(stat(proof["path"]) == {key: proof[key] for key in stat(proof["path"])}, "postprocessing_input_changed_after_read")
    totals = defaultdict(Counter)
    for roles in cells.values():
        for role, value in roles.items(): totals[role].update(value)
    value = dict(schema="opsd-selected-roles-v1", accepted=True, started_unix=started, finished_unix=time.time(), pid=os.getpid(),
                 counters=dict(counters), inputs=proofs, producer_sources=identities,
                 source=dict(stat(__file__), sha256=sha(__file__)), runtime=dict(python=sys.version, sqlite=sqlite3.sqlite_version), original_raw_files_reread=False,
                 global_roles={role: finish_role(counter) for role, counter in totals.items()},
                 cells=[dict(cell=key, population=dict(origin_cells[key]), roles={role: finish_role(counter) for role, counter in cells[key].items()}) for key in sorted(origin_cells)],
                 roles_private=dict(path=str(output / "roles.private.jsonl"), sha256=sha(output / "roles.private.jsonl")),
                 definitions=dict(role="Exactly one native predicted reference and full-distance native predicted student per eligible question; no role for hard all5_wrong.",
                                  origin="Stored pairs lack sibling IDs. All matching source rows remain ambiguity evidence; native predicted IDs remain well-defined and are not cancelled.",
                                  structure="Literal nonfinal think stack and producer reserved-tag stack are diagnostics, not the strict native final-action parser.",
                                  strict="Reuse accepted producer failure/reward metrics unchanged; invalid final action has no parser-defined gold answer.",
                                  lengths="response_char_count is characters; token_count is stored FULL RESPONSE token count, never policy-token count; missing token counts have a separate denominator."),
                 scope="CPU selected-role artifact audit only; no CUDA/HTTP/new inference/training/distance recomputation/population performance or causal claim.")
    (output / "summary.json").write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--selection-run", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    args = parser.parse_args(); os.umask(0o077)
    summary = run(args.selection_run, args.output, expected_summary_sha256=args.expected_summary_sha256)
    print(json.dumps(dict(accepted=summary["accepted"], counters=summary["counters"], summary_path=str(Path(args.output).resolve() / "summary.json"))))


if __name__ == "__main__":
    main()
