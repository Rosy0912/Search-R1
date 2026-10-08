"""Select six deterministic cells without retaining the full rollout file in RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import atomic_json, canonical_bytes, file_sha, load_science, object_sha, scan
else:
    from .common import atomic_json, canonical_bytes, file_sha, load_science, object_sha, scan

DATASETS = ("nq", "hotpotqa")
GROUPS = ("correct_correct", "correct_wrong", "all5_wrong")


def dataset_name(value):
    if value in ("hotpot", "hotpotqa"):
        return "hotpotqa"
    if value == "nq":
        return "nq"
    raise ValueError("Unexpected source dataset: " + repr(value))


def select(rollouts, pairs, train, output, per_cell=32, seed=42,
           expected_questions=169103, expected_pairs=100974, reward=None, science=None):
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    cells = {(dataset, group): [] for dataset in DATASETS for group in GROUPS}
    population = {key: 0 for key in cells}
    seen = set()
    current = []

    def accept_group():
        if not current:
            return
        question_id = current[0][0]["question_id"]
        if question_id in seen:
            raise ValueError("A question's sibling group reappeared: " + question_id)
        seen.add(question_id)
        rows = [item[0] for item in current]
        if len(rows) != 5 or sorted(row["sibling"] for row in rows) != list(range(5)):
            raise ValueError("Incomplete or duplicate five-sibling group: " + question_id)
        rows.sort(key=lambda row: row["sibling"])
        identity = {key: rows[0][key] for key in ("question_id", "question", "answers", "data_source")}
        if any(any(row[key] != value for key, value in identity.items()) for row in rows):
            raise ValueError("Sibling question identity changed: " + question_id)
        values = [row["reward"] for row in rows]
        if any(value not in (0, 1) for value in values):
            raise ValueError("Nonbinary rollout reward")
        group = "all5_wrong" if sum(values) == 0 else "correct_correct" if sum(values) == 5 else "correct_wrong"
        dataset = dataset_name(rows[0]["data_source"])
        key = (dataset, group)
        population[key] += 1
        selection_sha = hashlib.sha256(f"{seed}|{dataset}|{group}|{question_id}".encode()).hexdigest()
        record = dict(case_id=f"{dataset}-{group}-{selection_sha[:16]}", question_id=question_id,
                      dataset=dataset, group=group, selection_sha256=selection_sha,
                      original_siblings=[dict(row=row, provenance=provenance)
                                         for row, provenance in sorted(current, key=lambda item: item[0]["sibling"])],
                      original_pair=None, teacher_reference_status="not_applicable_no_successful_reference" if group == "all5_wrong" else "pending_pair_validation")
        cells[key].append(record)
        cells[key].sort(key=lambda case: (case["selection_sha256"], case["question_id"]))
        del cells[key][per_cell:]

    def visit_rollout(row, provenance):
        if current and row["question_id"] != current[0][0]["question_id"]:
            accept_group()
            current.clear()
        current.append((row, provenance))

    source_rollouts = scan(rollouts, visit_rollout)
    accept_group()
    if len(seen) != expected_questions or source_rollouts["rows"] != expected_questions * 5:
        raise ValueError("Full source rollout count differs from declared identity")
    if any(len(cell) != per_cell for cell in cells.values()):
        raise ValueError("One or more requested cells contain too few cases")
    cases = [case for dataset in DATASETS for group in GROUPS for case in cells[(dataset, group)]]
    selected = {case["question_id"]: case for case in cases}
    pairs_seen = set()

    def visit_pair(pair, provenance):
        question_id = pair["question_id"]
        if question_id in pairs_seen:
            raise ValueError("Duplicate original pair question_id")
        pairs_seen.add(question_id)
        case = selected.get(question_id)
        if case is None:
            return
        if case["group"] == "all5_wrong":
            raise ValueError("Hard excluded case unexpectedly has a mined pair")
        if pair["reference_reward"] != 1 or pair["student_reward"] != int(case["group"] == "correct_correct"):
            raise ValueError("Actual pair contradicts source sibling cell")
        siblings = [item["row"] for item in case["original_siblings"]]
        refs = [row for row in siblings if row["response"] == pair["reference"] and row["reward"] == 1]
        students = [row for row in siblings if row["response"] == pair["student"]
                    and row["policy_spans"] == pair["student_policy_spans"] and row["reward"] == pair["student_reward"]]
        if not refs or not students:
            raise ValueError("Pair response/provenance does not match original siblings")
        original = siblings[0]
        if any(pair[key] != original[key] for key in ("question", "answers")):
            raise ValueError("Pair question or answers changed")
        case["original_pair"] = dict(row=pair, provenance=provenance,
                                     reference_matching_siblings=[row["sibling"] for row in refs],
                                     student_matching_siblings=[row["sibling"] for row in students],
                                     reference_policy_spans=refs[0]["policy_spans"])
        case["teacher_reference_status"] = "successful_reference_verified_against_original_sibling"

    source_pairs = scan(pairs, visit_pair)
    if source_pairs["rows"] != expected_pairs:
        raise ValueError("Full source pair count differs from declared identity")
    train_seen = set()

    def visit_train(row, provenance):
        question_id = row["question_id"]
        if question_id in train_seen:
            raise ValueError("Duplicate source training question_id")
        train_seen.add(question_id)
        case = selected.get(question_id)
        if case is not None:
            original = case["original_siblings"][0]["row"]
            if any(original[key] != row[key] for key in ("question", "answers", "data_source")):
                raise ValueError("Original source training row disagrees with rollout")
            case["source_train_row"] = dict(row=row, provenance=provenance)

    source_train = scan(train, visit_train)
    if len(train_seen) != expected_questions or seen != train_seen:
        raise ValueError("Full source question ID sets do not match")
    for index, case in enumerate(cases):
        if "source_train_row" not in case or (case["group"] != "all5_wrong" and case["original_pair"] is None):
            raise ValueError("Incomplete selected case provenance")
        if reward is not None:
            for sibling in case["original_siblings"]:
                row = sibling["row"]
                actual = reward(row["response"], row["answers"], row["policy_spans"])
                if actual != row["reward"]:
                    raise ValueError("Strict protocol reward contradicts preserved sibling")
        case["case_index"] = index
        case["case_semantic_sha256"] = object_sha(case)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for case in cases:
            stream.write(canonical_bytes(case).decode() + "\n")
    manifest = dict(schema="opsd-independent-debug-cases-v1", created_unix=time.time(), seed=seed,
                    selection="smallest SHA256(seed|dataset|group|question_id); no order-dependent RNG",
                    per_cell=per_cell, selected_cases=len(cases),
                    population=[dict(dataset=d, group=g, questions=population[(d, g)]) for d in DATASETS for g in GROUPS],
                    sources=dict(rollouts=source_rollouts, pairs=source_pairs, train=source_train),
                    cases=dict(path=str(cases_path.resolve()), sha256=file_sha(cases_path), rows=len(cases)),
                    science=science, strict_reward_checked=reward is not None,
                    hard_reference="No successful reference; privileged teacher and OPSD loss are not applicable",
                    evaluation_scope="Stratified diagnostic sample; its EM is not the native512 or seven-benchmark EM")
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for key in ("rollouts", "pairs", "train", "output", "runtime"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--per-cell", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    core, science = load_science(args.runtime)
    manifest = select(args.rollouts, args.pairs, args.train, args.output, args.per_cell, args.seed,
                      reward=core.reward, science=science)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
