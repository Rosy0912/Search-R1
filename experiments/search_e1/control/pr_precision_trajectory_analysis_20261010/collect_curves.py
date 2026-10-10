"""Compare already-published R and PR aggregates; no remote/model access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    return list(csv.DictReader(raw.decode().splitlines())), {
        "path": str(path.resolve()), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original, r_source = load(args.reports / "final-cpu-debug-20261009/seven-group-70-slots.csv")
    current, pr_source = load(args.reports / "pr100-first30-native512-20261010/train-outer-curves.csv")
    r = [row for row in original if row["variant"] == "R"]
    pr = current[:10]
    assert len(r) == len(pr) == 10
    rows = []
    for a, b in zip(r, pr):
        assert int(a["outer_index"]) == int(b["outer_index"])
        assert int(a["sampled_trajectories"]) == int(b["sampled_trajectories"]) == 2560
        value = {"outer_number": int(a["outer_number"])}
        for group, row, hit_key in (("R", a, "train_strict_reward_hits"), ("PR", b, "train_strict_hits")):
            value[group] = {
                "strict_hits": int(row[hit_key]), "rollouts": 2560,
                "strict_em": int(row[hit_key]) / 2560,
                "pairs": int(row["usable_K"]),
                "actual_adam_cumulative": int(row["actual_adam_cumulative"]),
                "valid_policy_tokens": int(row["valid_policy_tokens"]),
                "raw_loss": float(row["raw_loss"]), "post_loss": float(row["post_loss"]),
                "all_wrong_questions": int(row["all_wrong_questions"] if group == "R" else row["sample_questions_all_wrong"]),
                "all_correct_questions": int(row["all_correct_questions"] if group == "R" else row["sample_questions_all_correct"]),
                "checkpoint_manifest_sha256": row["checkpoint_manifest_sha256"],
            }
        rows.append(value)
    totals = {}
    for group in ("R", "PR"):
        entries = [x[group] for x in rows]
        pairs = sum(x["pairs"] for x in entries)
        totals[group] = {
            "strict_hits": sum(x["strict_hits"] for x in entries), "rollouts": 25600,
            "strict_em": sum(x["strict_hits"] for x in entries) / 25600,
            "pair_exposure": pairs, "actual_adam": entries[-1]["actual_adam_cumulative"],
            "valid_policy_tokens": sum(x["valid_policy_tokens"] for x in entries),
            "exposure_weighted_raw_loss": sum(x["raw_loss"] * x["pairs"] for x in entries) / pairs,
            "exposure_weighted_post_loss": sum(x["post_loss"] * x["pairs"] for x in entries) / pairs,
        }
    output = {
        "scope": "Existing closed first-ten-outer aggregates, same question cohorts per corresponding outer.",
        "sources": {"R": r_source, "PR": pr_source}, "outers": rows, "totals": totals,
        "interpretation_limits": [
            "First outer sampling starts from the common initial checkpoint; later outers include online weight/data feedback.",
            "R and PR loss are recorded on different generated responses and numeric paths, not a fixed-response precision test.",
            "These changing TRAIN cohorts do not form a fixed validation EM curve; actual Adam count is distinct from outer count.",
            "No raw text, checkpoint, tensor, evaluation rows, or additional model generation is stored locally.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
