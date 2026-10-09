"""Plot accepted PR100 aggregate snapshots without reading raw experiment data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_small(path):
    before = path.stat()
    require(before.st_size <= 16 * 1024 * 1024, "Expected a small aggregate file")
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "Aggregate changed during read")
    return raw, {"path": str(path), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}


def finite(value):
    value = float(value)
    require(math.isfinite(value), "Nonfinite aggregate value")
    return value


def integer(value):
    number = finite(value)
    require(number >= 0 and number.is_integer(), "Invalid aggregate integer")
    return int(number)


def csv_records(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def match_csv(csv_rows, json_rows, columns):
    require(len(csv_rows) == len(json_rows), "CSV/summary row coverage differs")
    for left, right in zip(csv_rows, json_rows):
        for column in columns:
            if right[column] is None:
                require(left[column] == "", "CSV/summary empty loss differs")
            else:
                require(finite(left[column]) == finite(right[column]),
                        "CSV/summary value differs: " + column)


def load_report(directory):
    names = ("complete.json", "summary.json", "train-outer-curves.csv",
             "committed-adam-loss.csv", "native512-curves.csv")
    files = {name: read_small(directory / name) for name in names}
    complete = json.loads(files["complete.json"][0])
    summary = json.loads(files["summary.json"][0])
    require(complete.get("schema") == "opsd-pr100-readonly-cpu-report-complete-v1"
            and summary.get("schema") == "opsd-pr100-public-recorded-report-v1"
            and complete.get("accepted") is True and summary.get("accepted") is True,
            "Only accepted PR100 aggregate reports can be plotted")
    require(complete["coverage"] == summary["coverage"], "Report coverage differs")
    declared = {Path(ref["path"]).name: ref for ref in complete["artifacts"]}
    for name in names[1:]:
        require(name in declared and declared[name]["sha256"] == files[name][1]["sha256"]
                and declared[name]["bytes"] == files[name][1]["bytes"],
                "Aggregate differs from the accepted report: " + name)
    train = summary["train_outer_records"]
    evaluations = summary["heldout512_records"]
    updates = csv_records(files["committed-adam-loss.csv"][0])
    coverage = summary["coverage"]
    n = integer(coverage["training"]["completed_outers"])
    e = integer(coverage["evaluation"]["completed_evaluations"])
    adam = integer(coverage["training"]["actual_adam"])
    require(n <= 100 and len(train) == n
            and [r["outer_number"] for r in train] == list(range(1, n + 1)),
            "Training is not a continuous completed prefix")
    require(e <= 10 and len(evaluations) == e
            and [r["completed_outers"] for r in evaluations] == list(range(10, 10 * e + 1, 10))
            and 10 * e <= n and coverage["evaluation"]["formal_primary_slots"] == 512 * e,
            "Only complete 512-question evaluations may be plotted")
    require(len(updates) == adam and [integer(r["actual_adam"]) for r in updates] ==
            list(range(1, adam + 1)), "Actual Adam records are incomplete or repeated")
    match_csv(csv_records(files["train-outer-curves.csv"][0]), train,
              ("outer_number", "train_em", "raw_loss", "post_loss", "actual_adam_cumulative"))
    match_csv(csv_records(files["native512-curves.csv"][0]), evaluations,
              ("completed_outers", "strict_hits", "em", "nq_hits", "hotpot_hits"))
    for row in train:
        require(row["train_questions"] == 512 and row["sampled_trajectories"] == 2560
                and 0 <= row["train_strict_hits"] <= 2560
                and row["train_em"] == row["train_strict_hits"] / 2560,
                "TRAIN sampled EM denominator differs")
    for row in evaluations:
        require(0 <= row["strict_hits"] <= 512 and row["em"] == row["strict_hits"] / 512
                and row["nq_hits"] + row["hotpot_hits"] == row["strict_hits"],
                "Fixed512 EM denominator differs")
    # The reporter records the accepted source100 summary's path/SHA among its inputs.
    # Never invent a baseline from a remembered count or an incomplete evaluation.
    baseline_refs = [r for r in complete["input_json_references"]
                     if r.get("path", "").endswith("/zero/runs/summary.json")
                     and re.fullmatch("[a-f0-9]{64}", r.get("sha256", ""))]
    baseline = None
    if baseline_refs:
        hits = integer(coverage["evaluation"]["baseline_strict_hits"])
        baseline = finite(coverage["evaluation"]["baseline_em"])
        require(hits <= 512 and baseline == hits / 512, "Source100 baseline count differs")
        require(all(r["baseline_strict_hits"] == hits and r["baseline_em"] == baseline
                    for r in evaluations), "Evaluation baseline differs across closed points")
    return summary, train, updates, evaluations, baseline, baseline_refs, [files[k][1] for k in names]


def render(summary, train, updates, evaluations, baseline, output):
    # Keep matplotlib's disposable font cache outside the report/output directories.
    with tempfile.TemporaryDirectory(prefix="pr100-plot-cache-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "axes.titleweight": "bold", "savefig.facecolor": "white"})
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        a, b, c, d = axes.flat
        outer = [r["outer_number"] for r in train]
        a.plot(outer, [100 * r["train_em"] for r in train], "o", color="#1f6f8b", markersize=5)
        a.set(title="A  Pre-update sampled TRAIN EM", xlabel="Completed online outer", ylabel="Strict EM (%)")
        colors = {"raw_loss": "#777777", "post_loss": "#c45d1c"}
        labels = {"raw_loss": "Raw reverse KL", "post_loss": "Post-clipping loss"}
        for key in colors:
            usable = [r for r in train if r[key] is not None]
            b.plot([r["outer_number"] for r in usable], [finite(r[key]) for r in usable],
                   "o", color=colors[key], markersize=5, label=labels[key])
            c.plot([integer(r["actual_adam"]) for r in updates], [finite(r[key]) for r in updates],
                   ".", color=colors[key], markersize=4, alpha=.85, label=labels[key])
        b.set(title="B  Recorded loss per completed outer", xlabel="Completed online outer",
              ylabel="Equal-trajectory loss")
        c.set(title="C  Recorded loss for each actual Adam", xlabel="Actual cumulative Adam updates",
              ylabel="Equal-trajectory loss")
        for row in train[:-1]:
            c.axvline(row["actual_adam_cumulative"] + .5, color="#bbbbbb", linewidth=.6, linestyle=":")
        for ax in (b, c):
            ax.legend(frameon=False, fontsize=9)
        if baseline is not None:
            d.axhline(100 * baseline, color="#777777", linestyle="--", linewidth=1.2,
                      label="Referenced source100 baseline")
        if evaluations:
            d.plot([r["completed_outers"] for r in evaluations], [100 * r["em"] for r in evaluations],
                   "o", color="#417b43", markersize=6, label="Closed raw-checkpoint native512")
            for row in evaluations:
                d.annotate(str(row["strict_hits"]) + "/512", (row["completed_outers"], 100 * row["em"]),
                           xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        else:
            d.text(.5, .76, "No complete fixed512 evaluation at this snapshot", transform=d.transAxes,
                   ha="center", fontsize=10, color="#555555", wrap=True)
        d.set(title="D  Closed fixed512 validation EM", xlabel="Completed online outer", ylabel="Strict EM (%)")
        if baseline is not None or evaluations:
            d.legend(frameon=False, fontsize=9, loc="lower left")
        n = len(train)
        for ax in (a, b):
            ax.set_xlim(.5, max(1.5, n + .5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        c.set_xlim(.5, max(1.5, len(updates) + .5))
        c.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        d.set_xlim(0, max(10, n) + 2)
        d.set_xticks(list(range(10, max(10, n) + 1, 10)))
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.2)
            ax.yaxis.set_major_locator(MaxNLocator(5))
        fig.suptitle("PR100 | accepted completed snapshot: " + str(n) + " outers, " +
                     str(len(updates)) + " Adam updates, " + str(len(evaluations)) + " native512 evaluations",
                     fontsize=15, fontweight="bold", x=.075, ha="left")
        fig.text(.075, .016, "TRAIN EM: 512 different questions x 5 sampled trajectories per outer. Loss: recorded pre-update token means,\n"
                 "equally averaged over actual trajectories including tails. Points only; no interpolation, smoothing or partial validation.",
                 fontsize=9, color="#555555", va="bottom")
        fig.subplots_adjust(left=.075, right=.98, top=.89, bottom=.12, hspace=.38, wspace=.27)
        png = output / "pr100-em-loss-curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        return png, matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, help="Existing accepted aggregate snapshot")
    parser.add_argument("--output", required=True, help="New directory for PNG and provenance only")
    args = parser.parse_args()
    directory = Path(args.report_dir).resolve()
    output = Path(args.output).resolve()
    require(not output.exists(), "Never overwrite a plot snapshot")
    summary, train, updates, evaluations, baseline, baseline_refs, inputs = load_report(directory)
    output.mkdir(parents=True)
    png, version = render(summary, train, updates, evaluations, baseline, output)
    _, png_ref = read_small(png)
    _, source_ref = read_small(Path(__file__).resolve())
    provenance = {"schema": "opsd-pr100-aggregate-static-plot-v1", "accepted": True,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                  "source_code": source_ref, "matplotlib_version": version, "plot": png_ref,
                  "coverage": summary["coverage"], "training_run_id": summary["training_run_id"],
                  "pipeline_run_id": summary["pipeline_run_id"], "baseline_summary_references": baseline_refs,
                  "baseline_drawn": baseline is not None, "interpolation": False, "smoothing": False,
                  "partial_evaluation_rows": 0, "scope": "Aggregate-only rendering; sampled pre-update TRAIN EM on varying cohorts; recorded equal-trajectory raw/post-clipping losses; closed native512 only. No model, raw, tensor, token, network or bootstrap computation."}
    proof = output / "plot-provenance.json"
    proof.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _, proof_ref = read_small(proof)
    print(json.dumps({"accepted": True, "plot": png_ref, "provenance": proof_ref,
                      "completed_outers": len(train), "actual_adam": len(updates),
                      "completed_native512": len(evaluations)}))


if __name__ == "__main__":
    main()
