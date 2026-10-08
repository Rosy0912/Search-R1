"""Save own job scheduler states before rapid Slurm record purging.

Only metadata is read. There is no job submission, signal, asset loading, or
model interaction. Run only after the fixed dispatch journal confirms a job.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("job")
parser.add_argument("--intent", required=True, type=Path)
args = parser.parse_args()
root = Path("/dataset/shaoshitong/search-r1-rl/tree-grpo")
assert socket.gethostname().split(".")[0] == "h100-9"
assert re.fullmatch("[1-9][0-9]*", args.job)
assert args.intent.is_absolute() and root in args.intent.resolve().parents
journal = json.loads(Path(str(args.intent) + ".result.json").read_bytes())
assert journal["status"] == "SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE"
assert journal["job_id"] == args.job
output = root / "instruct300" / ("probe-scheduler-observation-" + args.job + ".jsonl")
terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
            "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
deadline = time.monotonic() + 1900
errors = 0
previous = None
with output.open("x") as stream:
    while time.monotonic() < deadline:
        try:
            query = subprocess.run(["scontrol", "show", "job", args.job],
                                   capture_output=True, text=True, timeout=10)
            row = dict(epoch=time.time(), job_id=args.job, returncode=query.returncode,
                       stdout=query.stdout, stderr=query.stderr, mutation=False)
        except subprocess.TimeoutExpired as error:
            row = dict(epoch=time.time(), job_id=args.job, returncode=None,
                       stdout=None, stderr=str(error), mutation=False)
        match = re.search(r"(?:^|\s)JobState=(\S+)", row["stdout"] or "")
        state = match.group(1) if row["returncode"] == 0 and match else None
        row["scheduler_state"] = state
        row["terminal_state_observed"] = state in terminal
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        if state != previous or row["terminal_state_observed"]:
            print(json.dumps(dict(job_id=args.job, state=state, evidence=str(output),
                                  terminal=row["terminal_state_observed"])), flush=True)
            previous = state
        if row["terminal_state_observed"]:
            break
        errors = 0 if state is not None else errors + 1
        if errors >= 3:
            print(json.dumps(dict(job_id=args.job, state="UNKNOWN", evidence=str(output),
                                  reason="Three unknown scheduler queries; no terminal acceptance")), flush=True)
            break
        time.sleep(15)
    else:
        print(json.dumps(dict(job_id=args.job, state="UNKNOWN", evidence=str(output),
                              reason="Observation deadline; no terminal acceptance")), flush=True)
