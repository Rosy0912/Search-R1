"""Fetch bounded JSON evidence pinned by a specific actual 3289 snapshot; no PT/data."""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RUN = "/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-policy-safety-calibration25-job3289-1791251062110516012"
ALLOWED = {"actual-vllm-backend-identity.json", "restore25-before-baseline.json",
           "restore25-before-rollout.json", "baseline-evaluation.json",
           "actual-probability-alignment.json", "probability-material-manifest.json",
           "probability-identity-ready.json", "post-sampling-pause-state.json",
           "ray-audit-environment-verification.json", "run-result.json"}
snapshot = Path(sys.argv[1]).resolve()
assert snapshot.parent == (Path.cwd() / "outputs/gigpo-e1-policy-safety").resolve()
actual = json.loads(snapshot.read_text())["actual"]
assert actual["job_id"] == "3289" and actual["actual_run"] == RUN
dest = Path("outputs/gigpo-e1-policy-safety/calibration3289-originals")
dest.mkdir(exist_ok=True)
records = {}
for row in actual["originals"]:
    name = Path(row["path"]).name
    if name not in ALLOWED:
        continue
    assert row["path"] == RUN + "/" + name and row["bytes"] < 1024 * 1024
    path = dest / name
    if path.exists():
        assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    else:
        records[name] = row["sha256"]
assert records, "No new immutable bounded JSON evidence to fetch"
code = """import base64,hashlib,json
from pathlib import Path
run=Path(RUN_VALUE)
assert run.resolve()==run and run.is_dir()
records=RECORDS_VALUE
out=[]
for name,expected in records.items():
 p=run/name
 assert p.is_file() and not p.is_symlink() and p.resolve().parent==run
 raw=p.read_bytes()
 assert len(raw)<1024*1024 and hashlib.sha256(raw).hexdigest()==expected
 json.loads(raw)
 out.append(dict(name=name,sha256=expected,raw=base64.b64encode(raw).decode()))
print(json.dumps(out))
""".replace("RUN_VALUE", repr(RUN)).replace("RECORDS_VALUE", repr(records))
script = "/usr/bin/python3 - <<'REMOTE'\n" + code + "\nREMOTE\n"
result = subprocess.run(["/usr/bin/python3", "work/remote.py", "h100-8"], input=script,
                        text=True, capture_output=True, timeout=180)
assert result.returncode == 0, result.stderr
fetched = json.loads(result.stdout)
assert len(fetched) == len(records)
for row in fetched:
    raw = base64.b64decode(row["raw"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == records[row["name"]]
    path = dest / row["name"]
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(path=str(path.resolve()), bytes=len(raw), sha256=row["sha256"])))
