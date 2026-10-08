"""Copy only immutable bounded JSON evidence from the actual calibration run."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

RUN = "/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-policy-safety-calibration25-job3289-1791251062110516012"
RECORDS = {
    "actual-vllm-backend-identity.json": "836a2cf5f62e6203f08b2543c2be25cfa704e54ffc9e37b6fb7f39b0f78755c0",
    "restore25-before-baseline.json": "5461d6245dac18e27b8e0f6659341390b2badbd738c94b01c45921c670287db5",
    "ray-audit-environment-verification.json": "c8df6642c924bf563acc857a0c4128c919e33f0d32df6c32f531f4f7bd64fffb",
}
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
""".replace("RUN_VALUE", repr(RUN)).replace("RECORDS_VALUE", repr(RECORDS))
script = "/usr/bin/python3 - <<'REMOTE'\n" + code + "\nREMOTE\n"
result = subprocess.run(["/usr/bin/python3", "work/remote.py", "h100-8"], input=script,
                        text=True, capture_output=True, timeout=180)
assert result.returncode == 0, result.stderr
records = json.loads(result.stdout)
assert len(records) == len(RECORDS)
dest = Path("outputs/gigpo-e1-policy-safety/calibration3289-originals")
dest.mkdir(exist_ok=True)
for record in records:
    raw = base64.b64decode(record["raw"], validate=True)
    assert hashlib.sha256(raw).hexdigest() == RECORDS[record["name"]]
    path = dest / record["name"]
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(path=str(path.resolve()), bytes=len(raw), sha256=record["sha256"])))
