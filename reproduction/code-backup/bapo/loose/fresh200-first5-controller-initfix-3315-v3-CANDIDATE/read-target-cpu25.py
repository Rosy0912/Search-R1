from pathlib import Path
import json,hashlib,subprocess
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');S='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()==S
record=json.loads((R/'logs/first5-v3-linux25-tests.actual.json').read_text());raw=(R/'logs/first5-v3-linux25-tests.actual.log').read_bytes();assert record['raw_log_sha256']==hashlib.sha256(raw).hexdigest() and record['passed']==25 and record['exit_code']==0 and record['native_acceptance']is False
B=R/'first5-evidence-tools-3315-v3';manifest=json.loads((B/'package-manifest.json').read_text());assert len(manifest['files'])==9
for n,sha in manifest['files'].items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==sha
print(json.dumps(dict(status='ACTUAL_LINUX25_CODE_ONLY_VERIFIED',record=record,raw_log=raw.decode(),raw_log_bytes=len(raw),installed_manifest=manifest,manifest_sha256=hashlib.sha256((B/'package-manifest.json').read_bytes()).hexdigest()),allow_nan=False))
