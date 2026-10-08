import json,hashlib,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');p=root/'candidates/tree-policy-safety-native-0980ec3-v2';raw=(p/'manifest.json').read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest=='b9c2e07714a745e462d3678328ad4d25a3a83958c1683787cf63e1562768137c';m=json.loads(raw)
for name,wanted in m['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==wanted,name
for name,wanted in m['external_helpers'].items():assert hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==wanted,name
wrapper=hashlib.sha256((p/'start.sh').read_bytes()).hexdigest();assert wrapper=='a35fa431fb838a3dc16021005fa940bfd06baf8b48de12f748e4fa4ed60882f4'
old=hashlib.sha256((root/'candidates/tree-policy-safety-native-0980ec3/manifest.json').read_bytes()).hexdigest();assert old=='ba9bba245a54e36c8026b21960eb4d644eefb97f930a1c2c2867269b8cbb1767'
print(json.dumps(dict(status='ACTUAL_TARGET_V2_PAYLOAD_SHA_PASS_NOT_NATIVE_EXECUTION',epoch=time.time(),target=str(p),code_commit=m['code_commit'],manifest_sha256=digest,wrapper_sha256=wrapper,payload_files_verified=len(m['files']),external_helpers=m['external_helpers'],old_manifest_preserved=old)),flush=True)
