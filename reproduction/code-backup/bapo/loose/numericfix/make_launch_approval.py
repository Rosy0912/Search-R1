import json,base64
from pathlib import Path
review=json.loads(Path('outputs/bapo-numericfix-independent-launch-review.json').read_text())
md=base64.b64encode(Path('outputs/bapo-numericfix-independent-launch-review.md').read_bytes()).decode()
s='''import pathlib,json,hashlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric')
review=REVIEW
report=root/'logs/bapo-numericfix-independent-launch-review.md'
report.write_bytes(base64.b64decode(MD))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[]
for name,sha in review['tools_sha256'].items():
 p=root/'tools'/name
 assert digest(p)==sha,name
 paths.append(p)
paths.extend([root/'tools/hardware-launch-hashes-numeric-v1.json', report])
paths.extend(root/'logs'/n for n in ['cpu-gate-numeric.json','cpu-full-numeric.txt','environment-acceptance.json','shared-inputs.json','isolation-setup.json','common-split-manifest.json','stage64-numeric-proof.json'])
proof=json.loads((root/'logs/stage64-numeric-proof.json').read_text().split('BAPO_STAGE64_DERIVATION_PROOF_JSON=')[-1])
p=pathlib.Path(proof['stage_file']);paths.extend([p,p.with_name('manifest.json')])
record=dict(status=review['status'],scope=review['scope'],source_commit=review['source_commit'],evidence=[dict(path=str(p),sha256=digest(p)) for p in paths])
(root/'logs/hardware-launch-approval-numeric.json').write_text(json.dumps(record,indent=2)+'\\n')
subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_hardware_launch.py')],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps(record))
'''.replace('REVIEW',repr(review)).replace('MD',repr(md))
Path('work/numericfix/install-launch-approval.py').write_text(s)
