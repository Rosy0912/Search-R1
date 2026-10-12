from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
p=root/'.aligned-em200-deployment.json'
print(json.dumps({'host':os.uname().nodename,'root':str(root.resolve()),'marker':json.loads(p.read_text()),'marker_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mount':json.loads(subprocess.check_output(['findmnt','-T',str(root),'-J'],text=True)),'scope':'READ_ONLY_OWN_TASK_SHARED_MARKER'}))
