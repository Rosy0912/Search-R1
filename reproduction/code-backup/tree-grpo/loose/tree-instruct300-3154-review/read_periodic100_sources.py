from pathlib import Path
import json,hashlib
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');C=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161';R=ROOT/'runs/tree-instruct300-3154';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((C/'manifest.json').read_bytes());assert sha(C/'manifest.json')=='aa9e3161ee8a211e6e311d2a87be9d456371f0f569d0e0c9c1703e3a575ba69f';proof=[]
for name in ['source/repo/search_r1/baselines/evaluation.py','source/repo/search_r1/baselines/bapo.py','source/repo/search_r1/qa_protocol.py','source/repo/verl/trainer/ppo/ray_trainer.py','validation_cost_hooks_instruct300.py']:
 p=C/name;assert sha(p)==m['files'][name];proof.append(dict(path=str(p),sha256=sha(p)))
s=json.loads((R/'output/checkpoints/tree-instruct300/training_state/global_step_100/manifest.json').read_bytes());driver=Path(s['driver']['path']);assert sha(driver)==s['driver']['sha256'];proof.append(dict(path=str(driver),sha256=sha(driver)))
h=Path(s['hf_checkpoint']);index=h/'model.safetensors.index.json';assert sha(index)==s['hf_identity']['files']['model.safetensors.index.json'];proof.append(dict(path=str(index),sha256=sha(index)))
print(json.dumps(dict(status='PERIODIC100_FROZEN_SOURCE_SMALL_FILE_SHA_PASS',proof=proof)))
