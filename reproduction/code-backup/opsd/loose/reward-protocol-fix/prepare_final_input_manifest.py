from pathlib import Path
import json,hashlib,subprocess,sys
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');repo=root/'repo';review=root/'tools/corrected-integration-review-9fb';source=review/'source-v3';sys.path.insert(0,str(source))
from search_r1.baselines.corrected_evaluation import prepare_inputs,load_inputs
module=source/'search_r1/baselines/corrected_evaluation.py';assert hashlib.sha256(module.read_bytes()).hexdigest()=='b2507bc228b8bc12c59576c4c599a6837d6301e348e187b2f5c5181ae17ecaef'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
config=repo/'verl/trainer/config/ppo_baselines.yaml';assert 'final_val_files: ${oc.env:SEARCH_R1_ASSETS,/data/search-r1}/data/paper-grpo/test.parquet' in config.read_text()
path=root/'assets-root/data/paper-grpo/test.parquet';output=review/'pregeneration-final-inputs.json';sha=prepare_inputs([str(path)],str(output));manifest=load_inputs(output,sha);counts={}
for row in manifest['records']:s=row['identity']['source'];counts[s]=counts.get(s,0)+1
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()==''
print(json.dumps(dict(scope='CPU pregeneration final data manifest only, no training/evaluation receipt',path=str(output),sha256=sha,count=len(manifest['records']),source_counts=counts,inputs=manifest['inputs'],original_training_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())))
