import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,hashlib,subprocess,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');U=R/'runs/tree-safety300-3268';out={'epoch':time.time(),'job_id':'3268','run':str(U)}
p=subprocess.run(['scontrol','show','job','3268'],text=True,capture_output=True);out['slurm']={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
files={}
for n in ['run-result.json','cost.json','launch-manifest.json','output/recovery-loaded.json','output/probability-calibration/ready.json']:
 p=U/n
 if p.exists():b=p.read_bytes();files[n]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
for pat in ['native-fullstate/rank-*.json','gpu-kl-proof/rank-*.json','output/validation-cost/*.json','output/probability-calibration/rank-*.json','output/entry-audit/*.json']:
 for p in U.glob(pat):
  b=p.read_bytes();files[str(p.relative_to(U))]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
out['files']=files
for n in ['stage.log','native-fullstate.log','resource-samples.jsonl']:
 p=U/n
 if p.exists():
  s=p.read_text();out[n]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tail':'\n'.join(s.splitlines()[-20:]),'lines':len(s.splitlines())}
out['counts']={n:len(list(U.glob(pat))) for n,pat in [('sampling_json','output/sampling-consistency/*.json'),('calibration_worker','output/probability-calibration/rank-*.json'),('directrank','output/tree-rank-audit/step-*.json'),('trajectory','output/trajectories/*manifest*')]}
print(json.dumps(out))
