from pathlib import Path
import json,hashlib,subprocess,os
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');S='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert os.uname().nodename=='h100-5'and R.resolve()==R
assert subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()==S
assert not subprocess.check_output(['git','-C',str(R/'repo'),'status','--porcelain'],text=True).strip()
names=['target-cpu-gate.json','target-cpu-gate.raw.log','context-budget-audit.json','context-budget-audit.raw.log','runtime-executable.json','runtime-tests.actual.json','runtime-tests.actual.log','frozen-policy-safety-config.json','user-fixed-policy-thresholds.json','latest-human-goal-behavior.json','isolation-setup.json','tools-manifest.json']
out={'source_commit':S,'root':str(R),'files':{}}
for n in names:
 p=R/'logs'/n;assert p.is_file()and not p.is_symlink();b=p.read_bytes();out['files'][n]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode()}
print(json.dumps(out))
