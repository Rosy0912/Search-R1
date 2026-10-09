"""Run the already-reviewed CPU collector only at unique formal3348's actual fifth pause."""
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
TOKEN='66cb18840a404586ab3333f8c393b8b7'
RUN=R/'runs'/('formal-aligned-em200-'+TOKEN+'-3348')
OUT=R/'evidence/first5-formal3348-cpu-20261009'
def require(v,m):
    if not v:raise RuntimeError(m)
require(os.uname().nodename=='h100-5','wrong node')
receipt=R/'scheduler-head/formal-submit-receipt.json'
require(receipt.resolve()==receipt and not receipt.is_symlink(),'noncanonical submission')
d=json.loads(receipt.read_text())
require(d.get('source')==SOURCE and d.get('job')=='3348' and d.get('token')==TOKEN and d.get('run')==str(RUN),'wrong formal submission')
require(RUN.resolve()==RUN and RUN.is_dir(),'wrong run')
helper=R/'tools/deployment/first5-cpu-v1/aligned_first5_cpu_evidence.py'
require(helper.resolve()==helper and hashlib.sha256(helper.read_bytes()).hexdigest()=='78d47e2653dd235f952c50b01cc1fc9baca525e11ecd17fa1a4e8202de9c8223','collector bytes differ')
log=RUN/'train.log';require(log.resolve()==log and log.is_file(),'original log absent')
text=log.read_text();pauses=[json.loads(line.split('FRESH200_STEP5_PAUSED_JSON=',1)[1]) for line in text.splitlines() if 'FRESH200_STEP5_PAUSED_JSON=' in line]
manifest=RUN/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json'
require(len(pauses)==1 and pauses[0].get('manifest_path')==str(manifest),'original fifth pause required')
require('FRESH200_STEP5_RELEASED_JSON=' not in text,'already released')
require(not OUT.exists() and not OUT.is_symlink(),'existing/unknown CPU extraction: inspect rather than retry')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'repo'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([str(R/'env/bin/python'),'-B',str(helper),'--output',str(OUT)],cwd=R/'repo',env=env,text=True,capture_output=True,timeout=1700)
print(json.dumps({'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'output':str(OUT),'scope':'ACTUAL_CPU_EXTRACTION_NOT_INDEPENDENT_ACCEPTANCE_OR_RELEASE'}))
require(p.returncode==0,'CPU collector subprocess failed')
proof=json.loads((OUT/'cpu-evidence.json').read_text())
require(proof.get('status')=='CPU_EVIDENCE_COLLECTED_NOT_INDEPENDENTLY_REVIEWED_NOT_RELEASE' and proof.get('issues')==[],'actual CPU evidence incomplete; preserve output')
