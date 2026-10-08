import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib,json,socket,subprocess,time
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');P=R/'instruct300/candidates/safety110-a632ba9-84e3e186';H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
assert socket.gethostname().split('.')[0]=='h100-9'; mh=H(P/'manifest.json');assert mh=='84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48';m=json.loads((P/'manifest.json').read_bytes())
for n,s in m['files'].items():assert H(P/n)==s,n
for n,s in m['external_files'].items():assert H(n)==s,n
assert H(m['parent_manifest']['path'])==m['parent_manifest']['sha256']
parent=json.loads(Path(m['parent_manifest']['path']).read_bytes());assert parent['identity']==m['parent_manifest']['identity'];assert parent['worker_states']==m['healthy110_metadata']['worker_states'];assert parent['driver']==m['healthy110_metadata']['driver'];assert parent['hf_identity']==m['healthy110_metadata']['hf_identity']
prior=json.loads((R/'runs/tree-recovery300-3234/run-result.json').read_bytes());assert prior['status']=='FAILED' and prior['cleanup_residual']=={} and prior['job_id']=='3234'
protected={**{p:'1502352181' for p in (1680782,1680783,1680784,1680785)},**{p:'1502398762' for p in (1696443,1696444,1696445,1696446)}}
for pid,t in protected.items():
 f=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split();assert f[19]==t and f[0] not in ('Z','X')
active=subprocess.check_output(['squeue','-h','-w','h100-9','-o','%i %j %T'],text=True)
for n in ('tree-instruct300','tree-recovery300','tree-safety300','tree-safety110-calibration','tree-policy-safety-native'):assert n not in active
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'repo',text=True).strip()=='4d133efeb1f675139d2558640f1e21d50780f133';assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'repo',text=True).strip()
print(json.dumps(dict(status='INDEPENDENT_H1009_METADATA_PRE_DISPATCH_PASS',epoch=time.time(),manifest_sha256=mh,files_verified=len(m['files']),external_pins_verified=len(m['external_files']),parent_manifest_sha256=H(m['parent_manifest']['path']),parent_step=110,prior_failed_cleanup_empty=True,protected_pid_starttimes=protected,squeue=active,wall_seconds=time.monotonic()-start,actual_gpu_used=False,actual_job_submitted=False,parent_large_weights_not_reread=True)))
