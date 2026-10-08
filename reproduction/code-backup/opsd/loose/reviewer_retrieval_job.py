import subprocess,shlex,json,sys,hashlib,math
from pathlib import Path
job=sys.argv[1]
code="""from pathlib import Path
import json,hashlib,subprocess
job=JOB;p=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-formal-'+job);f=p/'evidence/retrieval-comparison.json'
s=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True)
x={'job':job,'slurm':s.stdout,'slurm_rc':s.returncode,'exists':f.exists()}
if f.exists():
 b=f.read_bytes();x.update(raw=b.decode(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
print(json.dumps(x))
""".replace('JOB',repr(job))
r=subprocess.run(['python3','work/remote.py','python3 -c '+shlex.quote(code)],capture_output=True,text=True)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
x=json.loads(r.stdout);p=Path('work/opsd/reproduction/baselines/opsd-validation-20261003/evidence');(p/('job'+job+'_retrieval_current_reviewer_fetch.json')).write_text(r.stdout)
print(x['slurm']);print('exists',x['exists'])
if not x['exists']:raise SystemExit(3)
y=json.loads(x['raw']);old=json.loads((p/'job2994_retrieval_comparison_reviewer_raw.json').read_text());assert y['queries']==old['queries'] and len(y['queries'])==8
assert y['index_vectors']==21015324 and y['index_dimension']==768
out=[]
for i,q in enumerate(y['queries']):
 a=y['cpu_ids'][i];b=y['gpu_ids'][i];assert len(a)==len(b)==50 and len(set(a))==len(set(b))==50
 assert a[:3]==b[:3] and set(a)==set(b)
 assert y['content']['3']['cpu'][i]==y['content']['3']['gpu'][i]
 ca=dict(zip(a,y['content']['50']['cpu'][i]));cb=dict(zip(b,y['content']['50']['gpu'][i]));assert ca==cb
 sa=y['cpu_scores'][i];sb=y['gpu_scores'][i];assert all(math.isfinite(v) for v in sa+sb)
 assert all(v>=w for v,w in zip(sa,sa[1:])) and all(v>=w for v,w in zip(sb,sb[1:]))
 ma=dict(zip(a,sa));mb=dict(zip(b,sb));delta=max(abs(ma[k]-mb[k]) for k in ma);assert delta<.001
 out.append({'query':q,'top3_ids':a[:3],'top50_id_content_set_exact':True,'matched_score_max_abs_delta':delta,'tail_positions_moved':sum(v!=w for v,w in zip(a,b))})
(p/('job'+job+'_retrieval_current_independent_analysis.json')).write_text(json.dumps({'job_id':job,'comparison_sha256':x['sha256'],'raw_bytes':x['bytes'],'current_pretraining_retrieval_scope_accepted':True,'queries':out},indent=2)+'\n')
print('CURRENT_COMPARISON_INDEPENDENT_PASS',x['sha256'],x['bytes'],out)
