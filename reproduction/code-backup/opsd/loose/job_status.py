import glob,json,re,datetime
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-acceptance-2875')
files=glob.glob('/tmp/o4-2875/ray/session_latest/logs/worker-*-3072361.out')
assert len(files)==1,files
text=Path(files[0]).read_text(errors='replace')
text=re.sub(r'\x1b\[[0-9;]*m','',text)
steps=sorted(set(int(v) for v in re.findall(r'(?:^|\s)step:(\d+) - ',text) if int(v)>0))
exitfile=root/'exit.txt'
print(json.dumps(dict(job_id='2875',sample_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_outer_steps=steps,exit_code=exitfile.read_text().strip() if exitfile.exists() else None,training_finished=(root/'training-finished.txt').exists())))
