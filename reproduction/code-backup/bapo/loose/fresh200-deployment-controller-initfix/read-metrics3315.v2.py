from pathlib import Path
import json,datetime,subprocess,hashlib
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
r=json.loads((ROOT/'scheduler-head/formal200-receipt.json').read_text());assert r['source']==SOURCE and str(r['job_id'])=='3315' and r['nonce']=='bab8981d9fd14afc98923eb4ebb89e33'
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
run=ROOT/'runs'/('formal200-'+r['nonce']+'-3315')
markers=['BAPO_TRAIN_METRICS_JSON=','BASELINE_EVALUATION_METRICS_JSON=','FRESH200_STEP5_PAUSED_JSON=','FRESH200_COMPLETED_RUN_JSON=','SEARCH_R1_FINAL_METRICS_JSON=']
records={k:[] for k in markers}
with (run/'train.log').open() as f:
 for line in f:
  for marker in markers:
   if marker in line:
    try:value=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
    except ValueError:continue
    records[marker].append(value)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=SOURCE,job_id=3315,run_root=str(run),counts={k:len(v)for k,v in records.items()},latest={k:v[-3:]for k,v in records.items()}),allow_nan=False))
