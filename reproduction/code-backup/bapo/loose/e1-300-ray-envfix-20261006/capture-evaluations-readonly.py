import pathlib,json,hashlib,base64
run=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/runs/bapo-instruct300-formal300-3292')
assert (run/'source-commit.txt').read_text().strip()=='a2568d0df306ead6e059760e28b5bd53ee3f6876'
rows=[];offset=0
with (run/'train.log').open('rb')as f:
 for line in f:
  if b'BASELINE_EVALUATION_METRICS_JSON='in line or b'SEARCH_R1_FINAL_METRICS_JSON='in line:
   rows.append(dict(offset=offset,bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),line_base64=base64.b64encode(line).decode()))
  offset+=len(line)
print(json.dumps(dict(job_id=3292,remote_path=str(run/'train.log'),records=rows)))
