import pathlib,json
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177');stage=None;rows=[]
for line in (r/'train.log').read_text().splitlines():
 if 'BASELINE_EVALUATION_METRICS_JSON=' in line:
  x=json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]);
  if x.get('val/qa_count/all')==64:stage=x
 if 'BAPO_TRAIN_METRICS_JSON=' in line:
  t=json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]);
  if 50<=t['step']<=56:rows.append({'outer':t['step'],'stage64':stage,'outer_stage':t['metrics'].get('bapo/stage_validation')})
print(json.dumps({'stage_records':rows,'log_read_only':True}))
