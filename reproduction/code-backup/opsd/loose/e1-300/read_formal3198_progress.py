import pathlib,json,hashlib,datetime
H=pathlib.Path(__file__).resolve().parent
ns={'__file__':str(H/'deploy_formal_reviewed.py')};s=pathlib.Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns)
r=ns['remote']('cat /mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3198/train.log');rows=[];ev=[]
for line in r.stdout.decode().splitlines():
 if 'OPSD_E1_OUTER_JSON=' in line:rows.append(json.loads(line.split('OPSD_E1_OUTER_JSON=',1)[1]))
 if 'BASELINE_EVALUATION_METRICS_JSON=' in line:ev.append(json.loads(line.split('BASELINE_EVALUATION_METRICS_JSON=',1)[1]))
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');digest=hashlib.sha256(r.stdout).hexdigest();p=H/('formal3198-observed-'+str(len(rows))+'-'+stamp+'-'+digest[:12]+'-original.log');
with p.open('xb') as f:f.write(r.stdout)
for v in rows:
 m=v['metrics'];selected={k:m[k] for k in m if k.startswith('timing_s/') or k in ['actor/opsd_loss','actor/grad_norm','actor/lr','critic/rewards/mean','critic/score/mean','info/finish_ratio','info/valid_action_ratio','actor/optimizer_steps','actor/samples'] or k.startswith('actor/optimizer_steps/rank_')}
 print(json.dumps({'outer':v['actual_outer_updates'],'pid':v['driver_pid'],'source':v['source_commit'],'metrics':selected}))
print(json.dumps({'initial_evaluation':ev[0] if ev else None,'actual_count':len(rows),'raw_file':str(p)}))
(H/'formal3198-current-outer-records.json').write_text(json.dumps(rows,indent=2)+'\n')
