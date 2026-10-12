import pathlib,json,hashlib,time,subprocess
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
run=r/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
selected=[run/'checkpoints/actor/global_step_150',run/'checkpoints/algorithm/global_step_150',run/'checkpoints/fresh200_state/outer_150_COMPLETE']
files=[]
for d in selected:
 for f in sorted(d.rglob('*')):
  if f.is_file():
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
   files.append(dict(path=str(f),relative=str(f.relative_to(run)),size=f.stat().st_size,sha256=h.hexdigest()))
logs=[]
for f in run.glob('*'):
 if f.is_file() and f.name in ['train.log','resolved-config.yaml','source-commit.txt','driver-time.log','human-stop-final200-20261012.json','original-exit-code.txt']:
  logs.append(dict(path=str(f),relative='logs/'+f.name,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
for ev in (r/'runs').glob('eval150-final51713-*'):
 for f in ev.glob('final*.json'):
  logs.append(dict(path=str(f),relative='logs/'+ev.name+'/'+f.name,size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
metrics=[]
for line in (run/'train.log').open(errors='replace'):
 marker='BAPO_TRAIN_METRICS_JSON='
 if marker in line:
  try:
   v=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
   m=v.get('metrics',{});n=m.get('val/qa_count/all');
   if n==512:metrics.append(dict(outer=v.get('attempt_id'),em=m.get('val/em/all'),count=n))
  except Exception:pass
result=dict(epoch=time.time(),host='h100-11',session='01a0fcbd-41d8-7910-a410-372ce5c8870c',repo='Sutoonq/TMP_MODEL',prefix='h100-cleanup-20261012/bapo/',selected_outer=150,selection_basis='Highest 512 question EM among saved trained checkpoints',metrics=metrics,records=files+logs,total_bytes=sum(f['size'] for f in files+logs),preserve_latest_outer=200)
(r/'cleanup-best150-manifest-20261012.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
