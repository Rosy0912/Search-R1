"""Small read-only progress snapshot; never CUDA/import-model/signal/modify nodes."""
import pathlib,json,time,hashlib
S=pathlib.Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
def read(p):return json.loads(p.read_text())
def main():
 v=dict(observed_unix=time.time(),target='HKUST5',groups={},formal_test_slots=0)
 for g in ['G_POS','G_NEG','H_POS','H_NEG']:
  root=S/('group-'+g)/'runs';ledger=root/'training-em-25-complete.json';summaries=list((root/'outers').glob('outer-*/summary.json'))
  rows=read(ledger)['rows'] if ledger.exists() else [read(p) for p in sorted(summaries)]
  v['groups'][g]=dict(completed_outer=len(rows),optimizer_updates=rows[-1]['optimizer_updates'] if rows else 0,pair_exposure=rows[-1]['cumulative_pair_exposure'] if rows else 0)
  if rows:v['groups'][g]['last_summary']={k:rows[-1][k] for k in ['outer_index','train_correct','train_denominator','train_em','optimizer_updates','raw_checkpoint']}
  status=root/'status.json'
  if status.exists():
   t=read(status);v['groups'][g]['status']={k:t.get(k) for k in ['phase','outer_index','error','owner','updated_at_unix']}
  for p in sorted((root/'outers').glob('outer-*/sampling-attempt-*')):
   if (p.parent/'summary.json').exists():continue
   names=('rollouts-','generation-trace-','http-requests-')
   v['groups'][g].setdefault('pending_attempts',[]).append(dict(path=str(p),bytes={x.name:x.stat().st_size for x in p.iterdir() if x.is_file() and x.name.startswith(names)},last_worker_logs={x.name:x.read_text()[-500:] for x in p.glob('worker-*.log')}))
  memory=S/('group-'+g)/'receipts/migration-memory-live.json'
  if memory.exists():
   m=read(memory);v['groups'][g]['memory']=dict(age_seconds=time.time()-m['current']['sample_unix'],current=m['current'],samples=m['samples'],below=m['samples_below_limit'],minimum=m['minimum_actual_free_mib'])
 controller=S/'receipts/migration-live-controller.json'
 if controller.exists():
  r=read(controller);identity=r['identity'];p=pathlib.Path('/proc',str(identity['pid']));v['controller']=r
  if p.exists():
   t=(p/'stat').read_text().rsplit(')',1)[1].split();v['controller']['actual_kernel']=dict(state=t[0],starttick=int(t[19]),identity_matches=int(t[19])==identity['starttick'])
  else:v['controller']['actual_kernel']=dict(absent=True)
 v['completed_total']=sum(x['completed_outer'] for x in v['groups'].values());print(json.dumps(v,sort_keys=True))
if __name__=='__main__':main()
