import argparse,json,pathlib,re,datetime
p=argparse.ArgumentParser();p.add_argument('--outer',type=int,required=True);args=p.parse_args()
log=pathlib.Path('outputs/remote-evidence/r1-standard32b-direct100-3228-readonly.jsonl')
with log.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-2000000));rows=f.read().splitlines()
a=None
for row in rows:
 try:x=json.loads(row)
 except (json.JSONDecodeError,UnicodeDecodeError):continue
 for r in x.get('files',[]):
  if r['relative_path']=='../actual-outer.json' and 'data' in r and r['data']['actual_completed_outer_steps']==list(range(1,args.outer+1)):a=r;epoch=x['epoch'];control=x['scontrol']['stdout']
assert a is not None
pathlib.Path(f'outputs/standard3228-outer{args.outer}-watch-event.json').write_text(json.dumps({'scope':'DERIVED_EXISTING_WATCH_NOT_ORIGINAL_BYTE_COPY','epoch':epoch,'parent_actual_outer':a},ensure_ascii=False,indent=2)+'\n')
metrics=a['data']['records'][-1]['metrics'];costs={k:v for k,v in metrics.items() if k.startswith('timing_s/')}
p=pathlib.Path('outputs/STATE.json');s=json.loads(p.read_text());s['current_direct_teacher_run'].update(actual_completed_outer=args.outer,status=re.search(r'JobState=(\S+)',control).group(1),parent_actual_outer_ref={'path':a['original_path'],'sha256':a['sha256'],'size':a['size']},actual_slurm_snapshot_epoch=epoch,actual_slurm_runtime=re.search(r'RunTime=(\S+)',control).group(1),latest_outer_cost_seconds=costs,next_outer_lr=metrics['actor/lr']);saved=max([5]+[int(re.search(r'checkpoint(\d+)-',q.name).group(1)) for q in pathlib.Path('outputs').glob('standard3228-checkpoint*-live-presence.json') if int(re.search(r'checkpoint(\d+)-',q.name).group(1))<=args.outer]);s['current_status']=f'TEACHER3228_ACTUAL{args.outer}_100_HF{saved}_SAVED_STUDENT3224_WAITING';s['last_updated']=datetime.datetime.now().astimezone().isoformat();p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
rep=pathlib.Path('outputs/REPORT.md');rep.write_text(f'3228 actual [1..{args.outer}], SHA'+a['sha256']+f'. Running{args.outer}/100; physical HF{saved} retained. Parent receipt is not a new independent all8rank/fullmodel review. Student3224 still waitsafterok3228, TOPD0/300. No duplicate/new gate/algorithm change.\n\n'+rep.read_text())
m=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/topd');(m/'STATE.json').write_bytes(p.read_bytes());(m/'REPORT.md').write_bytes(rep.read_bytes());print(json.dumps({'outer':args.outer,'original_parent_sha256':a['sha256'],'costs':costs,'next_outer_lr':metrics['actor/lr']}))
