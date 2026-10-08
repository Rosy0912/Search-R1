"""Local original-log audit; no remote access or training control."""
import hashlib, json, os, types
from pathlib import Path
from datetime import datetime, timezone
B = Path(__file__).resolve().parent
SNAP = B / 'actual-snapshots/3261-20261005T225124138829Z'
PINS = {
 'common': '29592450920eb6bc4f556138f4d0a73757bf3fc05703f7860c3b2f69c02c0295',
 'gate': 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb',
 'config': '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'}
def raw(p, pin=None):
 assert p.resolve(strict=True) == p
 a=p.stat()
 with p.open('rb') as f:
  c=os.fstat(f.fileno()); r=f.read(); d=os.fstat(f.fileno())
 z=p.stat(); stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 assert stamp(a)==stamp(c)==stamp(d)==stamp(z)
 h=hashlib.sha256(r).hexdigest()
 assert pin is None or h==pin, str(p)
 return r,h
def module(p,pin):
 r,h=raw(p,pin);m=types.ModuleType('independent_'+p.stem);m.__file__=str(p)
 exec(compile(r,str(p),'exec',dont_inherit=True),m.__dict__);return m
common=module(B/'final-acceptance-f5c72c942641/author200_observation.py',PINS['common'])
gate=module(B.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py',PINS['gate'])
summary_raw,summary_sha=raw(SNAP/'observation-summary.json');summary=common.loads(summary_raw)
_,transport_sha=raw(SNAP/'observed-remote-original.json',summary['transport_original_sha256'])
assert summary['expected_source']==common.SOURCE and summary['job_id']=='3261' and summary['node']=='h100-10'
files={}
for name,rec in summary['files'].items():
 p=SNAP/'original-files'/name;assert p.is_relative_to(SNAP/'original-files')
 r,h=raw(p,rec['sha256']);assert len(r)==rec['bytes'];files[name]=(r,h)
assert len(files)==263
def j(name):return common.loads(files[name][0])
cfg=j('full_state/global_step_5/resolved-config.json');assert files['full_state/global_step_5/resolved-config.json'][1]==PINS['config']
assert files['bindings/data_manifest.json'][1]==common.DATA_SHA
assert summary['commands']['source_head']['stdout'].strip()==common.SOURCE
assert summary['commands']['source_porcelain']['stdout']==''
assert summary['commands']['source_head']['returncode']==summary['commands']['source_porcelain']['returncode']==0
workers=[]
for rank in range(8):
 r5=j(f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json')
 r50=j(f'checkpoints/actor/global_step_50/training_state/rank_{rank}.json')
 assert r50['rank']==rank and r50['step']==r50['scheduler_last_epoch']==50
 assert r50['status']=='FULL_STATE_SERIALIZED_OK'
 assert r50['pid']==r5['pid'] and r50['object_ids']==r5['object_ids']
 p=summary['processes'][str(r50['pid'])]
 assert p['in_observed_job'] is True and p['uids']==[0]*4 and '/job_3261/step_0' in p['cgroup']
 workers.append({'rank':rank,'pid':r50['pid'],'start_ticks':p['start_ticks']})
dp=summary['processes']['1533312'];assert dp['start_ticks']=='1829085316' and dp['in_observed_job'] is True
receipt=j('acceptance-5-receipt.json');assert files['acceptance-5-receipt.json'][1]==common.RECEIPT5_SHA
gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],j('bindings/public.json'))
closure=j('bindings/import_closure.json')
context={'gate':gate,'workers':workers,'checkpoint_warning_path':Path(closure['modules']['torch.utils._contextlib']['path']).parent/'checkpoint.py'}
streams={};boundaries={}
for name in ('ray-stdout-pid-1533312.out','train.log'):
 rec=summary['files'][name];assert rec['source_offset']==0 and rec['complete_original_at_start'] is True and rec['unchanged_during_read'] is True
 data=files[name][0];end=data.rfind(b'\n')+1;assert end>0
 streams[name]=common.parse_records(data[:end],context)
 boundaries[name]={'bytes':end,'sha256':hashlib.sha256(data[:end]).hexdigest(),'unaccepted_tail_bytes':len(data)-end}
driver=streams['ray-stdout-pid-1533312.out'];aggregate=streams['train.log']
join=common.join_log_prefixes(driver,aggregate,context,50)
assert join['authoritative_last_outer']==join['aggregate_last_outer']==50
evaluation=common.check_evaluation_sequence(driver,50)
assert common.canonical(aggregate['evaluations'])==common.canonical(driver['evaluations'])
assert [(e['kind'],e['step'],e['count']) for e in evaluation['events']]==[('dev',0,512),('stage',5,64),('dev',50,512),('stage',50,64)]
pair={}
for kind, before_step in (('dev',0),('stage',5)):
 a=next(e for e in evaluation['events'] if e['kind']==kind and e['step']==before_step)
 z=next(e for e in evaluation['events'] if e['kind']==kind and e['step']==50)
 assert a['source_counts']==z['source_counts'] and a['count']==z['count']
 pair[kind]={'before_step':before_step,'after_step':50,'count':a['count'],'source_counts':a['source_counts'],
  'fixed_configured_path':cfg['data']['val_files'] if kind=='dev' else cfg['baselines']['opsd_e1']['stage_files'],
  'metrics':{k:{'before':a['original_metrics']['val/'+k+'/all'],'after':z['original_metrics']['val/'+k+'/all'],
    'delta':z['original_metrics']['val/'+k+'/all']-a['original_metrics']['val/'+k+'/all']}
   for k in ('em','f1','format_rate','answer_format','generated_tokens','search_calls')}}
rows=driver['outers'];adam=[sum(r['metrics']['actor/optimizer_steps/rank_'+str(rank)] for r in rows) for rank in range(8)]
assert adam==[800]*8
last=rows[-1]['metrics']
report={'version':1,'status':'PASS_ACTUAL50_FIXED_LOG_AND_EVALUATION_BINDING_WITH_NEGATIVE_EFFECT_NOT_TENSOR_CONTENT',
 'reviewer':'author200_review','utc':datetime.now(timezone.utc).isoformat(),'snapshot':str(SNAP),'summary_sha256':summary_sha,
 'transport_sha256':transport_sha,'actual_verified_original_count':263,'source_commit':common.SOURCE,'config_sha256':PINS['config'],
 'data_manifest_sha256':common.DATA_SHA,'same_driver_pid':1533312,'same_driver_start_ticks':1829085316,'workers':workers,
 'complete_line_boundaries':boundaries,'log_join':join,'actual_outer_count':50,'actual_adam_calls_each_rank':adam,
 'all8rank_finite_grad_loss_and_teacherCUDA_clocks_constantLR_checked':True,'evaluations':evaluation,'same_fixed_set_pairs':pair,
 'last_training_reward':last['critic/rewards/mean'],'last_finish_ratio':last['env/finish_ratio'],'last_legal_action_ratio':last['env/ratio_of_valid_action'],
 'scientific_result':'Same512 dev metrics declined; separate same64 stage EM improved. Infrastructure progress is not evidence of effect improvement.',
 'limitations':['Configured fixed manifest/path/source-count binding, no independent Arrow question-ID reread or per-question generation rescoring.',
  'Training rewards change question cohort each outer; do not estimate paired reward improvement.',
  'Saved50 rank/driver status remains SERIALIZED metadata; no50 PT/Adam/HF tensor content or new-process restore audit.',
  'Actual25 repair completion remains independently pending; no GPU, SSH, job control, source modification, signature or terminal200 review performed.']}
print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
