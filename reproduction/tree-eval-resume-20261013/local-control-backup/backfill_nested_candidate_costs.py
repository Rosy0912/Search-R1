#!/usr/bin/env python3
"""Fixed-file H10018 candidate-cost collection; only credited, complete scalar output."""
import argparse,hashlib,importlib.util,json,math,os,shlex,time
from pathlib import Path
REMOTE = "import collections,hashlib,json,math,os,time\nfrom pathlib import Path\npin=PIN_DATA\nos.nice(15)\nstarted=time.time();cpu_started=time.process_time()\nfields=['new_generated_tokens','new_retained_model_tokens','new_prefill_tokens','retrieval_calls','scoring_calls','inherited_response_tokens','inherited_observation_tokens','inherited_raw_ledger_actions','generation_wall_seconds_amortized','scoring_wall_seconds_amortized']\nledger_fields=['generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_slots','raw_emitted_tokens','raw_retained_tokens','raw_discarded_tail_tokens','raw_inherited_leaf_tokens','semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds','scoring_calls','scoring_wall_seconds']\ndef finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)\ndef sums(rows,keys,getter=lambda row,key:row.get(key)):\n result={};known={};counts={}\n for key in keys:\n  values=[getter(row,key) for row in rows];good=[v for v in values if finite(v)]\n  counts[key]=len(good);known[key]=math.fsum(good) if good or not rows else None\n  result[key]=math.fsum(good) if len(good)==len(rows) else None\n  for dest in [result,known]:\n   if isinstance(dest[key],float) and dest[key].is_integer():dest[key]=int(dest[key])\n return result,known,counts\ndef summarize(rows):\n total,known,counts=sums(rows,fields)\n return {'candidate_count':len(rows),'field_sums':total,'field_known_sums':known,'field_record_counts':counts}\nchecks={'all':lambda x:True,'accepted':lambda x:x.get('accepted') is True,'discarded':lambda x:x.get('accepted') is False,\n 'fallback':lambda x:x.get('fallback') is True,'accepted_without_fallback':lambda x:x.get('accepted') is True and x.get('fallback') is not True,\n 'accepted_fallback':lambda x:x.get('accepted') is True and x.get('fallback') is True,\n 'unknown_outcome':lambda x:x.get('accepted') is not True and x.get('accepted') is not False}\ndef grouping(rows):return {key:summarize([row for row in rows if check(row)]) for key,check in checks.items()}\ndef ledger(rounds):return sums(rounds,ledger_fields,lambda row,key:row.get('cost_delta',{}).get(key))\ndef elapsed(rounds,key):\n vals=[row.get(key) for row in rounds]\n return math.fsum(vals) if all(finite(v) for v in vals) else None\ndef raw_meta(path,raw,before,after):\n return {'path':str(path),'resolved_path':str(path.resolve(strict=True)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns_before':before.st_mtime_ns,'mtime_ns_after':after.st_mtime_ns,'stable_read':before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns}\nrun=Path(pin['run']);audit=run/pin['audit']['path'];result={'summary_schema':'tree_candidate_cost_backfill_compact_v1','remote_observed_start_epoch':started,'source_commit':pin['source_commit'],'config_sha256':pin['config_sha256'],'unit':pin['unit'],'run_directory':str(run),'requested_outer':pin['outer'],'experiment_origin':pin.get('experiment_origin'),'origin_scope':pin.get('origin_scope'),'completed_rollout_count':None,'completed_rollout_steps':[],'rollouts':[],'skipped_audits':[],'source_pin':pin,'source_files':{},'remote_nice':os.getpriority(os.PRIO_PROCESS,0),'no_directory_traversal':True,'remote_files_written':False,'tensor_or_model_contents_read':False,'rewards_or_variances_recomputed':False}\ntry:\n before=audit.stat()\n if before.st_size>100000000:raise ValueError('Audit exceeds fixed one-file scalar aggregation budget')\n raw=audit.read_bytes();after=audit.stat();d=json.loads(raw);result['source_files']['audit']=raw_meta(audit,raw,before,after)\n if not result['source_files']['audit']['stable_read'] or len(raw)!=pin['audit']['bytes'] or before.st_mtime!=pin['audit']['mtime']:raise ValueError('Pinned completed audit bytes changed')\n if d.get('status')!='ROLLOUT_DIAGNOSTIC_COMPLETE' or d.get('global_step')!=pin['outer'] or d.get('attempt_id')!=pin['attempt_id']:raise ValueError('Selected file is not the expected completed outer/attempt')\n candidates=d.get('candidate_attempts');rounds=d.get('candidate_rounds')\n if not isinstance(candidates,list) or not isinstance(rounds,list):raise ValueError('Per-candidate/round records absent;costs stay unknown')\n costpath=run/'output/training-rollout-cost'/('attempt-%06d.json'%pin['attempt_id']);cb=costpath.stat();craw=costpath.read_bytes();ca=costpath.stat();cc=json.loads(craw);result['source_files']['controller']=raw_meta(costpath,craw,cb,ca)\n if not result['source_files']['controller']['stable_read'] or cc.get('status')!='TRAINING_ROLLOUT_COST_COMPLETE' or cc.get('candidate_outer')!=pin['outer'] or cc.get('code_commit')!=pin['source_commit'] or cc.get('config_sha256')!=pin['config_sha256']:raise ValueError('Actual controller source/config/outer identity does not match fixed pin')\n configpath=run/'resolved.yaml';qb=configpath.stat();qraw=configpath.read_bytes();qa=configpath.stat();result['source_files']['resolved_config']=raw_meta(configpath,qraw,qb,qa)\n if not result['source_files']['resolved_config']['stable_read'] or result['source_files']['resolved_config']['sha256']!=pin['config_sha256']:raise ValueError('Actual resolved configuration SHA differs')\n unique=[];seen=set();duplicates=0;missing_ids=0\n for ordinal,rr in enumerate(rounds):\n  identity=rr.get('round_id')\n  if identity is None:missing_ids+=1;unique.append(rr);continue\n  if identity in seen:duplicates+=1;continue\n  seen.add(identity);unique.append(rr)\n z={'audit_file':audit.name,'audit_file_bytes':after.st_size,'audit_file_mtime':after.st_mtime,'audit_file_sha256':result['source_files']['audit']['sha256'],'schema_version':d.get('schema_version'),'global_step':d.get('global_step'),'rollout_sequence':d.get('rollout_sequence'),'attempt_id':d.get('attempt_id'),'status':d.get('status'),'audit_algorithm_source_commit':d.get('algorithm_source_commit'),'audit_base_commit':d.get('base_commit'),'runtime_source_commit':pin['source_commit'],'config_sha256':pin['config_sha256'],'question_count':len(d['question_groups']) if isinstance(d.get('question_groups'),list) else None,'training_leaf_count':len(d['leaves']) if isinstance(d.get('leaves'),list) else None,'accepted_branch_count':len(d['branches']) if isinstance(d.get('branches'),list) else None,'original_root_leaf_count':sum(row.get('branch_sampling',{}).get('continuation')==0 for row in d['leaves']) if isinstance(d.get('leaves'),list) else None,'root_leaf_quotas':d.get('root_leaf_quotas'),'candidate_outcomes':grouping(candidates),'stages':{},'rounds':[],'duplicate_round_ids_excluded':duplicates,'rounds_missing_ids':missing_ids,'candidate_timing_scopes':dict(collections.Counter(row.get('timing_scope') for row in candidates)),'padding_cost_by_candidate_outcome':None,'fallback_overlap':{'accepted':sum(row.get('fallback') is True and row.get('accepted') is True for row in candidates),'discarded':sum(row.get('fallback') is True and row.get('accepted') is False for row in candidates),'unknown_accepted_flag':sum(row.get('fallback') is True and row.get('accepted') is not True and row.get('accepted') is not False for row in candidates)}}\n for stage in sorted({row.get('stage','UNKNOWN') for row in candidates}|{row.get('stage','UNKNOWN') for row in unique}):\n  cr=[row for row in candidates if row.get('stage')==stage];rr=[row for row in unique if row.get('stage')==stage];ls,lk,lc=ledger(rr)\n  z['stages'][stage]={'candidate_outcomes':grouping(cr),'round_count':len(rr),'generation_round_elapsed_seconds':elapsed(rr,'generation_wall_seconds') if not missing_ids else None,'scoring_round_elapsed_seconds':elapsed(rr,'scoring_wall_seconds') if not missing_ids else None,'round_controller_ledger':ls,'round_ledger_known_sums':lk,'round_ledger_field_record_counts':lc,'round_elapsed_by_outcome_seconds':None}\n for ordinal,rr in enumerate(unique,1):\n  cr=[row for row in candidates if row.get('round_id')==rr.get('round_id')] if rr.get('round_id') is not None else []\n  z['rounds'].append({'ordinal':ordinal,'stage':rr.get('stage'),'candidate_count':rr.get('candidate_count'),'accepted_count':sum(row.get('accepted') is True for row in cr) if rr.get('round_id') is not None else None,'discarded_count':sum(row.get('accepted') is False for row in cr) if rr.get('round_id') is not None else None,'fallback_count':sum(row.get('fallback') is True for row in cr) if rr.get('round_id') is not None else None,'generation_elapsed_seconds':rr.get('generation_wall_seconds'),'scoring_elapsed_seconds':rr.get('scoring_wall_seconds'),'controller_ledger':{key:rr.get('cost_delta',{}).get(key) for key in ledger_fields}})\n ls,lk,lc=ledger(unique);z['candidate_round_controller_ledger']=ls;z['candidate_round_ledger_known_sums']=lk;z['candidate_round_ledger_field_record_counts']=lc\n z['generation_candidate_round_elapsed_seconds']=elapsed(unique,'generation_wall_seconds') if not missing_ids else None\n z['scoring_candidate_round_elapsed_seconds']=elapsed(unique,'scoring_wall_seconds') if not missing_ids else None\n z['recorded_round_minus_candidate_residual']={key:ls[key]-z['candidate_outcomes']['all']['field_sums'][ck] if finite(ls[key]) and finite(z['candidate_outcomes']['all']['field_sums'][ck]) else None for key,ck in [('generated_tokens','new_generated_tokens'),('prefill_tokens','new_prefill_tokens'),('retrieval_calls','retrieval_calls'),('scoring_calls','scoring_calls')]}\n z['rollout_controller_cost']={key:cc.get(key) for key in ['status','attempt_id','candidate_outer','completed_outer_before','question_rows','costs','start_epoch','end_epoch','wall_seconds','trainer_process_cpu_seconds','code_commit','config_sha256','data_identity','model_identity','scope']}\n result.update(status='ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED',completed_rollout_count=1,completed_rollout_steps=[pin['outer']],rollouts=[z],schema_observed={'candidate_fields':sorted({key for row in candidates for key in row}),'round_fields':sorted({key for row in rounds for key in row}),'round_cost_delta_fields':sorted({key for row in rounds for key in row.get('cost_delta',{})})})\nexcept (OSError,ValueError,KeyError,TypeError) as exc:\n result.update(status='COST_FIELDS_UNKNOWN_SOURCE_UNAVAILABLE_OR_NOT_BOUND',error_type=type(exc).__name__,error=str(exc),completed_rollout_count=None)\n result['skipped_audits'].append({'path':str(audit),'reason':str(exc),'costs':None})\nresult['remote_observed_end_epoch']=time.time();result['collector_process_cpu_seconds']=time.process_time()-cpu_started;result['collector_wall_seconds']=time.time()-started\nresult['collector_cost_scope']='This read-only aggregation process only;not training service/outer CPU and not added to training costs.'\nencoded=json.dumps(result,ensure_ascii=False,separators=(',',':'))\nif len(encoded.encode())>90000:raise RuntimeError('Small scalar result exceeds90KB;no raw rows emitted')\nprint(encoded)\n"

def digest(raw):return hashlib.sha256(raw).hexdigest()
def small(value):
 raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode()
 if len(raw)>100000:raise ValueError('Scalar summary exceeds100KB; no fields discarded')
 return raw
def publish(base,mirrors,name,raw):
 for folder in [base]+mirrors:
  p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
def numeric(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def total(values):
 return {'total':math.fsum(values) if all(numeric(v) for v in values) else None,'known_sum':math.fsum(v for v in values if numeric(v)),'known_count':sum(numeric(v) for v in values),'required_count':len(values)}
def legacy_main(args):
 base=args.base.resolve();pinraw=args.pins.read_bytes();pins=json.loads(pinraw)
 allowed={'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4':range(3,6),'778b2b43dad90fc0c1c5d95d77c2326286011b73':range(6,20)}
 if sorted(p['outer'] for p in pins)!=list(range(3,20)):raise ValueError('Only fixed completed3..19 backfill authorized')
 for p in pins:
  if p['source_commit'] not in allowed or p['outer'] not in allowed[p['source_commit']]:raise ValueError('Unexpected task source/outer')
  if p['run']!='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+p['source_commit']:raise ValueError('Unexpected own run path')
  if Path(p['audit']['path']).parent.as_posix()!='output/tree-audit':raise ValueError('Unexpected audit file path')
  raw=(base/p['source_poll']).read_bytes()
  if digest(raw)!=p['source_poll_sha256']:raise ValueError('Fixed local source poll differs')
 index=json.loads((base/'PER_OUTER_METRICS_COMPACT.json').read_text());mirrors=[Path(p) for p in index['mirrors']]
 spec=importlib.util.spec_from_file_location('retained18_cost_route',args.ssh_helper);ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
 script_sha=digest(Path(__file__).read_bytes());records=[];parts=[]
 for pin in pins:
  name='candidate_cost_backfill/CANDIDATE_OUTER'+format(pin['outer'],'04d')+'.json';dest=base/name
  if dest.exists():
   item=json.loads(dest.read_text())
   if item.get('source_pin')!=pin:raise ValueError('Existing backfill part has different fixed source pin')
  else:
   source=REMOTE.replace('PIN_DATA',repr(pin));response=ssh.run('python3 -B -c '+shlex.quote(source),target='h100-10',timeout=60)
   if response.returncode:
    item={'status':'REMOTE_AGGREGATION_NOT_RETURNED_COST_UNKNOWN','requested_outer':pin['outer'],'source_pin':pin,'rollouts':[],'error':response.stderr[-1500:],'remote_returncode':response.returncode}
   else:item=json.loads(response.stdout)
   item['local_collector_sha256']=script_sha;item['remote_program_sha256']=digest(source.encode());item['fixed_pin_manifest_sha256']=digest(pinraw)
   item['interpretation']={'outcomes':'accepted/discarded are separate recorded flags;fallback overlaps accepted and is not an additive third bucket.','candidate_times':'Recorded equal amortized shares of parallel round elapsed,not independent outcome wall.','round_times':'Each original distinct round_id once;not multiplied by question count.','raw_round_residual':'Recorded round-vs-candidate residual remains unallocated;no outcome padding or wall split inferred.','inherited_prefix':'Repeated inherited token/action events,not unique generation or KV/FLOPs.','credit':'Backfill3..19 costs only;does not grant new outer credit or modify original1..19 book/cumulative.','absence':'A missing field has field_sums=null plus known_sum/record count,not zero. Observed empty outcome buckets have count0 and zero empty sums.'}
   publish(base,mirrors,name,small(item))
  raw=dest.read_bytes();parts.append({'outer':pin['outer'],'file':name,'bytes':len(raw),'sha256':digest(raw),'status':item['status'],'source_commit':pin['source_commit'],'config_sha256':pin['config_sha256'],'audit_file':pin['audit']['path']});records.append(item)
  r=item.get('rollouts',[None])[0] if item.get('rollouts') else None
  print(json.dumps({'outer':pin['outer'],'status':item['status'],'bytes':len(raw),'sha256':digest(raw),'candidate_count':r['candidate_outcomes']['all']['candidate_count'] if r else None,'actual_branch_scoring_calls':r['candidate_round_controller_ledger']['scoring_calls'] if r else None,'collector_cpu_seconds':item.get('collector_process_cpu_seconds')},ensure_ascii=False),flush=True)
 successful={x['requested_outer']:x['rollouts'][0] for x in records if x.get('status')=='ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED'}
 def values(fn):return [fn(successful[p['outer']]) if p['outer'] in successful else None for p in pins]
 rollup={'candidate_round_controller_ledger':{k:total(values(lambda r:r['candidate_round_controller_ledger'].get(k))) for k in ['generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_emitted_tokens','raw_retained_tokens','raw_discarded_tail_tokens','raw_inherited_leaf_tokens','scoring_calls','scoring_wall_seconds']},'candidate_outcomes':{}}
 for outcome in ['all','accepted','discarded','fallback','accepted_without_fallback','accepted_fallback','unknown_outcome']:
  rollup['candidate_outcomes'][outcome]={'candidate_count':total(values(lambda r:r['candidate_outcomes'][outcome]['candidate_count'])),'field_sums':{k:total(values(lambda r:r['candidate_outcomes'][outcome]['field_sums'][k])) for k in ['new_generated_tokens','new_retained_model_tokens','new_prefill_tokens','retrieval_calls','scoring_calls','inherited_response_tokens','inherited_observation_tokens','inherited_raw_ledger_actions','generation_wall_seconds_amortized','scoring_wall_seconds_amortized']}}
 result={'summary_schema':'tree_candidate_cost_backfill_index_v1','created_epoch':time.time(),'requested_completed_outers':list(range(3,20)),'actual_completed_cost_outers':sorted(successful),'unknown_outers':[p['outer'] for p in pins if p['outer'] not in successful],'parts':parts,'fixed_pin_manifest':{'file':'CANDIDATE_COST_BACKFILL_SOURCE_PINS_3_19.json','bytes':len(pinraw),'sha256':digest(pinraw)},'rollup3_through19':rollup,'missing_values_are_null':True,'old_books_or_through19_modified':False,'raw_rollout_or_tensor_transferred_to_Mac':False,'query_scope':'Fixed owned H10018 audit+controller+resolved config files only;serial nice15 CPU aggregation;no remote writes/scans/model/GPU/training calls.','cost_boundaries':'Per-outcome generation/scoring seconds are recorded amortized shares. Unique round/stage elapsed is separately recorded once. No padding allocation,real GPU cost,independent outcome wall,or reward/variance recomputation.','future_use':'Separate evidence supplement only. Original through19 and1..19 blocks remain unchanged. No automatic consumption into cost book in this one-off collector.'}
 publish(base,mirrors,'CANDIDATE_COST_BACKFILL_SOURCE_PINS_3_19.json',pinraw)
 raw=small(result);publish(base,mirrors,'CANDIDATE_COST_BACKFILL_THROUGH19.json',raw)
 lines=['# 已完成3–19轮候选成本补齐','', '只读取原H10018已固定文件，逐轮nice15低优先级CPU聚合；原始逐条账本仍留远端。原1–19块、旧through19、STATUS/REPORT不改。','', '|轮|source|候选/accepted/discarded/fallback重叠|候选新生成token|候选prefill|候选检索/评分|branch controller scoring次数/秒|唯一round生成/评分秒|','|---|---|---|---:|---:|---|---|---|']
 for pin in pins:
  r=successful.get(pin['outer'])
  if not r:lines.append('|'+str(pin['outer'])+'|'+pin['source_commit'][:8]+'|unknown|null|null|null|null|null|');continue
  c=r['candidate_outcomes'];s=c['all']['field_sums'];l=r['candidate_round_controller_ledger'];lines.append('|'+ '|'.join([str(pin['outer']),pin['source_commit'][:8],'/'.join(str(c[k]['candidate_count']) for k in ['all','accepted','discarded','fallback']),str(s['new_generated_tokens']),str(s['new_prefill_tokens']),str(s['retrieval_calls'])+'/'+str(s['scoring_calls']),str(l['scoring_calls'])+'/'+str(l['scoring_wall_seconds']),str(r['generation_candidate_round_elapsed_seconds'])+'/'+str(r['scoring_candidate_round_elapsed_seconds'])])+'|')
 lines+=['', '每轮JSON保留A1/B1/B2×accepted/discarded/fallback及其交叠、完整原scalar字段、coverage、每round原elapsed和native controller ledger，原文件SHA/字节/mtime及运行source/config绑定。缺失字段null；空bucket的观测0与缺失分开。','', '候选时间为原记录的并行round均分份额，不是独立候选/结果类别墙钟。round每ID只计一次，不乘512题。round包含DP padding的总账与candidate rows的差额仅列未分配residual，不猜padding分摊；继承prefix是重复事件量，不是新生成或唯一KV。','', '本摘要只补成本证据，不重算奖励/优势/方差、不授予训练进度，也不修改已冻结账本。新补齐的实际字段可由后续明确授权再接入本地工具。']
 publish(base,mirrors,'CANDIDATE_COST_BACKFILL_THROUGH19.md',('\n'.join(lines)+'\n').encode())
 print(json.dumps({'status':'BACKFILL_COMPLETE','known_outers':sorted(successful),'unknown_outers':result['unknown_outers'],'index_bytes':len(raw),'index_sha256':digest(raw),'actual_branch_scoring_calls3_19':rollup['candidate_round_controller_ledger']['scoring_calls'],'candidate_count3_19':rollup['candidate_outcomes']['all']['candidate_count']},ensure_ascii=False),flush=True)



SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + SOURCE
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
EXTENSIONS = 'CANDIDATE_COST_EXTENSION_INPUTS.json'

def fixed_publish(base,mirrors,name,raw):
 for folder in [base]+mirrors:
  path=folder/name
  if path.exists() and path.read_bytes()!=raw:raise ValueError('Immutable candidate evidence already exists with different bytes: '+str(path))
 publish(base,mirrors,name,raw)

def local_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def completed_pin(base,index,outer,local):
 rows=[r for r in index['effective_rows'] if r['outer']==outer and r.get('effective_history') is True]
 if not rows:return None
 if len(rows)!=1 or rows[0]['source_commit']!=SOURCE:raise ValueError('Credited outer does not have the fixed778 source')
 ref=rows[0];info=next(v for v in index['scalar_blocks'] if v['file']==ref['block_file']);raw=(base/info['file']).read_bytes()
 if len(raw)!=info['bytes'] or digest(raw)!=info['sha256']:raise ValueError('Credited local scalar block SHA/bytes differs')
 block=json.loads(raw);row=block['rows'][ref['block_row_index']]
 if row.get('outer')!=outer or row.get('source_commit')!=SOURCE:raise ValueError('Credited scalar row identity differs')
 full_name='FULL_OUTER'+str(outer)+'_PROGRESS_RECEIPT.json'
 if row.get('full_outer_evidence')!=full_name:raise ValueError('Credited row does not bind expected immutable FULL receipt')
 e=local.actual_inputs(base,base/full_name);full,poll=e['full'],e['poll']
 for name,binding in e['bindings'].items():
  if block['source_inputs'].get(name)!=binding:raise ValueError('Credited block source binding differs from immutable input')
 if e['source']!=SOURCE or full.get('config_sha256')!=CONFIG or full.get('run')!=RUN or full.get('unit')!=UNIT or poll.get('experiment_origin')!='fresh':raise ValueError('Expected own fixed778/config355 fresh-origin chain')
 controller=e['controller']
 if controller.get('status')!='TRAINING_ROLLOUT_COST_COMPLETE' or controller.get('candidate_outer')!=outer or controller.get('code_commit')!=SOURCE or controller.get('config_sha256')!=CONFIG:raise ValueError('No matching completed controller ledger')
 attempt=controller.get('attempt_id')
 if not isinstance(attempt,int) or attempt!=outer:raise ValueError('Unexpected fixed778 attempt mapping')
 import re
 choices=[x for x in poll.get('output_metadata',[]) if re.fullmatch('output/tree-audit/step_'+str(outer)+r'_rollout_\d+\.json',x.get('path',''))]
 if len(choices)!=1:raise ValueError('Expected exactly one pinned completed audit file')
 audit=choices[0]
 if not isinstance(audit.get('bytes'),int) or audit['bytes']<=0 or not numeric(audit.get('mtime')):raise ValueError('Pinned audit stat metadata incomplete')
 return {'outer':outer,'source_commit':SOURCE,'config_sha256':CONFIG,'run':RUN,'unit':UNIT,'attempt_id':attempt,'audit':audit,
  'source_poll':e['poll_name'],'source_poll_sha256':e['bindings'][e['poll_name']]['sha256'],
  'full_outer_receipt':full_name,'full_outer_receipt_sha256':e['bindings'][full_name]['sha256'],
  'native_metadata_readback':e['native_name'],'native_metadata_readback_sha256':e['bindings'][e['native_name']]['sha256'],
  'native_manifest_sha256':full['native_manifest_sha256'],'immutable_local_inputs':e['bindings'],
  'experiment_origin':'fresh','origin_scope':'Fresh experiment ancestry:0bef outer1,ba9 outer2..5,778 native5 continuation outer6 onward;not a new base-state reset or an exact executor equivalence claim.'}

def register(base,mirrors,info):
 path=base/EXTENSIONS
 registry=json.loads(path.read_bytes()) if path.exists() else {'summary_schema':'tree_candidate_cost_extension_inputs_v1','indexes':[],
  'scope':'Derived local input list only;each immutable per-outer index and scalar part is separately SHA bound. No training credit,remote call or historical book mutation.'}
 if registry.get('summary_schema')!='tree_candidate_cost_extension_inputs_v1':raise ValueError('Unexpected local candidate input list schema')
 old=[v for v in registry['indexes'] if v['outer']==info['outer']]
 if old and old!=[info]:raise ValueError('Existing registered outer has different immutable index')
 if not old:registry['indexes'].append(info);registry['indexes'].sort(key=lambda x:x['outer'])
 publish(base,mirrors,EXTENSIONS,small(registry))

def collect_extension(args):
 base=args.base.resolve(strict=True);index=json.loads((base/'PER_OUTER_METRICS_COMPACT.json').read_bytes());mirrors=[Path(x) for x in index['mirrors']]
 local=local_module(Path(__file__).with_name('append_nested_local_costs.py'),'local_cost_bindings');ssh=None;known=[];pending=[]
 for outer in range(args.first_outer,args.last_outer+1):
  pin=completed_pin(base,index,outer,local)
  if pin is None:
   pending.append(outer);print(json.dumps({'outer':outer,'status':'SKIPPED_NOT_ACTUALLY_CREDITED','remote_read':False,'candidate_cost':None}),flush=True);continue
  index_name='CANDIDATE_COST_EXTENSION_OUTER'+format(outer,'04d')+'.json'
  index_path=base/index_name;part_name='candidate_cost_backfill/CANDIDATE_OUTER'+format(outer,'04d')+'.json';dest=base/part_name
  if dest.exists():
   part_raw=dest.read_bytes();item=json.loads(part_raw)
   if item.get('source_pin')!=pin:raise ValueError('Existing candidate source pin differs;no overwrite/requery')
  else:
   if ssh is None:ssh=local_module(args.ssh_helper,'retained18_extension_route')
   source=REMOTE.replace('PIN_DATA',repr(pin));response=ssh.run('python3 -B -c '+shlex.quote(source),target='h100-10',timeout=60)
   if response.returncode:raise RuntimeError('Read-only aggregation did not return:'+response.stderr[-1500:])
   item=json.loads(response.stdout)
   item['local_collector_sha256']=digest(Path(__file__).read_bytes());item['remote_program_sha256']=digest(source.encode());item['fixed_pin_manifest_sha256']=digest(small([pin]))
   item['interpretation']={'outcomes':'accepted/discarded partition recorded outcomes;fallback overlaps accepted and is not an additive third bucket.',
    'origin':pin['origin_scope'],'credit':'Only already credited completed outer cost supplement;no new outer/write credit and no old book or cumulative rewrite.',
    'candidate_times':'Recorded equal amortized shares of parallel round elapsed;not independent per-outcome wall.',
    'round_times':'Each distinct original round_id once;not multiplied by512 questions or1024 root rows.',
    'padding':'Original native/controller totals may include DP padding. Candidate outcome padding allocation remains null;round-minus-candidate residual is unallocated.',
    'inherited_prefix':'Repeated inherited token/action events,not new unique generation/KV/FLOPs.',
    'absence':'Missing field_sums remain null with known_sum/count;observed empty buckets have actual zero empty sums.',
    'gpu_cost':'Physical shared8GPU capacity is not actual GPU-active cost or exclusive billing.'}
   if item.get('status')!='ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED':
    print(json.dumps({'outer':outer,'status':item.get('status'),'error':item.get('error'),'candidate_cost':None,'registered':False}),flush=True);pending.append(outer);continue
   part_raw=small(item);fixed_publish(base,mirrors,part_name,part_raw)
  if item.get('status')!='ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED':raise ValueError('Existing candidate part is not complete;not reused as known')
  info={'outer':outer,'file':part_name,'bytes':len(part_raw),'sha256':digest(part_raw),'status':item['status'],'source_commit':SOURCE,'config_sha256':CONFIG,'audit_file':pin['audit']['path']}
  if index_path.exists():
   index_raw=index_path.read_bytes();v=json.loads(index_raw)
   if v.get('parts')!=[info] or v.get('requested_completed_outers')!=[outer]:raise ValueError('Existing immutable extension index differs')
  else:
   v={'summary_schema':'tree_candidate_cost_backfill_index_v1','created_epoch':time.time(),'requested_completed_outers':[outer],'actual_completed_cost_outers':[outer],'unknown_outers':[],
    'parts':[info],'experiment_origin':'fresh','source_commit':SOURCE,'config_sha256':CONFIG,'run_directory':RUN,
    'cost_boundaries':item['interpretation'],'old_blocks_and_cumulative_files_modified':False,'raw_rollout_or_tensor_transferred_to_Mac':False,
    'future_use':'SHA-bound local cumulative supplement only;per-outer original audit stays remote. Incomplete/uncredited outers are never queried.'}
   index_raw=small(v)
  fixed_publish(base,mirrors,index_name,index_raw)
  r=item['rollouts'][0];c=r['candidate_outcomes'];f=c['all']['field_sums'];ledger=r['candidate_round_controller_ledger']
  md=('# 第'+str(outer)+'轮候选成本补齐\n\nsource778/config355；fresh母链continuation；仅已计入完成轮，原账本留H10018。\n\n'
   +'候选 all/accepted/discarded/fallback重叠：'+ '/'.join(str(c[k]['candidate_count']) for k in ['all','accepted','discarded','fallback'])+'。\n\n'
   +'候选新生成/prefill/检索/评分：'+ '/'.join(str(f[k]) for k in ['new_generated_tokens','new_prefill_tokens','retrieval_calls','scoring_calls'])+'。branch评分实际'+str(ledger['scoring_calls'])+'次 / '+str(ledger['scoring_wall_seconds'])+'秒。\n\n'
   +'JSON按A1/B1/B2及结果保留全部原scalar、coverage、SHA/bytes。候选时间是原记录的round均分份额；round每ID计一次。padding分摊、独立结果墙钟、真实GPU计费未知；不从差额推算。fallback与accepted重叠，继承prefix是重复事件。\n\n'
   +'无新增训练信用；旧through20及1–20块未改。索引SHA '+digest(index_raw)+'；原audit SHA '+item['source_files']['audit']['sha256']+'。\n').encode()
  fixed_publish(base,mirrors,index_name[:-5]+'.md',md)
  register(base,mirrors,{'outer':outer,'file':index_name,'bytes':len(index_raw),'sha256':digest(index_raw)})
  known.append(outer);print(json.dumps({'outer':outer,'status':item['status'],'index_file':index_name,'index_sha256':digest(index_raw),'part_bytes':len(part_raw),'part_sha256':digest(part_raw),'candidate_count':c['all']['candidate_count'],'candidate_field_sums':f,'actual_branch_controller':ledger,'audit_sha256':item['source_files']['audit']['sha256'],'collector_cpu_seconds':item.get('collector_process_cpu_seconds'),'remote_write':False},ensure_ascii=False),flush=True)
 print(json.dumps({'status':'CANDIDATE_EXTENSION_RANGE_COMPLETE','requested_outers':[args.first_outer,args.last_outer],'actual_completed_cost_outers':known,'skipped_pending_or_unknown_outers':pending,'registry_file':EXTENSIONS,'registry_sha256':digest((base/EXTENSIONS).read_bytes()) if (base/EXTENSIONS).exists() else None,'old_book_and_cumulative_modified':False}),flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--pins',type=Path);ap.add_argument('--ssh-helper',type=Path,required=True)
 ap.add_argument('--first-outer',type=int,default=3);ap.add_argument('--last-outer',type=int,default=19);args=ap.parse_args()
 args.base=args.base.resolve(strict=True)
 if (args.first_outer,args.last_outer)==(3,19):
  existing=args.base/'CANDIDATE_COST_BACKFILL_THROUGH19.json'
  if existing.exists():
   print(json.dumps({'status':'LEGACY3_19_PRESERVED_NO_REMOTE_READ','file':existing.name,'sha256':digest(existing.read_bytes())}));return
  args.pins=args.pins or args.base/'CANDIDATE_COST_BACKFILL_SOURCE_PINS_3_19.json';legacy_main(args);return
 if not 20<=args.first_outer<=args.last_outer<=200 or args.pins is not None:raise ValueError('Extension requires explicit20..200 range;fixed pins derive only from credited immutable local receipts')
 collect_extension(args)

if __name__=='__main__':main()
