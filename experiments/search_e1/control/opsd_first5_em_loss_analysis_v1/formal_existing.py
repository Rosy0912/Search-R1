from pathlib import Path
from collections import Counter
import json, hashlib, gzip, tarfile, datetime, itertools, sys, time
E=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008')
start=time.time()
O=E/'em-loss-relation-20261009'/'formal512-existing-evidence-20261009-v2'
O.mkdir(parents=True,exist_ok=False)
def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p,x):
 p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def bind(p,b=None):
 if b is None: b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
manifestpath=E/'goal-continuation-20261009-26/native-stratified-five-inputs/five-closed-model-inputs.json'
mb=manifestpath.read_bytes(); manifest=json.loads(mb)
names=['bf16_zero','C0','C1','M']
allrows={}; sourceproofs=[bind(manifestpath,mb)]; summaries={}; nativefields={}; modelproofs={}
for name in names:
 m=manifest['models'][name]
 for field in ['records','summary']:
  p=Path(m[field]['path']); b=p.read_bytes()
  assert sha(b)==m[field]['sha256'] and len(b)==m[field]['bytes'],(name,field,'identity')
  sourceproofs.append(bind(p,b))
  if field=='records': rows=[json.loads(l) for l in b.splitlines() if l.strip()]
  else: summary=json.loads(b)
 rows=sorted([r for r in rows if r['version']==m['version']],key=lambda r:r['validation_index'])
 assert len(rows)==512 and len({r['question_id'] for r in rows})==512
 assert [r['validation_index'] for r in rows]==list(range(512))
 assert all(r['primary'] is True and r['status']=='complete' and r['validation_index']==r['rank']+8*r['slot'] for r in rows)
 assert Counter(r['dataset'] for r in rows)=={'nq':256,'hotpotqa':256}
 assert summary['accepted'] is True
 allrows[name]=rows; summaries[name]=summary
 assert sum(r['protocol_em'] for r in rows)==summary['models'][m['version']]['overall']['strict_hits']
allidentity=[[r['question_id'],r['row_semantic_sha256'],r['dataset']] for r in allrows['bf16_zero']]
assert all([[r['question_id'],r['row_semantic_sha256'],r['dataset']] for r in allrows[n]]==allidentity for n in names)
cohort_sha=sha(json.dumps(allidentity,ensure_ascii=False,separators=(',',':')).encode())
proofdirs={
 'bf16_zero':E/'common-bf16-zero-closed-backup-20261009',
 'C0':E/'c0-native512-closed-backup-20261009',
 'C1':E/'c1-closed-native512-compressed-20261009',
 'M':E/'M-closed-native512-compressed-de7b77e-20261009'}
validated=[]; counters={}; table=[]
scorekeys={'scores','sequences_scores','logits','logprobs','log_probs','probabilities','output_scores'}
for name in names:
 pp=proofdirs[name]/'complete-sha-verified.json'
 pb=pp.read_bytes(); proof=json.loads(pb)
 assert proof['accepted'] is True
 sourceproofs.append(bind(pp,pb))
 target={}
 for r in allrows[name]:
  if name=='bf16_zero': local=proofdirs[name]/'files'/('worker-'+str(r['rank']))/('slot-'+format(r['slot'],'02d'))/'native-capture.json'
  elif name=='C0': local=proofdirs[name]/'files'/r['version']/('worker-'+str(r['rank']))/('slot-'+format(r['slot'],'02d'))/'native-capture.json'
  else:
   prefix='/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/'
   assert r['trace']['path'].startswith(prefix)
   local=r['trace']['path'][len(prefix):]
  target[str(local)]=r
 cc=Counter(); ec=Counter(); generate_keysets=Counter(); checked=0; generate_kwargs=Counter()
 def consume(location,b,r):
  global checked
  assert len(b)==r['trace']['bytes'] and sha(b)==r['trace']['sha256'],(name,r['validation_index'],'trace SHA')
  c=json.loads(b)
  for field in ['version','question_id','validation_index','rank','slot']:
   assert c[field]==r[field],(name,field)
  assert c['protocol']['protocol_em']==r['protocol_em']
  assert c['protocol']['protocol_valid_final_action']==r['protocol_valid_final_action']
  assert c['protocol']['failure_class']==r['failure_class']
  assert c['native_result']['search_calls']==r['search_calls']
  assert c['native_result']['termination']==r['termination']
  assert c['processed_per_token_scores_recorded'] is False
  assert c['primary_ordinary_prompt'] is True and c['teacher_forced'] is False and c['privileged_reference'] is False
  assert c['return_objects_modified'] is False and c['generation_options_modified'] is False and c['hooks_restored'] is True
  assert c['logging_errors']==[]
  gc=0
  for event in c['events']:
   ec[event['kind']]+=1
   if event['kind']=='model_generate':
    assert not (scorekeys & set(event))
    assert isinstance(event['output_ids'],list) and event['output_type']=='torch.Tensor'
    assert event['kwargs']['do_sample'] is False and event['kwargs']['max_new_tokens']==500
    assert not any(event['kwargs'].get(k) for k in ['output_scores','return_dict_in_generate','output_logits'])
    gc+=1
    generate_keysets['|'.join(sorted(event.keys()))]+=1
    generate_kwargs[json.dumps({k:v for k,v in event['kwargs'].items() if k!='stopping_criteria'},sort_keys=True)]+=1
  cc['captures_verified']+=1; cc['generation_calls']+=gc
  cc['processed_score_unavailable']+=1
  compact={k:r[k] for k in ['validation_index','question_id','row_semantic_sha256','dataset','protocol_em','protocol_valid_final_action','failure_class','search_calls','termination','rank','slot','version','run_id']}
  compact.update({'model':name,'formal_weight_stage':'common_zero' if name=='bf16_zero' else 'raw_after_10_outer_rounds','native_result_sha256':r['native_result_sha256'],'native_trace_sha256':r['trace']['sha256'],'native_trace_bytes':r['trace']['bytes'],'generation_calls':gc,'formal_greedy':True,'generation_probability_available':False,'processed_per_token_scores_recorded':False,'answer_confidence':None,'pass_probability_estimate':None,'confidence_unavailable_reason':'Only raw generation IDs/kwargs/processors and behavioral trace were recorded; no generate-time logprob/logit/score or full distribution. Text is not parsed as calibrated confidence.'})
  table.append(compact)
  validated.append({'model':name,'validation_index':r['validation_index'],'source_local_location':str(location),'native_trace_sha256':sha(b),'bytes':len(b)})
 if name in ['bf16_zero','C0']:
  for local,r in target.items(): consume(local,Path(local).read_bytes(),r)
 else:
  ap=Path(proof['archive']['path'])
  st=ap.stat()
  for k,actual in [('bytes',st.st_size),('device',st.st_dev),('inode',st.st_ino),('mtime_ns',st.st_mtime_ns),('ctime_ns',st.st_ctime_ns)]:
   assert actual==proof['archive'][k],(name,'archive metadata continuity',k)
  mm={m['relative']:m for m in proof['members']}
  found=set()
  with gzip.open(ap,'rb') as gf:
   with tarfile.open(fileobj=gf,mode='r|') as tf:
    for member in tf:
     if member.name in target:
      assert member.isfile() and member.name not in found
      b=tf.extractfile(member).read()
      assert member.size==mm[member.name]['bytes'] and sha(b)==mm[member.name]['sha256']
      consume(str(ap)+'::'+member.name,b,target[member.name]); found.add(member.name)
   while gf.read(1024*1024): pass
  assert found==set(target),(name,'missing tar captures')
  sourceproofs.append({'path':str(ap),'original_accepted_archive_sha256':proof['archive']['sha256'],'fresh_selected_native_member_SHA':True,'fresh_selected_members':len(found),'fresh_gzip_EOF_CRC':True,'full_archive_SHA_reread':False,'archive_metadata_continuity_checked':True})
 counters[name]={'verification':dict(cc),'event_counts':dict(ec),'generate_event_key_sets':dict(generate_keysets),'generate_kwargs_excluding_stop_state':dict(generate_kwargs)}
 assert cc['captures_verified']==512
table.sort(key=lambda r:(names.index(r['model']),r['validation_index']))
longpath=O/'formal512-outcomes.private.jsonl'
longpath.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for r in table))
wide=[]
for i in range(512):
 row={'validation_index':i,'question_id':allidentity[i][0],'row_semantic_sha256':allidentity[i][1],'dataset':allidentity[i][2],'models':{}}
 for n in names:
  r=allrows[n][i]
  row['models'][n]={k:r[k] for k in ['protocol_em','protocol_valid_final_action','failure_class','search_calls','termination']}
 wide.append(row)
widepath=O/'formal512-paired-outcomes.private.jsonl'
widepath.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for r in wide))
aggregates={}
for n in names:
 rows=allrows[n]
 def agg(rr):
  return {'questions':len(rr),'strict_hits':sum(r['protocol_em'] for r in rr),'em':sum(r['protocol_em'] for r in rr)/len(rr),'final_invalid':sum(not r['protocol_valid_final_action'] for r in rr),'failure_counts':dict(Counter('null' if r['failure_class'] is None else r['failure_class'] for r in rr)),'search_calls':dict(Counter(str(r['search_calls']) for r in rr)),'termination':dict(Counter(r['termination'] for r in rr)),'mean_search_calls':sum(r['search_calls'] for r in rr)/len(rr)}
 aggregates[n]={'overall':agg(rows),'datasets':{d:agg([r for r in rows if r['dataset']==d]) for d in ['nq','hotpotqa']}}
pairs={}
for base,new in itertools.combinations(names,2):
 cells={'gained':[],'lost':[],'preserved_correct':[],'still_wrong':[]}
 for a,b in zip(allrows[base],allrows[new]):
  key='gained' if not a['protocol_em'] and b['protocol_em'] else 'lost' if a['protocol_em'] and not b['protocol_em'] else 'preserved_correct' if a['protocol_em'] else 'still_wrong'
  cells[key].append({'base':a,'new':b})
 pairs[base+'__to__'+new]={'counts':{k:len(v) for k,v in cells.items()},'net_hits':len(cells['gained'])-len(cells['lost']),'delta_em':(len(cells['gained'])-len(cells['lost']))/512,'lost_new_failure_counts':dict(Counter(x['new']['failure_class'] for x in cells['lost'])),'lost_new_search_calls':dict(Counter(str(x['new']['search_calls']) for x in cells['lost'])),'search_changed_by_cell':{k:sum(x['base']['search_calls']!=x['new']['search_calls'] for x in v) for k,v in cells.items()}}
for p in [
 Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/control/opsd_ablation_eval_v1/hooks.py'),
 Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/control/opsd_ablation_eval_v1/worker.py')]: sourceproofs.append(bind(p))
summary={'schema':'existing-formal512-em-loss-evidence-audit-v1','accepted':True,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_started_unix':start,'execution_ended_unix':time.time(),'scope':'Local CPU read-only extraction of four already accepted formal greedy native512 results; no inference, scoring, model loading, transfer, Node write, GPU, training, queue modification, or Git mutation. Outputs private. Formal C0/C1/M are raw after 10 outer rounds, not a 5-round checkpoint evaluation.','models':names,'primary_slots':2048,'unique_questions':512,'equal_question_row_dataset_identity_across_all_4_models':True,'ordered_common_cohort_identity_sha256':cohort_sha,'native_capture_count_verified':len(validated),'native_capture_scope':'2048 selected existing captures independently fresh SHA matched to accepted primary records; C1 and M decompressed per member without extracting whole archives, gzip EOF/CRC checked. No full archive rehash or full backup arithmetic claim.','probability_availability':{'greedy_one_draw_per_model_per_question':True,'generation_logprob_available':False,'generation_processed_per_token_score_available':False,'complete_next_token_distribution_available':False,'numeric_confidence_available':False,'text_confidence_inference_performed':False,'single_question_pass_probability_estimable_from_this_evidence':False,'native_EM_is_cohort_empirical_deterministic_hit_rate':True,'TRAIN_5fresh_success_frequency_definition':'s/5 is empirical success frequency at the outer-start sampling policy with the TRAIN stochastic generation protocol. Observed pass@5 is 1(s>0); neither equals formal greedy single-question pass probability or true success probability without repeated matched draws.','TRAIN_heldout_join':'No cross-cohort loss/outcome join performed. QIDs contain train prefixes; actual identity overlap requires explicit source cohort sets.'},'failure_histogram_null_normalization':'The JSON key null represents original failure_class=None; source record/capture values remain null in outcome tables. Histograms are descriptive accepted protocol classifications, no rescoring.', 'aggregate_outcomes':aggregates,'paired_outcomes':pairs,'capture_schema_counters':counters,'source_proofs':sourceproofs,'output_tables':[bind(longpath),bind(widepath)]}
dump(O/'summary.private.json',summary)
dump(O/'native-trace-fresh-SHA.private.json',{'captures':validated,'captures_count':len(validated),'scope':'Fresh selected 2048 native capture byte identity only; private source locations and no prompt bodies.'})
print(json.dumps({'output_dir':str(O),'accepted':True,'native_captures_verified':len(validated),'equal_identity':True,'formal_hits':{n:aggregates[n]['overall']['strict_hits'] for n in names},'probability_available':False,'tables':[bind(longpath),bind(widepath)],'summary':bind(O/'summary.private.json'),'trace_manifest':bind(O/'native-trace-fresh-SHA.private.json')},ensure_ascii=False,sort_keys=True))
