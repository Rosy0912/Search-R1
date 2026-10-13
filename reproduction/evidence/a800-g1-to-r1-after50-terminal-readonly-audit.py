import sys,json,time,hashlib,math,os,subprocess
from pathlib import Path
from collections import Counter
p=json.load(sys.stdin);b=Path(p['base']);started=time.time()
def read(rel,cap=64*1024*1024):
 q=b/rel;assert q.resolve()==q and q.is_file() and q.stat().st_size<=cap
 raw=q.read_bytes();return json.loads(raw),{'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def digest(q):
 h=hashlib.sha256()
 with Path(q).open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 return h.hexdigest()
m,mf=read('logs/run-manifest.json',16*1024*1024)
assert all(m[k]==p[k] for k in ('run_id','execution_id','source_commit'))
repo=b/'repo';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==p['source_commit']
assert all(digest(repo/name)==sha for name,sha in m['source_sha256'].items())
sys.path.insert(0,str(repo))
from reproduction.common8.validation_identity import expected
import importlib.util
helper=Path(p['ownership_helper_path']);assert digest(helper)==p['ownership_helper_sha256']
spec=importlib.util.spec_from_file_location('readonly_ownership_helpers',helper);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
identity=helpers.identity;owned_session_survivors=helpers.owned_session_survivors
proof,pf=read('logs/full-run-acceptance.json');assert pf['sha256']==p['full_acceptance_sha256']
cleanup,cf=read('logs/cleanup-status.json',4*1024*1024);assert cf['sha256']==p['cleanup_sha256']
evaluation,ef=read('logs/final-identity.json',16*1024*1024)
assert proof['run_id']==cleanup['run_id']==evaluation['run_id']==m['run_id']
assert proof['structural_accepted'] is True and proof['contract_kind']=='fresh200_warmup57'
assert proof['actual_outer_updates']==200 and proof['per_rank_optimizer_opportunities']==4000
assert all(v==3993 for v in proof['per_rank_actual_optimizer_writes'].values()) and len(proof['per_rank_actual_optimizer_writes'])==4
record,frame,ids=expected(b,'final');sources=frame['data_source'].tolist()
assert evaluation['split']=='final' and evaluation['final'] is True and evaluation['step']==200
assert evaluation['question_ids']==ids and evaluation['data_sources']==sources
assert evaluation['validation_sha256']==record['sha256'] and evaluation['counts']==record['source_counts']
ids_sha=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
assert ids_sha==proof['shared_data']['products']['final']['ordered_probe_ids_sha256']
assert ids_sha=='fc85a32c0ab3a931badf19d3c70ae7bdd8379f3240c118caaf144080c4c91121'
em=evaluation['em'];f1=evaluation['f1'];assert len(ids)==len(em)==len(f1)==51713
assert all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in em)
assert all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in f1)
assert all(a<=c for a,c in zip(em,f1))
correct=math.fsum(em);weighted_em=correct/len(em);weighted_f1=math.fsum(f1)/len(f1)
assert correct==evaluation['correct'] and weighted_em==evaluation['weighted_em']==proof['final_weighted_em']
assert weighted_f1==evaluation['weighted_f1']==proof['final_weighted_f1']
counts=dict(Counter(sources));assert len(counts)==7 and counts['nq']==3610 and sum(counts.values())==51713
sc={s:math.fsum(v for z,v in zip(sources,em) if z==s) for s in counts};assert sc==proof['final_correct'] and counts==proof['final_counts']
checkpoints={}
for step in (5,50,100,150,200):
 checkpoint=Path(m['checkpoint_dir'])/'actor'/('global_step_'+str(step));rel=str((checkpoint/'complete.json').relative_to(b));d,df=read(rel,4*1024*1024)
 assert d==proof['checkpoint_proofs'][str(step)] and d['run_id']==m['run_id'] and d['step']==step and d['finite_all_saved_tensors'] is True
 assert Path(d['checkpoint']).resolve()==checkpoint.resolve() and d['validated_tensor_count']==435
 hashes={name:digest(checkpoint/name) for name in d['files_sha256']}
 assert hashes==d['files_sha256']
 checkpoints[str(step)]={'complete_receipt':df,'tensor_count':435,'producer_all_finite':True,'current_files_sha256':hashes,'actual_current_files_match_saved_proof':True}
assert proof['final_saved_tensor_count']==435
up=b/'logs/actual-updates.jsonl';raw=up.read_bytes();assert len(raw)<2*1024*1024 and hashlib.sha256(raw).hexdigest()==p['ledger_sha256']
rows=[json.loads(v) for v in raw.splitlines()];assert [v['step'] for v in rows]==list(range(1,201))
assert all(v['run_id']==m['run_id'] and v['opportunities_processed']==20 and 1<=v['actual_optimizer_writes']<=20 for v in rows)
assert sum(v['actual_optimizer_writes'] for v in rows)==3993 and rows[-1]['optimizer_writes_after']==3993
led,lf=read('logs/process-ledger.json',256*1024*1024);claim,plf=read('logs/pipeline-launch-claim.json',4*1024*1024)
assert led['run_id']==claim['run_id']==m['run_id'] and claim['execution_id']==m['execution_id']
records=[claim['pipeline_identity']];leaders=[];jobs={}
if records[0]['sid']==records[0]['pid']:leaders.append(records[0])
for name,job in led['jobs'].items():
 records+=list(job['ledger'].values());leaders.append(job['leader']);jobs[name]={'state':job['state'],'exit_code':job.get('exit_code'),'leader':{k:job['leader'][k] for k in ('pid','start_ticks','sid','uid')}}
 assert job['state']=='STOPPED',name
assert jobs['trainer']['exit_code']==jobs['final-acceptance']['exit_code']==0
live=[]
for rec in records:
 try:current=identity(rec['pid'])
 except (FileNotFoundError,ProcessLookupError):continue
 if current['state'] not in ('Z','X') and all(current[k]==rec[k] for k in ('start_ticks','sid','uid')):live.append({k:current[k] for k in ('pid','start_ticks','sid','uid')})
assert not live
survivors,reused=owned_session_survivors(leaders);assert not survivors
assert cleanup['execution_id']==m['execution_id'] and cleanup['execution_backend']=='ssh_local' and cleanup['cleaned'] is True
assert cleanup['scope']=='only recorded project PID births and private local corpus cache'
corpus,cof=read('logs/corpus-cache-cleanup.json',4*1024*1024);cc,ccf=read('logs/corpus-cache-claim.json',4*1024*1024)
assert corpus['execution_id']==m['execution_id'] and corpus['run_id']==m['run_id'] and corpus['cleaned'] is True and corpus['original_assets_removed'] is False
assert cc['phase']=='CLEANED' and cc.get('original_assets_owned') is False
root=Path(corpus['root']);assert root==b/'cache/retrieval'/m['execution_id'] and not root.exists()
pipeline,pipf=read('logs/pipeline-status.json',4*1024*1024);assert pipeline['run_id']==m['run_id'] and pipeline['stage']=='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP'
assert Path(m['initial_model']).is_dir() and Path(m['reference_model']).is_dir()
assert cleanup['slurm_job_id'] is None and corpus['slurm_job_id'] is None
assert cc['run_id']==m['run_id'] and cc['execution_id']==m['execution_id'] and cc['execution_backend']=='ssh_local' and cc['uid']==os.getuid()
assert not root.is_symlink()
assert cc['removed_file_sizes']==corpus['removed_file_sizes']
removed=corpus['removed_file_sizes'];removed_sha=hashlib.sha256(json.dumps(removed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assets,assetsf=read('logs/retrieval-assets.json',4*1024*1024)
assert assets['run_id']==m['run_id'] and assets['execution_id']==m['execution_id'] and assets['accepted'] is True and assets['temporary_original_assets'] is False
originals={}
for name,rec in assets['files'].items():
 q=Path(rec['path']);st=q.stat();assert q.is_file() and not q.is_symlink() and (st.st_dev,st.st_ino,st.st_size)==(rec['device'],rec['inode'],rec['size'])
 originals[name]={'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'metadata_matches_original_admission':True}
cuda=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
cuda_rows=[tuple(x.strip() for x in line.split(',')) for line in cuda.splitlines() if line.strip()]
remaining_cuda,reused_cuda=helpers.remaining_cuda_contexts(records,cuda_rows);assert not remaining_cuda
assert proof['optimizer_completion_rule']=='all20_opportunities_processed_and_at_least_one_write'
assert set(proof['scheduler_receipts'])=={'0','1','2','3'}
scheduler={}
for rank,s in proof['scheduler_receipts'].items():
 q=Path(s['receipt_path']);assert q.resolve()==q and b in q.parents and q.stat().st_size<64*1024*1024
 assert digest(q)==s['observed_file_sha256']
 assert s['complete_outer_optimizer_opportunities']==4000 and s['cumulative_actual_optimizer_writes']==3993
 prefix=b'\n'.join(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode() for row in s['selected_rows'])+b'\n'
 assert hashlib.sha256(prefix).hexdigest()==s['selected_rows_sha256']
 scheduler[rank]={k:s[k] for k in ('receipt_path','selected_rows_sha256','observed_file_sha256','complete_outer_optimizer_opportunities','cumulative_actual_optimizer_writes')}
attempts=proof['formal_inline_all_attempts'];assert attempts['structural_accepted'] is True and attempts['completed_outer_count']==200 and attempts['actual_optimizer_writes_per_rank']==3993
alog=Path(m['train_log']);assert alog.stat().st_size<64*1024*1024
assert b'SEARCH_G1_FATAL_SCHEDULER_EVIDENCE=' not in alog.read_bytes()
attempt_projection={k:v for k,v in attempts.items() if k not in ('attempts',)}
extra={'ownership_helper_path':str(helper),'ownership_helper_sha256':p['ownership_helper_sha256'],'helper_source_commit':p['ownership_helper_source_commit'],'remaining_owned_cuda_contexts':0,'reused_recorded_pid_cuda_contexts':reused_cuda,'corpus_removed_file_count':len(removed),'corpus_removed_byte_sum':sum(removed.values()),'corpus_removed_file_sizes_sha256':removed_sha,'original_retrieval_assets':originals,'retrieval_assets_admission':assetsf,'scheduler_receipts':scheduler,'formal_inline_all_attempts':attempt_projection,'fatal_scheduler_console_marker_absent':True}
out={'role':'actual_read_only_predecessor_final_identity_checkpoint_and_owned_cleanup_audit','started_unix':started,'checked_unix':time.time(),'identity':{k:p[k] for k in ('base','run_id','execution_id','source_commit')},'run_manifest':mf,'source_files_checked':len(m['source_sha256']),'actual_source_hashes_match':True,'full_acceptance':pf,'final_identity':ef,'final_ordered_ids_sha256':ids_sha,'final_ordered_ids_match_canonical':True,'final_data_sources_match_canonical_order':True,'final_count':len(ids),'final_unique_normalized_ids':len(set(ids)),'duplicates_preserved':True,'binary_em_finite_f1_verified':True,'final_correct':correct,'final_em':weighted_em,'final_f1':weighted_f1,'counts':counts,'correct_by_source':sc,'source_em':proof['final_source_em'],'source_f1':proof['final_source_f1'],'checkpoint_proofs':checkpoints,'final_canonical_tensor_count':435,'canonical_finite_evidence':'actual producer complete435finite plus successful pinned verify_full canonical tensor inspection; this read hashes current files without loading tensors or rerunning evaluation','updates_ledger_sha256':p['ledger_sha256'],'continuous_complete_outers':200,'processed_opportunities':4000,'successful_adam':3993,'skips':7,'process_ledger':lf,'pipeline_claim':plf,'jobs':jobs,'recorded_births_checked':len(records),'surviving_same_births':0,'surviving_owned_session_processes':0,'reused_session_ids':reused,'cleanup':cf,'corpus_cleanup':cof,'corpus_claim':ccf,'private_corpus_root_absent':True,'original_assets_removed':False,'official_model_reference_directories_retained':True,'pipeline':pipf,'pipeline_stage':pipeline['stage'],'evaluations_launched':False,'model_tensor_scans_launched':False,'trainer_launched_or_restarted':False,'remote_signals_sent':False,'gpu_claim_or_gpu_operation_performed':False,'structural_terminal_accepted':True}
out.update(extra)
print(json.dumps(out,allow_nan=False))
