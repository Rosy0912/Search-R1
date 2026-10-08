#!/usr/bin/env python3
"""Terminal-only read-only evidence/CPU HF wrapper. Never signs acceptance or trains."""
import argparse,ast,base64,csv,hashlib,importlib.util,io,json,math,os,pathlib,re,resource,subprocess,sys,tarfile,time
P=pathlib.Path
ROOT=P('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');RUN=ROOT/'runs/bapo-instruct300-formal300-3292'
SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876';REFSHA='a27674a26f4958f9514c19be6f9f5ba5fab9bfd640c586b5bf293266da29ee74'
PREFIXSHA='57b05f48a27ab5fd5c81e3e524f71569c6675f8c3ca234ac105b0c677e2c3dc0';MANIFESTSHA='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46'
WATCHSHA='53ed622dcce4aa5475d80f6489683860a3f43d7d1b6e4c6aa5657649d2a75a80';GUARDSHA='c0334988153496305b57aaecafb908d28d14705791b4632792f03e135c96e8c3'
MODEL=ROOT/'assets/qwen2.5-3b-instruct-aa8e725';MODEL_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44';DATA_SHA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
COUNTS={'nq':3610,'triviaqa':11313,'popqa':14267,'hotpotqa':7405,'2wikimultihopqa':12576,'musique':2417,'bamboogle':125}
CONTINUITY='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def filebind(p):
 p=P(p);need(p.is_file()and not p.is_symlink()and p.resolve()==p,'Canonical ordinary original '+str(p));a=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 b=p.stat();need((a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino),'Original changed '+str(p));return dict(path=str(p),bytes=b.st_size,sha256=h.hexdigest())
def strictjson(raw):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'Duplicate JSON key');d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def reference(p):
 need(filebind(p)['sha256']==REFSHA,'Exact reviewed Instruct reference tool required');s=importlib.util.spec_from_file_location('read_only_reviewed_reference',p);m=importlib.util.module_from_spec(s);sys.dont_write_bytecode=True;s.loader.exec_module(m);return m

def scope_functions(p):
 """Only explicit outer-range + saturated LR substitutions; numerical oracle unchanged."""
 m=reference(p);text=P(p).read_text();tree=ast.parse(text);ns=dict(m.__dict__)
 for name in ('validate_metrics','validate_allrank_reporting'):
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);src=ast.get_source_segment(text,node)
  replacements={'list(range(1,6))':'list(range(1,len(train)+1))','len(ranks)==40':'len(ranks)==len(train)*8','outer in range(1,6)':'outer in range(1,len(train)+1)','range(1,6)}':'range(1,len(train)+1)}','1e-6*(outer-1)/114':'1e-6*min((outer-1)/114,1)','1e-6*outer/114':'1e-6*min(outer/114,1)','len(train)==5 and len(ranks)==len(train)*8':'0<len(train)<=300 and len(ranks)==len(train)*8'}
  for old,new in replacements.items():src=src.replace(old,new)
  exec(compile(src,'reviewed_reference_explicit_outer_scope','exec'),ns)
 return ns['validate_metrics'],ns['validate_allrank_reporting']
def records(raw,marker):
 result=[];offset=0
 for line in raw.splitlines(keepends=True):
  clean=re.sub(rb'\x1b\[[0-9;]*m',b'',line)
  if marker.encode()in clean:
   payload=clean.split(marker.encode(),1)[1].strip();result.append(dict(event=strictjson(payload),offset=offset,bytes=len(line),sha256=hashlib.sha256(line).hexdigest()))
  offset+=len(line)
 return result

def validate_logs(prefix,current,reference_path,final=True):
 old=P(prefix).read_bytes();new=P(current).read_bytes();need(len(old)==959681 and hashlib.sha256(old).hexdigest()==PREFIXSHA,'Exact inherited25 prefix required')
 olddriver=records(old,'BAPO_TRAIN_METRICS_JSON=');newdriver=records(new,'BAPO_TRAIN_METRICS_JSON=');oldrank=records(old,'ACTOR_RANK_UPDATE_JSON=');newrank=records(new,'ACTOR_RANK_UPDATE_JSON=')
 need([v['event']['step']for v in olddriver]==list(range(1,26)),'Inherited exact1..25');steps=[v['event']['step']for v in newdriver];need(steps==list(range(26,26+len(steps)))and len(steps)<=275,'Current3292 exact26..n withoutduplicates')
 need(not final or steps==list(range(26,301)),'Final requires275 actualnewouter');end=25+len(steps);need(len(oldrank)==200 and len(newrank)==len(steps)*8,'Inherited200/new8rank perouter')
 for row in oldrank:need(row['event']['scheduler_last_epoch']in range(1,26),'Wrong inherited rank scope')
 for row in newrank:need(row['event']['scheduler_last_epoch']in steps,'Wrong current rank scope')
 train=[x['event']for x in olddriver+newdriver];need(all(type(v['step'])is int for v in train),'Exactintegerdriver clock');ranks=[x['event']for x in oldrank+newrank];metrics,allrank=scope_functions(reference_path);stage=metrics(train,ranks);proof=allrank(train,ranks)
 # Preserve all original source/offset/bytes bindings; merged view never replaces actual train.log.
 provenance=[dict(source=str(path),marker=marker,**row)for path,raw in ((prefix,old),(current,new))for marker in ('BAPO_TRAIN_METRICS_JSON=','ACTOR_RANK_UPDATE_JSON=')for row in records(raw,marker)]
 return dict(inherited_outer=25,new_outer=len(steps),actual_outer_updates=end,rank_records=len(ranks),optimizer_calls_per_rank=20*end,expected_bapo=stage,allrank_reporting=proof,provenance=provenance,sampling_continuity=CONTINUITY,final_scope_complete=bool(final and end==300))
def final_metrics(raw):
 rows=records(raw,'SEARCH_R1_FINAL_METRICS_JSON=');need(len(rows)==1,'Unique actual51713 final aggregate');d=rows[0]['event'];evaluations=records(raw,'BASELINE_EVALUATION_METRICS_JSON=');need(evaluations and evaluations[-1]['event']==d,'Final actual evaluation aggregate mismatch')
 need(type(d.get('val/qa_count/all'))is int and d.get('val/qa_count/all')==51713,'Final51713 aggregate count')
 for source,count in COUNTS.items():
  need(type(d.get('val/qa_count/'+source))is int and d['val/qa_count/'+source]==count and type(d.get('val/evaluated_count/'+source))is int and d['val/evaluated_count/'+source]==count,'Final per-source count '+source)
 for source in ['all']+list(COUNTS):
  for key in ('em','f1','answer_format','format_rate'):
   value=d.get('val/'+key+'/'+source);need(type(value)in(int,float)and math.isfinite(value)and 0<=value<=1,'Final finite/range '+key+'/'+source)
  for key in ('generated_tokens','search_calls'):
   value=d.get('val/'+key+'/'+source);need(type(value)in(int,float)and math.isfinite(value)and value>=0,'Final finite cost '+key)
 for key in ('em','f1','answer_format','format_rate','generated_tokens','search_calls'):
  expected=math.fsum(d['val/'+key+'/'+src]*count for src,count in COUNTS.items())/51713;need(math.isclose(d['val/'+key+'/all'],expected,rel_tol=1e-9,abs_tol=1e-12),'Final weightedaggregate '+key)
 need(all(type(v)not in(int,float)or math.isfinite(v)for v in d.values()),'Nonfinite final metric');return dict(metrics=d,original_record=rows[0],per_answer_recomputed=False,evidence_scope='Actual production aggregate only; no persisted51713 answers')
def terminal(raw):
 text=raw.decode();field=lambda key:re.findall(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',text)
 for k,v in [('JobId','3292'),('JobName','bapo-instruct300-formal300'),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5'),('NumNodes','1'),('TresPerNode','gres:gpu:h100:8'),('JOB_GRES','gpu:h100:8')]:need(field(k)==[v],'Actual terminal '+k)
 need(len(field('RunTime'))==1,'Terminal time');t=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)',field('RunTime')[0]);need(t is not None,'RunTime units');day,h,m,s=map(int,(t[1]or 0,t[2],t[3],t[4]));need(m<60 and s<60,'RunTime units');seconds=day*86400+h*3600+m*60+s;need(seconds>0,'RunTime positive')
 for key in ('TRES','AllocTRES'):
  rows=field(key);need(len(rows)==1 if key=='TRES'else len(rows)<=1,'Actual '+key)
  if rows:
   pairs=[x.split('=',1)for x in rows[0].split(',')];need(all(len(x)==2 for x in pairs)and len(pairs)==len({x[0]for x in pairs}),'Duplicate TRES');values=dict(pairs);need(values.get('node')=='1'if key=='TRES'else 'node'not in values or values['node']=='1','TRES node');need(all(v=='8'for k,v in values.items()if k.startswith('gres/gpu')),'Conflicting GPU TRES')
 return dict(job_id=3292,reserved_seconds=seconds,gpu_count=8,reserved_gpu_hours=seconds*8/3600,new_outer=275,inherited_outer=25,monetary_price_assumed=False)
def validate_checkpoint(m,complete,run):
 need(m['run_root']==str(run)and m['source_commit']==SOURCE,'Final manifest source/run')
 for key,value in [('global_step',300),('actual_outer_updates',300),('horizon',300),('warmup_steps',114)]:need(type(m[key])is int and m[key]==value,'Final manifest '+key)
 mp=run/'checkpoints/full_state/global_step_300/manifest.json';need(strictjson(mp.read_bytes())==m and strictjson(mp.with_name('checkpoint-complete.json').read_bytes())==complete,'Loaded actualmanifest/complete bytes');need(all(type(complete[k])is int for k in ('step','actual_outer_updates','world_size')),'Complete integer clocks');need(complete['status']=='FULL_STATE_CHECKPOINT_COMPLETE'and complete['step']==complete['actual_outer_updates']==300 and complete['source_commit']==SOURCE and complete['world_size']==8 and complete['generation_rng_saved']is True,'Complete300 actualmarker')
 need(complete['manifest_path']==str(mp)and complete['manifest_sha256']==filebind(mp)['sha256'],'Complete marker manifest binding');lineage=complete['recovery_lineage'];resume=run/'recovery-manifest.json';need(lineage['old_run_root']==strictjson(resume.read_bytes())['old_run_root']and lineage['old_step']==25 and lineage['manifest_path']==str(resume)and lineage['manifest_sha256']==filebind(resume)['sha256'],'Complete lineage25')
 items=m['files'];need(items and len({x['path']for x in items})==len(items),'Checkpoint unique members')
 actual=[]
 for row in items:
  need(type(row['bytes'])is int and row['bytes']>0 and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Typedfullmemberbinding');p=P(row['path']);need(run in p.parents and row==filebind(p),'Actual fullmember binding changed');actual.append(row)
 configpath=mp.parent/'resolved-config.json';need(filebind(configpath)['sha256']==m['config_sha256'],'Resolved config binding');cfg=strictjson(configpath.read_bytes());need(cfg['trainer']['total_training_steps']==301 and cfg['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfg['trainer']['default_local_dir']==str(run/'checkpoints')and cfg['baselines']['algorithm_source_commit']==SOURCE and cfg['baselines']['bapo']['stage_validation_size']==64 and cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1,'Actual301/114/currentrun/stage64/TP1')
 expected=set()
 for child in (run/'checkpoints/actor/global_step_300',run/'checkpoints/algorithm/global_step_300',mp.parent):
  for p in child.rglob('*'):
   need(not p.is_symlink(),'No checkpoint symlink');
   if p.is_file()and p.name not in ('manifest.json','checkpoint-complete.json'):expected.add(str(p))
 expected.add(str(run/'source-commit.txt'));need(expected=={x['path']for x in items},'Manifest must enumerate complete checkpointtree, not selectedfiles')
 member={x['path']:x for x in items};reports=m['rank_roundtrips'];need(len(reports)==8 and {r['rank']for r in reports}==set(range(8))and len({r['pid']for r in reports})==8,'Native actual8rank')
 for r in reports:
  need(all(type(r[k])is int for k in ('rank','world_size','step','scheduler_last_epoch','native_state_version','optimizer_state_entries')),'Native integer schema')
  p=run/'checkpoints/actor/global_step_300/training_state'/('rank_%d.pt'%r['rank']);need(r['path']==str(p)and r['sha256']==member[str(p)]['sha256']and strictjson(p.with_suffix('.json').read_bytes())==r,'Native member/report binding');need(r['status']=='FULL_STATE_SERIALIZED_OK'and r['step']==r['scheduler_last_epoch']==300 and r['world_size']==8 and r['native_state_version']==3 and r['roundtrip']is False,'Actualserialized nativev3, not fictitiousreload')
  need(r['model_state_dict_type']=='SHARDED_STATE_DICT'and type(r['model_dtensor_entries'])is int and r['model_dtensor_entries']>0 and r['optimizer_state_entries']>0,'Native model/Adam schema');need(r['state_digest']==r['reloaded_state_digest']and re.fullmatch('[0-9a-f]{64}',r['state_digest']),'Nativeproducer serialized digest');need(type(r['pid'])is int and r['pid']>0,'Native actualPID');top=r['generation_rng_topology'];need(top['version']==1 and top['torch_random_states_dtype']=='torch.uint8'and top['torch_random_states_numel']>0 and top['device_mesh']==dict(shape=[8,1],names=['dp','infer_tp'],ranks=[[i]for i in range(8)]) and top['torch_random_states_shape']==[16]and top['torch_random_states_numel']==16 and top['gen_random_states_dtype']=='torch.uint8'and top['gen_random_states_shape']==[16]and top['gen_random_states_numel']==16,'Native generation RNG topology');need(r['generation_rng_before_digest']==r['generation_rng_after_digest']and re.fullmatch('[0-9a-f]{64}',r['generation_rng_before_digest']),'Savedgeneration RNG unchanged')
 actor=cfg['actor_rollout_ref']['actor'];rollout=cfg['actor_rollout_ref']['rollout'];need(actor['ppo_mini_batch_size']==128 and actor['ppo_micro_batch_size']==16 and actor['ppo_epochs']==1 and actor['loss_reduction']=='sequence_mean'and actor['optim']['lr']==1e-6 and actor['optim']['warmup_style']=='constant'and rollout['log_prob_micro_batch_size']==32 and rollout['n_agent']==5 and cfg['data']['train_batch_size']==512,'Actual originaltrainingbudget');need(cfg['baselines']['bapo']==dict(stage_validation_files=str(ROOT.parent/'bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data/stage.parquet'),stage_validation_size=64,stage_validation_freq=1,patience=5,alpha=.05,bonus=.5,resample_attempts=2),'OriginalBAPO stage protocol')
 need(strictjson(mp.read_bytes())==m and strictjson(mp.with_name('checkpoint-complete.json').read_bytes())==complete,'Manifest/complete changed during fullmemberhash')
 return dict(members=actual,config=cfg,reports=reports,manifest=filebind(mp),complete=filebind(mp.parent/'checkpoint-complete.json'),fresh_gpu_reload_performed=False)
def cpu_driver(run,expected_bapo):
 import torch
 from search_r1.baselines.full_state import state_digest
 path=run/'checkpoints/full_state/global_step_300/driver.pt';payload=torch.load(path,map_location='cpu',weights_only=False);r=strictjson(path.with_name('driver-reload.json').read_bytes());need(r['status']=='DRIVER_SERIALIZED_OK'and r['step']==300 and r['sha256']==filebind(path)['sha256']and r['state_digest']==state_digest(payload)and r['path']==str(path)and r['pid']==payload['driver_pid']and r['sampler_cursor']==153600,'Actualdriver serialized proof');old=strictjson((run/'recovery-witness/driver.json').read_bytes());need(payload['driver_pid']==old['new_pid']and old['target_job_id']==3292 and old['target_run_root']==str(run)and old['target_source_commit']==SOURCE,'Actualdriver identity lineage')
 need(payload['global_step']==payload['actual_outer_updates']==300 and payload['next_global_step']==301 and payload['epoch']==0 and payload['bapo']==expected_bapo and payload['rng']['cuda']==[],'Driver300 clock/BAPO/CPU RNG');s=payload['sampler'];need(s['size']==169028 and s['cursor']==153600 and s['epoch']==0 and len(s['order'])==s['size']and set(s['order'])==set(range(s['size'])),'Actual sampler permutation/cursor')
 return dict(state_digest=state_digest(payload),sampler_digest=state_digest(s),actual_outer_updates=300,driver_status=r['status'],fresh_gpu_reload_performed=False)
def new_json(p,value):
 with P(p).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--mode',choices=('prefix-check','terminal-evidence','cpu-hf'),required=True);a.add_argument('--reference-tool',type=P,required=True);a.add_argument('--prefix',type=P,required=True);a.add_argument('--current-log',type=P,default=RUN/'train.log');a.add_argument('--terminal',type=P);a.add_argument('--output-dir',type=P,required=True);a.add_argument('--cpu-threads',type=int,default=8);args=a.parse_args();need(1<=args.cpu_threads<=32,'CPUthreads');need(not args.output_dir.exists(),'Exclusive fresh evidence directory');args.output_dir.mkdir(parents=True);start=time.monotonic();cpu=time.process_time()
 result=dict(status='UNSIGNED_REQUIRES_FINAL_INDEPENDENT_REVIEW',mode=args.mode,job_id=3292,source_commit=SOURCE,run_root=str(RUN),final_accepted=False,actual_cpu_hf_executed=False,actual_final_gpu_reload_performed=False,issues=[],sampling_continuity=CONTINUITY,tool=filebind(P(__file__).resolve()))
 try:
  result['logs']=validate_logs(args.prefix,args.current_log,args.reference_tool,args.mode!='prefix-check')
  if args.mode!='prefix-check':
   need(args.current_log==RUN/'train.log'and args.terminal is not None,'Only actualterminal same3292');need(ROOT in args.terminal.parents,'Owned capturedterminal');result['terminal']=filebind(args.terminal);result['cost']=terminal(args.terminal.read_bytes());need((RUN/'source-commit.txt').read_text().strip()==SOURCE and (ROOT/'source-expected.txt').read_text().strip()==SOURCE,'Actualsource');need(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Frozen source unchanged');need(filebind(ROOT/'logs/tools-manifest.json')['sha256']==MANIFESTSHA,'Frozen36');need('TRAIN_RETURNED_SUCCESSFULLY'in (RUN/'phase.log').read_text(),'Actualsuccessfulphase');need(strictjson((RUN/'own-process-cleanup.json').read_bytes())['remaining']==[],'Actualscopedcleanup')
   mp=RUN/'checkpoints/full_state/global_step_300/manifest.json';result['checkpoint']=validate_checkpoint(strictjson(mp.read_bytes()),strictjson(mp.with_name('checkpoint-complete.json').read_bytes()),RUN);state=strictjson((RUN/'checkpoints/algorithm/global_step_300/state.json').read_bytes());need(state['method']=='bapo'and state['global_step']==state['actual_outer_updates']==300 and state['algorithm_source_commit']==SOURCE and state['bapo']==result['logs']['expected_bapo'],'Algorithmstate300');result['final_metrics']=final_metrics(args.current_log.read_bytes());restore=[strictjson((RUN/'recovery-witness'/('rank_%d.json'%i)).read_bytes())for i in range(8)];need(all(r['pid']==restore[r['rank']]['new_pid']and restore[r['rank']]['actual_env']['SLURM_JOB_ID']=='3292'and restore[r['rank']]['target_run_root']==str(RUN)and restore[r['rank']]['target_source_commit']==SOURCE for r in result['checkpoint']['reports']),'FinalrankPID/source lineageactual3292');native=[v['event']for v in records(args.current_log.read_bytes(),'FULL_STATE_RANK_JSON=')if v['event'].get('step')==300];need(len(native)==8 and sorted(native,key=lambda r:r['rank'])==sorted(result['checkpoint']['reports'],key=lambda r:r['rank']),'Finalactual native producer markers')
   need(filebind(ROOT/'tools/guard_journal.py')['sha256']==GUARDSHA and filebind(ROOT/'tools/resource_watch.py')['sha256']==WATCHSHA,'Exactguard tools');sp=importlib.util.spec_from_file_location('actual_guard_journal',ROOT/'tools/guard_journal.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g);result['guardian']=g.replay(RUN/'resource-watch.jsonl',3292,str(RUN),SOURCE,ROOT/'tools/resource_watch.py')
   # All terminal logs and members remain remote; archive small evidence only, bind big native files fullstream.
   names=('source-commit.txt','resolved-config.yaml','slurm-allocation.txt','gpus-before.csv','processes-before.csv','gpus-exit.csv','cpu-process-monitor.jsonl','gpu-monitor.csv','process-monitor.csv','train.log','phase.log','retriever.log','resource-watch.jsonl','own-process-cleanup.json','launch-approval.json','tools-manifest.json','cpu-gate.json','recovery-manifest.json','recovery-permission.json','recovery-permission-check.json','permission-resolved-config.yaml','hardware-receipt-check.log','retrieval-assets-locality.json','runtime-profile.txt','retriever-health.json')
   originals=[filebind(RUN/name)for name in names]+[filebind(p)for p in sorted((RUN/'recovery-witness').glob('*.json'))]
   originals+=[row for row in result['checkpoint']['members']if P(row['path']).suffix not in ('.pt','.safetensors')]
   originals+=[result['checkpoint']['manifest'],result['checkpoint']['complete'],result['terminal'],filebind(args.prefix)]
   originals=list({row['path']:row for row in originals}.values())
   need(filebind(ROOT/'common-data/manifest.json')['sha256']==DATA_SHA,'Actualfinal split source/hash');dm=strictjson((ROOT/'common-data/manifest.json').read_bytes());need(dm['splits']['final']['rows']==51713 and dm['splits']['final']['source_counts']==COUNTS,'Finalactualseven-source split');result['data_protocol']=dm;result['raw_inventory']=originals
   with tarfile.open(args.output_dir/'small-originals.tar.gz','x:gz')as archive:
    for row in originals:archive.add(row['path'],arcname=('terminal/slurm-terminal.txt'if row['path']==str(args.terminal)else 'lineage/train-prefix1-25.log'if row['path']==str(args.prefix)else str(P(row['path']).relative_to(ROOT))),recursive=False)
   need(all(filebind(P(x['path']))==x for x in originals),'Raw originals changed during archive')
   if args.mode=='cpu-hf':
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(args.cpu_threads),MKL_NUM_THREADS=str(args.cpu_threads));sys.path.insert(0,str(ROOT/'repo'));import torch;torch.set_num_threads(args.cpu_threads);need(not torch.cuda.is_initialized(),'No CUDA');result['driver']=cpu_driver(RUN,result['logs']['expected_bapo']);ref=reference(args.reference_tool);ck=RUN/'checkpoints/actor/global_step_300';need({p.name for p in ck.iterdir()if p.is_file()}==ref.HF_FILES,'ExactHF12');need(filebind(MODEL.resolve()/'VERIFIED-MODEL-MANIFEST.json')['sha256']==MODEL_SHA and filebind(ROOT/'common-data/manifest.json')['sha256']==DATA_SHA,'Officialshared inputs');mm=strictjson((MODEL.resolve()/'VERIFIED-MODEL-MANIFEST.json').read_bytes());need(mm['repo']=='Qwen/Qwen2.5-3B-Instruct'and mm['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','OfficialInstruct');bases=[filebind(MODEL.resolve()/x['file'])for x in mm['files']];need(all(b['sha256']==v['sha256']and b['bytes']==v['bytes']for b,v in zip(bases,mm['files'])),'Officialmodel fullhash');hstart=time.monotonic();hc=time.process_time();result['cpu_hf']=ref.hf_proof(MODEL.resolve(),ck);result['cpu_hf_cost']=dict(wall_seconds=time.monotonic()-hstart,process_cpu_seconds=time.process_time()-hc,monetary_price_assumed=False);need(all(filebind(P(b['path']))==b for b in bases),'Officialbase changed duringCPUHF');post=validate_checkpoint(strictjson(mp.read_bytes()),strictjson(mp.with_name('checkpoint-complete.json').read_bytes()),RUN);need(post==result['checkpoint'],'Checkpoint changed acrossactualCPUHF');result['actual_cpu_hf_executed']=True
  result['extraction_complete']=True
 except Exception as exc:result['issues'].append(type(exc).__name__+': '+str(exc));result['extraction_complete']=False
 result['cost_scope']='Current3292 reserved GPU and extractor CPU only; old effective25/failed runs/hardware cost must remain separately reported'
 result['cost_actual']=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,platform=sys.platform,extra_gpu_execution=False);new_json(args.output_dir/'evidence.json',result);print(json.dumps(dict(extraction_complete=result['extraction_complete'],issues=result['issues'],final_accepted=False)));return 0 if result['extraction_complete']else 2
if __name__=='__main__':raise SystemExit(main())
