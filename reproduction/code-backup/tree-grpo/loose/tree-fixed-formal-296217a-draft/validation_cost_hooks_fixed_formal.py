"""External validation-only cost observer; frozen training files remain untouched."""
import functools,hashlib,json,math,os,pathlib,re,resource,time,sys


def sha(path):
 return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def wrap_trainer(Trainer,Manager,root,final_expected_rows=None):
 """Called in the actual Ray main task; tests may supply isolated fake classes."""
 assert not getattr(Trainer,'_tree_cost_observer_installed',False)
 original_validate=Trainer._validate;original_final=Trainer._prepare_final_validation
 sequence=[0]
 @functools.wraps(original_final)
 def prepare_final(self,*args,**kwargs):
  result=original_final(self,*args,**kwargs)
  if final_expected_rows is not None:assert len(self.val_dataset)==final_expected_rows, 'Final evaluation dataset coverage differs from fixed protocol'
  self._tree_cost_final_phase=True
  return result
 @functools.wraps(original_validate)
 def validate(self,*args,**kwargs):
  assert self.config.do_search,'Observer covers the original search validation path only'
  sequence[0]+=1;seq=sequence[0];pid=os.getpid()
  phase='final' if getattr(self,'_tree_cost_final_phase',False) else ('pretrain' if self.global_steps==0 else 'periodic')
  directory=pathlib.Path(root);directory.mkdir(parents=True,exist_ok=True)
  stem=f'validation-{pid}-{seq}';ledger=directory/(stem+'.jsonl');report=directory/(stem+'.json')
  assert not ledger.exists() and not report.exists()
  counters=dict(generation_attempts=0,generation_completed=0,generation_failed=0,submitted_rows=0,returned_rows=0,submitted_prefill_token_events=0,returned_nonpad_model_token_events=0,real_sample_rows=0,active_row_events=0,submitted_padding_row_events=0,http_attempts=0,http_completed=0,http_failed=0,http_query_events=0,http_returned_query_rows=0)
  start=time.time();clock=time.perf_counter();cpu=time.process_time();event=0
  def emit(kind,**fields):
   nonlocal event
   event+=1
   record=dict(event=event,kind=kind,epoch=time.time(),pid=pid,validation_sequence=seq,global_step=int(self.global_steps),phase=phase,**fields)
   with ledger.open('a') as f:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
  pad=int(self.tokenizer.pad_token_id);eos=self.tokenizer.eos_token_id
  emit('validation_start',dataset_rows=len(self.val_dataset),pad_token_id=pad,eos_token_id=eos,observer_source_sha256=sha(__file__),job_id=os.environ.get('SLURM_JOB_ID'),run=str(directory.parent),count_convention='raw returned response positions != PAD; excludes PAD and also EOS when PAD==EOS; engine emission lengths unavailable')
  wg=self.actor_rollout_wg;original_generate=wg.generate_sequences;had_own='generate_sequences' in vars(wg)
  original_loop=Manager.run_llm_loop;original_padding=Manager._generate_with_gpu_padding;original_search=Manager._batch_search
  manager_owned={name:name in vars(Manager) for name in ['run_llm_loop','_generate_with_gpu_padding','_batch_search']}
  @functools.wraps(original_generate)
  def generate(batch,*a,**kw):
   call=counters['generation_attempts']+1;rows=int(batch.batch['input_ids'].shape[0]);prefill=int(batch.batch['attention_mask'].sum().item())
   counters['generation_attempts']+=1;counters['submitted_rows']+=rows;counters['submitted_prefill_token_events']+=prefill
   emit('generation_start',call=call,rows=rows,prefill_token_events=prefill);t=time.perf_counter()
   try:out=original_generate(batch,*a,**kw)
   except BaseException as err:
    counters['generation_failed']+=1
    try:emit('generation_failed',call=call,error_type=type(err).__name__,wall_seconds=time.perf_counter()-t)
    except BaseException:print('TREE_VALIDATION_COST_FAILURE_RECORD_UNAVAILABLE',file=sys.stderr,flush=True)
    raise
   tokens=int((out.batch['responses']!=pad).sum().item());returned=int(out.batch['responses'].shape[0])
   counters['generation_completed']+=1;counters['returned_rows']+=returned;counters['returned_nonpad_model_token_events']+=tokens
   emit('generation_completed',call=call,rows=returned,nonpad_model_token_events=tokens,wall_seconds=time.perf_counter()-t)
   return out
  @functools.wraps(original_loop)
  def loop(manager,gen_batch,initial_input_ids,*a,**kw):
   if manager.actor_rollout_wg is wg:
    rows=int(gen_batch.batch['input_ids'].shape[0]);counters['real_sample_rows']+=rows;emit('real_sample_batch',rows=rows)
   return original_loop(manager,gen_batch,initial_input_ids,*a,**kw)
  @functools.wraps(original_padding)
  def padding(manager,active_batch,*a,**kw):
   if manager.actor_rollout_wg is wg:
    rows=int(active_batch.batch['input_ids'].shape[0]);g=int(manager.config.num_gpus);extra=(-rows)%g
    counters['active_row_events']+=rows;counters['submitted_padding_row_events']+=extra;emit('gpu_padding_submission',active_rows=rows,padding_rows=extra)
   return original_padding(manager,active_batch,*a,**kw)
  @functools.wraps(original_search)
  def search(manager,queries,*a,**kw):
   if manager.actor_rollout_wg is not wg:return original_search(manager,queries,*a,**kw)
   call=counters['http_attempts']+1;counters['http_attempts']+=1;counters['http_query_events']+=len(queries)
   emit('http_start',call=call,queries=len(queries));t=time.perf_counter()
   try:out=original_search(manager,queries,*a,**kw)
   except BaseException as err:
    counters['http_failed']+=1
    try:emit('http_failed',call=call,error_type=type(err).__name__,wall_seconds=time.perf_counter()-t)
    except BaseException:print('TREE_VALIDATION_COST_FAILURE_RECORD_UNAVAILABLE',file=sys.stderr,flush=True)
    raise
   counters['http_completed']+=1;counters['http_returned_query_rows']+=len(out['result'])
   emit('http_completed',call=call,query_result_rows=len(out['result']),wall_seconds=time.perf_counter()-t)
   return out
  result=None;failure=None;wg.generate_sequences=generate;Manager.run_llm_loop=loop;Manager._generate_with_gpu_padding=padding;Manager._batch_search=search
  try:
   result=original_validate(self,*args,**kwargs)
   actual=sum(int(v) for k,v in result.items() if k.startswith('val/evaluated_count/'))
   assert actual==len(self.val_dataset)==counters['real_sample_rows'],'Observed validation coverage differs from actual dataset'
   assert counters['generation_attempts']==counters['generation_completed'] and counters['generation_failed']==0
   assert counters['http_attempts']==counters['http_completed'] and counters['http_failed']==0
   emit('validation_completed',evaluated_rows=actual)
   return result
  except BaseException as err:
   failure=type(err).__name__
   try:emit('validation_failed',error_type=failure)
   except BaseException:print('TREE_VALIDATION_COST_FAILURE_RECORD_UNAVAILABLE',file=sys.stderr,flush=True)
   raise
  finally:
   if had_own:wg.generate_sequences=original_generate
   else:del wg.generate_sequences
   for name,original in [('run_llm_loop',original_loop),('_generate_with_gpu_padding',original_padding),('_batch_search',original_search)]:
    if manager_owned[name]:setattr(Manager,name,original)
    else:delattr(Manager,name)
   try:
    summary=dict(status='VALIDATION_COST_COMPLETE' if failure is None else 'VALIDATION_COST_FAILED',phase=phase,pid=pid,validation_sequence=seq,global_step=int(self.global_steps),job_id=os.environ.get('SLURM_JOB_ID'),run=str(directory.parent),start_epoch=start,end_epoch=time.time(),wall_seconds=time.perf_counter()-clock,trainer_process_cpu_seconds=time.process_time()-cpu,trainer_process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),dataset_rows=len(self.val_dataset),evaluated_rows=None if failure else sum(int(v) for k,v in result.items() if k.startswith('val/evaluated_count/')),counters=counters,ledger_sha256=sha(ledger),observer_source_sha256=sha(__file__),error_type=failure,pad_token_id=pad,eos_token_id=eos,limitations='Nonpad returned token count matches Tree ledger convention; PAD==EOS excludes EOS; genuine emitted PAD positions and exact engine emission lengths unavailable. Prefill is submitted repeated token events, not unique/cached tokens or FLOPs. CPU time only this Ray trainer process; remote workers/encoder/index separate. Wall includes original validation and observer overhead.')
    pending=report.with_suffix('.pending')
    with pending.open('x') as f:
     json.dump(summary,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.link(pending,report);pending.unlink()
   except BaseException:
    if failure is None:raise
    print('TREE_VALIDATION_COST_FAILURE_REPORT_UNAVAILABLE',file=sys.stderr,flush=True)
 Trainer._validate=validate;Trainer._prepare_final_validation=prepare_final;Trainer._tree_cost_observer_installed=True


def install():
 from verl.trainer.ppo.ray_trainer import RayPPOTrainer
 from search_r1.llm_agent.generation import LLMGenerationManager
 import inspect
 source=inspect.getsourcefile(RayPPOTrainer)
 # Source hash is pinned by the reviewed entry/receipt; keep a runtime receipt too.
 expected=os.environ['TREE_VALIDATION_TRAINER_SHA256'];assert sha(source)==expected
 output=os.environ['SEARCH_R1_BASELINE_OUTPUT'];path=pathlib.Path(output).resolve()
 job=os.environ['SLURM_JOB_ID'];assert re.fullmatch(r'[1-9][0-9]*',job)
 assert path==pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs')/('tree-fixed-formal-'+job)/'output', 'Observer output must match this fixed-formal Slurm job'
 wrap_trainer(RayPPOTrainer,LLMGenerationManager,path/'validation-cost',final_expected_rows=51713)
 print('TREE_VALIDATION_COST_INSTALLED_JSON='+json.dumps(dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],trainer_source_sha256=expected,observer_source_sha256=sha(__file__),output=str(path))),flush=True)


def measured_remote(original,installer=install):
 import ray
 @ray.remote
 def measured_task(config):
  installer()
  return original(config)
 return measured_task
