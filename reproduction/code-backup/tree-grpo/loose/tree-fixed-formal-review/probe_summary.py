import ast,json,tempfile,pathlib,hashlib,sys,contextlib,io,copy
from unittest.mock import patch
R=pathlib.Path.cwd();O=R/'work/tree-fixed-formal-review';SRC=R/'work/summarize_tree_fixed_formal_296217a.py';COMMIT='296217a4b676a26a92fc729f5d9882b785dbfa96';remote='/dataset/shaoshitong/search-r1-rl/tree-grpo';cases=[]
with tempfile.TemporaryDirectory() as d:
 root=pathlib.Path(d).resolve();run=root/'runs/tree-fixed-formal-200';cpu=root/'runs/tree-fixed-formal-cpu-200-201';run.mkdir(parents=True);cpu.mkdir();out=root/'out.json';cfg='same-config\n';(run/'resolved.yaml').write_text(cfg);(run/'output/entry-audit').mkdir(parents=True);(run/'output/entry-audit/entry-config.yaml').write_text(cfg)
 result=dict(status='ACTUAL_STAGE_COMPLETED_UNREVIEWED',training_body_exit_code=0,entry_config_verified=True,cleanup_residual={},code_commit=COMMIT,job_id='200');cpures=dict(status='HF500_FULL_RELOAD_COMPLETE_PENDING_REVIEW',exit_code=0,parent_job_id='200',cpu_job_id='201');hf=dict(status='HF_RELOAD_PARAMETER_DELTA_ACCEPTED',all_parameters_finite=True,cpu_forward_logits_finite=True,tokenizer_vocab_equal=True,bos_eos_equal=True,changed_parameters=100,changed_tensors=2,max_abs_parameter_delta=.001,checkpoint=str(run/'output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_500'),hf_loading_info={'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]})
 metrics={};sources=[f's{i}' for i in range(7)];keys=['f1','em','answer_format','format_rate','generated_tokens','search_calls'];values=[.25,.125,.5,.5,10,1]
 for s,n in zip(sources,[100]*6+[51113]):
  metrics[f'val/evaluated_count/{s}']=metrics[f'val/qa_count/{s}']=n
  for k,v in zip(keys,values):metrics[f'val/{k}/{s}']=v
 metrics['val/qa_count/all']=51713
 for k,v in zip(keys,values):metrics[f'val/{k}/all']=v
 base={'actor/optimizer_steps':12,'actor/grad_norm':1,'actor/pg_loss':.001,'actor/entropy_loss':1,'actor/pg_clipfrac':.1,'actor/ppo_kl':.001,'rollout/generated_tokens':100,'rollout/prefill_tokens':200,'rollout/retrieval_calls':10,'rollout/trajectories':3072,'rollout/padded_rows':0,'rollout/context_mismatches':0}
 lines=['step:'+str(s)+' - '+' - '.join(k+':'+str(v) for k,v in {**base,'training/actual_outer_updates':s}.items()) for s in range(1,501)]
 rankdir=run/'output/tree-rank-audit';rankdir.mkdir();prof=rankdir/'memory-profile';prof.mkdir()
 for s in range(1,501):
  for rank in range(8):
   j=dict(outer_step=s,rank=rank,world_size=8,local_rows=384,local_minibatch=32,local_microbatch=8,epochs=1,code_commit=COMMIT,job_id='200',successful_optimizer_calls=12,finite_grad_norms=12,existing_grad_norms=[1]*12);(rankdir/f'step-{s:06d}-rank-{rank:02d}.json').write_text(json.dumps(j));(prof/f'step-{s:06d}-rank-{rank:02d}.json').write_text('{}')
 vdir=run/'output/validation-cost';vdir.mkdir();reports=[]
 for seq,(phase,step,rows) in enumerate([('pretrain',0,64)]+[('periodic',s,64) for s in range(50,501,50)]+[('final',500,51713)],1):
  ledger=vdir/f'validation-300-{seq}.jsonl';ledger.write_text('fixture ledger only\n');j=dict(status='VALIDATION_COST_COMPLETE',dataset_rows=rows,evaluated_rows=rows,phase=phase,global_step=step,pid=300,validation_sequence=seq,ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest());reports.append(j);(vdir/f'{seq:02d}.json').write_text(json.dumps(j))
 (run/'cost.json').write_text('{}');(cpu/'slurm-cpu-terminal.txt').write_text('JobId=201 JobState=COMPLETED ExitCode=0:0')
 class Remap(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str) and node.value.startswith(remote):return ast.copy_location(ast.Constant(node.value.replace(remote,str(root))),node)
   return node
 code=compile(ast.fix_missing_locations(Remap().visit(ast.parse(SRC.read_text()))),str(SRC),'exec')
 for name in ['success','failed_parent','failed_HF_collector_exit0','wrong_parent','wrong_checkpoint','nonfinite_HF','missing_loading_key','bad_terminal','wrong_final_count','wrong_periodic','bad_ledger','rank_wrong_job']:
  rs=copy.deepcopy(result);cs=copy.deepcopy(cpures);hs=copy.deepcopy(hf);ms=copy.deepcopy(metrics)
  if name=='failed_parent':rs['status']='FAILED'
  if name=='failed_HF_collector_exit0':cs['status']='PARENT_FAILED_HF_ACCEPTANCE_NOT_ATTEMPTED'
  if name=='wrong_parent':cs['parent_job_id']='999'
  if name=='wrong_checkpoint':hs['checkpoint']=str(run/'output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_499')
  if name=='nonfinite_HF':hs['all_parameters_finite']=False
  if name=='missing_loading_key':hs['hf_loading_info']['missing_keys']=['missing.weight']
  if name=='wrong_final_count':ms['val/qa_count/all']=64
  (run/'run-result.json').write_text(json.dumps(rs));(cpu/'result.json').write_text(json.dumps(cs));(cpu/'hf500-full-reload.json').write_text(json.dumps(hs));(run/'stage.log').write_text('\n'.join(lines)+'\nSEARCH_R1_FINAL_METRICS_JSON='+json.dumps(ms)+'\n');(cpu/'slurm-cpu-terminal.txt').write_text('JobId=201 JobState='+('FAILED ExitCode=1:0' if name=='bad_terminal' else 'COMPLETED ExitCode=0:0'))
  if name=='wrong_periodic':p=vdir/'02.json';j=json.loads(p.read_text());j['global_step']=49;p.write_text(json.dumps(j))
  if name=='bad_ledger':p=vdir/'validation-300-1.jsonl';p.write_text('changed ledger')
  if name=='rank_wrong_job':p=rankdir/'step-000001-rank-00.json';j=json.loads(p.read_text());j['job_id']='999';p.write_text(json.dumps(j))
  error=None
  with patch.object(sys,'argv',[str(SRC),'--run',str(run),'--hf-cpu-run',str(cpu),'--output',str(out)]),contextlib.redirect_stdout(io.StringIO()):
   try:exec(code,{'__name__':'__main__'})
   except (AssertionError,ValueError) as e:
    import traceback
    error=traceback.format_exc()
  assert (error is None)==(name=='success'),(name,error)
  cases.append({'case':name,'accepted':error is None,'error':error})
  if out.exists():out.unlink()
  if name=='wrong_periodic':(vdir/'02.json').write_text(json.dumps(reports[1]))
  if name=='bad_ledger':(vdir/'validation-300-1.jsonl').write_text('fixture ledger only\n')
  if name=='rank_wrong_job':j['job_id']='200';(rankdir/'step-000001-rank-00.json').write_text(json.dumps(j))
(O/'SUMMARY_CASES.json').write_text(json.dumps({'status':'PASS_TOY_CPU_EXTRACTOR_ONLY','source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'cases':cases,'scope':'Exact summary AST with only remote root remapped, actual parse_console500, real4000 toy rank files/4000 empty presence-only profile files/12 toy ledger reports/seven artificial datasets; no actual scientific evidence or GPU/HF'},indent=2));print('12 summary positive/negative extraction cases passed; failed collector exit0 cannot make summary success')
