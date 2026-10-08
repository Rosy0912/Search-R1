import importlib.util,json,tempfile,pathlib,yaml
chat=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');sp=importlib.util.spec_from_file_location('v',chat/'work/validate_tree_stage.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
results=[]
with tempfile.TemporaryDirectory() as temp:
 run=pathlib.Path(temp);log=run/'stage.log'
 def lines(n):return '\n'.join('(TaskRunner pid=123) '+ 'step:'+str(i)+' - '+' - '.join(k+':'+str(i if k=='training/actual_outer_updates' else 1e-12 if k=='actor/grad_norm' else 1) for k in sorted(v.REQUIRED)) for i in range(1,n+1))
 for name,text,expected,ok in [('two',lines(2),2,True),('five',lines(5),5,True),('two_not_five',lines(2),5,False),('five_not_two',lines(5),2,False),('duplicate',lines(2)+'\n'+lines(2),2,False),('nonfinite',lines(2).replace('1e-12','nan'),2,False),('zero_optimizer',lines(2).replace('actor/optimizer_steps:1','actor/optimizer_steps:0'),2,False),('zero_grad',lines(2).replace('1e-12','0'),2,False),('missing_metric',lines(2).replace('actor/ppo_kl:1 - ',''),2,False)]:
  log.write_text(text)
  try:v.parse_console(log,expected);accepted=True
  except (AssertionError,ValueError):accepted=False
  assert accepted==ok,name;results.append({'name':name,'fixture_only':True,'accepted':accepted})
 log.write_text(lines(2));cfg={'trainer':{'total_training_steps':3,'smoke_only':True,'default_local_dir':str(run/'checkpoints')},'baselines':{'method':'tree_grpo','base_commit':'base','algorithm_source_commit':'algo'}}
 (run/'resolved.yaml').write_text(yaml.safe_dump(cfg));(run/'cost.json').write_text(json.dumps({'stage':'two','exit_code':0,'allocated_gpus':8,'start_epoch':1,'end_epoch':2,'job_id':9}))
 ck=run/'checkpoints/actor/global_step_2';ck.mkdir(parents=True);(ck/'model.safetensors').write_text('NOT_REAL_WEIGHTS')
 st=run/'checkpoints/algorithm/global_step_2';st.mkdir(parents=True);state={'method':'tree_grpo','global_step':2,'base_commit':'base','algorithm_source_commit':'algo','resume_supported':False};(st/'state.json').write_text(json.dumps(state));v.inspect(run,'two');results.append({'name':'structure_only','accepted_structure':True,'real_hf_not_tested':True})
 state['global_step']=5;(st/'state.json').write_text(json.dumps(state))
 try:v.inspect(run,'two');raise RuntimeError('wrong state passed')
 except AssertionError:results.append({'name':'wrong_state_refused','accepted':False})
 try:v.inspect(run/'absent','two');raise RuntimeError('absent run passed')
 except AssertionError:results.append({'name':'absent_artifacts_refused','accepted':False})
(chat/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/evidence/stage-validator-probe.json').write_text(json.dumps(results,indent=2));print(len(results),'fixture checks passed; no real stage accepted')
