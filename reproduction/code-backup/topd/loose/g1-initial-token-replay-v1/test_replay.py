import importlib.util,pathlib,json,pytest
s=importlib.util.spec_from_file_location('replay',pathlib.Path(__file__).with_name('replay.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
ROOT=pathlib.Path(__file__).resolve().parents[2]
DUMP=ROOT/'work/g1-32b-fallback-candidate-v3/ts_probe_dump.py'
RAW=ROOT/'outputs/remote-evidence/g1-initial3124/initial/outer-0'
def test_original_source_identity_drift_refused(tmp_path):
 p=tmp_path/'dump.py';p.write_text('def normalize(x): return x')
 with pytest.raises(ValueError,match='source drift'):r.grader_functions(p)
def test_real3124_original_grader_replay():
 if not RAW.exists():pytest.skip('actual raw3124 not packaged in native CPU fixtures')
 result=r.labels(r.load(RAW/'generation-trace.json'),r.load(RAW/'generation-token-trace.json'),r.grader_functions(DUMP))
 assert result['closedbook_questions']==300 and result['valid_rollouts']==1276
 assert result['all_original_contexts_and_labels_identical']
def test_real3124_label_tamper_refused():
 if not RAW.exists():pytest.skip('actual raw3124 not packaged in native CPU fixtures')
 trace=r.load(RAW/'generation-trace.json');trace['flips'][0]=1-trace['flips'][0]
 with pytest.raises(ValueError,match='counterfactual labels differ'):r.labels(trace,r.load(RAW/'generation-token-trace.json'),r.grader_functions(DUMP))

def test_replay_loop_preserves_original_stop_truncation_and_EOS():
 tok=type('Tok',(),{'eos_token_id':9})()
 def increment(tok,allids,prev,prefix,read,**kwargs):return ['x'],{1:'<answer>yes',2:'</answer>extra',9:''}[allids[-1]],0,0
 def stop(seq,count,p):
  if not count:return None
  for text in p.stop:
   at=seq.output_text.find(text)
   if at>=0:seq.output_text=seq.output_text[:at+len(text)];return text
 f={'convert_prompt_ids_to_tokens':lambda *a,**k:([],0,0),'detokenize_incrementally':increment,'_check_stop_strings':stop};sampling={'stop':['</answer>'],'include_stop_str_in_output':True}
 assert r.decode_original(tok,{'prompt_token_ids':[1],'token_ids':[1,2]},sampling,f)==('<answer>yes</answer>','</answer>')
 with pytest.raises(ValueError,match='tokens after original stop'):r.decode_original(tok,{'prompt_token_ids':[1],'token_ids':[1,2,9]},sampling,f)
