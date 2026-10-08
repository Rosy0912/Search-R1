"""Independent fault-injection tests; only external observer files, no production edit."""
import json,types
import pytest
import validation_cost_hooks as hooks
import test_validation_cost_hooks as fixture

def fail_event(monkeypatch,kind):
 original=json.dumps
 def dumps(value,*a,**kw):
  if isinstance(value,dict) and value.get('kind')==kind:raise OSError('observer ledger failure')
  return original(value,*a,**kw)
 monkeypatch.setattr(hooks.json,'dumps',dumps)

def test_generation_error_not_replaced_by_failure_event_io(tmp_path,monkeypatch):
 error=RuntimeError('genuine GPU generation failure');T=fixture.setup_case(error);t=T();old=t.actor_rollout_wg.generate_sequences;hooks.wrap_trainer(T,fixture.Manager,tmp_path);fail_event(monkeypatch,'generation_failed')
 with pytest.raises(BaseException) as caught:t._validate()
 assert caught.value is error
 assert t.actor_rollout_wg.generate_sequences==old and 'generate_sequences' not in vars(t.actor_rollout_wg)

def test_search_error_not_replaced_by_failure_event_io(tmp_path,monkeypatch):
 error=RuntimeError('genuine original HTTP failure');T=fixture.setup_case();t=T()
 def bad_search(self,queries):raise error
 monkeypatch.setattr(fixture.Manager,'_batch_search',bad_search);hooks.wrap_trainer(T,fixture.Manager,tmp_path);fail_event(monkeypatch,'http_failed')
 with pytest.raises(BaseException) as caught:t._validate()
 assert caught.value is error and fixture.Manager._batch_search is bad_search

def test_original_exception_survives_final_ledger_hash_failure(tmp_path,monkeypatch):
 error=RuntimeError('original validation failure');T=fixture.setup_case(error);t=T();old=t.actor_rollout_wg.generate_sequences;hooks.wrap_trainer(T,fixture.Manager,tmp_path);original=hooks.sha
 def sha(path):
  if str(path).endswith('.jsonl'):raise OSError('ledger vanished')
  return original(path)
 monkeypatch.setattr(hooks,'sha',sha)
 with pytest.raises(BaseException) as caught:t._validate()
 assert caught.value is error and t.actor_rollout_wg.generate_sequences==old
 assert not list(tmp_path.glob('*.json'))

def test_inherited_manager_and_instance_callable_exact_restore(tmp_path,monkeypatch):
 class Child(fixture.Manager):pass
 monkeypatch.setattr(fixture,'Manager',Child)
 T=fixture.setup_case();t=T();original=t.actor_rollout_wg.generate_sequences;t.actor_rollout_wg.generate_sequences=original
 before=dict(vars(Child));hooks.wrap_trainer(T,Child,tmp_path);result=t._validate()
 assert result is t.results and t.actor_rollout_wg.__dict__['generate_sequences'] is original
 for n in ['run_llm_loop','_generate_with_gpu_padding','_batch_search']:assert (n in vars(Child))==(n in before)

def test_final_wrong_dataset_rejected_before_generation(tmp_path):
 T=fixture.setup_case();t=T();hooks.wrap_trainer(T,fixture.Manager,tmp_path,final_expected_rows=51713)
 with pytest.raises(AssertionError):t._prepare_final_validation()
 assert not t.actor_rollout_wg.calls and not getattr(t,'_tree_cost_final_phase',False) and not list(tmp_path.iterdir())

def test_success_ledger_hash_failure_cannot_publish_complete(tmp_path,monkeypatch):
 T=fixture.setup_case();t=T();old=t.actor_rollout_wg.generate_sequences;hooks.wrap_trainer(T,fixture.Manager,tmp_path);original=hooks.sha
 def sha(path):
  if str(path).endswith('.jsonl'):raise OSError('successful cost ledger unreadable')
  return original(path)
 monkeypatch.setattr(hooks,'sha',sha)
 with pytest.raises(OSError):t._validate()
 assert t.actor_rollout_wg.generate_sequences==old and not list(tmp_path.glob('*.json'))

def test_empty_http_batches_count_requests_without_fabricating_queries(tmp_path,monkeypatch):
 class EmptyManager(fixture.Manager):
  def run_llm_loop(self,gen_batch,initial_input_ids):
   out=self._generate_with_gpu_padding(gen_batch);assert self._batch_search([])=={'result':[]};return out
 monkeypatch.setattr(fixture,'Manager',EmptyManager);T=fixture.setup_case();t=T();hooks.wrap_trainer(T,EmptyManager,tmp_path);t._validate()
 report=json.loads(next(tmp_path.glob('*.json')).read_text());c=report['counters'];assert c['http_attempts']==c['http_completed']==2 and c['http_query_events']==c['http_returned_query_rows']==0
 assert c['submitted_rows']==16 and c['returned_nonpad_model_token_events']==48

def test_nonmatching_manager_search_is_passthrough_not_counted(tmp_path,monkeypatch):
 class OtherManager(fixture.Manager):
  def run_llm_loop(self,gen_batch,initial_input_ids):
   other=type(self).__new__(type(self));other.actor_rollout_wg=object();result=other._batch_search(['unrelated']);assert result=={'result':[[]]}
   return self._generate_with_gpu_padding(gen_batch)
 monkeypatch.setattr(fixture,'Manager',OtherManager);T=fixture.setup_case();t=T();hooks.wrap_trainer(T,OtherManager,tmp_path);t._validate()
 c=json.loads(next(tmp_path.glob('*.json')).read_text())['counters'];assert c['http_attempts']==c['http_query_events']==0
