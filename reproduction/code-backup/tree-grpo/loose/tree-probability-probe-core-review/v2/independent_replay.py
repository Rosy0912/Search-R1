import pathlib,importlib.util,json,torch
D=pathlib.Path(__file__).resolve().parent;S=D.parents[1]/'tree-probability-probe-core-v2';T=S/'tests/baselines/test_probability_probe_capture.py'
s=importlib.util.spec_from_file_location('core_v2_independent_fixture',T);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);P=m.P;torch.set_num_threads(4)
def target(a,name='w',group=None):
 r=dict(logical_name=name,module=a.actor_module,attribute='weight',expected_cast_digest=P.tensor_digest(a.actor_module.weight))
 if group:r['expected_storage_alias_group']=group
 return r
def refused(trace):
 try:trace.receipt()
 except ValueError:return True
 raise AssertionError('invalid receipt published')
cases=[]
a,b=m.Actor(),m.Actor();tr=P.EffectiveWeightTrace([target(a,'a','tied'),target(b,'b','tied')])
try:
 with tr:P.capture_original_call(a,m.batch(),1.);P.capture_original_call(b,m.batch(),1.)
except ValueError as e:assert 'storage alias differs' in str(e)
else:raise AssertionError('alias not refused')
assert refused(tr);cases.append('original_alias_failure_receipt_rejected')
a=m.Actor();tr=P.EffectiveWeightTrace([target(a)]);f=a._forward_micro_batch
def failure(data,temperature):f(data,temperature);raise RuntimeError('original entry failed after module')
a._forward_micro_batch=failure
try:
 with tr:P.capture_original_call(a,m.batch(),1.)
except RuntimeError:pass
else:raise AssertionError('original error missing')
assert len(tr.records)==1 and refused(tr);cases.append('original_forward_exception_receipt_rejected')
a=m.Actor();tr=P.EffectiveWeightTrace([target(a)]);assert refused(tr)
with tr:
 c=P.capture_original_call(a,m.batch(),1.);assert refused(tr)
r=tr.receipt();assert r['trace_completed_successfully'] is True and r['tied_alias_identity_proven'] is False and r['acceptance'] is False
v=P.compare_same_logits(c,m.NATIVE,m.BRANCH);assert v['metadata']['selected_native_exactly_matches_original'];cases.append('legitimate_complete_trace_original_call_and_normalization_pass')
try:
 with tr:pass
except ValueError:pass
else:raise AssertionError('incomplete reuse accepted')
assert refused(tr);cases.append('reuse_without_forward_cannot_publish_stale_receipt')
contract=json.loads((S/'docs/TREE_PROBE_SOURCE_CONTRACT.json').read_text());assert all(P.file_sha(S/p)==h for p,h in contract['unchanged_source_sha256'].items())
assert not torch.cuda.is_initialized()
out=dict(status='INDEPENDENT_CORE_V2_COUNTEREXAMPLES_FIXED_CPU_PASS',module_sha256=P.file_sha(m.MODULE),cases=cases,unchanged_source_pins_match=True,cuda_initialized=False,actual_model_loaded=False,gpu_ready=False)
(D/'INDEPENDENT_REPLAY_PROOF.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
