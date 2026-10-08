import pathlib,importlib.util,json,hashlib,torch
D=pathlib.Path(__file__).resolve().parent;T=D.parent/'tree-probability-probe-source/tests/baselines/test_probability_probe_capture.py'
s=importlib.util.spec_from_file_location('core_review_fixture',T);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);P=m.P
torch.set_num_threads(4);proof=[]
a,b=m.Actor(),m.Actor()
targets=[dict(logical_name=str(i),module=x.actor_module,attribute='weight',expected_cast_digest=P.tensor_digest(x.actor_module.weight),expected_storage_alias_group='tied') for i,x in enumerate([a,b])]
trace=P.EffectiveWeightTrace(targets)
try:
 with trace:P.capture_original_call(a,m.batch(),1.);P.capture_original_call(b,m.batch(),1.)
except ValueError as e:assert 'storage alias differs' in str(e)
else:raise AssertionError('alias failure not seen')
r=trace.receipt();proof.append(dict(case='receipt_after_failed_alias_exit',published=True,receipt=r))
a=m.Actor();original=a._forward_micro_batch
trace=P.EffectiveWeightTrace([dict(logical_name='w',module=a.actor_module,attribute='weight',expected_cast_digest=P.tensor_digest(a.actor_module.weight))])
def failed(data,temp):
 original(data,temp);raise RuntimeError('after native forward')
a._forward_micro_batch=failed
try:
 with trace:P.capture_original_call(a,m.batch(),1.)
except RuntimeError:pass
else:raise AssertionError('forward failure not seen')
r=trace.receipt();proof.append(dict(case='receipt_after_original_forward_exception',published=True,receipt=r))
assert not torch.cuda.is_initialized()
out={'status':'TWO_TRACE_RECEIPT_LIFECYCLE_COUNTEREXAMPLES_CONFIRMED','source_sha256':P.file_sha(m.MODULE),'tests':proof,'cpu_threads':4,'cuda_initialized':False,'real_model_loaded':False}
(D/'INDEPENDENT_COUNTEREXAMPLES.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
