"""Independent loader AST and fabricated CPU tensor checks; no actual HF/PT/GPU."""
import ast,hashlib,json,sys,tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import torch
ROOT=Path.cwd();CANDIDATE=ROOT/'work/tree-probability-probe-runner';HERE=ROOT/'work/tree-probability-probe-runner-review/loader-child'
sys.path.insert(0,str(CANDIDATE))
from search_r1.baselines import probability_probe_loader as L
from search_r1.baselines.probability_probe_capture import tensor_digest
SOURCE=CANDIDATE/'verl/workers/fsdp_workers.py';OLD=ROOT/'work/tree-safety110-a632ba9-v2/source/repo/verl/workers/fsdp_workers.py'
LOADER=CANDIDATE/'search_r1/baselines/probability_probe_loader.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(LOADER)=='939e83ba293e59f9fc9906cb77ad43732eee44640fc20307c36a4a36bc8d738d'
assert sha(SOURCE)=='bee26eb985f5245bc6f3bb9db658b29a5d1994d88c9c46b337dab27efb2e4432'
cases=[]
def ok(name,condition,detail=None):
 assert condition,(name,detail);cases.append(dict(name=name,passed=True,detail=detail))
def refused(name,call,match):
 try:call()
 except ValueError as e:ok(name,match in str(e),str(e))
 else:raise AssertionError(name+' was unexpectedly accepted')
def dump(x):return ast.dump(x,include_attributes=False)
method=next(n for n in next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ActorRolloutRefWorker').body if isinstance(n,ast.FunctionDef) and n.name=='_build_model_optimizer')
before,after,receipt=L.split_original_constructor(SOURCE,sha(SOURCE));prefix=before.body[0]
i=next(i for i,n in enumerate(method.body) if n.lineno==receipt['original_barrier_line']);j=next(i for i,n in enumerate(method.body) if n.lineno==receipt['omitted_optimizer_conditional_line'])
ok('frozen_original_worker_full_bytes_unchanged',SOURCE.read_bytes()==OLD.read_bytes())
ok('signature_defaults_and_argument_order_unchanged',dump(prefix.args)==dump(method.args))
ok('HF_prefix_statements_exact_original_AST',[dump(x) for x in prefix.body[:-1]]==[dump(x) for x in method.body[:i]])
ok('barrier_MP_FSDP_statements_exact_original_AST',[dump(x) for x in after.body]==[dump(x) for x in method.body[i:j]])
ok('prefix_returns_only_local_namespace',dump(prefix.body[-1])==dump(ast.parse('return locals()').body[0]))
compile(before,str(SOURCE),'exec');compile(after,str(SOURCE),'exec')
ok('both_AST_parts_compile',True)
kept=ast.Module(body=prefix.body[:-1]+after.body,type_ignores=[])
kept_calls=[getattr(n.func,'attr',None) for n in ast.walk(kept) if isinstance(n,ast.Call)]
ok('no_optimizer_scheduler_backward_update_calls_in_kept_AST',not any(x in kept_calls for x in ['AdamW','backward','step','update_actor','generate_sequences']) and not any(isinstance(n,ast.Name) and n.id=='get_constant_schedule_with_warmup' for n in ast.walk(kept)))
original_fsdps=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FSDP']
kept_fsdps=[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FSDP']
ok('FSDP_call_kwargs_exact_original',len(original_fsdps)==len(kept_fsdps)==1 and dump(original_fsdps[0])==dump(kept_fsdps[0]),[k.arg for k in kept_fsdps[0].keywords])

class Tiny(torch.nn.Module):
 def __init__(self,extras=0):
  super().__init__();self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Embedding(7,3)
  self.lm_head=torch.nn.Linear(3,7,bias=False);self.lm_head.weight=self.model.embed_tokens.weight
  self.extra=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(extras)])
 def forward(self,x):return x

# Fabricated full logical-count model: 2 serialized aliases + 433 distinct weights.
# Every tiny checkpoint value is BF16; actual model masters are real FP32 casts.
model=Tiny(433)
for p in model.parameters():p.data=p.data.to(torch.bfloat16).to(torch.float32)
saved={n:p.detach().to(torch.bfloat16).clone() for n,p in model.named_parameters(remove_duplicate=False)}
reads=[]
class Shard:
 def __enter__(self):return self
 def __exit__(self,*a):return False
 def keys(self):return list(saved)
 def get_tensor(self,name):reads.append(name);return saved[name]
fake=ModuleType('safetensors');fake.safe_open=lambda *a,**k:Shard()
with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'safetensors':fake}):
 (Path(tmp)/'fake.safetensors').touch();index={'weight_map':{n:'fake.safetensors' for n in saved}}
 targets,inventory=L.exhaustive_hf_cast_inventory(model,tmp,index)
 ok('synthetic_435_serialized_434_unique_all_keys_read',inventory['serialized_count']==435 and inventory['unique_parameter_count']==434 and set(reads)==set(saved) and len(reads)==435)
 ok('both_serialized_tied_keys_retained',all(n in reads and n in inventory['logical_records'] for n in ['model.embed_tokens.weight','lm_head.weight']))
 ok('actual_FP32_master_full_cast_bytes_bound',all(v['serialized']['dtype']=='torch.bfloat16' and v['loaded_master']['dtype']=='torch.float32' and v['loaded_master']==tensor_digest(dict(model.named_parameters(remove_duplicate=False))[n]) for n,v in inventory['logical_records'].items()))
 ok('effective_BF16_digests_bound_separately',all(v['expected_effective_bf16']['dtype']=='torch.bfloat16' and v['expected_effective_bf16']==tensor_digest(saved[n].to(torch.bfloat16)) for n,v in inventory['logical_records'].items()))
 ok('serialized_to_unique_maps_only_real_tied_pair',inventory['serialized_to_unique']['model.embed_tokens.weight']==inventory['serialized_to_unique']['lm_head.weight'] and len(set(inventory['serialized_to_unique'].values()))==434)
 bad=dict(index['weight_map']);bad.pop('lm_head.weight')
 refused('deleting_head_to_fit_count_refused',lambda:L.exhaustive_hf_cast_inventory(model,tmp,dict(weight_map=bad)),'count changed')
 saved['lm_head.weight'][0,0]+=1
 refused('changed_second_tied_serialized_value_refused',lambda:L.exhaustive_hf_cast_inventory(model,tmp,index),'master differs')

# Window claim uses alias primitive + complete shape/stride/offset equality,
# not object identity and not numerical equality/address coincidence over time.
m=Tiny();original=m.model.embed_tokens.weight
m.lm_head.weight=torch.nn.Parameter(original.detach().view_as(original))
with L.SameWindowTiedAliasTrace(m,m.model.embed_tokens,m.lm_head) as trace:m(torch.tensor([1]))
r=trace.receipt()
ok('different_Tensor_objects_same_full_storage_view_is_valid_alias',m.lm_head.weight is not m.model.embed_tokens.weight and r['tied_alias_identity_proven'])
ok('same_window_receipt_no_historical_identity_claim',r['historical_3268_identity_proven'] is False and r['records'][0]['same_window_alias'] is True)
m.lm_head.weight=torch.nn.Parameter(original.detach().clone())
trace=L.SameWindowTiedAliasTrace(m,m.model.embed_tokens,m.lm_head)
refused('equal_independent_storage_cannot_publish',lambda:trace.__enter__().__exit__(None,None,None),'absent/failed') # zero forwards fails regardless of value
trace=L.SameWindowTiedAliasTrace(m,m.model.embed_tokens,m.lm_head)
def equal_copy_forward():
 with trace:m(torch.tensor([1]))
refused('equal_copy_actual_callback_refused',equal_copy_forward,'alias evidence')
refused('failed_equal_copy_trace_receipt_refused',trace.receipt,'failed/incomplete')
m.lm_head.weight=original
trace=L.SameWindowTiedAliasTrace(m,m.model.embed_tokens,m.lm_head)
def two_calls():
 with trace:m(torch.tensor([1]));m(torch.tensor([2]))
refused('two_forward_windows_cannot_claim_one_window',two_calls,'alias evidence')
trace=L.SameWindowTiedAliasTrace(m,m.model.embed_tokens,m.lm_head)
with trace:m(torch.tensor([1]))
first=trace.receipt()
def failed_reuse():
 with trace:raise RuntimeError('synthetic failure before callback')
try:failed_reuse()
except RuntimeError:pass
refused('successful_old_trace_not_reusable_after_failure',trace.receipt,'failed/incomplete')
ok('all_alias_hooks_removed_after_success_and_failures',len(m._forward_pre_hooks)==0)
result=dict(status='INDEPENDENT_LOADER_AST_AND_TINY_CPU_CASES_PASSED',loader_sha256=sha(LOADER),original_worker_sha256=sha(SOURCE),frozen_worker_sha256=sha(OLD),construction_receipt=receipt,case_count=len(cases),cases=cases,actual_model_forwards=0,synthetic_cpu_model_forward_calls=5,actual_optimizer_writes=0,actual_HF_assets_loaded=False,actual_remote_calls=0,actual_GPU_or_FSDP_executed=False,scope='AST comparison to pinned original source and fabricated tiny CPU models/readers only. Full logical count uses 435 small generated checkpoint tensors, not HF assets. No historical actor or actual same-FSDP-window identity proof.')
(HERE/'INDEPENDENT_LOADER_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],case_count=len(cases))))
