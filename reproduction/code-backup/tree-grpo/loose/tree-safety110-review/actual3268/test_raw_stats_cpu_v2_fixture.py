"""Actual CPU execution of reviewer script on tiny files; never production PT."""
import ast, pathlib, tempfile, hashlib, json, contextlib, io, os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import torch
torch.set_num_threads(4)
D=pathlib.Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
src=(D/'raw_stats_cpu_v2.py').read_text()
old=(D/'raw_stats_cpu.py').read_bytes()
try:compile(old,'preserved_old','exec')
except SyntaxError as e:old_failure={'type':type(e).__name__,'message':str(e)}
else:raise AssertionError('old unexpectedly valid')
results=[]
with tempfile.TemporaryDirectory(prefix='tree-raw-v2-fixture-') as td:
 R=pathlib.Path(td);P=R/'output/probability-calibration';P.mkdir(parents=True)
 sample=torch.tensor([[-2.,-1.],[-3.,-4.]])
 current=sample+torch.tensor([[.02,-.03],[.001,.005]])
 d=(current.double()-sample.double()).flatten().repeat(8)
 decision=dict(valid_tokens=d.numel(),invalid_tokens=0,numeric_bound_exceeded=0,max_abs_delta=float(d.abs().max()),mean_abs_delta=float(d.abs().mean()),rms_delta=float(d.square().mean().sqrt()),sampled_k3_mean=float((torch.expm1(d)-d).mean()),allowed=False,reasons=['raw_turn_sampling_actor_max_abs_delta'])
 workers=[]
 for rank in range(8):
  sp=R/f'rank-{rank}.pt';jp=R/f'rank-{rank}.json'
  torch.save(dict(sampled_log_probs=sample,old_log_probs=current,responses=torch.tensor([[2,3],[4,5]]),input_ids=torch.tensor([[1,2,3],[1,4,5]]),attention_mask=torch.ones(2,3,dtype=torch.long),distribution={'temperature':1.}),sp)
  rec=dict(rank=rank,saved_rows=2,local_rows=2,enforced=False,generation_call=1,decision=decision)
  jp.write_text(json.dumps(rec))
  workers.append(dict(rank=rank,job_id='3268',code_commit='a632ba91840fb116aacfc0e770681a860647b295',config_sha256='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b',scheduler_step=110,restored_optimizer_steps=[2500],optimizer_committed=0,scheduler_advanced=False,diagnostic_entries=[dict(path=str(jp),sha256=sha(jp.read_bytes()),snapshot_path=str(sp),snapshot_sha256=sha(sp.read_bytes()))]))
 raw=json.dumps({'workers':workers},indent=2).encode();(P/'ready.json').write_bytes(raw)
 tree=ast.parse(src)
 changed=0
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='R' for t in n.targets):
   n.value=ast.Call(func=ast.Attribute(value=ast.Name(id='pathlib',ctx=ast.Load()),attr='Path',ctx=ast.Load()),args=[ast.Constant(str(R))],keywords=[]);changed+=1
 assert changed==1
 code=compile(ast.fix_missing_locations(tree),'actual_raw_v2_fixture','exec')
 out=io.StringIO()
 with contextlib.redirect_stdout(out):exec(code,{'__name__':'__main__'})
 report=json.loads(out.getvalue());assert report['ready_sha256']==sha(raw) and report['all_token_occurrences']==32 and report['calls'][0]['physical_rows']==16 and report['raw_max_row_token_bound']==8000 and not report['cuda_initialized']
 results.append(dict(case='actual_cpu_positive_and_exact_ready_bytes_binding',passed=True,ready_sha256=sha(raw),valid_tokens=32,physical_rows=16,max_abs_delta=report['all_max_abs']))
 sp=R/'rank-0.pt';sp.write_bytes(sp.read_bytes()+b'tamper')
 try:
  with contextlib.redirect_stdout(io.StringIO()):exec(code,{'__name__':'__main__'})
 except AssertionError:results.append(dict(case='snapshot_sha_tamper_rejected_before_load',passed=True))
 else:raise AssertionError('tamper accepted')
proof=dict(status='TINY_CPU_FIXTURE_PASS_NOT_ACTUAL3268_PT_ACCEPTANCE',old_script_sha256=sha(old),new_script_sha256=sha(src.encode()),only_change='remove one unmatched close parenthesis before allow_nan=False in final print',original_syntax_error=old_failure,new_ast_compile=True,trajectory_ast_compile=True,trajectory_sha256=sha((D/'tree_trajectory_cpu.py').read_bytes()),cases=results,cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads(),fixture_scope='8 ranks represented by CPU files, 16 rows/32 tokens; no distributed GPU/model/optimizer; production Ready/large PT not read')
compile((D/'tree_trajectory_cpu.py').read_text(),'tree_trajectory_cpu.py','exec')
assert not torch.cuda.is_initialized()
(D/'CPU_SCRIPT_V2_FIXTURE_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
