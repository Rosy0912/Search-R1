import pathlib,subprocess,ast,types,torch,json,hashlib,copy
R=pathlib.Path(__file__).resolve().parents[2];G=R.parent/'users-shaoshitong-documents-remote-codex-control/work/gigpo';D=pathlib.Path(__file__).parent;COMMIT='4dc50b033929fc26715fee3526bffbf6055254d5'
get=lambda name:subprocess.check_output(['git','show',COMMIT+':'+name],cwd=G)
raw=get('search_r1/baselines/gradient_norm.py');(D/'candidate_gradient_norm.py').write_bytes(raw);ns={};exec(compile(raw,'candidate_gradient_norm.py','exec'),ns);clip=ns['clip_grad_norm64']
rawkl=get('verl/trainer/ppo/core_algos.py');tree=ast.parse(rawkl);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kl_penalty');k={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'candidate_core_algos','exec'),dict(torch=torch),k);kl=k['kl_penalty']
# Demonstrate two distinct possible numerical mechanisms, not actual failure cause.
x=torch.tensor([-100.],requires_grad=True);orig=(torch.exp(-x)+x-1).clamp(-10,10);orig.sum().backward();assert not torch.isfinite(x.grad).all()
y=torch.tensor([-100.],requires_grad=True);fixed=kl(y,torch.zeros_like(y),'low_var_kl');fixed.sum().backward();assert fixed.item()==10 and y.grad.item()==0
for dtype in [torch.float32,torch.float64,torch.bfloat16]:
 delta=torch.tensor([-10000.,-20.,-11.,-10.,-2.,0.,1.,2.,4.,5.,80.,90.,10000.],dtype=dtype);p=(-delta).requires_grad_();v=kl(p,torch.zeros_like(p),'low_var_kl');v.sum().backward();assert torch.isfinite(v).all() and torch.isfinite(p.grad).all()
 oracle_delta=delta.double().clamp(-11,4);oracle=(oracle_delta.exp()-oracle_delta-1).clamp(-10,10);assert torch.allclose(v.double(),oracle,rtol=.03 if dtype==torch.bfloat16 else 1e-5,atol=1e-6)
for bad in [float('nan'),float('inf'),-float('inf')]:
 try:kl(torch.tensor([bad]),torch.zeros(1),'low_var_kl')
 except ValueError:pass
 else:raise AssertionError('nonfinitehidden')
m=torch.nn.Linear(500,1,bias=False);m.weight.grad=torch.full_like(m.weight,1.6e20);assert torch.isfinite(m.weight.grad).all() and not torch.isfinite(torch.linalg.vector_norm(m.weight.grad.float()));before=m.weight.grad.clone();oracle=torch.linalg.vector_norm(before.double());n=clip(m,1);torch.testing.assert_close(n,oracle);torch.testing.assert_close(m.weight.grad,before*(1/(oracle+1e-6)).float())
for dtype in [torch.float32,torch.float64,torch.bfloat16]:
 m=torch.nn.Linear(8,4).to(dtype)
 for p in m.parameters():p.grad=torch.randn_like(p)
 original=copy.deepcopy(m)
 for p,q in zip(m.parameters(),original.parameters()):q.grad=p.grad.clone()
 rng=torch.get_rng_state().clone();n=clip(m,1);native=torch.nn.utils.clip_grad_norm_(original.parameters(),1);assert torch.equal(rng,torch.get_rng_state())
 for p,q in zip(m.parameters(),original.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=.03 if dtype==torch.bfloat16 else 1e-5,atol=1e-6)
m=torch.nn.Linear(2,1,bias=False);m.weight.grad=torch.tensor([[float('nan'),1.]])
try:clip(m,1)
except ValueError:pass
else:raise AssertionError('badgradienthidden')
assert torch.isnan(m.weight.grad[0,0]) and m.weight.grad[0,1]==1
out=dict(status='CPU_CANDIDATE_MECHANISMS_VERIFIED_NOT_TREE_ROOT_CAUSE_OR_DEPLOYMENT_ACCEPTANCE',candidate_commit=COMMIT,gradient_source_sha256=hashlib.sha256(raw).hexdigest(),ordinary_clip_tolerance_equivalence=True,FP64_norm_overflow_counterexample=True,saturated_KL_backward_counterexample=True,nonfinite_refused=True,RNG_unchanged=True,scope='Unsharded real CPU tensors only; no real FSDP collective/NCCL/GPU proof')
(D/'candidate-cpu-evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
