from pathlib import Path
p=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-native-numeric-296217a/numerics-gpu-fixture.py')
s=p.read_text().replace("sys.path.insert(0,str(Path(__file__).parent/'repo'))", "EXPECTED_SOURCE='FIXED_SOURCE_PENDING'\nrepo=Path(os.environ['SEARCH_R1_REPO']).resolve()\nassert repo==Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric/repo')\nsys.path.insert(0,str(repo))\nimport subprocess\nassert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==EXPECTED_SOURCE\nassert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()")
s=s.replace('from search_r1.baselines.gradient_norm import clip_grad_norm64', "from search_r1.baselines.gradient_norm import clip_grad_norm64\nfrom verl.trainer.ppo import core_algos\nassert Path(gradient_norm.__file__).resolve()==repo/'search_r1/baselines/gradient_norm.py'\nassert Path(core_algos.__file__).resolve()==repo/'verl/trainer/ppo/core_algos.py'")
start=s.index('import importlib.util\n');end=s.index("out=Path(os.environ['NUMERIC_FIXTURE_OUTPUT'])",start)
kl='''# Actual GPU KL values and backward derivatives against the clipped FP64 function.
points=torch.cat((torch.linspace(-12,5,400,device=device),torch.tensor([-10000.,-11.,4.,80.,89.,90.,99.,10000.],device=device)))
current=(-points).requires_grad_()
actual=core_algos.kl_penalty(current,torch.zeros_like(current),'low_var_kl')
# Analytic saturated branches are constant10; bounded FP64 exp avoids oracle overflow.
oracle_current=(-points.double()).clamp(min=-700).detach().requires_grad_()
x=-oracle_current
oracle=(x.exp()-x-1).clamp(-10,10)
torch.testing.assert_close(actual.double(),oracle,rtol=1e-5,atol=1e-6)
actual.sum().backward();oracle.sum().backward()
torch.testing.assert_close(current.grad.double(),oracle_current.grad,rtol=1e-5,atol=1e-6)
assert torch.isfinite(current.grad).all()
records.append(dict(case='gpu_low_var_kl_saturated_forward_backward_fp64_oracle',points=points.numel(),max_value_error=float((actual.double()-oracle).abs().max()),max_grad_error=float((current.grad.double()-oracle_current.grad).abs().max())))
for bad in (float('nan'),float('inf'),float('-inf')):
 caught=False
 try:core_algos.kl_penalty(torch.tensor([bad],device=device),torch.zeros(1,device=device),'low_var_kl')
 except ValueError as error:caught='Nonfinite' in str(error)
 assert caught
records.append(dict(case='gpu_low_var_kl_nonfinite_inputs_refused',count=3))
'''
s=s[:start]+kl+s[end:]
s=s.replace(',diagnostics_records=diagnostics_records', ',source_commit=EXPECTED_SOURCE,core_algos_file=core_algos.__file__')
r=Path('work/numericfix/remote');r.mkdir(parents=True,exist_ok=True);(r/'numerics-gpu-fixture.py').write_text(s)
