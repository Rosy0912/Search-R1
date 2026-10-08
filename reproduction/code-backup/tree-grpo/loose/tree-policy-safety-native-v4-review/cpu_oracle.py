import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
import pathlib,hashlib,json,math,importlib.util
import torch
torch.set_num_threads(4)
P=pathlib.Path('work/tree-policy-safety-native-v4');V=pathlib.Path('work/tree-policy-safety-native-v3');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest.json')=='0935f213c61b70ea06faa2737ce31211693a4f9e848429a3b1c27ae7d76f4b1e'
assert sha(P/'start.sh')=='1f1ab84e04e31759c8724ae635dd91c6dcf4976c5af0d2c406d5487130621d59'
m=json.loads((P/'manifest.json').read_bytes())
for p,h in m['files'].items():assert sha(P/p)==h,p
changes=[p for p in m['files'] if sha(P/p)!=sha(V/p)];assert set(changes)=={'native-run.py','numerics-gpu-fixture.py'},changes
assert sha(P/'numerics-gpu-fixture.py')=='05005e51111457523ec9979201028b976e871d4b8f93125111d8a6211f6d6645'
spec=importlib.util.spec_from_file_location('fixed_safety',P/'repo/search_r1/baselines/policy_safety.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
inputs=[];stats=[];samples=[]
for rank in range(8):
 d=torch.linspace(-.2,1.2,rank,dtype=torch.float64);w=torch.full_like(d,(rank+1)/100);inputs.extend(zip(d.tolist(),w.tolist()))
 s,q=M.update_statistics(torch.zeros_like(d),d,-torch.ones_like(d),torch.ones_like(d),w);stats.append(s);samples.append(q)
globalstats={k:sum(s[k] for s in stats) for k in stats[0]};allowed=M.admission(globalstats,10.,1.);kl=M.admission(globalstats,1e-12,1.);tail=M.admission(globalstats,10.,.01)
manualmass=sum(w for d,w in inputs);manualkl=sum((math.expm1(d)-d)*w for d,w in inputs)/manualmass;tailmass=sum(w for d,w in inputs if d>math.log(3.));manualtail=tailmass/manualmass
assert abs(allowed['mass']-manualmass)<1e-14 and abs(allowed['sampled_old_policy_kl']-manualkl)<1e-14 and abs(allowed['negative_tail_fraction']-manualtail)<1e-14
assert allowed['allowed'] and kl['reasons']==['sampled_old_policy_kl'] and tail['reasons']==['negative_tail_fraction']
q=M.global_negative_quantiles(torch.cat(samples));assert q['quantile_sample_count']==28
stats=[]
for rank in range(8):
 d=torch.linspace(-.2,1.2,rank,dtype=torch.float64);w=torch.full_like(d,(rank+1)/100)
 if rank==7:d[0]=31.
 s,_=M.update_statistics(torch.zeros_like(d),d,-torch.ones_like(d),torch.ones_like(d),w);stats.append(s)
bound=M.admission({k:sum(s[k] for s in stats) for k in stats[0]},10.,1.);assert bound['overflow_count']==1 and bound['sampled_old_policy_kl'] is None and not bound['allowed']
proof={'status':'V4_LOCAL_BYTE_AND_CPU_ORACLE_PASS_NOT_CUDA','manifest_files':len(m['files']),'payload_diff':changes,'global_weighted_reference':{'mass':manualmass,'kl':manualkl,'tail_weight_fraction':manualtail,'negative_count':28},'quantiles':q,'allowed':allowed,'low_kl_refusal':kl,'low_tail_refusal':tail,'one_rank_bound_refusal':bound,'native_pending':True}
pathlib.Path('work/tree-policy-safety-native-v4-review/LOCAL_CPU_ORACLE.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
