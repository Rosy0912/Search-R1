from pathlib import Path
import importlib.util,json,torch
p=Path('work/tree-instruct300-runtime-v2/source/repo/search_r1/baselines/training_state.py');s=importlib.util.spec_from_file_location('state_review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
mod=torch.nn.Linear(2,1,bias=False);opt=torch.optim.AdamW(mod.parameters(),lr=.01);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
x=torch.ones(1,2);mod(x).sum().backward();opt.step();sched.step()
param=next(mod.parameters());param._local_shard=param.detach().clone();saved=m.worker_snapshot(mod,opt,sched,{'rank':0});before=param.detach().clone()
with torch.no_grad():param.add_(1);param._local_shard.add_(1)
m.restore_worker(mod,opt,sched,saved,{'rank':0});visible=m.digest(m.worker_snapshot(mod,opt,sched,{'rank':0}))==m.digest(saved);internal=torch.equal(param._local_shard,before)
assert visible and not internal
rec={'status':'CPU_INDEPENDENT_STORAGE_RESTORE_GAP_REPRODUCED','visible_snapshot_exact_after_restore':visible,'internal_local_shard_restored':internal,'max_local_shard_delta':float((param._local_shard-before).abs().max()),'scope':'Real restore_worker with synthetic independent _local_shard storage; CPU only, does not establish actual3150 root cause or FSDP kernel determinism'}
Path('work/tree-instruct300-3150-review/LOCAL_SHARD_RESTORE_PROBE.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec)
