import sys,json,hashlib,time,torch
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
sys.path.insert(0,str(ROOT/'first5-evidence-tools-3315-v4'))
from first5_helpers import bind_source,digest
wall=time.perf_counter();cpu=time.process_time();bind_source()
from search_r1.baselines.fresh200_contract import outer_contract,validate_rank_updates
from search_r1.baselines.fresh200_driver import Fresh200Progress
p=RUN/'checkpoints/diagnostics/attempt_50_candidate_outer_50/all-rank-write-ledger.json'
raw=p.read_bytes();x=json.loads(raw)
assert hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()=='6b1424eeba13ca43c69eb13babdea3d322be08df2b2d5252a8122a1913da2748'
assert x['driver_pid']==760122 and x['attempt_id']==50 and x['candidate_outer']==50
validate_rank_updates(x['rank_write_ledger'],50)
assert all(e['optimizer_steps']==10 and e['processed_opportunities']==10 and e['skipped_opportunities']==0 for e in x['rank_write_ledger'])
assert x['progress']['actual_adam_writes_per_rank']==[990]*8
previous=json.loads((RUN/'checkpoints/diagnostics/attempt_49_candidate_outer_49/all-rank-write-ledger.json').read_bytes())
assert previous['progress']['actual_adam_writes_per_rank']==[980]*8
assert all(e['optimizer_steps']==20 for e in previous['rank_write_ledger'])
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_READONLY_ACTUAL_OUTER50_LEDGER_SOURCE',ledger_path=str(p),ledger_sha256=hashlib.sha256(raw).hexdigest(),semantic_sha256='6b1424eeba13ca43c69eb13babdea3d322be08df2b2d5252a8122a1913da2748',contract49=outer_contract(49),contract50=outer_contract(50),previous_progress=previous['progress'],progress=x['progress'],source_sha256={r:digest(ROOT/'repo'/r) for r in ['search_r1/baselines/fresh200_contract.py','verl/trainer/ppo/ray_trainer.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py']},cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,whole200_accepted=False),allow_nan=False))
