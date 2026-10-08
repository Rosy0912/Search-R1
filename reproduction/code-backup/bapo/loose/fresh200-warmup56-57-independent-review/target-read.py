import sys,json,hashlib,time,torch
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');RUN=R/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
sys.path.insert(0,str(R/'first5-evidence-tools-3315-v4'))
from first5_helpers import bind_source,digest
wall=time.perf_counter();cpu=time.process_time();bind_source()
from search_r1.baselines.fresh200_contract import validate_rank_updates,outer_contract
expected={'56': '8893bf32fa28bbac45cf88cfaabd6ed0a78332a9ed1796ebc10f311edfa66349', '57': '7dcd2861a02c946c4e3305f9243ca061b036518724ac3800cee549f9a5e03eab'}
results=[]
for n in (56,57):
 p=RUN/('checkpoints/diagnostics/attempt_%d_candidate_outer_%d/all-rank-write-ledger.json'%(n,n));raw=p.read_bytes();doc=json.loads(raw)
 assert hashlib.sha256(json.dumps(doc,sort_keys=True,allow_nan=False).encode()).hexdigest()==expected[str(n)]
 validate_rank_updates(doc['rank_write_ledger'],n)
 assert doc['driver_pid']==760122 and doc['progress']['completed_outers']==n and doc['progress']['actual_adam_writes_per_rank']==[1050 if n==56 else 1060]*8
 results.append(dict(outer=n,ledger_path=str(p),raw_sha256=hashlib.sha256(raw).hexdigest(),semantic_sha256=expected[str(n)],progress=doc['progress'],contract=outer_contract(n)))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_READONLY_TARGET_ACTUAL56_57_LEDGER_SOURCE',results=results,source_sha256={r:digest(R/'repo'/r) for r in ['search_r1/baselines/fresh200_contract.py','verl/workers/fsdp_workers.py','verl/utils/torch_functional.py']},cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,whole200_accepted=False),allow_nan=False))
