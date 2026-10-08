import pathlib,json,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');print(json.dumps({'epoch':time.time(),'files':[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size} for p in R.rglob('*') if p.is_file() and not any(k in str(p.relative_to(R)) for k in ('runtime/','cache/'))]}))
