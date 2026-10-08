import pathlib,json,hashlib
p=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177/logs/independent-step5-extraction-firstactual-20261004/evidence.json')
print(json.dumps({'exists':p.exists(),'bytes':p.stat().st_size if p.exists()else 0}))
if p.exists():print(p.read_text())
