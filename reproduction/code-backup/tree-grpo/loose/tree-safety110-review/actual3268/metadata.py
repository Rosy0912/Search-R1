import pathlib,json,hashlib,time,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');out={'epoch':time.time(),'files':{}}
for n in ['resolved.yaml','output/entry-audit/entry-config.yaml','output/entry-audit/entry.jsonl','retriever-coexistence.json','retrieval-gate.json','output/recovery-loaded.json','stage.log','resource-samples.jsonl','process-samples.jsonl']:
 p=R/n
 if p.exists():
  b=p.read_bytes();out['files'][n]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
print(json.dumps(out))
