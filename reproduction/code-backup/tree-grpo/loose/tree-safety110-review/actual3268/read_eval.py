import pathlib,json,hashlib,time,re
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');files={}
for p in [R/'output/validation-cost/validation-564819-1.json',R/'output/validation-cost/validation-564819-1.jsonl']:
 b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
b=(R/'stage.log').read_bytes();lines=[l for l in b.decode().splitlines() if re.search(r'(?<![\w/])step:110 - ',l) or 'Initial validation metrics:' in l]
print(json.dumps({'epoch':time.time(),'files':files,'score_lines':lines,'stage_sha256':hashlib.sha256(b).hexdigest()}))
