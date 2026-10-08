import sys,json,base64,gzip,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;j=json.loads(gzip.decompress(base64.b64decode(Path(sys.argv[1]).read_bytes())));(O/(Path(sys.argv[1]).stem+'_DECODED.json')).write_text(json.dumps(j,indent=2));root='/dataset/shaoshitong/search-r1-rl/tree-grpo/'
for k,v in j['files'].items():
 assert k.startswith(root) and hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'];rel=Path(k[len(root):]);assert '..' not in rel.parts;p=O/'actual3103-files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v['text'])
print('SHA verified/extracted',len(j['files']))
