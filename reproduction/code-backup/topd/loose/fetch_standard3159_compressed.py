import sys,pathlib,json,base64,hashlib,gzip
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
code=r'''
import pathlib,json,base64,hashlib,gzip
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v3/evidence');rows=[]
for p in sorted(r.glob('*.jsonl')):
 b=p.read_bytes();assert len(b)<40*1024**2;c=gzip.compress(b,compresslevel=6,mtime=0);rows.append(dict(name=p.name,path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),compressed_size=len(c),compressed_sha256=hashlib.sha256(c).hexdigest(),data=base64.b64encode(c).decode()))
print(json.dumps(rows))
'''
r=send(code,timeout=180);assert r.returncode==0,r.stderr.decode();rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-r1-profile3159')
for row in rows:
 c=base64.b64decode(row.pop('data'));assert hashlib.sha256(c).hexdigest()==row['compressed_sha256'];b=gzip.decompress(c);assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];(out/row['name']).write_bytes(b)
(out/'compressed-fetch-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print('REAL_FULL_JSONL_SHA_VERIFIED',len(rows),sum(x['size'] for x in rows),'compressed',sum(x['compressed_size'] for x in rows))
