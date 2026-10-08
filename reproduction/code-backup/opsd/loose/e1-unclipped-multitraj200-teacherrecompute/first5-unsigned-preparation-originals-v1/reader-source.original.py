P='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304/evidence/first5-actual-review-v1'
NAMES=['unsigned-step5-payload.json', 'receipt-preparation-audit.json', 'immutable-first5-train-prefix.out', 'producer-start.original.json', 'producer-exit.original.json', 'producer.stdout.original', 'producer.stderr.original', 'evidence-spec.json', 'root-review.json', 'independent-review.json']
import pathlib,json,base64,hashlib,os,stat
p=pathlib.Path(P);out={};assert p.resolve(strict=True)==p and p.stat().st_uid==os.getuid()==0
for name in NAMES:
 f=p/name;assert f.resolve(strict=True)==f;before=f.stat();assert stat.S_ISREG(before.st_mode) and before.st_uid==0 and before.st_nlink==1 and before.st_size<2000000
 with f.open('rb') as x:raw=x.read();opened=os.fstat(x.fileno())
 after=f.stat();keys=['st_dev','st_ino','st_size','st_uid','st_nlink','st_mtime_ns','st_ctime_ns'];assert len({tuple(getattr(s,k) for k in keys) for s in (before,opened,after)})==1
 out[name]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'physical':str(f),'stat':{k:getattr(before,k) for k in keys}}
print(json.dumps(out,sort_keys=True))
