"""Head-only read-only peer relay, strict target immutable SHA acceptance."""
import hashlib,json,pathlib,shlex,subprocess,time
source='/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/wiki/'
target='/dataset/shaoshitong/search-r1-rl/tree-grpo/assets-root/assets/wiki/'
assets=[('wiki-18.jsonl.gz',5123307260,'7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db'),('part_ab',21609402413,'b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11'),('part_aa',42949672960,'a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023')]
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
records=[]
for name,size,expected in assets:
 print('READ_ONLY_PEER_COPY_START',name,flush=True)
 code="""import hashlib,pathlib,sys,time,json
p=pathlib.Path(DEST);tmp=p.with_name(p.name+'.peer-copy.'+str(time.time_ns()));h=hashlib.sha256();size=0
with tmp.open('xb') as out:
 for chunk in iter(lambda:sys.stdin.buffer.read(8*1024**2),b''):
  out.write(chunk);h.update(chunk);size+=len(chunk)
assert size==SIZE and h.hexdigest()==SHA, 'Peer source bytes differ from immutable manifest; retain copy'
if p.exists():p.rename(p.with_name(p.name+'.previous.'+str(time.time_ns())))
tmp.replace(p)
print(json.dumps(dict(path=str(p),size=size,sha256=h.hexdigest(),source='read-only h100-11 TOPD own immutable assets')))
""".replace('DEST',repr(target+name)).replace('SIZE',repr(size)).replace('SHA',repr(expected))
 start=time.time()
 src=subprocess.Popen(['ssh',*opts,'h100-11','cat '+shlex.quote(source+name)],stdout=subprocess.PIPE)
 dst=subprocess.Popen(['ssh',*opts,'h100-9','/usr/bin/python3.10 -c '+shlex.quote(code)],stdin=src.stdout,stdout=subprocess.PIPE,text=True)
 src.stdout.close();out,_=dst.communicate();rc=src.wait()
 assert rc==0 and dst.returncode==0, 'Failed peer relay; keep own target partial'
 print(out.strip(),flush=True);record=json.loads(out);record['wall_seconds']=time.time()-start;records.append(record)
 pathlib.Path('/dev/shm/shaoshitong-tree-grpo-control-20261003/peer-assets-copied.json').write_text(json.dumps(records,indent=2))
print('ALL_PEER_ASSETS_TARGET_SHA_VERIFIED',flush=True)
verify=subprocess.run(['ssh',*opts,'h100-9','/usr/bin/python3.10 /dataset/shaoshitong/search-r1-rl/tree-grpo/tools/copy_direct_assets.py > /dataset/shaoshitong/search-r1-rl/tree-grpo/logs/peer-assets-full-verify.log 2>&1'])
assert verify.returncode==0, 'Complete immutable asset verification failed'
print('ALL_IMMUTABLE_ASSETS_READY_FOR_DERIVATION',flush=True)
