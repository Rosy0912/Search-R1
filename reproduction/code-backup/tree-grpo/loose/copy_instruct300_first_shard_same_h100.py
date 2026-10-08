import subprocess,shlex,json,time
src='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/assets/qwen2.5-3b-instruct-aa8e725/model-00001-of-00002.safetensors'
dst='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725/model-00001-of-00002.safetensors'
sha='67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2';size=3968658944
opts=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
def call(node,code):return opts+[node,'/usr/bin/python3.10 -c '+shlex.quote(code)]
proof=json.loads(subprocess.check_output(call('h100-5','import pathlib,hashlib,json;p=pathlib.Path('+repr(src)+');h=hashlib.sha256();\nwith p.open("rb") as f:\n for b in iter(lambda:f.read(8388608),b""):h.update(b)\nprint(json.dumps({"bytes":p.stat().st_size,"sha256":h.hexdigest()}))'),text=True));assert proof=={'bytes':size,'sha256':sha};print(json.dumps({'source_verified':proof,'node':'h100-5'}),flush=True)
offset=int(subprocess.check_output(call('h100-9','import pathlib;p=pathlib.Path('+repr(dst+'.part')+');print(p.stat().st_size if p.exists() else 0)'),text=True));assert offset<=size
reader='import sys;f=open('+repr(src)+',"rb");f.seek('+str(offset)+');\nfor b in iter(lambda:f.read(8388608),b""):sys.stdout.buffer.write(b)'
writer='''import pathlib,fcntl,sys,os,hashlib,json
p=pathlib.Path(DEST);t=p.with_name(p.name+'.part');lock=(p.parent.parent/(p.parent.name+'.tree-upload.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not p.exists();assert (t.stat().st_size if t.exists() else 0)==OFFSET
with t.open('ab' if OFFSET else 'xb') as f:
 for b in iter(lambda:sys.stdin.buffer.read(8388608),b''):f.write(b)
 f.flush();os.fsync(f.fileno())
assert t.stat().st_size==SIZE
h=hashlib.sha256()
with t.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==EXPECTED
os.link(t,p);t.unlink();print(json.dumps({'status':'SAME_H100_FIRST_SHARD_TARGET_SHA_PASS','bytes':SIZE,'sha256':EXPECTED,'path':str(p)}),flush=True)
'''.replace('DEST',repr(dst)).replace('OFFSET',str(offset)).replace('SIZE',str(size)).replace('EXPECTED',repr(sha))
print(json.dumps({'status':'SAME_H100_COPY_STARTED','resume_offset':offset,'epoch':time.time()}),flush=True)
a=subprocess.Popen(call('h100-5',reader),stdout=subprocess.PIPE)
b=subprocess.Popen(call('h100-9',writer),stdin=a.stdout);a.stdout.close();br=b.wait();ar=a.wait();assert ar==br==0,(ar,br)
