import tarfile,re,json,time
p='/mnt/data/shaoshitong/cleanup-backups-20261012-gigpo/necessary-logs.tar'; patterns={'hf_token':re.compile(rb'hf_[A-Za-z0-9]{20,}'),'github_token':re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),'openai_key':re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{20,}'),'private_key':re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----')};hits=[];files=0;size=0
with tarfile.open(p,'r|') as t:
 for m in t:
  if not m.isfile():continue
  files+=1;size+=m.size;f=t.extractfile(m);tail=b'';types=set()
  while True:
   data=f.read(1024*1024)
   if not data:break
   data=tail+data
   for name,pattern in patterns.items():
    if pattern.search(data):types.add(name)
   tail=data[-256:]
  if types:hits.append({'member':m.name,'patterns':sorted(types)})
print(json.dumps({'epoch':time.time(),'files':files,'bytes':size,'credential_pattern_hits':hits}))
