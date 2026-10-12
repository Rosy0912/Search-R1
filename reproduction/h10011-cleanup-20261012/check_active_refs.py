import pathlib,json,os,subprocess,re,time
base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
refs=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  maps=(p/'maps').read_text(errors='replace')
  hits=[l.split()[-1] for l in maps.splitlines() if str(base) in l]
  if hits:refs.append(dict(pid=p.name,kind='mmap',paths=hits))
  if str(base/'cache') in cmd:refs.append(dict(pid=p.name,kind='cmd_cache_reference'))
 except (OSError,PermissionError):pass
j=subprocess.run(['scontrol','show','job','-o','3387'],capture_output=True,text=True).stdout
fields={k:re.search(r'(?:^|\s)'+k+r'=(\S+)',j).group(1) for k in ['WorkDir','Command','StdOut'] if re.search(r'(?:^|\s)'+k+r'=(\S+)',j)}
print(json.dumps(dict(epoch=time.time(),references=refs,job3387=fields)))
